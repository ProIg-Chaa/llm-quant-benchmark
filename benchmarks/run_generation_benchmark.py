#!/usr/bin/env python3

import argparse
import csv
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch


def bootstrap_vendor_path() -> Optional[Path]:
    project_root = Path(__file__).resolve().parents[1]
    vendor_path = os.environ.get("LLM_QUANT_VENDOR_PATH")
    candidate = Path(vendor_path) if vendor_path else project_root / ".vendor" / "stage3"
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        return candidate
    return None


VENDOR_PATH = bootstrap_vendor_path()
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


CSV_FIELDS = [
    "timestamp",
    "model_name",
    "model_path",
    "backend",
    "quant_method",
    "weight_dtype",
    "batch_size",
    "prompt_id",
    "input_tokens",
    "max_new_tokens",
    "generated_tokens",
    "ttft_ms",
    "total_latency_ms",
    "decode_tokens_per_s",
    "request_tokens_per_s",
    "peak_gpu_mem_mb",
    "status",
    "error_msg",
]


class TimingTextIteratorStreamer(TextIteratorStreamer):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.generation_start_time: Optional[float] = None
        self.first_token_time: Optional[float] = None

    def set_generation_start_time(self, start_time: float) -> None:
        self.generation_start_time = start_time

    def put(self, value):
        if (
            self.generation_start_time is not None
            and self.first_token_time is None
            and not self.next_tokens_are_prompt
        ):
            if hasattr(value, "numel") and value.numel() > 0:
                self.first_token_time = time.perf_counter()
        super().put(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Stage 1 FP16 generation benchmark.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--precision", default="fp16")
    parser.add_argument(
        "--quant-method",
        default="fp16",
        choices=["fp16", "bnb_int8", "bnb_int4", "awq", "gptq"],
        help="Inference weight format. Stage 2 adds bitsandbytes; Stage 3 adds AWQ/GPTQ.",
    )
    parser.add_argument("--backend", default="transformers")
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measure-runs", type=int, default=3)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--sample-out", required=True)
    return parser.parse_args()


def timestamp_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def read_prompts(prompt_file: Path) -> List[Dict[str, str]]:
    prompts: List[Dict[str, str]] = []
    with prompt_file.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompt_id = item.get("prompt_id")
            prompt = item.get("prompt")
            if not prompt_id or not prompt:
                raise ValueError(f"Invalid prompt entry on line {line_no}: {item}")
            prompts.append({"prompt_id": prompt_id, "prompt": prompt})
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_file}")
    return prompts


def resolve_torch_dtype(precision: str) -> torch.dtype:
    precision = precision.lower()
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported precision: {precision}")


def resolve_weight_dtype(quant_method: str, precision: str) -> str:
    if quant_method == "fp16":
        return precision
    if quant_method == "bnb_int8":
        return "int8"
    if quant_method == "bnb_int4":
        return "int4"
    if quant_method == "awq":
        return "int4"
    if quant_method == "gptq":
        return "int4"
    raise ValueError(f"Unsupported quant_method: {quant_method}")


def format_prompt(tokenizer, prompt_text: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt_text}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt_text


def load_model_and_tokenizer(
    model_path: str,
    precision: str,
    quant_method: str,
    device: torch.device,
) -> Tuple[object, object]:
    torch_dtype = resolve_torch_dtype(precision)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    common_kwargs = {
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }

    if quant_method == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            **common_kwargs,
        )
        model.eval()
        model.to(device)
    elif quant_method in {"bnb_int8", "bnb_int4"}:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:
            raise RuntimeError(
                "bitsandbytes quantization requires transformers BitsAndBytesConfig. "
                "Install bitsandbytes first."
            ) from exc

        if quant_method == "bnb_int8":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=False,
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map={"": 0} if device.type == "cuda" else "cpu",
            torch_dtype=torch_dtype,
            **common_kwargs,
        )
        model.eval()
    elif quant_method in {"awq", "gptq"}:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map={"": 0} if device.type == "cuda" else "cpu",
            torch_dtype=torch_dtype,
            **common_kwargs,
        )
        model.eval()
    else:
        raise ValueError(f"Unsupported quant_method: {quant_method}")

    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    return model, tokenizer


def generate_once(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int,
    device: torch.device,
) -> Dict[str, object]:
    formatted_prompt = format_prompt(tokenizer, prompt_text)
    model_inputs = tokenizer([formatted_prompt], return_tensors="pt", padding=True)
    model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
    input_tokens = int(model_inputs["input_ids"].shape[1])

    streamer = TimingTextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=120.0,
    )
    generated: Dict[str, object] = {"output_ids": None, "error": None}

    def _worker() -> None:
        try:
            with torch.inference_mode():
                generated["output_ids"] = model.generate(
                    **model_inputs,
                    streamer=streamer,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        except Exception as exc:
            generated["error"] = exc
            streamer.end()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    synchronize_if_needed(device)
    start_time = time.perf_counter()
    streamer.set_generation_start_time(start_time)

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    chunks: List[str] = []
    for text_chunk in streamer:
        chunks.append(text_chunk)

    worker.join()
    synchronize_if_needed(device)
    end_time = time.perf_counter()

    error = generated["error"]
    if error is not None:
        raise error

    output_ids = generated["output_ids"]
    if output_ids is None:
        raise RuntimeError("Model.generate returned no outputs.")

    generated_tokens = int(output_ids.shape[1] - input_tokens)
    new_token_ids = output_ids[0, input_tokens:]
    output_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)

    ttft_ms: Optional[float]
    if streamer.first_token_time is None:
        ttft_ms = None
    else:
        ttft_ms = (streamer.first_token_time - start_time) * 1000.0

    total_latency_ms = (end_time - start_time) * 1000.0
    decode_time_s = max((end_time - (streamer.first_token_time or start_time)), 1e-9)
    total_time_s = max((end_time - start_time), 1e-9)
    decode_tokens_per_s = generated_tokens / decode_time_s if generated_tokens > 0 else 0.0
    request_tokens_per_s = generated_tokens / total_time_s if generated_tokens > 0 else 0.0

    return {
        "input_tokens": input_tokens,
        "generated_tokens": generated_tokens,
        "ttft_ms": ttft_ms,
        "total_latency_ms": total_latency_ms,
        "decode_tokens_per_s": decode_tokens_per_s,
        "request_tokens_per_s": request_tokens_per_s,
        "peak_gpu_mem_mb": peak_memory_mb(device),
        "output_text": output_text,
        "streamed_text": "".join(chunks),
    }


def append_csv_rows(csv_path: Path, rows: Iterable[Dict[str, object]]) -> None:
    ensure_parent(csv_path)
    file_exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_jsonl(sample_path: Path, records: Iterable[Dict[str, object]]) -> None:
    ensure_parent(sample_path)
    with sample_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_base_row(args: argparse.Namespace, prompt_id: str) -> Dict[str, object]:
    return {
        "timestamp": timestamp_now(),
        "model_name": args.model_name,
        "model_path": args.model_path,
        "backend": args.backend,
        "quant_method": args.quant_method,
        "weight_dtype": resolve_weight_dtype(args.quant_method, args.precision),
        "batch_size": args.batch_size,
        "prompt_id": prompt_id,
        "input_tokens": None,
        "max_new_tokens": args.max_new_tokens,
        "generated_tokens": None,
        "ttft_ms": None,
        "total_latency_ms": None,
        "decode_tokens_per_s": None,
        "request_tokens_per_s": None,
        "peak_gpu_mem_mb": None,
        "status": "error",
        "error_msg": "",
    }


def main() -> int:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Stage 1 benchmark only supports --batch-size 1.")

    model_path = Path(args.model_path)
    prompt_file = Path(args.prompt_file)
    csv_out = Path(args.csv_out)
    sample_out = Path(args.sample_out)

    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file does not exist: {prompt_file}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prompts = read_prompts(prompt_file)

    print(f"[info] loading model from {model_path}")
    print(f"[info] device={device}")
    print(f"[info] vendor_path={VENDOR_PATH if VENDOR_PATH else '<none>'}")
    model, tokenizer = load_model_and_tokenizer(
        str(model_path),
        args.precision,
        args.quant_method,
        device,
    )
    print("[info] model loaded")

    sample_prompt = prompts[0]["prompt"]
    for warmup_idx in range(args.warmup_runs):
        print(f"[warmup] run={warmup_idx + 1}/{args.warmup_runs}")
        _ = generate_once(
            model=model,
            tokenizer=tokenizer,
            prompt_text=sample_prompt,
            max_new_tokens=min(args.max_new_tokens, 16),
            device=device,
        )

    sample_records: List[Dict[str, object]] = []
    seen_prompt_ids = set()

    for prompt in prompts:
        prompt_id = prompt["prompt_id"]
        prompt_text = prompt["prompt"]
        print(f"[measure] prompt_id={prompt_id}")

        rows: List[Dict[str, object]] = []
        for measure_idx in range(args.measure_runs):
            row = build_base_row(args, prompt_id)
            row["timestamp"] = timestamp_now()
            try:
                result = generate_once(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                )
                row.update(
                    {
                        "input_tokens": result["input_tokens"],
                        "generated_tokens": result["generated_tokens"],
                        "ttft_ms": f"{result['ttft_ms']:.3f}" if result["ttft_ms"] is not None else "",
                        "total_latency_ms": f"{result['total_latency_ms']:.3f}",
                        "decode_tokens_per_s": f"{result['decode_tokens_per_s']:.3f}",
                        "request_tokens_per_s": f"{result['request_tokens_per_s']:.3f}",
                        "peak_gpu_mem_mb": f"{result['peak_gpu_mem_mb']:.3f}",
                        "status": "ok",
                        "error_msg": "",
                    }
                )
                if prompt_id not in seen_prompt_ids:
                    sample_records.append(
                        {
                            "timestamp": row["timestamp"],
                            "model_name": args.model_name,
                            "quant_method": args.quant_method,
                            "prompt_id": prompt_id,
                            "prompt": prompt_text,
                            "output": result["output_text"],
                        }
                    )
                    seen_prompt_ids.add(prompt_id)
            except Exception as exc:
                row["error_msg"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            print(
                f"[measure] prompt_id={prompt_id} run={measure_idx + 1}/{args.measure_runs} "
                f"status={row['status']}"
            )
        append_csv_rows(csv_out, rows)

    if sample_records:
        append_jsonl(sample_out, sample_records)

    print(f"[done] csv_out={csv_out}")
    print(f"[done] sample_out={sample_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
