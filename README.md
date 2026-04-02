# llm-quant-benchmark

Stage 1 builds a runnable baseline for local LLM inference benchmarking with a fixed model, fixed prompts, and a fixed output schema. Stage 2 extends the same pipeline to `bitsandbytes` INT8 and INT4 without changing the result schema.
Stage 3 extends the same benchmark to pre-quantized local `AWQ` and `GPTQ` model directories.

## Stage 1 Scope

- Fixed base model: `Qwen2.5-3B-Instruct`
- Fixed backend: `transformers`
- Fixed precision baseline: `FP16`
- Fixed prompt set: `prompts/generation_smoke.jsonl`
- Fixed decoding policy: `do_sample=False`, `use_cache=True`
- Fixed output artifacts: CSV metrics plus JSONL generation samples

## Project Layout

```text
llm-quant-benchmark/
├── benchmarks/
├── prompts/
├── report/
├── results/
│   ├── raw/
│   └── samples/
├── scripts/
└── README.md
```

## Stage 1 Commands

Check the runtime and dependency status:

```bash
micromamba run -n llm_opt python scripts/check_env.py
```

Run the FP16 baseline benchmark:

```bash
micromamba run -n llm_opt python benchmarks/run_generation_benchmark.py \
  --model-path /share/home/wangzixu/liudinghao/gushuo/proj/transfoemer-llm/model/Qwen2.5-3B-Instruct \
  --model-name qwen2.5-3b-instruct \
  --precision fp16 \
  --backend transformers \
  --prompt-file prompts/generation_smoke.jsonl \
  --batch-size 1 \
  --max-new-tokens 64 \
  --warmup-runs 1 \
  --measure-runs 3 \
  --csv-out results/raw/fp16_baseline.csv \
  --sample-out results/samples/fp16_baseline.jsonl
```

## Stage 2 Commands

Install only `bitsandbytes` into `llm_opt` without upgrading existing packages:

```bash
bash scripts/install_bitsandbytes.sh
```

Run the entire Stage 2 comparison in one shot:

```bash
bash scripts/run_stage2_bitsandbytes.sh
```

Run INT8:

```bash
micromamba run -n llm_opt python benchmarks/run_generation_benchmark.py \
  --model-path /share/home/wangzixu/liudinghao/gushuo/proj/transfoemer-llm/model/Qwen2.5-3B-Instruct \
  --model-name qwen2.5-3b-instruct \
  --precision fp16 \
  --quant-method bnb_int8 \
  --backend transformers \
  --prompt-file prompts/generation_smoke.jsonl \
  --batch-size 1 \
  --max-new-tokens 64 \
  --warmup-runs 1 \
  --measure-runs 3 \
  --csv-out results/raw/bnb_int8.csv \
  --sample-out results/samples/bnb_int8.jsonl
```

Run INT4:

```bash
micromamba run -n llm_opt python benchmarks/run_generation_benchmark.py \
  --model-path /share/home/wangzixu/liudinghao/gushuo/proj/transfoemer-llm/model/Qwen2.5-3B-Instruct \
  --model-name qwen2.5-3b-instruct \
  --precision fp16 \
  --quant-method bnb_int4 \
  --backend transformers \
  --prompt-file prompts/generation_smoke.jsonl \
  --batch-size 1 \
  --max-new-tokens 64 \
  --warmup-runs 1 \
  --measure-runs 3 \
  --csv-out results/raw/bnb_int4.csv \
  --sample-out results/samples/bnb_int4.jsonl
```

Generate a markdown comparison table:

```bash
micromamba run -n llm_opt python scripts/summarize_results.py \
  --inputs results/raw/fp16_baseline.csv results/raw/bnb_int8.csv results/raw/bnb_int4.csv \
  --markdown-out report/stage2_bitsandbytes_summary.md \
  --title "Stage 2 BitsAndBytes Benchmark Summary"
```

## Stage 3 Commands

Install Stage 3 runtime deps into a repo-local vendor directory instead of modifying `llm_opt`:

```bash
bash scripts/install_stage3_quant_deps.sh
```

Run local AWQ and GPTQ benchmarks:

```bash
bash scripts/run_stage3_prequantized.sh
```

## Output Schema

Every experiment row uses the same CSV schema:

```text
timestamp
model_name
model_path
backend
quant_method
weight_dtype
batch_size
prompt_id
input_tokens
max_new_tokens
generated_tokens
ttft_ms
total_latency_ms
decode_tokens_per_s
request_tokens_per_s
peak_gpu_mem_mb
status
error_msg
```

Metric definitions in Stage 1:

- `ttft_ms`: time from `generate()` start to the first generated token observed by the streamer
- `total_latency_ms`: time from `generate()` start to generation completion
- `decode_tokens_per_s`: `generated_tokens / decode_time`
- `request_tokens_per_s`: `generated_tokens / total_time`
- `peak_gpu_mem_mb`: `torch.cuda.max_memory_allocated()` after resetting peak stats before each request

## Current Environment Notes

`llm_opt` currently has `torch==2.4.0+cu124`, `transformers==4.45.2`, and `bitsandbytes==0.43.3` available. Stage 3 uses a repo-local vendor directory for `autoawq`, `auto_gptq`, and `optimum` so the shared environment does not need to be upgraded.

For Stage 2, the benchmark keeps the same prompt file, batch size, and output schema. The only controlled change is `--quant-method`:

- `fp16`
- `bnb_int8`
- `bnb_int4`
- `awq`
- `gptq`

## Validation Checklist

- `scripts/check_env.py` prints package availability and CUDA visibility
- `benchmarks/run_generation_benchmark.py` loads from a local model path only
- Benchmark outputs:
  - `results/raw/fp16_baseline.csv`
  - `results/samples/fp16_baseline.jsonl`
- CSV rows can be reused by later quantization methods without schema changes
