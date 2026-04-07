# llm-quant-benchmark

A reproducible LLM inference quantization benchmark project comparing `FP16`, `bitsandbytes INT8`, `bitsandbytes INT4`, `AWQ`, and `GPTQ` on the same base model and prompt set.

The project is organized as an engineering experiment rather than a one-off notebook. It fixes model, prompts, decoding policy, and output schema, then records memory, latency, throughput, and generation samples in reusable CSV/JSONL artifacts.

## What This Project Measures

- `GPU memory`: peak allocated memory during a request
- `TTFT`: time to first token
- `Total latency`: end-to-end generation time
- `Decode throughput`: generated tokens per second after first token
- `Request throughput`: generated tokens per second over the whole request
- `Output samples`: fixed prompts for qualitative comparison

## Experiment Setup

- Base model: `Qwen2.5-3B-Instruct`
- Backend: `transformers`
- Prompt set: `prompts/generation_smoke.jsonl`
- Batch size: `1`
- Max new tokens: `64`
- Warmup runs: `1`
- Measure runs: `3`
- Decoding policy: `do_sample=False`, `use_cache=True`

## Final Result Summary

Source: [report/stage3_all_quant_summary.md](/share/home/wangzixu/liudinghao/gushuo/proj/llm-quant-benchmark/report/stage3_all_quant_summary.md)

| quant_method | rows | avg_ttft_ms | avg_total_latency_ms | avg_decode_tokens_per_s | avg_request_tokens_per_s | avg_peak_gpu_mem_mb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fp16 | 9 | 33.557 | 1841.770 | 35.406 | 34.760 | 6009.961 |
| bnb_int8 | 9 | 244.580 | 12975.846 | 4.948 | 4.855 | 3398.132 |
| bnb_int4 | 9 | 59.987 | 2428.496 | 27.026 | 26.358 | 2140.986 |
| awq | 9 | 55.881 | 3022.898 | 19.577 | 19.208 | 1984.557 |
| gptq | 9 | 95.042 | 4868.792 | 13.408 | 13.147 | 2095.575 |

## Key Findings

- `FP16` delivered the best latency and throughput, but used the most memory.
- `bitsandbytes INT4` had the best overall memory/throughput tradeoff in this stack.
- `AWQ` achieved the lowest memory footprint.
- `GPTQ` also reduced memory significantly, but was slower than `AWQ` and `bnb_int4`.
- `bitsandbytes INT8` reduced memory, but was dramatically slower in this environment and is not a good default deployment choice here.

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

## Core Commands

Check environment:

```bash
/share/home/wangzixu/.local/share/mamba/envs/llm_opt/bin/python scripts/check_env.py
```

Run FP16 baseline:

```bash
/share/home/wangzixu/.local/share/mamba/envs/llm_opt/bin/python benchmarks/run_generation_benchmark.py \
  --model-path /share/home/wangzixu/liudinghao/gushuo/proj/transfoemer-llm/model/Qwen2.5-3B-Instruct \
  --model-name qwen2.5-3b-instruct \
  --precision fp16 \
  --quant-method fp16 \
  --backend transformers \
  --prompt-file prompts/generation_smoke.jsonl \
  --batch-size 1 \
  --max-new-tokens 64 \
  --warmup-runs 1 \
  --measure-runs 3 \
  --csv-out results/raw/fp16_baseline.csv \
  --sample-out results/samples/fp16_baseline.jsonl
```

Install `bitsandbytes` only:

```bash
bash scripts/install_bitsandbytes.sh
```

Run Stage 2 (`FP16 + bnb INT8 + bnb INT4`):

```bash
bash scripts/run_stage2_bitsandbytes.sh
```

Install Stage 3 runtime deps into repo-local vendor path:

```bash
bash scripts/install_stage3_quant_deps.sh
```

Run Stage 3 (`AWQ + GPTQ`):

```bash
bash scripts/run_stage3_prequantized.sh
```

Regenerate the final summary table:

```bash
/share/home/wangzixu/.local/share/mamba/envs/llm_opt/bin/python scripts/summarize_results.py \
  --inputs results/raw/fp16_baseline.csv results/raw/bnb_int8.csv results/raw/bnb_int4.csv results/raw/awq.csv results/raw/gptq.csv \
  --markdown-out report/stage3_all_quant_summary.md \
  --title "Stage 3 All Quant Benchmark Summary"
```

## Output Schema

Every benchmark row uses the same schema:

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

## Reports

- [report/stage2_bitsandbytes_summary.md](/share/home/wangzixu/liudinghao/gushuo/proj/llm-quant-benchmark/report/stage2_bitsandbytes_summary.md)
- [report/stage3_all_quant_summary.md](/share/home/wangzixu/liudinghao/gushuo/proj/llm-quant-benchmark/report/stage3_all_quant_summary.md)
- [report/final_report.md](/share/home/wangzixu/liudinghao/gushuo/proj/llm-quant-benchmark/report/final_report.md)
- [report/resume_and_interview.md](/share/home/wangzixu/liudinghao/gushuo/proj/llm-quant-benchmark/report/resume_and_interview.md)
