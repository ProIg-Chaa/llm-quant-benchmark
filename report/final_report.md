# LLM Quantization Benchmark Report

## Summary

This project benchmarks five inference deployment variants on the same `Qwen2.5-3B-Instruct` model:

- `FP16`
- `bitsandbytes INT8`
- `bitsandbytes INT4`
- `AWQ`
- `GPTQ`

The benchmark fixes model, prompt set, decoding policy, batch size, and output length, then compares memory, latency, and throughput under a unified measurement pipeline.

## Experiment Design

Controlled variables:

- Base model: `Qwen2.5-3B-Instruct`
- Prompt file: `prompts/generation_smoke.jsonl`
- Batch size: `1`
- Max new tokens: `64`
- Warmup runs: `1`
- Measure runs: `3`
- Decode policy: greedy decoding with `do_sample=False`

Metrics:

- `peak_gpu_mem_mb`
- `ttft_ms`
- `total_latency_ms`
- `decode_tokens_per_s`
- `request_tokens_per_s`

Outputs:

- Raw metrics in `results/raw/*.csv`
- Generation samples in `results/samples/*.jsonl`

## Final Results

| quant_method | avg_ttft_ms | avg_total_latency_ms | avg_decode_tokens_per_s | avg_request_tokens_per_s | avg_peak_gpu_mem_mb |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp16 | 33.557 | 1841.770 | 35.406 | 34.760 | 6009.961 |
| bnb_int8 | 244.580 | 12975.846 | 4.948 | 4.855 | 3398.132 |
| bnb_int4 | 59.987 | 2428.496 | 27.026 | 26.358 | 2140.986 |
| awq | 55.881 | 3022.898 | 19.577 | 19.208 | 1984.557 |
| gptq | 95.042 | 4868.792 | 13.408 | 13.147 | 2095.575 |

## Relative Change vs FP16

| quant_method | memory reduction | TTFT change | total latency change | decode throughput change |
| --- | ---: | ---: | ---: | ---: |
| bnb_int8 | 43.5% lower | 628.8% higher | 604.5% higher | 86.0% lower |
| bnb_int4 | 64.4% lower | 78.8% higher | 31.9% higher | 23.7% lower |
| awq | 67.0% lower | 66.5% higher | 64.1% higher | 44.7% lower |
| gptq | 65.1% lower | 183.2% higher | 164.4% higher | 62.1% lower |

## Analysis

### 1. FP16 is still the latency/throughput upper bound

`FP16` achieved the best TTFT, the best end-to-end latency, and the best decode throughput. This is expected because it avoids quantization-specific kernel overhead and compatibility layers.

### 2. bnb INT4 is the best tradeoff in this stack

`bnb_int4` reduced memory from `6009.961 MB` to `2140.986 MB` while keeping latency and throughput in a still-usable range. It was the strongest practical compromise between deployment efficiency and runtime performance.

### 3. AWQ has the lowest memory footprint

`AWQ` achieved the smallest peak memory usage at `1984.557 MB`, slightly below `bnb_int4` and `gptq`. This makes it attractive for memory-constrained single-GPU deployment.

### 4. GPTQ works, but is slower than AWQ and bnb INT4 here

`GPTQ` reduced memory substantially, but its TTFT and total latency were noticeably worse than `AWQ` and `bnb_int4`. In this environment it is usable, but not the best default deployment option.

### 5. bnb INT8 is a bad default choice in this environment

`bnb_int8` reduced memory versus `FP16`, but the latency penalty was extreme. This experiment is a good reminder that lower precision does not automatically imply faster inference. Kernel implementation quality and backend integration matter more than bit width alone.

## Engineering Work Done

- Built a unified benchmark runner for `FP16`, `bnb INT8`, `bnb INT4`, `AWQ`, and `GPTQ`
- Standardized metric output into reusable CSV/JSONL artifacts
- Added environment inspection and stage-specific dependency checks
- Added result aggregation into markdown summary tables
- Fixed multiple compatibility issues across `autoawq` and `auto_gptq` vendor code so local pre-quantized models could run under the current stack

## Limitations

- Quality evaluation is still based on generation samples only; perplexity or task accuracy is not yet added
- Measurements are from a fixed prompt set and `batch_size=1`
- Results are environment-specific and should not be generalized across backends without re-running

## Recommended Next Steps

- Add perplexity evaluation on a held-out text corpus
- Add batch throughput experiments for `batch_size > 1`
- Add longer-context tests to separate prefill bottlenecks from decode bottlenecks
- Optionally benchmark `vLLM` or `TensorRT-LLM` on the same model for backend-level comparison
