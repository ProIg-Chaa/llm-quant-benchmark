# Resume and Interview Notes

## Resume Version

Built an end-to-end LLM quantization benchmark system for `Qwen2.5-3B-Instruct`, comparing `FP16`, `bitsandbytes INT8/INT4`, `AWQ`, and `GPTQ` under a unified `transformers`-based inference pipeline. Standardized memory, TTFT, total latency, and throughput collection into CSV/JSONL artifacts, and reproduced that `bnb INT4` delivered the best memory/performance tradeoff while `AWQ` achieved the lowest memory footprint. Resolved multiple real-world compatibility issues across `autoawq` and `auto_gptq` vendor code to make pre-quantized local models runnable in the benchmark stack.

## Short Resume Bullets

- Designed and implemented a reusable LLM inference quantization benchmark covering `FP16`, `bnb INT8/INT4`, `AWQ`, and `GPTQ`.
- Standardized GPU memory, TTFT, total latency, and throughput collection into a unified evaluation pipeline and report format.
- Benchmarked `Qwen2.5-3B-Instruct` and identified `bnb INT4` as the best deployment tradeoff and `AWQ` as the lowest-memory option in the tested stack.
- Debugged and patched `autoawq` and `auto_gptq` compatibility issues to support local pre-quantized model loading and evaluation.

## Interview Talking Points

### Why this project matters

The main value is not “I ran quantization once”, but “I built a controlled and repeatable deployment benchmark”. That is the difference between model tinkering and infra engineering.

### What was controlled

- Same base model
- Same prompt set
- Same decode policy
- Same batch size and output length
- Same metric schema across all methods

### What I learned

- Lower precision does not guarantee lower latency
- Memory savings and runtime savings are separate optimization targets
- Backend/kernel maturity is often more important than the nominal quantization bit width
- Quantized model evaluation often fails first on tooling and compatibility, not on theory

### Strongest project takeaway

In this setup, `bnb INT4` was the best practical deployment compromise, `AWQ` minimized memory, and `bnb INT8` underperformed badly on latency despite lower memory than `FP16`.

### If asked about hard parts

- `autoawq` imported unsupported `qwen3` modules under the current `transformers` version
- `auto_gptq` had multiple loading-path issues, including optional dependency coupling and local config handling
- The benchmark needed a unified interface across native `transformers`, `bitsandbytes`, `AWQ`, and `GPTQ` loaders

### If asked how to extend it

- Add perplexity and task evaluation
- Add batch and long-context experiments
- Compare against optimized serving backends such as `vLLM` or `TensorRT-LLM`
