#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/share/home/wangzixu/.local/share/mamba/envs/llm_opt/bin/python"
export LLM_QUANT_VENDOR_PATH="${PROJECT_ROOT}/.vendor/stage3"

AWQ_MODEL="/share/home/wangzixu/liudinghao/gushuo/proj/transfoemer-llm/model/Qwen2.5-3B-Instruct-AWQ"
GPTQ_MODEL="/share/home/wangzixu/liudinghao/gushuo/proj/transfoemer-llm/model/Qwen2.5-3B-Instruct-GPTQ-Int4"
PROMPT_FILE="${PROJECT_ROOT}/prompts/generation_smoke.jsonl"

"${PYTHON_BIN}" benchmarks/run_generation_benchmark.py \
  --model-path "${AWQ_MODEL}" \
  --model-name qwen2.5-3b-instruct-awq \
  --precision fp16 \
  --quant-method awq \
  --backend transformers \
  --prompt-file "${PROMPT_FILE}" \
  --batch-size 1 \
  --max-new-tokens 64 \
  --warmup-runs 1 \
  --measure-runs 3 \
  --csv-out results/raw/awq.csv \
  --sample-out results/samples/awq.jsonl

"${PYTHON_BIN}" benchmarks/run_generation_benchmark.py \
  --model-path "${GPTQ_MODEL}" \
  --model-name qwen2.5-3b-instruct-gptq-int4 \
  --precision fp16 \
  --quant-method gptq \
  --backend transformers \
  --prompt-file "${PROMPT_FILE}" \
  --batch-size 1 \
  --max-new-tokens 64 \
  --warmup-runs 1 \
  --measure-runs 3 \
  --csv-out results/raw/gptq.csv \
  --sample-out results/samples/gptq.jsonl

"${PYTHON_BIN}" scripts/summarize_results.py \
  --inputs results/raw/fp16_baseline.csv results/raw/bnb_int8.csv results/raw/bnb_int4.csv results/raw/awq.csv results/raw/gptq.csv \
  --markdown-out report/stage3_all_quant_summary.md \
  --title "Stage 3 All Quant Benchmark Summary"
