#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/share/home/wangzixu/.local/share/mamba/envs/llm_opt/bin/python"
MODEL_PATH="/share/home/wangzixu/liudinghao/gushuo/proj/transfoemer-llm/model/Qwen2.5-3B-Instruct"
MODEL_NAME="qwen2.5-3b-instruct"
PROMPT_FILE="prompts/generation_smoke.jsonl"

"${PYTHON_BIN}" benchmarks/run_generation_benchmark.py \
  --model-path "${MODEL_PATH}" \
  --model-name "${MODEL_NAME}" \
  --precision fp16 \
  --quant-method fp16 \
  --backend transformers \
  --prompt-file "${PROMPT_FILE}" \
  --batch-size 1 \
  --max-new-tokens 64 \
  --warmup-runs 1 \
  --measure-runs 3 \
  --csv-out results/raw/fp16_baseline.csv \
  --sample-out results/samples/fp16_baseline.jsonl

"${PYTHON_BIN}" benchmarks/run_generation_benchmark.py \
  --model-path "${MODEL_PATH}" \
  --model-name "${MODEL_NAME}" \
  --precision fp16 \
  --quant-method bnb_int8 \
  --backend transformers \
  --prompt-file "${PROMPT_FILE}" \
  --batch-size 1 \
  --max-new-tokens 64 \
  --warmup-runs 1 \
  --measure-runs 3 \
  --csv-out results/raw/bnb_int8.csv \
  --sample-out results/samples/bnb_int8.jsonl

"${PYTHON_BIN}" benchmarks/run_generation_benchmark.py \
  --model-path "${MODEL_PATH}" \
  --model-name "${MODEL_NAME}" \
  --precision fp16 \
  --quant-method bnb_int4 \
  --backend transformers \
  --prompt-file "${PROMPT_FILE}" \
  --batch-size 1 \
  --max-new-tokens 64 \
  --warmup-runs 1 \
  --measure-runs 3 \
  --csv-out results/raw/bnb_int4.csv \
  --sample-out results/samples/bnb_int4.jsonl

"${PYTHON_BIN}" scripts/summarize_results.py \
  --inputs results/raw/fp16_baseline.csv results/raw/bnb_int8.csv results/raw/bnb_int4.csv \
  --markdown-out report/stage2_bitsandbytes_summary.md
