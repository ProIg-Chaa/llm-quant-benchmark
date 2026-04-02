#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/share/home/wangzixu/.local/share/mamba/envs/llm_opt/bin/python"
TARGET_DIR="${PROJECT_ROOT}/.vendor/stage3"

mkdir -p "${TARGET_DIR}"

# Stage 3 policy: keep llm_opt stable; install AWQ/GPTQ runtime deps into repo-local vendor path.
"${PYTHON_BIN}" -m pip install --upgrade --no-deps \
  --target "${TARGET_DIR}" \
  autoawq==0.2.9 \
  optimum==2.1.0 \
  auto-gptq==0.7.1
