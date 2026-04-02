#!/usr/bin/env bash
set -euo pipefail

# Stage 2 policy: only add bitsandbytes, do not upgrade existing packages.
micromamba run -n llm_opt python -m pip install --no-deps bitsandbytes==0.43.3
