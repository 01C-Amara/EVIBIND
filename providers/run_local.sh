#!/usr/bin/env bash
# Run the injection benchmark against a local OpenAI-compatible server
# (vLLM, Ollama, LM Studio).
# Usage: ./providers/run_local.sh http://localhost:11434/v1 qwen3:8b
set -euo pipefail
BASE_URL="${1:?usage: run_local.sh <base-url> <model>}"
MODEL="${2:?usage: run_local.sh <base-url> <model>}"
cd "$(dirname "$0")/.."
python bench/run_bench.py live \
  --base-url "$BASE_URL" \
  --api-key "local" \
  --model "$MODEL" \
  --label "$MODEL" \
  --out "bench/results/${MODEL//[:\/]/_}.json"
python bench/make_charts.py
