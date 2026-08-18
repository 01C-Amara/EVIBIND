#!/usr/bin/env bash
# Run the injection benchmark live against any OpenRouter model.
# Usage: OPENROUTER_API_KEY=... ./providers/run_openrouter.sh openai/gpt-5.6-luna
set -euo pipefail
MODEL="${1:?usage: run_openrouter.sh <vendor/model>}"
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}"
cd "$(dirname "$0")/.."
python bench/run_bench.py live \
  --base-url "https://openrouter.ai/api/v1" \
  --api-key "$OPENROUTER_API_KEY" \
  --model "$MODEL" \
  --label "$MODEL" \
  --out "bench/results/${MODEL//\//_}.json"
python bench/make_charts.py
