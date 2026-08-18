#!/usr/bin/env bash
# Run the injection benchmark live against xAI Grok.
# Usage: XAI_API_KEY=... ./providers/run_grok.sh [model]
set -euo pipefail
MODEL="${1:-grok-4}"
: "${XAI_API_KEY:?set XAI_API_KEY}"
cd "$(dirname "$0")/.."
python bench/run_bench.py live \
  --base-url "https://api.x.ai/v1" \
  --api-key "$XAI_API_KEY" \
  --model "$MODEL" \
  --label "$MODEL" \
  --out "bench/results/${MODEL//\//_}.json"
python bench/make_charts.py
