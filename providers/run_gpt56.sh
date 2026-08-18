#!/usr/bin/env bash
# Run the injection benchmark live against OpenAI GPT-5.6 tiers.
# Usage: OPENAI_API_KEY=... ./providers/run_gpt56.sh [gpt-5.6-luna|gpt-5.6-terra|gpt-5.6-sol]
set -euo pipefail
MODEL="${1:-gpt-5.6-luna}"
: "${OPENAI_API_KEY:?set OPENAI_API_KEY}"
cd "$(dirname "$0")/.."
python bench/run_bench.py live \
  --base-url "https://api.openai.com/v1" \
  --api-key "$OPENAI_API_KEY" \
  --model "$MODEL" \
  --label "$MODEL" \
  --out "bench/results/${MODEL}.json"
python bench/make_charts.py
