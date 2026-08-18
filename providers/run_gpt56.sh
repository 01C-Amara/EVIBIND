#!/usr/bin/env bash
# Run InjectBench live against an OpenAI GPT-5.6 tier.
#
#   ./providers/run_gpt56.sh [gpt-5.6-luna|gpt-5.6-terra|gpt-5.6-sol]
#
# These tiers refuse function tools on /v1/chat/completions unless reasoning is
# disabled ("Function tools with reasoning_effort are not supported ... use
# /v1/responses or set reasoning_effort to 'none'"), so they are driven through
# the Responses API and rendered back into chat-completion shape for scoring.
#
# The key is read from .env when present, so it never reaches a command line.
set -euo pipefail
MODEL="${1:-gpt-5.6-luna}"
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  KEY="file:.env"
else
  : "${OPENAI_API_KEY:?set OPENAI_API_KEY or create a .env}"
  KEY="env:OPENAI_API_KEY"
fi

python bench/run_bench.py live \
  --base-url "https://api.openai.com/v1" \
  --api-key "$KEY" \
  --model "$MODEL" \
  --label "$MODEL" \
  --endpoint responses \
  --reasoning-effort "${REASONING_EFFORT:-medium}" \
  --concurrency "${CONCURRENCY:-8}" \
  --out "bench/results/${MODEL}.json"
python bench/make_charts.py
