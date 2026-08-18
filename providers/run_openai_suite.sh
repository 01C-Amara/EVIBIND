#!/usr/bin/env bash
# Run InjectBench across the OpenAI line-up in one go.
#
#   ./providers/run_openai_suite.sh
#
# The key is read from .env (never passed on a command line). The GPT-5.6
# tiers refuse function tools on /chat/completions, so they are driven through
# /v1/responses; everything else uses chat completions.
set -uo pipefail
cd "$(dirname "$0")/.."

KEY="file:.env"
run() { # model, endpoint, effort
  local model="$1" endpoint="$2" effort="${3:-}"
  echo "=== $model ($endpoint) ==="
  python bench/run_bench.py live \
    --base-url "https://api.openai.com/v1" --api-key "$KEY" \
    --model "$model" --label "$model" \
    --endpoint "$endpoint" ${effort:+--reasoning-effort "$effort"} \
    --concurrency "${CONCURRENCY:-8}" \
    --out "bench/results/${model}.json" 2>&1 | tail -16
}

run gpt-5.6-terra responses medium
run gpt-5.6-luna  responses medium
run gpt-5.6-sol   responses medium
run gpt-5.4-mini  chat
run gpt-5.4-nano  chat
run gpt-4.1-mini  chat

python bench/make_charts.py
