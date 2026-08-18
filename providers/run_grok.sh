#!/usr/bin/env bash
# Run InjectBench live against xAI Grok over the HTTP API.
#
#   XAI_API_KEY=... ./providers/run_grok.sh [model]
#
# This needs an xAI *API key*. If your Grok access is a grok.com subscription
# used through the `grok` CLI, there is no API key and no OpenAI-compatible
# endpoint to point at -- use the CLI driver instead:
#
#   python bench/run_grok_cli.py --model grok-4.6 --concurrency 5
#
set -euo pipefail
MODEL="${1:-grok-4}"
: "${XAI_API_KEY:?set XAI_API_KEY, or use bench/run_grok_cli.py for the CLI}"
cd "$(dirname "$0")/.."
python bench/run_bench.py live \
  --base-url "https://api.x.ai/v1" \
  --api-key "env:XAI_API_KEY" \
  --model "$MODEL" \
  --label "$MODEL" \
  --concurrency "${CONCURRENCY:-8}" \
  --out "bench/results/${MODEL//\//_}.json"
python bench/make_charts.py
