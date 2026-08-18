#!/usr/bin/env bash
# Run InjectBench against every provider that is reachable here, then
# regenerate the README charts.
#
#   export XAI_API_KEY=...        # Grok over the xAI API
#   export OPENAI_API_KEY=...     # GPT-5.6 Luna / Terra / Sol
#   export OPENROUTER_API_KEY=... # anything on OpenRouter
#   ./providers/run_all.sh
#
# Keys can also come from a .env file next to this repo (never committed). Keys
# are passed as env: specs, so they do not appear in the process command line.
set -uo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a; . ./.env; set +a
fi

ran=0
run() { # label, base-url, key-var, model, [endpoint]
  local label="$1" url="$2" keyvar="$3" model="$4" endpoint="${5:-chat}"
  if [[ -z "${!keyvar:-}" ]]; then
    echo "skip  $label (no $keyvar)"
    return
  fi
  echo "run   $label ($model, $endpoint)"
  if python bench/run_bench.py live --base-url "$url" --api-key "env:$keyvar" \
      --model "$model" --label "$model" --endpoint "$endpoint" \
      ${REASONING_EFFORT:+--reasoning-effort "$REASONING_EFFORT"} \
      --concurrency "${CONCURRENCY:-8}" \
      --out "bench/results/${model//\//_}.json"; then
    ran=$((ran + 1))
  else
    echo "FAILED $label — see the error above" >&2
  fi
}

# The GPT-5.6 tiers reject function tools on /chat/completions; they go through
# the Responses API instead. Everything else uses chat completions.
run "xAI Grok"        "https://api.x.ai/v1"          XAI_API_KEY        "${GROK_MODEL:-grok-4}"
run "OpenAI GPT-5.6"  "https://api.openai.com/v1"    OPENAI_API_KEY     "${OPENAI_MODEL:-gpt-5.6-terra}" responses
run "OpenAI GPT-5.6"  "https://api.openai.com/v1"    OPENAI_API_KEY     "gpt-5.6-luna"                   responses
run "OpenAI GPT-5.4"  "https://api.openai.com/v1"    OPENAI_API_KEY     "gpt-5.4-mini"
run "OpenRouter"      "https://openrouter.ai/api/v1" OPENROUTER_API_KEY "${OPENROUTER_MODEL:-openai/gpt-5.6-luna}"

if (( ran > 0 )); then
  python bench/make_charts.py
  echo
  echo "$ran run(s) complete. Results in bench/results/, charts in assets/."
else
  echo
  echo "No provider keys found. Set XAI_API_KEY / OPENAI_API_KEY / OPENROUTER_API_KEY."
  echo
  echo "Grok through a grok.com subscription (no API key):"
  echo "  python bench/run_grok_cli.py --model grok-4.6 --concurrency 5"
  echo
  echo "No key at all:"
  echo "  python bench/mock_provider.py --port 8099 --mode last-mention &"
fi
