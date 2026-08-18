#!/usr/bin/env bash
# Run InjectBench against every provider whose key is present in the
# environment, then regenerate the README charts.
#
#   export XAI_API_KEY=...        # Grok
#   export OPENAI_API_KEY=...     # GPT-5.6 Luna / Terra
#   export OPENROUTER_API_KEY=... # anything on OpenRouter
#   ./providers/run_all.sh
#
# Keys can also come from a .env file next to this repo (never committed).
set -uo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a; . ./.env; set +a
fi

ran=0
run() { # label, base-url, key, model
  local label="$1" url="$2" key="$3" model="$4"
  if [[ -z "${key:-}" ]]; then
    echo "skip  $label (no key)"
    return
  fi
  echo "run   $label ($model)"
  if python bench/run_bench.py live --base-url "$url" --api-key "$key" \
      --model "$model" --label "$model" \
      --out "bench/results/${model//\//_}.json"; then
    ran=$((ran + 1))
  else
    echo "FAILED $label — see the error above" >&2
  fi
}

run "xAI Grok"        "https://api.x.ai/v1"        "${XAI_API_KEY:-}"        "${GROK_MODEL:-grok-4}"
run "OpenAI GPT-5.6"  "https://api.openai.com/v1"  "${OPENAI_API_KEY:-}"     "${OPENAI_MODEL:-gpt-5.6-terra}"
run "OpenAI GPT-5.6"  "https://api.openai.com/v1"  "${OPENAI_API_KEY:-}"     "gpt-5.6-luna"
run "OpenRouter"      "https://openrouter.ai/api/v1" "${OPENROUTER_API_KEY:-}" "${OPENROUTER_MODEL:-openai/gpt-5.6-luna}"

if (( ran > 0 )); then
  python bench/make_charts.py
  echo
  echo "$ran run(s) complete. Results in bench/results/, charts in assets/."
else
  echo
  echo "No provider keys found. Set XAI_API_KEY / OPENAI_API_KEY / OPENROUTER_API_KEY,"
  echo "or try the offline path:  python bench/mock_provider.py --port 8099 &"
fi
