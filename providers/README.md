# Provider runners

One-command benchmark runs against common providers. Each script calls
`bench/run_bench.py live` and writes a result JSON that
`bench/make_charts.py` picks up automatically.

These need network access and your own API key — run them on your machine,
not in a sandbox.

| script | provider | env var |
|---|---|---|
| `run_grok.sh` | xAI (Grok) | `XAI_API_KEY` |
| `run_gpt56.sh` | OpenAI GPT-5.6 Luna / Terra | `OPENAI_API_KEY` |
| `run_openrouter.sh` | any OpenRouter model | `OPENROUTER_API_KEY` |
| `run_local.sh` | vLLM / Ollama (OpenAI-compatible) | — |

## Grok via the `grok` CLI (offline path)

If you prefer your local `grok` CLI to raw API calls, use the export/score
path: `python bench/run_bench.py export` writes `bench/results/requests.jsonl`;
answer each request with the CLI (one OpenAI-style completion per line, keyed
by `case_id`); then `python bench/run_bench.py score --responses ... --label grok`.

## Gateway upstreams

The same base URLs work for the gateway itself — protection is identical for
every provider:

```bash
export EVIBIND_UPSTREAM_BASE_URL="https://api.x.ai/v1"        # Grok
export EVIBIND_UPSTREAM_BASE_URL="https://api.openai.com/v1"  # GPT-5.6 Luna/Terra
export EVIBIND_UPSTREAM_BASE_URL="https://openrouter.ai/api/v1"
export EVIBIND_UPSTREAM_BASE_URL="http://localhost:11434/v1"  # Ollama
```

## Running Grok from Windows (the `grok.exe` CLI)

The CLI is a Windows binary, so run this from PowerShell or WSL on the machine
where it is installed — not from a Linux sandbox that only mounts your folders.

```powershell
# 1. export the 150 model-visible prompts
python bench\run_bench.py export --out bench\results\requests.jsonl

# 2. answer each one with the CLI, writing one JSON line per case:
#    {"case_id": "...", "response": {<OpenAI-style chat completion>}}
#    (any script that loops over requests.jsonl works)

# 3. score both arms and refresh the charts
python bench\run_bench.py score --responses bench\results\responses_grok.jsonl `
    --label grok-4 --out bench\results\grok-4.json
python bench\make_charts.py
```

If the CLI exposes an OpenAI-compatible endpoint, skip the export/score dance
and point `run_bench.py live` at it directly.

## No key, no network

`bench/mock_provider.py` serves the same API locally and implements three
selector behaviours (`last-mention`, `first-mention`, `aligned`). It is how CI
keeps the live path working, and it reproduces the weak-selector row in the
README without any credentials.
