# Provider runners

One-command benchmark runs against common providers. Each script calls
`bench/run_bench.py live` and writes a result JSON that
`bench/make_charts.py` picks up automatically.

These need network access and your own credentials — run them on your machine,
not in a sandbox.

| script | provider | credential |
|---|---|---|
| `run_openai_suite.sh` | the whole OpenAI line-up in one go | `.env` or `OPENAI_API_KEY` |
| `run_gpt56.sh` | one GPT-5.6 tier (Luna / Terra / Sol) | `.env` or `OPENAI_API_KEY` |
| `run_grok.sh` | xAI Grok over the HTTP API | `XAI_API_KEY` |
| `run_openrouter.sh` | any OpenRouter model | `OPENROUTER_API_KEY` |
| `run_local.sh` | vLLM / Ollama (OpenAI-compatible) | — |
| `run_all.sh` | everything reachable, then charts | whatever is set |

Keys can be passed as `file:PATH` or `env:NAME` instead of a literal, so they
never appear in a command line or shell history:

```bash
python bench/run_bench.py live --api-key file:.env  ...
python bench/run_bench.py live --api-key env:XAI_API_KEY ...
```

A `.env` must be `VAR=value`, one per line — a bare key on its own line cannot
be sourced by `run_all.sh`.

## Two OpenAI quirks the runners handle

**The GPT-5.6 tiers refuse function tools on `/v1/chat/completions`:**

```text
Function tools with reasoning_effort are not supported for gpt-5.6-terra in
/v1/chat/completions. To use function tools, use /v1/responses or set
reasoning_effort to 'none'.
```

Pass `--endpoint responses` (as `run_gpt56.sh` does) to drive them through the
Responses API; replies are rendered back into chat-completion shape, so scoring
is identical.

**OpenAI rejects a `tool` message that answers no assistant tool call.** 60
InjectBench cases hand the model an untrusted tool result without one, so
`--dialect openai` (the default) inserts the minimal assistant call that makes
the transcript well formed. Untrusted content is passed through byte for byte.
`--dialect native` sends cases verbatim.

## Grok

The `grok` CLI authenticates against a **grok.com subscription**, not an xAI API
key, and exposes no OpenAI-compatible endpoint — so `run_grok.sh` is only usable
if you separately hold an `XAI_API_KEY`. Check with `grok models`; if it says
"You are logged in with grok.com", use the CLI driver:

```bash
python bench/run_grok_cli.py --model grok-4.6 --concurrency 5
python bench/run_bench.py score \
    --responses bench/results/grok-4.6.responses.jsonl \
    --label grok-4.6 --out bench/results/grok-4.6.json
python bench/make_charts.py
```

The driver presents the tool as its real JSON Schema and constrains the CLI's
structured-output mode to it, so Grok fills the same slots under the same
constraints. It is an emulation of tool calling, not the native mechanism —
see [`docs/FINDINGS.md`](../docs/FINDINGS.md) §9 for what that does and does not
license. The CLI is a Windows binary, so run it from PowerShell or WSL on the
machine where it is installed.

## Gateway upstreams

The same base URLs work for the gateway itself:

```bash
export EVIBIND_UPSTREAM_BASE_URL="https://api.openai.com/v1"    # GPT-5.x
export EVIBIND_UPSTREAM_BASE_URL="https://api.x.ai/v1"          # Grok
export EVIBIND_UPSTREAM_BASE_URL="https://openrouter.ai/api/v1"
export EVIBIND_UPSTREAM_BASE_URL="http://localhost:11434/v1"    # Ollama
```

To see the whole path run against a live model in one command:

```bash
python examples/live_gateway_demo.py --model gpt-5.4-nano
```

## No key, no network

`bench/mock_provider.py` serves the same API locally and implements three
selector behaviours (`last-mention`, `first-mention`, `aligned`). It is how CI
keeps the live path working, and it reproduces the weak-selector row in the
README without any credentials.
