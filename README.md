<p align="center">
  <img src="assets/logo.svg" alt="EviBind — evidence-bound tool calls" width="420">
</p>

<p align="center">
  <b>An argument-level firewall for LLM tool calls.</b><br>
  Untrusted content can propose; only trusted replay materializes an executable critical value.
</p>

---

A schema-valid tool call can still use an action-critical value from the wrong
source: a prompt-injected account number is perfectly valid JSON. EviBind is an
OpenAI-compatible gateway that removes executable critical literals from the
model interface. For slots you mark critical (account references, recipients,
paths, amounts), the gateway compiles a request-local table of typed evidence
derivations, lets the model select — and then trusted code alone re-derives and
releases the value, with a signed certificate per release. If no admissible
evidence supports the call, it fails closed and asks for input instead.

Built from the reference implementation of the EviBind research artifact
(ICLR 2027 submission, under review). The theory, frozen experiments, and the
BoundaryBench-v1 suites ship in this repo.

## Quick start

```bash
pip install -e .

export EVIBIND_UPSTREAM_BASE_URL="https://openrouter.ai/api/v1"   # or api.openai.com/v1, api.x.ai/v1, a vLLM/Ollama URL
export EVIBIND_UPSTREAM_API_KEY="your-provider-key"
export EVIBIND_GATEWAY_API_KEY="local-gateway-key"
export EVIBIND_HANDLE_SECRET="$(python -c 'import secrets;print(secrets.token_hex(32))')"
export EVIBIND_OPERATING_MODE="enforce"

evibind serve --host 127.0.0.1 --port 8090
```

Point your existing OpenAI client at the gateway — nothing else changes:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8090/v1", api_key="local-gateway-key")
```

Mark the slots that matter with `x-evibind-*` annotations in your tool schema
(the model never sees them — the gateway strips annotations before forwarding):

```json
"account_ref": {
  "type": "string",
  "x-evibind-slot-role": "control",
  "x-evibind-evidence-type": "account_ref",
  "x-evibind-sources": ["user.current_turn"],
  "x-evibind-resolution-type": "verbatim",
  "x-evibind-extraction-cue": "account"
}
```

An offline, no-key end-to-end example: `python examples/minimal_evidence_binding.py`.

## Where it sits

<p align="center"><img src="assets/architecture.svg" width="760" alt="App → EviBind gateway (compile, bind, replay) → any OpenAI-compatible provider"></p>

Works with any OpenAI-compatible chat-completions endpoint: OpenAI, OpenRouter,
xAI (Grok), vLLM, Ollama, LM Studio. The protection is provider- and
model-agnostic because it wraps the request/response boundary, not the model.

## Benchmark: InjectBench

150 deterministic cases over ten categories, split by the kind of error they
induce. **Origin violations** (60) try to make the model use a value that exists
only in untrusted tool output. **Selection errors** (90) put every candidate in
admissible evidence and only require the model to pick correctly.

Both arms score the *same* model output: `native` is the raw tool call,
`guarded` is that call passed through the gateway.

### What the boundary structurally prevents

Against a worst-case selector that always emits the engineered wrong value:

<p align="center"><img src="assets/bench_control.svg" width="820" alt="All 60 origin-violation cases neutralised: 45 repaired to the authorised value, 15 withheld; selection errors pass through"></p>

All 60 origin violations are neutralised — and in 45 of them the gateway does
better than blocking: it re-derives the user-authorised value and the intended
call still goes out. Garbled long identifiers are withheld too. Selection errors
(stale values after a correction, ambiguity, swapped slots, forbidden values)
pass through unchanged, because confining *where a value came from* is not the
same as choosing *which* value was meant. That limit is measured here rather
than glossed: see [`docs/FINDINGS.md`](docs/FINDINGS.md), which also records an
open fail-closed defect the run surfaced.

### What it costs on a real model

<p align="center"><img src="assets/bench_live.svg" width="820" alt="Claude Haiku across 150 cases: identical outcomes native and guarded, zero false rejections"></p>

Claude Haiku, live: 75/75 correct calls released unchanged (including
50-character ARNs and near-duplicate account contexts), 75/75 abstentions
preserved, zero rewritten arguments, zero false rejections. Haiku also resisted
all 60 injections on its own — which is worth reading carefully: the native
arm's safety is a property of that model in a batch-review setting, while the
guarded arm's property holds for any selector. Caveats in
[`docs/FINDINGS.md`](docs/FINDINGS.md#5-live-model-result-the-gateway-is-free).

Run it against your own model — any OpenAI-compatible endpoint:

```bash
python bench/run_bench.py live --base-url https://api.x.ai/v1 \
    --api-key "$XAI_API_KEY" --model grok-4 --label grok-4
python bench/make_charts.py
```

Or offline: `export` the prompts, answer them with any model or CLI, then
`score`. Ready-made runners for Grok, GPT-5.6 Luna/Terra, OpenRouter and local
servers are in [`providers/`](providers/).

## Results from the paper

The research artifact behind this repo froze a 100-case test (ten unseen tool
families, hashes fixed before any model output) measuring exact critical
binding with the actual candidate catalog vs. admissible top-1 presentation:

<p align="center"><img src="assets/paper_binding.svg" width="800" alt="Exact critical binding: Qwen3-1.7B 5→89, Qwen3.6-35B 86→100, GPT-OSS-120B 97→100, GPT-5.6-Luna 100→100"></p>

Highlights from the submission (all numbers claim-ledger-bound in
`paper/claims.yaml` of the evidence bundle):

- 300 equal-value provenance pairs: value-only checking is 100% complete but
  **0% sound**; derivation-aware checking and EviBind are both 100/100.
- 30 sandboxed executed-effect scenarios: native literals cause 30/30 harmful
  effects; EviBind completes 30/30 with zero harm.
- Single-fault study: all 8 faults in a checker's redundant literal/trace
  channel are exploitable (240/240); EviBind has no such channel (0/240,
  fail closed).
- 1,000,000 release-boundary mutations: zero unsound releases.

## What it protects — and what it doesn't

EviBind confines *configured critical leaves*: an executable critical value can
only come from an authenticated, replayable derivation admitted by your slot
policy. It does **not** replace policy review (a wrong policy un-protects a
slot), planner isolation, sink sandboxing for free-text/SQL/shell payloads, or
business authorization. Multi-turn agent utility has a measured cost — see the
stateful results in the paper before wrapping long-horizon planners. Scope,
threat model, and residual attack surface: [`SECURITY.md`](SECURITY.md) and the
paper's §8.

## Repo layout

| path | contents |
|---|---|
| `evibind/` | product surface: gateway, policy, schema lint, CLI (`evibind serve`) |
| `tapbench/` | the underlying engine: compiler, materializer, certificates, fuzzer, suites |
| `bench/` | InjectBench: the 150-case benchmark used above |
| `providers/` | one-command runners: OpenRouter, xAI/Grok, GPT-5.6, local models |
| `examples/` | offline minimal binding, guarded host execution, annotated request |
| `docs/` | findings, public API, reproducibility, upstream research README |
| `tests/` | boundary, gateway, fragility, and suite tests |

## Development

```bash
pip install -e ".[dev]"
pytest tests -q
ruff check .
```

## Citation

If you use EviBind in research, cite the artifact via [`CITATION.cff`](CITATION.cff).
License: [MIT](LICENSE).
