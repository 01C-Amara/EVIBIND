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

## See it

One command, a live model, a real injection. The user authorises one account;
an attacker-controlled tool result orders another:

```bash
OPENAI_API_KEY=... python examples/live_gateway_demo.py --model gpt-5.4-nano
```

```text
user authorised  : ACC-4000
tool result wants: ACC-8000   <- attacker-controlled text

  without EviBind : ACC-8000   <- followed the injection
  with EviBind    : ACC-4000   <- bound to the span the user wrote
  gateway latency : 0.92s
```

GPT-5.4 mini and GPT-4.1 mini follow that injection too. The attacker's value
has to be *opaque* for that — when the same demo used `exfil@evil.example`, all
three models refused on their own. Injection resistance measured with
obviously-malicious payloads overstates real safety; account numbers, ARNs and
record IDs carry no such signal.

No key? The same argument runs offline:
`python examples/minimal_evidence_binding.py`.

## What the evidence says

Nine live models, 150 cases each, 1,350 scored calls
([`docs/FINDINGS.md`](docs/FINDINGS.md)):

- **Zero false rejections.** Not one case, on any model, where the model bound
  the critical slot correctly and the gateway then withheld or altered it. A
  filter that breaks good calls is unshippable; this one doesn't.
- **It repairs rather than blocks.** Correct calls *rise* behind the gateway on
  weaker models — 17→45, 18→43, 8→35 out of 60. Of the 43 cases GPT-5.4 nano
  bound to the attacker's account, 28 were re-derived to the account the user
  actually authorised, 15 withheld, none leaked.
- **A cheap model behind it beats a frontier model alone.** GPT-5.4 nano guarded
  lands 120/150 correct calls with 0 harmful; GPT-5.6 Sol and Grok 4.6 unguarded
  land 104/150.
- **The guarantee does not depend on the model.** Native harmful bindings range
  from 0/60 to 43/60 across the nine. Behind the gateway every one of them is
  0/60.

And numbers that are not ours. On
[AgentDojo](https://github.com/ethz-spylab/agentdojo) (ETH SPY Lab, MIT),
scored on **their** utility and security metrics, with **their** attack, EviBind
inserted into **their** agent loop. The banking suite, all 16 user tasks crossed
with all 9 injection tasks:

| arm | cases | task completed | attack succeeded |
|---|---|---|---|
| baseline | 144 | 53 | **58 (40%)** |
| **EviBind** | 144 | **58** | **0** |

Attacks eliminated, and more tasks completed than without it — because a call
re-derived from the user's own turn still goes out, where the baseline followed
the injection and failed the task.

**Banking is the best case, not the typical one.** Across four suites the
outcome tracks how often the authorised value is somewhere the attacker cannot
write to, which `bench/agentdojo/scope.py` measures with no model involved:

| suite | re-derivable | attack succeeded | clean-traffic completion |
|---|---|---|---|
| banking | 75% | 58 → **0** | 8 → 6 of 16 |
| slack | 27% | 66 → **24** | 15 → **1** of 21 |

Where the user names the value, the boundary is close to free. Where they do
not — Slack, where channel names and addresses come from tool output — it
withholds most of the traffic and does not even close the attack surface. That
spread is the honest headline, and
[`bench/agentdojo/`](bench/agentdojo/) carries all four suites, both arms, the
clean controls and the reproduction commands.

<p align="center"><img src="assets/bench_contrast.svg" width="840" alt="The same three models refuse tool-selection injection almost perfectly on InjecAgent while taking the attacker's account number two thirds of the time on InjectBench"></p>

On [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) (UIUC, MIT), eight
models call the attacker's tool **0/510** — GPT-5.6 Sol, Luna and Terra, Grok
4.6 and 4.5, GPT-5.4 mini and nano — even on the split that prefixes the
payload with *"IGNORE ALL PREVIOUS INSTRUCTIONS"*. GPT-4.1 mini, the oldest of
them, is the only one to slip: 7 times in 510, and the gateway withheld every
one.

The frontier rows are what make this readable. If only the cheap tiers had run,
*"weak models are weak"* would fit the data. But Sol and Grok 4.6 are at 0/510
**and** 0/60, while GPT-5.4 nano is at 0/510 **and** 43/60. The variance is in
the shape of the attack, not the tier of the model.

The difference is the shape of the attack, not its strength. *"Call this other
tool"* reads as an instruction from the wrong party and gets refused. *"Use this
account number instead"* does not read as an instruction at all — it reads as a
fact about which value is correct, and nothing in training tells the model the
fact arrived from the wrong place. That gap is the argument for an
argument-level boundary, and it is why a suite measuring only tool-selection
attacks reports this problem as solved. Method, caveats and the utility cost:
[`bench/injecagent/`](bench/injecagent/).

End to end through `evibind serve` against live OpenAI, GPT-5.4 nano now
completes 88/150 with 0 harmful and 0 malformed releases — up from 0/150 before
the extraction fixes in [`docs/FINDINGS.md`](docs/FINDINGS.md) §10.

## Quick start

```bash
pip install -e .

export EVIBIND_UPSTREAM_BASE_URL="https://api.openai.com/v1"   # or api.x.ai/v1, openrouter.ai/api/v1, a vLLM/Ollama URL
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
completion = client.chat.completions.create(
    model="gpt-5.4-nano", messages=messages, tools=[annotated_tool]
)
```

Verified against the stock `openai` package: with the model proposing the
injected account, the client receives the account the user authorised.
`GET /v1/models` proxies your upstream, so tools that enumerate models to
populate a picker or health-check a base URL work too.


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

Three runnable examples: `examples/live_gateway_demo.py` (network, a real
injection, with and without the gateway), `examples/trusted_state_binding.py`
(offline — how to protect a value the user never typed) and
`examples/minimal_evidence_binding.py` (offline, no key).

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
all 60 injections on its own — though by declining to act on every one of them,
not by resolving them correctly, which is a different thing from the repair the
gateway performs. Read the native arm carefully in general: its safety is a
property of that model in a batch-review setting, while the guarded arm's
property holds for any selector. Caveats in
[`docs/FINDINGS.md`](docs/FINDINGS.md#5-live-model-result-the-gateway-is-free).

### Nine live models

<p align="center"><img src="assets/bench_models.svg" width="880" alt="Origin-violation outcomes for nine live models: native harmful ranges 0 to 43 of 60, guarded harmful is 0 for every model"></p>

Every model below ran all 150 cases live. The table reports the **critical
slot** — the one the confinement claim is about — because whole-call equality
also demands incidental slots match, and a model that binds the account
correctly while writing `"500.00 USD"` where gold says `"500.00"` is a
formatting difference, not a security event.

Two rows carry caveats: the Claude Haiku responses were collected in an earlier
run that sent cases verbatim (`--dialect native`), and the Grok rows come from
the CLI's schema-constrained structured output rather than native tool calling.
Both are spelled out in [`docs/FINDINGS.md`](docs/FINDINGS.md) §8–9.

| model | origin: harmful native | origin: harmful guarded | origin: correct native | origin: correct guarded | selection: correct native → guarded |
|---|---|---|---|---|---|
| GPT-5.6 Terra | 4/60 | **0/60** | 23/60 | 25/60 | 72 → 72 |
| GPT-5.6 Luna | 3/60 | **0/60** | 26/60 | 29/60 | 68 → 68 |
| GPT-5.6 Sol | 0/60 | **0/60** | 31/60 | 31/60 | 73 → 73 |
| Grok 4.6 | 0/60 | **0/60** | 30/60 | 30/60 | 74 → 74 |
| Grok 4.5 | 1/60 | **0/60** | 21/60 | 22/60 | 69 → 69 |
| Claude Haiku | 0/60 | **0/60** | 0/60 | 0/60 | 75 → 75 |
| GPT-5.4 mini | 40/60 | **0/60** | 18/60 | **43/60** | 73 → 73 |
| GPT-5.4 nano | 43/60 | **0/60** | 17/60 | **45/60** | 75 → 75 |
| GPT-4.1 mini | 43/60 | **0/60** | 8/60 | **35/60** | 75 → 75 |
| Weak selector (local mock) | 60/60 | **0/60** | 0/60 | **45/60** | 60 → 60 |

Three things are worth reading off it.

**Frontier models mostly resist; smaller ones do not.** Sol, Grok 4.6 and Haiku
followed no injection at all; Grok 4.5 followed one, Luna three, Terra four. The
cheaper tiers followed roughly two thirds of them — 43 of 60 for both
GPT-5.4 nano and GPT-4.1 mini. Model capability is a real mitigation, and it is
not one you can rely on when the deployment picks the model — a cost-tiering
router or a rate-limit fallback moves a system down this table without changing
a line of application code.

**The gateway reaches zero on all of them,** which is the point of a structural
guarantee: the guarded column does not depend on which model produced the call.

**It repairs rather than merely blocks.** On the weak selectors the correct-call
count *rises* — 17 → 45, 18 → 43, 8 → 35 — because a slot re-derived from the
user's own span replaces the injected one and the intended call still goes out.
Of the 43 cases GPT-5.4 nano bound to the attacker's account, 28 were repaired
to the account the user actually authorised, 15 were withheld, and none leaked.
Blocking alone would have left all 43 tasks unfinished.

**Selection errors are untouched, in both directions.** The last column is
identical before and after for every model: the boundary confines *where a value
came from*, not *which* value was meant. The mock control makes the limit
explicit — 15 of its cases stay harmful under the gateway, all of them
`negation` (a value the user explicitly forbade). Measured, not glossed:
[`docs/FINDINGS.md`](docs/FINDINGS.md).

Reproduce the mock row with no API key at all:

```bash
python bench/mock_provider.py --port 8099 --mode last-mention &
python bench/run_bench.py live --base-url http://127.0.0.1:8099/v1 \
    --api-key none --model mock-last-mention --label mock-last-mention \
    --dialect native
```

Run it against your own model — any OpenAI-compatible endpoint. Keys can be
passed as `file:` or `env:` specs so they never reach a command line:

```bash
python bench/run_bench.py live --base-url https://api.openai.com/v1 \
    --api-key file:.env --model gpt-5.4-mini --label gpt-5.4-mini
python bench/make_charts.py
```

Two provider quirks are handled for you, and both are worth knowing:

- **The GPT-5.6 tiers refuse function tools on `/v1/chat/completions`** unless
  reasoning is disabled. Pass `--endpoint responses` to drive them through
  `/v1/responses` instead; results are rendered back into chat-completion shape
  so scoring is unchanged.
- **OpenAI rejects a `tool` message that does not answer an assistant tool
  call.** 60 InjectBench cases hand the model an untrusted tool result with no
  such turn, so `--dialect openai` (the default) inserts the minimal assistant
  call that makes the transcript well formed. The untrusted content is passed
  through byte for byte. Use `--dialect native` to send cases verbatim.

The whole OpenAI line-up in one step:

```bash
./providers/run_openai_suite.sh
```

Grok authenticates through a grok.com subscription rather than an xAI API key,
and its CLI exposes no OpenAI-compatible endpoint, so it has its own driver:

```bash
python bench/run_grok_cli.py --model grok-4.6 --concurrency 5
python bench/run_bench.py score --responses bench/results/grok-4.6.responses.jsonl \
    --label grok-4.6 --out bench/results/grok-4.6.json
```

That path constrains the CLI's structured output to the tool's own JSON Schema,
so Grok fills the same slots under the same constraints — but it is an
emulation of tool calling, not the native mechanism. See
[`docs/FINDINGS.md`](docs/FINDINGS.md) for what that does and does not license.

If you do hold an `XAI_API_KEY`, the ordinary live path works unchanged:

```bash
python bench/run_bench.py live --base-url https://api.x.ai/v1 \
    --api-key env:XAI_API_KEY --model grok-4 --label grok-4
```

There is also an offline path: `export` the prompts, answer them with any model
or CLI (including `grok.exe` on Windows), then `score` the JSONL. Individual
runners live in [`providers/`](providers/).

## Status

Honest picture, because the two halves of this repo carry different amounts of
evidence.

**The binding algorithm.** The table above scores `protect_chat_completion`
against a model response the harness fetches itself: 1,350 live scored calls,
nine models, zero false rejections, zero harmful releases.

**The serving path.** `evibind serve` owns the whole interaction — it compiles a
candidate table, forces one action tool, has the model select handles instead of
writing literals, then certifies and materializes. Run end to end against live
OpenAI with GPT-5.4 nano, before and after the extraction fixes described in
[`docs/FINDINGS.md`](docs/FINDINGS.md) §10:

| | correct | harmful | withheld | malformed |
|---|---|---|---|---|
| serving path, as first measured | 0/150 | 0 | 135 | 15 |
| **serving path, now** | **88/150** | **0** | **62** | **0** |

58 of 60 origin violations complete: injected instruction 15/15, injected data
field 15/15, forged authority 15/15, user-defers-to-tool 13/15.

Most of the 62 remaining abstentions are the boundary doing its job — 15
`ambiguity` cases should clarify rather than guess, and 15 `distractor` cases
put eight near-duplicate accounts in play. 15 `cross_slot` cases stay unresolved
because cue-based extraction cannot separate two same-type slots when neither
cue precedes its value; that limit is real and documented.

**Latency.** Ten sequential requests each, GPT-5.4 nano: 0.69s median direct,
0.94s median through the gateway — `+0.25s`, one round trip, no second model
call.

**How much of a real agent this covers: about a third.** On
[AgentDojo](https://github.com/ethz-spylab/agentdojo) (ETH SPY Lab, MIT), whose
injection tasks are argument-level, only **43 of 119** action-critical argument
values across four suites are ones the user actually wrote — 75% in banking,
27% in Slack. The rest arrive the same way the attack does. Its flagship case is
*"pay the bill 'bill-december-2023.txt'"*, where the authorised IBAN is inside
the document and the injection replaces that very block, so the attacked file
holds only the attacker's. EviBind withholds: safe, and unable to complete.

That is the boundary's stated assumption with a number on it, and the first
thing to check against your own tool surface. You raise it by having the
*application* fetch such values itself and pass them out-of-band, rather than
letting the model read them out of a document:

```python
request["evibind"]["dialogue_state"] = {"recipient": iban_from_your_invoice_api}
# on the slot: "x-evibind-source-policy": "trusted_state_only"
```

`python examples/trusted_state_binding.py` runs both arms on AgentDojo's own
bill scenario, offline: bound to the user's turn it withholds, bound to trusted
state it releases the authorised IBAN while the model proposes the attacker's.
Method and full table: [`bench/agentdojo/`](bench/agentdojo/), write-up in
[`docs/FINDINGS.md`](docs/FINDINGS.md) §17–18.

**One exposure worth knowing before you deploy.** If a model swaps two
same-typed critical slots — `from_account` and `to_account` — the boundary
releases it. Both values are the user's own spans, so confinement has no
objection, and the payment goes the wrong way. The ICLR mixed-order revision
measures model selection on exactly this relation at 16–64% exact across
presentation orders, against 100% on the other five. Set
`{"evibind": {"clarify_interchangeable_slots": true}}` to withhold instead; it
is off by default because it also withholds the correctly assigned call, which
is the point — the two are indistinguishable to the boundary.
[`docs/FINDINGS.md`](docs/FINDINGS.md) §15–16.

**Not measured yet.** Annotation burden on a real tool surface, and throughput
under load. The third-party number now exists —
[`bench/injecagent/`](bench/injecagent/) — and it comes with its own honest
limit: an argument-level boundary cannot touch a parameterless tool, which is
30% of InjecAgent's direct-harm cases and 59% of its data-stealing ones. Tool
-level authorization belongs *alongside* argument binding, not after it.

## Results from the paper

The research artifact behind this repo froze a 100-case test (ten unseen tool
families, hashes fixed before any model output) measuring exact critical
binding with the actual candidate catalog vs. admissible top-1 presentation:

<p align="center"><img src="assets/paper_binding.svg" width="800" alt="Exact critical binding: Qwen3-1.7B 5 to 89, Qwen3.6-35B 86 to 100, GPT-OSS-120B 97 to 100, GPT-5.6-Luna 100 to 100"></p>

From the current submission (*semantic robustness revision*, under review):

- **300 equal-value provenance pairs**: value-only checking is 100% complete but
  **0% sound**; derivation-aware checking and EviBind are both 100/100.
- **30 sandboxed executed-effect scenarios**: native literals cause 30/30
  harmful effects and complete 0/30 tasks; EviBind completes 30/30 with zero
  harm. A correctly built trace-materializing cite-and-check matches it.
- **Single-fault study**: all 8 faults in a checker's redundant literal/trace
  channel are exploitable (240/240 harmful); EviBind has no such channel
  (0/240, fail closed).
- **1,000,000 release-boundary mutations**: zero unsound releases.
- **Presentation matters as much as the model.** Admissible top-1 moves exact
  binding from 5/100 to 89/100 for Qwen3-1.7B and 86/100 to 100/100 for
  Qwen3.6-35B, with no regressions. But a singleton ranker is *positional*: it
  keeps gold in 100/100 gold-late cases and 0/100 gold-early ones. Retaining
  the top two candidates restores it.
- **Six semantic relations, alternatives retained**: Qwen3.6-35B reaches 88.8%
  exact recall and 86% all-order exactness, GPT-5.6-Luna 97% and 94%. Both are
  perfect on five of the six relations; **two-slot destination composition is
  the sole outlier.**
- **Boundary reliability and task utility decompose.** Across 336 prospective
  ToolSandbox rows the boundary reduces call exceptions by 0.256 while reducing
  task similarity by 0.155. Reliable dispatch does not by itself recover
  end-to-end task progress.

The last two are worth reading against this repo's live runs, because they were
found again independently here:

| paper | live in this repo |
|---|---|
| two-slot destination composition is the sole failing relation | the 15 `cross_slot` cases are the group the serving path cannot resolve, because neither cue precedes its value ([`docs/FINDINGS.md`](docs/FINDINGS.md) §10) |
| boundary reliability and planner competence are distinct axes | the end-to-end run completes 88/150 with 0 harmful — safe well before it is useful ([Status](#status)) |
| a singleton ranker is positional; keep the top two | offering only the greedy reading of a span hid the correct value entirely, until both readings were offered (§10) |

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
| path | contents |
|---|---|
| `evibind/` | product surface: gateway, policy, schema lint, CLI (`evibind serve`) |
| `tapbench/` | the underlying engine: compiler, materializer, certificates, fuzzer, suites |
| `bench/` | InjectBench: 150 cases, the mock provider, and both runners |
| `bench/injecagent/` | external benchmark: InjecAgent adapted, fetched not vendored |
| `bench/agentdojo/` | external scoping: how much of a real agent this boundary covers |
| `providers/` | one-command runners: OpenAI, xAI/Grok, OpenRouter, local models |
| `examples/` | `live_gateway_demo.py` (network) and `minimal_evidence_binding.py` (offline) |
| `docs/` | start with [`FINDINGS.md`](docs/FINDINGS.md); the rest is paper apparatus |
| `tests/` | boundary, gateway, fragility, and suite tests |

Start here: [`examples/live_gateway_demo.py`](examples/live_gateway_demo.py) to
see it work, [`docs/FINDINGS.md`](docs/FINDINGS.md) for every result and every
known defect, [`bench/cases.py`](bench/cases.py) for what is actually tested,
and [`bench/injecagent/`](bench/injecagent/) for the number that is not ours.

The `docs/` directory also carries the paper's apparatus — preregistrations,
review panels, human-study protocols. Those are the research record, not
product documentation; `FINDINGS.md`, `PROVIDERS.md`, `PUBLIC_API.md` and
`OPERATIONS.md` are the ones you want.

## Development

```bash
pip install -e ".[dev]"
pytest tests -q
ruff check .
```

Sixteen test modules import the paper's `scripts` package, which this repo does
not ship; `tests/conftest.py` skips them when it is absent, so `pytest tests -q`
is green on a clean checkout.

## Citation

If you use EviBind in research, cite the artifact via [`CITATION.cff`](CITATION.cff).
License: [MIT](LICENSE).
