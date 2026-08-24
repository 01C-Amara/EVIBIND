<p align="center">
  <img src="assets/logo.svg" width="360" alt="EviBind">
</p>

<p align="center">
  Authenticated evidence binding for fail-closed LLM tool execution.
</p>

<p align="center">
  <a href="https://github.com/01C-Amara/EVIBIND/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/01C-Amara/EVIBIND/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-2563eb.svg"></a>
  <img alt="Python 3.11-3.13" src="https://img.shields.io/badge/python-3.11--3.13-0f766e.svg">
</p>

EviBind replaces model-written critical arguments with references to
request-local evidence. Trusted code authenticates those references, replays
their derivations, validates the tool contract, and materializes the executable
call atomically. The model can select policy-admissible evidence; it cannot
invent or rewrite protected values at the release boundary.

The repository deliberately has two complete usage paths:

| Path | Audience | Stable entry point | Evidence standard |
|---|---|---|---|
| Product | application and platform engineers | `evibind` package and CLI | tests, conformance, package and container gates |
| Research | reviewers and researchers | `scripts/`, `tapbench/`, and versioned evidence bundles | frozen protocols, case-level artifacts, digests, and claim audit |

These paths share the same compiler and materializer. Benchmark-only labels,
gold answers, and model outputs never enter the runtime API.

> **Status:** research-grade alpha. The release boundary and public APIs are
> tested, but production deployment still requires policy review, least
> privilege, complete mediation, a shared nonce store for multiple replicas,
> and ordinary tool authorization. Read [SECURITY.md](SECURITY.md) before
> connecting EviBind to effects.

## How it works

<p align="center">
  <img src="assets/architecture.svg" width="780" alt="Application to EviBind compile, bind, replay, and dispatch boundary to an OpenAI-compatible model">
</p>

1. The application marks authority- or effect-bearing JSON leaves and states
   which evidence roots and transforms are admissible.
2. EviBind compiles typed, destination-bound candidates from immutable messages,
   versioned trusted state, defaults, and schema enums.
3. The model selects opaque handles through a literal-free internal action.
4. Trusted code authenticates and replays each selected derivation, assembles a
   closed argument object, and emits a materialization certificate.
5. The host SDK dispatches only that reconstructed manifest.

EviBind proves a confinement property: a released protected leaf was produced
by a current, policy-admissible derivation for that exact destination. It does
**not** prove intended-candidate correctness, business authorization, policy
completeness, tool correctness, or downstream effect safety.

## Quick start

Python 3.11–3.13 is supported.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/minimal_evidence_binding.py
```

The offline example needs no provider or key. It presents the same email value
through an admissible user span and an inadmissible tool-output span; EviBind
admits one candidate, rejects the other, materializes the call, and replays the
certificate.

For the OpenAI-compatible gateway, copy `.env.example` to `.env`, populate it
outside version control, and export the values through your process manager:

```bash
export EVIBIND_UPSTREAM_BASE_URL="https://api.openai.com/v1"
export EVIBIND_UPSTREAM_API_KEY="..."
export EVIBIND_GATEWAY_API_KEY="..."
export EVIBIND_HANDLE_SECRET="$(python -c 'import secrets; print(secrets.token_hex(32))')"
export EVIBIND_OPERATING_MODE="enforce"

evibind serve --host 127.0.0.1 --port 8090
```

Point an existing OpenAI client at the local boundary:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8090/v1",
    api_key="your-local-gateway-key",
)
response = client.chat.completions.create(
    model="your-model",
    messages=messages,
    tools=[annotated_tool],
)
```

Private `x-evibind-*` annotations define the protected leaves. They are removed
before the provider sees the public tool schema:

```json
{
  "account_ref": {
    "type": "string",
    "x-evibind-slot-role": "control",
    "x-evibind-evidence-type": "account_ref",
    "x-evibind-sources": ["user.current_turn"],
    "x-evibind-resolution-type": "verbatim",
    "x-evibind-extraction-cue": "account"
  }
}
```

Scaffold and inspect a policy before using enforcement:

```bash
evibind init --request request.json --output request.evibind.json
evibind lint-schema --strict --request request.evibind.json
evibind inspect --request request.evibind.json --output inspection.json
```

`init` is conservative scaffolding, not an authorization decision. An owner
must review inferred criticality, sources, ambiguity behavior, and versions.

## Host-owned dispatch

The gateway can protect an OpenAI-compatible response, while
`GuardedToolExecutor` additionally owns complete mediation and dispatches only
the certified manifest:

```python
from evibind import GuardedToolExecutor

executor = GuardedToolExecutor(
    {"transfer": transfer_handler},
    handle_secret=server_secret,
)
turn = executor.prepare(request_payload)
provider_response = call_provider(turn.upstream_payload)
result = turn.complete(provider_response)
```

A guarded turn is single-use. Missing bindings, stale versions, malformed
actions, unknown tools, extra calls, failed effect confirmation, or certificate
mismatch fail closed before a handler runs.

## Evidence at a glance

The ICLR 2027 artifact separates direct ActionIR studies from compatibility
replay over ordinary literal tool calls.

| Evidence layer | Frozen result | Supports |
|---|---|---|
| OriginBench | 300 equal-value provenance pairs; EviBind and dispatch-atomic trace materialization are jointly sound/complete, value-only checking is not | representation theorem and reference construction |
| EffectSuite | EviBind completes 30/30 policy-admissible mock effects with 0 harm; native literals cause 30/30 harmful effects | enforced boundary behavior |
| Implementation fragility | 8/8 redundant literal/trace faults expose cite-and-check, 0/8 expose EviBind’s absent channel; shared-boundary faults expose both | representation-specific fault surface |
| Boundary fuzz | 1,000,000 authenticated release mutations, 0 unsound releases | tested boundary robustness |
| Fresh-family binding | Qwen3-1.7B: 5% → 89%; Qwen3.6-35B-A3B: 86% → 100% with verified top-1 | direct ActionIR usability under the frozen construction |
| Multi-relation binding | Qwen3.6-35B-A3B 88.8% and GPT-5.6-Luna 97% exact recall; destination composition remains the outlier | alternative-preserving semantic binding |
| Needle 2 replay | 17 harmful native releases in 59 released calls; 0 in 17 replay-gated releases | confidence and provenance are complementary |
| AgentDojo banking replay | successful attacks 6/144 → 0/144; task completion 57/144 → 58/144 | narrow external compatibility evidence |

Needle and AgentDojo use compatibility replay: their models emit ordinary
literal calls, then EviBind re-derives or withholds protected values. They do not
test direct model generation of ActionIR. Full methods, uncertainty, and scope
are in [docs/RESEARCH_EVIDENCE.md](docs/RESEARCH_EVIDENCE.md).

## Reproduce and verify

Run the deterministic mechanism evidence with no model or network:

```bash
python scripts/reproduce_public_artifact.py \
  --output-dir reproduced/mechanism \
  --fuzz-trials 10000
```

Use `--full` for the paper-scale one-million-trial fuzz. Outputs are
deterministically ordered and content-addressed in `SHA256SUMS`.

The full paper archive remains a release asset rather than Git history. Verify
it against the checked-in release record:

```bash
python scripts/verify_evidence_bundle.py \
  EviBind_ICLR_2027_evidence_bundle_20260821_v8.zip \
  --sidecar EviBind_ICLR_2027_evidence_bundle_20260821_v8.zip.sha256 \
  --release-metadata evidence/paper-v8.json
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the four evidence
tiers, exact commands, and claim-to-artifact map.

## Operating boundary

EviBind confines configured protected leaves. Applications still need:

- business authorization and approval for high-impact effects;
- policy review and independent versioning of schemas, state, and transforms;
- least-privilege tool credentials held outside the model and gateway;
- complete mediation and a dispatcher that never executes native audit output;
- TLS, request limits, sandboxed sinks, and secret-safe logging; and
- a linearizable shared nonce store and idempotency keys for multiple replicas.

Ambiguity is not evidence. When several admissible candidates remain, retain
alternatives for model/user disambiguation or abstain. A learned top-1 ranker is
a presentation heuristic, not proof of user intent.

## Documentation

| Document | Purpose |
|---|---|
| [Public API](docs/PUBLIC_API.md) | stable imports, low-level construction, and research surface |
| [Operations](docs/OPERATIONS.md) | policy initialization, modes, replay, and provider envelopes |
| [Architecture](docs/ARCHITECTURE.md) | trust boundary, module ownership, and dual-use invariants |
| [Security](SECURITY.md) | deployment requirements, residual risks, and reporting |
| [Research evidence](docs/RESEARCH_EVIDENCE.md) | study taxonomy, results, uncertainty, and limits |
| [Reproducibility](docs/REPRODUCIBILITY.md) | deterministic replay and model-backed protocols |
| [Findings](docs/FINDINGS.md) | detailed engineering and benchmark record |
| [Contributing](CONTRIBUTING.md) | development, scientific, and release gates |

## Repository layout

| Path | Responsibility |
|---|---|
| `evibind/` | stable package facade, core evidence types, host SDK, schema lint, CLI |
| `tapbench/` | shared engine plus research algorithms; APIs are artifact-versioned unless re-exported by `evibind` |
| `scripts/` | deterministic reproduction, evidence verification, and research entry points |
| `evidence/` | small signed-off release records; large archives stay in release assets |
| `bench/` | live and third-party compatibility studies with explicit provenance |
| `examples/` | provider-free and networked integrations |
| `tests/` | product, boundary, artifact, and regression tests |
| `docs/` | operator and research documentation |

## Development

```bash
python -m pip install -e ".[dev,research]"
python -m pytest -q
python -m ruff check .
python -m build
python scripts/audit_release_archives.py dist/*.whl dist/*.tar.gz
```

CI tests Python 3.11–3.13, runs the offline example and deterministic evidence
smoke, builds wheel and source distributions, installs the wheel in a clean
environment, audits release contents, and builds the container.

## Citation and license

If EviBind supports published work, cite the software release and accompanying
paper using [CITATION.cff](CITATION.cff). EviBind is available under the
[MIT License](LICENSE).
