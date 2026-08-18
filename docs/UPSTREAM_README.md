# EviBind: Evidence-Bound Argument Materialization

**Why the name?** EviBind is a portmanteau of *evidence* and *binding*. The
system binds each action-critical argument to the specific, replayable evidence
derivation that authorizes it. The name refers to that security boundary, not
to evidence retrieval or citation generation.

EviBind changes the model's action representation: for authority-bearing tool
arguments, the model selects request-scoped evidence handles instead of
generating executable literals. Trusted code alone materializes values after
checking their source, evidence type, destination, request, policy epoch, state
version, expiry, and final tool contract.

This prevents unsupported literal construction under the configured evidence
policy. It does not prove that the model chose the correct tool or candidate,
that the user intended a mentioned value, that the user is authorized, or that
the tool's effect is desirable or correctly implemented.

## Quick Start

The fastest end-to-end example is deterministic, offline, and requires no model
or API key:

```bash
python -m pip install -e .
python examples/minimal_evidence_binding.py
python scripts/reproduce_public_artifact.py \
  --output-dir /tmp/evibind-public-reproduction \
  --fuzz-trials 10000
```

For the OpenAI-compatible gateway:

```bash
python -m pip install -e .

export EVIBIND_UPSTREAM_BASE_URL="https://api.openai.com/v1"
export EVIBIND_UPSTREAM_API_KEY="your-provider-key"
export EVIBIND_GATEWAY_API_KEY="local-gateway-key"
export EVIBIND_HANDLE_SECRET="replace-with-at-least-32-random-bytes"
export EVIBIND_OPERATING_MODE="enforce"

evibind serve --host 127.0.0.1 --port 8090
```

The stable in-process Python surface uses the product namespace:

```python
from evibind import EviBindGateway, GatewayConfig, lint_tool_schemas
```

Then use the normal OpenAI client and change only its base URL:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8090/v1",
    api_key="local-gateway-key",
)

response = client.chat.completions.create(
    model="your-upstream-model",
    messages=[
        {"role": "user", "content": "Create an event on 2026-08-02."}
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "create_calendar_event",
                "description": "Create a calendar event.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "x-evibind-slot-role": "control",
                            "x-evibind-resolution-type": "normalizable",
                            "x-evibind-evidence-type": "iso_date",
                            "x-evibind-sources": ["user.current_turn"],
                            "x-evibind-extraction-cue": "date",
                        }
                    },
                    "required": ["date"],
                    "additionalProperties": False,
                },
            },
        }
    ],
)
```

Before starting the gateway, lint and fingerprint the private contract:

```bash
evibind lint-schema --strict --request examples/openai_request.json
```

For an unannotated request, initialize and inspect a conservative policy locally:

```bash
evibind init --request request.json --output request.evibind.json
evibind inspect --request request.evibind.json --output inspection.json
```

Initialization is a review scaffold, not an authorization decision. See
[`docs/OPERATIONS.md`](OPERATIONS.md) for modes, certificate replay, and
provider-envelope adapters.

For host-owned, complete-mediation tool dispatch, see
[`docs/HOST_SDK.md`](HOST_SDK.md).

For versioned clarification, trust labels, and exact-manifest effect
confirmation, see
[`docs/STATE_AND_EFFECTS.md`](STATE_AND_EFFECTS.md).
For the versioned one-call matched-compute suite, compiler/selector metrics, and
payload-bound replay protocol, see [`docs/EVIBENCH.md`](EVIBENCH.md).
The canonical ICLR 2027 paper source, formal guarantee, and executable claim
ledger live in `paper/` in the research bundle (not shipped in this repo). Start with
[`docs/REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the evidence tiers and
exact commands, and [`docs/PUBLIC_API.md`](PUBLIC_API.md) for the supported
runtime and research surfaces.

The gateway compiles and filters evidence candidates before inference. It removes
the executable tools and private `x-evibind-*` annotations from the upstream
request, then makes one model call with a forced `evibind_action` schema whose
domains contain only valid handle IDs. The response remains OpenAI-compatible:
EviBind replaces the internal action call with the trusted materialized tool call
and adds a top-level `evibind` decision record.

`EVIBIND_HANDLE_SECRET` is optional for local development; when absent, each
gateway process creates an ephemeral 256-bit secret.

The transport targets the OpenAI Chat Completions contract used by hosted providers and
local servers such as llama.cpp, vLLM, and SGLang. The release suite exercises protocol
fixtures plus a live llama.cpp backend; each additional provider should pass conformance
before deployment. Omit `EVIBIND_UPSTREAM_API_KEY` when a local server needs no key.

## Container

The image runs as non-root and requires gateway authentication for its public
container bind:

```bash
docker build -t evibind .
docker run --rm -p 8090:8090 \
  -e EVIBIND_UPSTREAM_BASE_URL="https://api.openai.com/v1" \
  -e EVIBIND_UPSTREAM_API_KEY="your-provider-key" \
  -e EVIBIND_GATEWAY_API_KEY="local-gateway-key" \
  evibind
```

`GET /health` is unauthenticated and exposes only status and gateway version.

## What The Gateway Checks

Before the model sees an action domain, EviBind:

1. compiles typed derivations from UTF-8 message spans, versioned state, schema
   defaults, and deterministic versioned transforms;
2. removes candidates that fail origin, destination, evidence-type, value-class,
   transform, state-version, or policy-epoch checks;
3. issues opaque MAC-authenticated handles bound to the request digest, tool,
   JSON Pointer destination, derivation and value digests, state versions,
   transform versions, expiry, and nonce;
4. asks the model once to select `call`, `need_input`, or `no_tool` and, for a
   call, only destination-bound handle IDs;
5. replays selected derivations and validates the complete materialized action
   against the original closed JSON contract before releasing it.

Holding the selected handles fixed, arbitrary free-form model text cannot change
an authority-bearing materialized value. Certificates serialize to JSON and can
be replayed with the same policy and state inputs. Full certificate details are
returned only when server-authorized diagnostics are enabled.

EviBind replaces the upstream model's internal action call but does not itself
execute the resulting tool. The application retains credentials and execution
authority and must use only the gateway response. Setting
`EVIBIND_CONTROLLER_MODE=legacy_literal` restores the former post-hoc literal
path for research reproduction; it does not provide the v2 action-space
guarantee.

## Evidence

The paper's evidence is deliberately split by claim:

- **Information and boundary behavior.** OriginBench-300 contains 300
  equal-valued, differently authorized derivation pairs. CheckerAttack-40 and
  EffectSuite-30 test citation/normalization attacks, state TOCTOU, and executed
  payment/filesystem/SQL effects. EviBind completes 30/30 authorized scripted
  effects with zero harm. Trace-materializing atomic cite-and-check matches only
  after making the authenticated trace authoritative at atomic dispatch; the
  reject-only checker completes none. In Fragility-12, all eight mutations of
  the baseline's additional literal/trace coherence channel are exploitable
  (240/240 harmful effects), whereas EviBind has no critical literal for those
  operators to expose and fails closed 240/240. Four shared trusted-boundary
  violations expose both systems and bound this result. One million
  structure-aware mutations over 26 fields produce zero unsound releases.
- **Fresh-family binding.** On a prospectively specified 100-case test,
  ranker-selected top-1 presentation among verifier-admissible candidates moves
  Qwen3-1.7B from 5% to 89%, Qwen3.6-35B-A3B from 86% to 100%,
  and open GPT-OSS-120B from 97% to 100%. GPT-5.6-Luna, the light tier of a
  hosted frontier family, is 100% in the original arms and 94% under both
  gold+3 and gold+7 hard-distractor stress.
- **Position robustness.** A prospective 200-request stress shows that the
  train-only ranker retains gold in 100/100 gold-late but 0/100 gold-early
  cases. With full catalogs, Qwen3.6-35B is exact on all four gold-early orders
  and on 100/100, 100/100, 87/100, and 83/100 gold-late cases; Qwen3-1.7B is
  index-sensitive. Both models release 0/100 when compact presentation omits
  gold, making the failure safe but unusable. A separately frozen 800-output
  top-2 extension retains gold in all 200 paired requests. Qwen3.6-35B reaches
  100/100/100/100 recall for gold-early and 100/100/90/86 for gold-late, with
  90% all-order exactness. Its four visible candidates match the unpruned
  actual condition, so this is an alternative-retention result rather than a
  prompt-compression result.
- **Stateful task utility.** A matched intervention reduces tool exceptions but
  does not recover task similarity. It rules out two repaired local mechanisms
  as sufficient explanations; the remaining losses manifest as multi-turn
  trajectory divergence.

These are synthetic mechanism and systems results, not human-subject evidence
or a universal production guarantee. The proxy/subagent role-play corpus is not
reported as independent human authorship. BoundaryBench-v1 and the paper
evidence bundle are versioned release assets; see
[`docs/REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

The verifier establishes structural admissibility, not which admissible value
the user intended. Compact top-1 presentation is therefore an opt-in usability
policy, not a security verdict. The measured top-2 fallback preserves the
minimal alternatives in the present two-slot design; a deployment should expose
multiple admissible candidates or abstain whenever selector confidence or
semantic coverage is insufficient. Verification still applies independently to
every released call.

## Product Surface

- `evibind serve`: OpenAI-compatible HTTP gateway at `/v1/chat/completions`.
- `evibind.GuardedToolExecutor`: compile with full host context and dispatch
  only a single-use, trusted materialized call through registered handlers.
- `evibind chat --request request.json`: compile candidates, make one upstream
  handle-selection call, and protect one request from the CLI.
- `evibind lint-schema --strict --request request.json`: validate annotations, close
  public contracts, and emit private/provider schema fingerprints.
- `evibind init --request in.json --output out.json`: scaffold a conservative
  typed private policy without overwriting existing files by default.
- `evibind inspect --request request.json`: compile candidates, missing slots,
  and rejection reasons without a provider call.
- `evibind.execution.ExecutionCoordinator`: recompile clarification turns over
  immutable compare-and-swap execution records.
- `evibind.effects`: require exact-manifest confirmation before release.
- `evibind replay --request request.json --certificate response.json`: verify
  one diagnostic certificate using `EVIBIND_HANDLE_SECRET`.
- `tapbench`: benchmark generation, scoring, conformance, and analysis commands.
- `GET /health`: unauthenticated liveness check with no provider details.
- `EVIBIND_ALLOW_DIAGNOSTICS=true` plus
  `evibind.include_diagnostics=true`: include the full replayable resolution trace.

The v2 gateway supports non-streaming, single-call Chat Completions requests. It
enforces `n=1`, sets `parallel_tool_calls=false`, and fails closed if an upstream
returns multiple choices or proposed calls. Streaming, parallel calls, multi-turn
tool plans, native Anthropic/Google HTTP transports, and production telemetry are
on the roadmap in `docs/PRODUCT.md`; pure envelope adapters are available now.
Provider-specific startup requirements and the validated compatibility matrix
are in `docs/PROVIDERS.md`.

## Development

```bash
python -m pip install -r requirements/reproducibility.txt
python -m pip install -e . --no-deps
python -m pytest -q
python -m tapbench.cli conformance
python examples/minimal_evidence_binding.py
python scripts/reproduce_public_artifact.py \
  --output-dir /tmp/evibind-public-reproduction \
  --fuzz-trials 10000
```

Use `--full` on the reproduction script to run the paper's one-million-trial
boundary fuzz. Model-backed experiments require separately obtained model/API
access and must use the frozen configs and no-retry rules documented in the
reproducibility guide. Protocols, model pins, amendments, response digests, and
discipline gates are retained rather than silently regenerated.

## Security And Scope

EviBind refuses non-loopback binding unless `EVIBIND_GATEWAY_API_KEY` is set.
Provider credentials are read from the environment; redirects are rejected, and
upstream error bodies are redacted unless diagnostics are explicitly enabled.
Request and upstream response bodies are bounded at 10 MiB, and released calls
are rebuilt from validated canonical fields. EviBind is an admission layer, not
a sandbox, authorization service, or substitute for least-privilege tool
credentials. See `SECURITY.md` before a networked deployment.

EviBind is released under the MIT License. Contributions are welcome; see
`CONTRIBUTING.md`.
