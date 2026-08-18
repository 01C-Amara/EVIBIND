# EviBind Product Plan

## Product Thesis

Most LLM gateways route requests, normalize provider APIs, track cost, or apply
content guardrails. EviBind should occupy a narrower and defensible layer: an
evidence-admission sidecar between model generation and tool execution.

The user supplies an existing model API and JSON tool schemas annotated with
slot evidence policies. EviBind compiles valid request-scoped derivations before
inference, replaces executable tools with a single handle-selection Action IR,
and alone materializes the selected values. The model never has an executable
literal channel for authority-bearing arguments.

The product metric is the exact-task utility and selective-risk/coverage frontier,
plus compiler recall, one-call token cost, and deterministic gate latency.

## Current MVP

The repository now contains a usable v2:

- OpenAI-compatible `/v1/chat/completions` gateway;
- host-owned, single-use guarded execution through an exact registered tool
  handler map;
- any OpenAI-compatible remote or local upstream;
- one model call over `call`, `need_input`, and `no_tool` Action IR modes;
- typed span, state, default, enum, composite, and versioned-transform
  derivations;
- local origin, destination, evidence-type, transform, value-class, state, and
  policy filtering before model selection;
- opaque MAC-authenticated handles bound to request, tool, destination,
  derivation, value digest, policy epoch, state versions, transform versions,
  expiry, and nonce;
- trusted materialization, recursive JSON-contract validation, and JSON
  certificate replay;
- conservative `init`, recursive strict linting, provider-free candidate
  inspection, and authenticated archived replay commands;
- server-controlled `enforce`, fail-closed `assist`, and explicitly
  non-enforcing shadow `audit` modes;
- dependency-free envelope adapters for OpenAI Chat and Responses, Anthropic
  Messages, and Google Interactions and `generateContent`;
- immutable compare-and-swap execution records and an in-process clarification
  coordinator that recompiles request-bound handles on every user turn;
- source trust labels plus a global trust floor for authority/effect values;
- private effect classes and exact-manifest, short-lived confirmation challenges
  with atomic process-local single-use enforcement;
- fail-closed handling for malformed, unmediated, multiple, or expired calls;
- environment-only provider credentials and mandatory authentication for
  non-loopback binding;
- fail-closed upstream redirects, bounded request/response bodies, redacted
  upstream errors, and canonical released-call serialization;
- zero new runtime dependencies beyond the existing package.

The `evibind.host` SDK owns candidate compilation, protection, exact-manifest
effect gating, and registered-handler dispatch inside one atomically single-use
turn. This is the recommended full-context integration for applications that
can route every tool through EviBind.

The gateway replaces the provider's internal `evibind_action` call with a
materialized OpenAI-compatible tool call. It never holds tool credentials or
executes the call. Complete mediation in gateway mode therefore still requires
the application to use only the returned materialized call.

## API Contract

EviBind accepts a standard Chat Completions body plus an optional private `evibind`
object:

```json
{
  "model": "provider-model",
  "messages": [{"role": "user", "content": "Pay amount=20"}],
  "tools": [],
  "evibind": {
    "policy_epoch": "2026-07",
    "dialogue_state": {},
    "include_diagnostics": false
  }
}
```

Recommended parameter annotations are:

- `x-evibind-evidence-type`: a reusable type such as `email_address`,
  `phone_number`, `iso_date`, `currency_amount`, `person_ref`, `account_ref`,
  `schema_enum`, or `opaque_registry_id`;
- `x-evibind-sources`: exact admissible origins such as `user.current_turn` or
  `state.contacts`;
- `x-evibind-transforms`: audited pure transform IDs allowed for the slot;
- `x-evibind-extraction-cue`: an exact schema-authorized cue such as `amount` for
  request text like `amount=20`;
- `x-evibind-criticality`: `target`, `control`, `content`, or `effect`;
- `x-evibind-value-class`: `authority_bearing`, `opaque_content`, or
  `effect_bearing`;
- `x-evibind-on-ambiguity`: `clarify`, `reject`, or `confirm`.

EviBind infers only stable schema formats and conservative name/type cases.
Unknown action-critical slots fail closed and must declare an evidence type.
Production schemas should be annotated and linted with
`evibind lint-schema --strict --request request.json`; the report fingerprints
both the private contract and the provider-visible stripped schema.

## Positioning

EviBind should integrate with existing gateways rather than compete on routing,
billing, or observability. Deploy it after a provider gateway and before the
application's tool dispatcher:

```text
application -> EviBind candidate compiler -> provider model
                    |                            |
                    |<---- handle Action IR -----|
                    v
          verifier + materializer -> tool dispatcher
```

The open-source core should remain provider-neutral and self-hostable. A hosted
offering can add managed schema linting, calibration, replay dashboards, signed
certificates, and compliance retention without closing the verifier.

## Naming Due Diligence

EviBind replaces the working name TAP-R, which collides with Tree of Attacks with
Rubric Based Scoring in LLM red teaming. As checked on 2026-07-27, exact-name
searches found no EviBind repository on GitHub and the `evibind` endpoints on
PyPI and npm returned 404. This is a collision screen, not trademark clearance.

## Beta Milestones

1. Completed: policy initializer, recursive schema linter, private/provider
   fingerprints, and local candidate inspector.
2. Completed: pure OpenAI Responses, Anthropic Messages, and Google tool-call
   envelope adapters around one canonical Action IR. Native HTTP transports and
   live-provider certification remain pending.
3. Completed: synchronous host SDK with full conversation/state context,
   registered handler coverage, atomically single-use turns, materialized
   manifest verification, and exact-manifest effect confirmation.
4. Pending: buffer streaming arguments and release only after certification.
5. Partially completed: immutable state versions, clarification recompilation,
   and per-effect authorization ship for single calls. Persisted multi-turn and
   multi-call orchestration remain pending.
6. Pending: export OpenTelemetry and Prometheus counters without logging request
   contents by default.
7. Partially completed: authenticated witness certificates and deterministic
   offline replay ship; a compact whole-certificate signature is pending.
8. Partially completed: request-scoped effect policies and confirmation hooks
   ship. Tenant risk budgets and a distributed approval/nonce store remain
   pending.
9. Pending: publish typed Python and TypeScript response models.

## Evaluation Before A Production Claim

Every release should report:

- accepted-call exact precision and its interval;
- call coverage and non-call outcome accuracy;
- unsupported action-critical rate;
- precision/coverage curves across risk budgets;
- latency and model-call overhead;
- results by tool family, schema shape, language, provider, and model size;
- failures under prompt injection, corrections, negation, stale state, and
  cross-slot span conflicts.

The first frozen product-path benchmark replays identical ToolSandbox requests
through native provider tool calling and EviBind on held-out stateful families.
It reduces tool-call exceptions but also reduces official milestone similarity;
that negative result is a boundary on task-effectiveness claims, not evidence of
end-to-end agent safety. BFCL remains the recognizable external accuracy anchor.

## Non-Goals

The host SDK is a narrow registered-handler dispatcher, not a network sandbox,
IAM system, prompt-injection solution, general planner, or proof that a model's
direct answer is correct. The HTTP gateway does not own execution. A dispatched
call is only as trustworthy as its compiler, schema contract, state source,
handler registration, and certificate implementation.
