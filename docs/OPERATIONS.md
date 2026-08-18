# Policy Operations

EviBind's operator workflow is local-first: initialize a conservative private
policy, review and lint it, inspect the compiled request domain without calling
a model, then run the gateway in an explicitly selected mode.

## Initialize And Inspect

Initialize a copy of an existing Chat Completions request:

```bash
evibind init \
  --request request.json \
  --output request.evibind.json \
  --policy-epoch 2026-07

evibind lint-schema \
  --strict \
  --request request.evibind.json

evibind inspect \
  --request request.evibind.json \
  --output inspection.json
```

`init` does not overwrite an existing output unless `--force` is supplied. It
closes object schemas that did not declare `additionalProperties`, assigns
standard evidence types from stable schema/name cues, adds exact sources,
criticality, value class, ambiguity handling, and extraction cues, and records a
policy epoch. This is a scaffold, not an authorization decision: an owner must
review inferred sources and criticality before enforcement.

`inspect` performs no provider call. It reports the request digest, policy and
contract versions, valid candidate handles, missing required destinations, and
local rejection reasons. Candidate rows intentionally omit the server-side
materialized value, though their display labels can contain request excerpts and
must still be treated as sensitive operational output.

## Operating Modes

Set the server-controlled mode with `EVIBIND_OPERATING_MODE`:

- `enforce` (default): the model selects only evidence handles. EviBind
  materializes valid calls and withholds invalid or incomplete calls.
- `assist`: the same enforced action space, with a user-facing clarification
  message when runtime-derived required evidence is missing.
- `audit`: the provider receives the original native tool schemas. EviBind
  preserves the provider's executable literal call and attaches a shadow
  `would_release` decision.

Audit mode is deliberately non-enforcing. Its response sets `enforced=false`,
sets `selective_guarantee` to `null`, and carries a warning. Do not connect audit
responses to a production tool dispatcher unless native model calls are already
authorized to execute.

`audit` and `assist` require the v2 `one_call` policy compiler. The
`legacy_literal` research controller can run only in `enforce`.

## Stateful Clarification And Effects

The `evibind.execution` API provides immutable state versions and recompiles a
new handle domain after each user clarification. The `evibind.effects` API and
`evibind.effect_policies` request option can require exact-manifest confirmation
after materialization but before release. See
[`STATE_AND_EFFECTS.md`](STATE_AND_EFFECTS.md) for the state machine, request
shape, trust labels, replay behavior, and multi-replica boundary.

## Certificate Replay

Production replay requires a stable secret:

```bash
export EVIBIND_HANDLE_SECRET="at-least-32-random-bytes-kept-stable"
export EVIBIND_ALLOW_DIAGNOSTICS=true
```

Request `evibind.include_diagnostics=true` to receive the full materialization
certificate, then verify it offline:

```bash
evibind replay \
  --request request.evibind.json \
  --certificate protected-response.json \
  --output replay.json
```

The replay command finds exactly one certificate, authenticates each witness,
re-evaluates every derivation from the supplied request, policy epoch, defaults,
and versioned state, verifies transform and state versions, reconstructs the
manifest digest, and rechecks the current JSON contract. Archived replay permits
an expired witness by default because expiry governs live selection; MAC,
request, policy, state, transform, value, destination, and contract bindings
remain mandatory.

## Provider Envelope Adapters

`evibind.adapters` provides pure, dependency-free envelope mappings for:

- OpenAI Chat Completions;
- OpenAI Responses;
- Anthropic Messages;
- Google Gemini Interactions;
- Google Gemini legacy `generateContent`.

The adapters encode the single `evibind_action` tool and normalize provider tool
calls back to the canonical materializer envelope. They do not yet add native
HTTP transports, authentication, retry behavior, streaming buffers, or live
provider certification. The shipped gateway endpoint remains OpenAI Chat
Completions-compatible.
