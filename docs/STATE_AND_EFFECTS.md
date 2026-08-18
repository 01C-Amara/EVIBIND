# Stateful Execution And Effect Authorization

EviBind separates four decisions that ordinary tool calling often collapses:

1. compile request-local evidence;
2. select and materialize an argument manifest;
3. authorize the external effect;
4. dispatch the authorized action.

The separation matters because valid arguments do not imply that an effect is
authorized.

## Immutable Execution Graph

`evibind.execution` exposes an immutable, compare-and-swap execution record.
The supported path is:

```text
created
  -> selecting
  -> awaiting_clarification -> selecting
  -> materialized
  -> awaiting_confirmation -> authorized
  -> dispatched
```

`no_tool`, `failed`, and `cancelled` are terminal alternatives. Every transition
requires the caller's expected state version. Records validate the complete
transition chain, reject events after terminal nodes, and reject request-digest
changes outside the clarification edge.

`ExecutionCoordinator` compiles a one-call session, applies one model response,
and can append a user clarification. Clarification always recompiles the
candidate domain against a new request digest. Handles from the previous turn
therefore cannot be selected in the new turn. If earlier user evidence should
remain admissible, the reviewed slot policy must explicitly include
`user.prior_turn`; this is not silently inferred by the coordinator.

The coordinator is an in-process orchestration surface. It does not persist
records or dispatch tools. A production service should store each immutable
record with optimistic concurrency using `state_version`.

## Trust Labels

The core labels derivation roots as:

- `user_explicit` for the current user turn;
- `user_context` for prior user turns;
- `state_authorized` for versioned state;
- `schema_owned` for defaults and enumerations;
- `tool_untrusted`, `model_untrusted`, or `unknown` for untrusted roots.

Trust labels are diagnostics, not a substitute for the slot source policy.
There is also a non-overridable floor: authority-bearing and effect-bearing
values cannot be compiled from untrusted tool/model roots even if a schema
mistakenly lists that source. Explicitly configured untrusted roots may still
supply `opaque_content`; that content does not acquire authority.

Released one-call summaries contain a trust assessment without materialized
values. `explicitly_effect_authorizing=true` only means that at least one root
came from the current user turn and no root was labeled untrusted. It does not
replace effect confirmation.

## Effect Policies

Declare private effect policies under the request's `evibind` object:

```json
{
  "evibind": {
    "effect_policies": {
      "pay_invoice": {
        "effect_class": "external_write",
        "confirmation": "required",
        "ttl_seconds": 300
      }
    }
  }
}
```

Supported classes are `read_only`, `reversible_write`, `external_write`, and
`irreversible`. If `confirmation` is omitted, `read_only` defaults to
`not_required` and every write class defaults to `required`. Unknown tools,
fields, classes, confirmation policies, and TTLs fail before the provider call.

When a required-confirmation action materializes, EviBind removes its executable
tool call and returns `decision=confirmation_required` with a challenge token.
The token contains no arguments. It is authenticated and bound to:

- the request digest;
- tool ID;
- exact materialized manifest digest;
- policy epoch;
- effect class and complete effect-policy digest;
- expiry and a random nonce.

Resubmit the same request and effect policy with:

```json
{
  "evibind": {
    "effect_confirmation": "<challenge token>"
  }
}
```

EviBind releases only if the new materialized action has the exact bound
manifest and every other binding still matches. A changed request, tool,
argument manifest, policy epoch, effect class, or policy rejects the token.
Successful consumption is atomic and single-use within one gateway process.

## Deployment Boundaries

The built-in consumed-nonce store is process-local and intentionally has no
pretend distributed guarantee. A multi-process or multi-replica deployment must
replace it with a shared, atomic, TTL-backed nonce store before claiming global
single-use confirmation. A shared implementation can satisfy
`evibind.effects.ConsumedNonceStore`, be passed to
`EffectAuthorizer(nonce_store=...)`, and then be injected into the gateway.
confirmed authorization does not make an external tool itself exactly-once.

Effect policies require the one-call controller in `enforce` or `assist` mode.
They are rejected in non-enforcing `audit` mode. Challenges are bearer
credentials: keep `Cache-Control: no-store`, avoid logging them, use TLS, and
limit gateway access.
