# Host-Owned Guarded Execution

The `evibind.host` API is the full-context integration path for applications
that want EviBind to own the boundary immediately before tool execution. It
combines four operations in one single-use turn:

1. compile locally valid, request-scoped evidence candidates;
2. replace executable tools with the one-call `evibind_action` handle schema;
3. verify and materialize the provider response; and
4. dispatch only the materialized call through a registered handler.

The host supplies the complete messages, private slot policy, trusted state, and
tool handlers. A provider never receives private `x-evibind-*` annotations or
executable authority-bearing literals.

## Minimal integration

```python
from evibind.host import GuardedToolExecutor


def pay_invoice(arguments):
    return billing.pay(amount=arguments["amount"])


executor = GuardedToolExecutor(
    {"pay_invoice": pay_invoice},
    handle_secret=secret_from_a_secret_manager,
)

turn = executor.prepare(request)
provider_payload = turn.upstream_payload
provider_response = model_client.chat_completions(provider_payload)
outcome = turn.complete(provider_response)

if outcome.executed:
    print(outcome.tool_id, outcome.manifest_digest, outcome.result)
else:
    print(outcome.decision, outcome.protected_response)
```

`turn.complete(...)` is the only dispatch path. It admits at most one call and
is atomically single-use, including when validation or a handler fails. Do not
send the materialized call in `outcome.protected_response` to a second generic
tool dispatcher.

Every request tool must have a registered handler before the model is called.
Handlers receive an isolated copy of the exact arguments covered by the action
manifest. Handler mutation cannot change the recorded manifest or arguments.
The synchronous v1 SDK rejects coroutine handlers and awaitable results.

## Effect confirmation

Request-level `evibind.effect_policies` work in the host SDK exactly as in the
gateway. For a confirmation-required effect, the first completed turn returns
`decision="confirmation_required"` and executes nothing. Present the complete
manifest to the user, copy the returned challenge token into
`evibind.effect_confirmation`, prepare a new turn, and submit a new one-call
proposal. The token is bound to the request, tool, materialized manifest, policy
epoch, effect policy, expiry, and nonce. It is atomically single-use.

The confirmation approves the exact manifest, not the tool name in general.
Changing an argument, request, policy epoch, or effect class invalidates it.

## Guarantee boundary

The host SDK supports only `enforce` and `assist`. It refuses `audit` because an
audit response preserves model-generated executable literals.

This path can provide complete mediation when the application:

- gives EviBind the evidence-bearing conversation and trusted state;
- registers every executable tool only with `GuardedToolExecutor`;
- calls tools only through `GuardedTurn.complete`; and
- does not retain another unguarded executor path.

The OpenAI-compatible gateway remains a replace-the-call integration: complete
mediation there depends on the application executing only the returned
materialized call. A plain server-side MCP proxy normally lacks the host
conversation and therefore cannot claim full evidence binding without an
explicit evidence-context extension. The host SDK is the recommended adapter
surface for Agents SDK, MCP host, FastMCP, and similar integrations.

EviBind admission does not establish user authorization, tool-selection
correctness, candidate intent, effect desirability, or tool implementation
correctness. Business authorization, sandboxing, and application policy remain
separate controls.
