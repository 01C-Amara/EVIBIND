# Security Policy

EviBind is an admission layer. It does not replace tool authorization, network
sandboxing, least-privilege credentials, rate limits, or human approval for
high-impact actions.

## Deployment Requirements

- Non-loopback binding is refused unless `EVIBIND_GATEWAY_API_KEY` is set.
- Supply provider credentials only through `EVIBIND_UPSTREAM_API_KEY` or the local
  provider's own secret mechanism.
- Set `EVIBIND_HANDLE_SECRET` to at least 32 random bytes for production and
  protect it like a signing key. Without it, handles use a process-local
  ephemeral key and certificates cannot be verified after restart.
- Upstream redirects are rejected so provider authorization is never forwarded
  to a redirected origin. Request and upstream response bodies are capped at
  10 MiB.
- Terminate TLS at a trusted reverse proxy for any remote deployment.
- Keep tool credentials in the application or dispatcher; never give them to the
  model or EviBind gateway.
- Treat schema contracts, reference context, and dialogue state as security
  inputs. Version and authorize them independently of user text.
- Full diagnostics require `EVIBIND_ALLOW_DIAGNOSTICS=true`; do not enable
  them for untrusted clients because traces can contain request spans and
  materialized values. Upstream error bodies are redacted unless this operator
  flag is enabled.
- Treat `EVIBIND_OPERATING_MODE=audit` as non-enforcing. It preserves native
  executable literals for shadow comparison, sets `enforced=false`, and must not
  feed a production dispatcher unless those native calls are independently
  authorized.
- Authority- and effect-bearing values have a global trust floor: tool/model
  output cannot supply them even when a schema mistakenly lists that untrusted
  source. Untrusted roots may supply explicitly configured opaque content only.
- Deterministic verification establishes that a candidate is structurally
  admissible for a destination; a learned or heuristic ranker does not establish
  user intent. Enable compact top-1 presentation only under a calibrated policy.
  The prospective position study makes retaining alternatives or abstaining a
  measured requirement under selector uncertainty; its top-2 arm is the minimal
  alternative-preserving fallback for the evaluated two-slot construction.
- Removing the critical model literal eliminates literal/trace coherence faults;
  it does not make the trusted boundary implementation-proof. Authentication,
  replay, destination scoping, complete mediation, and sink sandboxing remain
  required. The public fragility study includes symmetric shared-boundary bypass
  controls specifically to prevent a broader interpretation.
- Treat effect-confirmation challenges as bearer credentials. They are bound to
  the request, tool, exact argument manifest, policy, epoch, and expiry, but must
  still be protected with TLS and excluded from logs.
- Built-in confirmation replay tracking is atomic but process-local. Use a
  shared TTL-backed nonce store before deploying multiple workers or replicas,
  and use dispatcher idempotency keys because external effects are not made
  exactly-once by EviBind.
- Persist immutable execution records with optimistic concurrency. Never accept
  a client-supplied state transition without checking its expected version and
  full transition history.

In enforce and assist modes, the model receives only the internal
handle-selection tool. Released calls are rebuilt by replaying authenticated
request/tool/destination-bound derivations and checking the final JSON contract.
Duplicate tool names and legacy, malformed, unmediated, parallel, expired, or
extra upstream call fields are rejected or not
forwarded. Responses use
`Cache-Control: no-store` and `X-Content-Type-Options: nosniff`.

The v2 gateway intentionally enforces `n=1` and rejects streaming, multiple
completion choices, and multiple tool calls. It does not inspect direct answers
for prompt injection or data leakage.

## Reporting A Vulnerability

Do not open a public issue for a vulnerability that exposes credentials, bypasses
the release gate, or allows unauthorized execution. Send a private report to the
repository maintainers with a minimal reproducer, affected version, and impact.
Please avoid including real provider keys or sensitive request contents.

## Public Release Hygiene

Release archives must pass `scripts/audit_release_archives.py`. The audit rejects
generated experiment trees, generated paper outputs, machine-local paths,
credential-like material, private keys, participant records, and proxy-human
handoff data. Provider-backed reproduction must read credentials from the
environment and keep raw response artifacts out of public logs until they have
been reviewed for sensitive content.
