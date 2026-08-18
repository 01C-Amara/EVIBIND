# EviBind public API and algorithm map

EviBind has three surfaces with different stability guarantees.

## Stable product surface

Import product-facing objects from `evibind`, not internal modules:

```python
from evibind import (
    EviBindGateway,
    GatewayConfig,
    GuardedToolExecutor,
    lint_tool_schemas,
)
```

- `EviBindGateway` and `GatewayConfig` expose the OpenAI-compatible admission
  gateway.
- `GuardedToolExecutor` owns complete mediation and dispatches only the trusted
  materialized manifest.
- `lint_tool_schemas` validates private evidence annotations and the public
  closed contract.
- The `evibind` CLI exposes `serve`, `chat`, `lint-schema`, `init`, `inspect`,
  and `replay`.

Breaking changes to this surface require a package-version change and changelog
entry. Provider-specific envelopes are normalized before the evidence boundary;
they do not alter the trusted materialization semantics.

## Low-level evidence-binding surface

`evibind.core` exposes the paper's reference construction:

1. `EvidenceContext` fixes messages, versioned state, and policy epoch.
2. `CandidateProposal` describes a typed derivation for one tool destination.
3. `compile_candidates` evaluates proposals and rejects unauthorized origins,
   stale state, type violations, destination mismatch, and policy mismatch.
4. `HandleIssuer` or `Ed25519HandleIssuer` authenticates request-local,
   destination-bound witnesses.
5. The model emits an `ActionProposal` containing handle IDs rather than
   executable critical literals.
6. `materialize` verifies and replays the selected derivations, assembles the
   closed argument object, and returns a `MaterializationCertificate`.
7. `replay_materialization` independently checks the certificate against the
   same policy and state inputs.

Run `examples/minimal_evidence_binding.py` for a deterministic example. The core
objects are versioned through constants such as `ACTION_IR_VERSION`,
`BINDING_WITNESS_VERSION`, and `MATERIALIZER_VERSION`. Serialized artifacts must
be rejected when their version is unsupported; do not coerce them silently.

## Research surface

`tapbench` contains reusable evaluation algorithms and is versioned by artifact
rather than by a strict library compatibility promise:

- `equal_value_benchmark`: equal-valued, differently authorized derivations;
- `adversarial_boundary`: citation/normalization, state-TOCTOU, and executed
  effect scenarios;
- `boundary_fuzz`: deterministic mutation of authenticated release fields;
- `verified_ranker`: train-only candidate ranking followed by deterministic
  structural verification. Verification certifies that a derivation is
  admissible for a destination; it does not certify intendedness;
- `candidate_position_robustness`: paired mention-order construction,
  deterministic catalog permutations, and family-cluster analysis;
- `confirmatory` and `confirmatory_inference`: prospective case construction and
  response analysis;
- `stateful_failure_taxonomy`: mutually exclusive stateful failure attribution;
- `nonce_store`: exactly-once nonce interface and process-local reference store;
  and
- `paper_audit`: claim-ledger and manuscript consistency checking.

Use the scripts in `scripts/` as the canonical command-line entry points for
research artifacts. Each output should carry an explicit schema version,
complete parameters, deterministic ordering, and a checksum. Model-backed
scripts must preserve every attempted row and must not retry unless a published
protocol explicitly allows it.

Treat a ranker's top-1 output as an optional presentation heuristic. When its
confidence or semantic coverage is inadequate, preserve the admissible set for
model/user disambiguation or abstain; never relabel top-1 as verified intent.

## Security contract

EviBind confines configured authority- and effect-bearing argument leaves. It
does not determine user intent, business authorization, policy completeness,
tool correctness, or downstream effect safety. Applications must keep tool
credentials outside the model/gateway, use least privilege, mediate every tool
call, and provide a shared linearizable nonce store for multi-worker deployment.
See `SECURITY.md` for the complete deployment boundary.
