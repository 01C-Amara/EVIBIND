# Architecture and repository contract

EviBind is both an installable enforcement component and the executable
reference construction evaluated by the paper. The same semantics serve both
roles, but their interfaces, data, and stability promises are intentionally
different.

## Runtime data flow

1. The application supplies messages, versioned trusted state, tool contracts,
   and a reviewed private evidence policy.
2. The compiler enumerates typed derivations for each protected JSON Pointer and
   rejects inadmissible roots, transforms, versions, and value classes.
3. The issuer authenticates request-, tool-, destination-, and policy-bound
   witnesses. Only opaque handles enter the model-facing ActionIR catalog.
4. The model chooses `call`, `need_input`, or `no_tool`. A call contains tool and
   handle identifiers, not executable protected literals.
5. The materializer verifies every witness, replays its derivation against the
   current immutable context, assembles the closed arguments, and emits a
   certificate.
6. The host SDK verifies the manifest binding and invokes the registered handler
   at most once. An effect policy may insert an exact-manifest confirmation gate.

The model/provider is outside the trusted computing base. The compiler,
authentication and replay code, policy and version registries, nonce store,
closed-contract validator, and complete dispatcher are inside it.

## Module ownership

| Layer | Modules | Compatibility |
|---|---|---|
| Stable product facade | `evibind`, `evibind.core`, `evibind.host`, `evibind.schema` | semantic version and changelog |
| Shared trusted engine | gateway, one-call controller, policy and effect modules currently under `tapbench` | internal unless re-exported by `evibind` |
| Research algorithms | benchmark generators, scorers, rankers, analyses under `tapbench` | frozen by artifact schema and digest |
| Experiment orchestration | `scripts/`, `bench/`, `providers/` | protocol-specific; never imported by runtime request paths |
| Evidence releases | `evidence/*.json` plus external archives | immutable, content-addressed |

The wheel includes `tapbench` because the current stable facade uses its shared
engine. That does not make every `tapbench` symbol a supported product API.
Consumers should import from `evibind`; moving shared engine code behind that
facade remains an internal refactor.

## Dual-use invariants

- Product code never reads gold labels, benchmark case metadata, preregistration
  files, archived model responses, or paper claims.
- Research drivers call the same public compiler and materializer used by the
  gateway; they do not maintain a permissive shadow implementation.
- Every scientific output has a schema version, deterministic ordering, complete
  parameters, and cryptographic digests. Timing metadata is excluded from
  deterministic comparisons.
- Hosted model regeneration is not assumed bit-exact. Frozen cases, prompts,
  catalogs, decoding settings, raw envelopes, and no-retry rules are retained.
- Large evidence archives are release assets. Git stores the verifier and exact
  release record, preventing an archive from being silently substituted.
- `audit` mode is observational and never feeds the production dispatcher.
- A passing certificate establishes policy-admissible derivation and replay,
  not business authorization or intendedness.

## Release boundaries

A product release must pass unit and integration tests, conformance, wheel and
source-distribution installation, archive hygiene, and container build. A paper
evidence release additionally requires the claim-ledger audit, internal digest
manifest, canonical PDF digest, frozen protocols and amendments, privacy scan,
and a checked-in `evidence/*.json` record.

The two releases can share a version but are not interchangeable: a package
release is an executable dependency; an evidence bundle is an immutable record
of exactly what supports the manuscript.
