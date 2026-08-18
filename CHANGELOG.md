# Changelog

All notable EviBind changes are documented here.

## Unreleased

- Ran InjectBench live against nine models: GPT-5.6 Terra/Luna/Sol,
  GPT-5.4 mini/nano, GPT-4.1 mini, and Grok 4.6/4.5. Native harmful bindings
  range from 0/60 to 43/60 across the origin-violation set; behind the gateway
  every model reaches 0/60, and on the weaker tiers the correct-call count rises
  (17→45, 18→43, 8→35) because the slot is re-derived rather than merely
  blocked. Cross-model chart in `assets/bench_models.svg`.
- Added `native_slot` / `guarded_slot` scoring for the critical slot alone.
  Whole-call equality was demoting correct bindings to `other` whenever a model
  wrote `"500.00 USD"` for an incidental amount slot, which hid the result the
  benchmark exists to measure.
- Added `bench/adapters.py`: repairs `tool` messages that answer no assistant
  call (OpenAI rejects them; 60 cases were affected), drives the GPT-5.6 tiers
  through `/v1/responses` since they refuse function tools on chat completions,
  and runs cases concurrently with bounded retries.
- Added `bench/run_grok_cli.py` for Grok access via a grok.com subscription,
  where there is no API key and no OpenAI-compatible endpoint.
- Added `bench/summarize_all.py` and `providers/run_openai_suite.sh`.
- `--api-key` now accepts `file:PATH` and `env:NAME`, so credentials stay off
  command lines and out of shell history.
- **Fixed: the gateway could not use `api.openai.com` as an upstream at all.**
  OpenAI rejects a function schema with a top-level `oneOf`, which is what the
  action tool emitted, so every request 400'd before the model was consulted.
  The branch union now sits under one required `action` property; both proposal
  parsers accept either shape, so existing certificates still replay. Read the
  branches with the new `action_branches()` helper rather than indexing `oneOf`.
  Verified live: 150/150 cases served end to end against `gpt-5.4-nano`.
- Added `bench/run_gateway_e2e.py` and `examples/live_gateway_demo.py`, which
  exercise `evibind serve` itself rather than the offline binding path.
- **Fixed: cue-based extraction offered only the greedy reading of a span**,
  so `"...account ACC-4000 - that is..."` proposed `"ACC-4000 -"` and `"I have"`
  but never `"ACC-4000"`. Junk candidates made the slot look ambiguous, an
  ambiguous required slot was reported as a missing destination, and a missing
  destination removed the `call` branch from the action schema — leaving every
  model, from nano to Sol, with `need_input` as its only legal answer. Both
  readings are now offered narrowest first, a greedy reading whose trailing
  token has no alphanumeric character is dropped, the narrow reading strips
  sentence punctuation, and the cue must be a whole word (`account` no longer
  matched inside `accounts`, offering the leftover `"s"`).
- **Fixed: `account_ref`, `order_ref`, `event_ref` and `opaque_registry_id`
  validated with `nonempty_string`**, so any phrase was an admissible
  identifier. They now require a single identifier token carrying at least one
  digit or separator — which admits `ACC-4000`, an IBAN, `/safe/report-000` and
  an ARN, and rejects `I`, `the`, `I have` and `ACC-4000 -`. This narrows
  admissibility only: a value that fails is withheld, never substituted.
  `person_ref` is deliberately unchanged, because names are multi-word.
- Corrected two `bench/cases.py` annotation errors the end-to-end run exposed:
  `amount` was typed `currency_amount`, a *structured* type that validates
  `{"amount": 240.0, "currency": "USD"}` and can never validate the string the
  schema declares; and `recipient`/`path` carried extraction cues absent from
  their own prose, though their evidence types have shape patterns that locate
  the value without one.
- End-to-end result: GPT-5.4 nano through `evibind serve` against live OpenAI
  goes from 0/150 correct, 15 malformed to **86/150 correct, 0 harmful, 0
  malformed**. All four origin-violation families complete.
- Measured gateway overhead: `+0.25s` median against `gpt-5.4-nano`
  (0.69s direct, 0.94s guarded), one round trip, no second model call.
- Fixed `pytest tests -q`, the command CI runs: sixteen test modules import the
  paper's `scripts` package, which this repo does not ship, so collection failed
  outright. `tests/conftest.py` now skips them when `scripts` is absent.
- Fixed `.env` handling — the file held a bare key with no `VAR=` prefix, so
  `run_all.sh` could never source it.
- Defined EviBind explicitly as evidence binding and documented the stable
  product, low-level reference, and research API surfaces.
- Added a deterministic provider-free evidence-binding example and a one-command
  public reproduction driver for OriginBench, CheckerAttack, EffectSuite, and
  boundary fuzzing.
- Added a tested environment snapshot, claim-to-artifact map, citation metadata,
  complete paper reproduction guide, and CI smoke tests for published commands.
- Extended release-archive auditing to reject credential material, private keys,
  participant records, proxy-human handoffs, generated outputs, and machine-local
  paths.
- Made BoundaryBench-v1 self-contained by including the boundary fuzzer and
  correcting every embedded command to include its required output arguments.
- Split the atomic cite-and-check effects baseline into reject-only and
  trace-materializing variants; the latter matches EviBind's 30/30 safe
  completions while retaining a redundant executable-literal field.
- Added Fragility-12, a prospectively specified single-fault study separating
  eight redundant literal/trace coherence mutations from four symmetric shared
  trusted-boundary controls, and integrated it into provider-free reproduction
  and BoundaryBench.
- Added a prospective mention-order and catalog-permutation robustness study,
  including family-cluster inference, gold-index accuracy, index-selection bias,
  and within-case consistency metrics.
- Added a separately frozen 800-output admissible top-2 extension that retains
  gold in all paired requests, reaches 90% all-order exactness for Qwen3.6-35B,
  and distinguishes alternative retention from prompt compression.
- Clarified throughout the public contract that deterministic verification
  certifies structural admissibility rather than intendedness; compact top-1 is
  opt-in and must abstain or expose alternatives under selector uncertainty.
- Added a host-owned, single-use guarded executor that compiles with full
  conversation/state context and dispatches only exact materialized manifests.
- Integrated fail-closed registered handlers and single-use effect confirmation.
- Implemented all seven preregistered one-call powered EviBench conditions,
  keeping unsafe comparison conditions confined to the evaluation harness.
- Added corpus, policy, authorship, blind double-annotation, independent
  adjudication, and family-disjoint split freeze gates for the powered study.
- Hash-pinned the four preregistered local model artifacts and froze model,
  seed, context, decoding, payload-digest, and replay configuration.
- Added a machine-audited readiness gate that permits powered outcome
  generation only after the independent human corpus evidence is frozen.

## 0.3.0 - 2026-07-30

- Replaced the default literal-first gateway with one model call over a forced
  handle-selection Action IR; retained the old path only as explicit
  `legacy_literal` research compatibility mode.
- Added a typed evidence-derivation IR for message spans, versioned state,
  defaults, enums, pure transforms, tuples, and arrays.
- Added reusable evidence types and explicit authority-bearing, opaque-content,
  and effect-bearing value classes.
- Added local candidate prefiltering, request/tool/destination/state-bound
  MAC-authenticated handles, trusted nested materialization, recursive JSON
  contract checks, literal-noninterference tests, and JSON certificate replay.
- Added fail-closed coverage for unmediated tools, generated literals,
  cross-slot reuse, stale state, expiry, tampering, and untrusted provenance.
- Added conservative policy initialization, recursive nested-schema linting,
  provider-free candidate inspection, and authenticated offline replay CLIs.
- Added explicit `audit`, `enforce`, and `assist` operating modes; audit is
  labeled non-enforcing and preserves native calls only for shadow evaluation.
- Added canonical envelope adapters for OpenAI Chat/Responses, Anthropic
  Messages, and Google Gemini Interactions/`generateContent`.
- Added deterministic mutation/property tests, archived-expiry replay coverage,
  and stricter nested-policy and value-class consistency checks.
- Added an immutable compare-and-swap execution graph and in-process
  clarification coordinator; every clarification recompiles against a new
  request digest and invalidates prior handles.
- Added derivation trust labels and a non-overridable ban on laundering
  tool/model output into authority- or effect-bearing values.
- Added effect classes, exact-manifest HMAC confirmation challenges, atomic
  process-local single-use consumption, and tamper/expiry/replay/concurrency
  fault tests.
- Added the source-defined, digest-pinned EviBench v1 diagnostic suite with
  separate compiler and selector metrics, selective Wilson intervals, and
  explicit correction/negation failure preservation.
- Added one-call matched-compute native/full conditions, safe prompt-only
  ablations, deterministic evaluation handles, and payload-bound replay.
- Added a canonical post-verdict paper centered on evidence-bound
  materialization, with an explicit materialization-confinement theorem,
  literal-noninterference corollary, trusted-boundary assumptions, and
  non-guarantees.
- Added an executable paper claim audit and a separate powered-extension
  preregistration that keeps the 13-case EviBench suite diagnostic-only.
- Expanded maintained CI coverage to the evidence IR, one-call runtime, state,
  effects, adapters, EviBench, paper audit, and their release-critical tests.
- Explicitly excluded and archive-audited generated LaTeX outputs while
  retaining canonical paper sources in the source distribution.

## 0.2.0 - 2026-07-28

- Renamed the public project and CLI from the TAP-R working title to EviBind.
- Added an OpenAI-compatible admission gateway for non-streaming, single-call
  Chat Completions requests.
- Added request-local source, destination-role, semantic-extent, and joint
  contract checks with fail-closed non-call outcomes.
- Added `evibind lint-schema --strict` with private and provider-visible schema
  fingerprints.
- Added gateway authentication, upstream credential isolation, request-size
  limits, private-annotation stripping, and authenticated non-loopback binding.
- Rejected duplicate tool names, multi-choice requests and responses, upstream
  redirects, and oversized upstream bodies; redacted provider error details by
  default and rebuilt released calls from canonical validated fields.
- Added `no-store`/`nosniff` response headers, suppressed the Python runtime
  version header, and rejected credentials embedded in upstream base URLs.
- Added provider guidance, Docker packaging, Python 3.11-3.13 CI, 16 conformance
  fixtures, and the frozen research harness.
- Made the clean CI matrix self-contained by adding NumPy to the dev extra,
  isolating heavyweight LoRA tests, removing generated-work fixture dependencies,
  and restoring Python 3.11 analyzer compatibility.
- Added packaged, path-free experiment defaults so wheels and source archives can
  load the benchmark configuration without private local model pins.
- Added prospective certificate-hierarchy and stateful ToolSandbox evaluations,
  including transparent negative and diagnostic-exclusion reporting.
- Completed the two-model native-tool BFCL reference as an all-diagnostic
  artifact, with an integrity-only amendment that emits no unavailable-comparator
  coefficients and preserves every generated row.
- Added an explicit source-distribution allowlist so generated experiment artifacts
  and caches cannot enter release archives.
- Added a CI archive audit that rejects generated paths and machine-local
  absolute paths in both wheels and source distributions.
- Made optional FC-RewardBench dependencies lazy and exposed the documented
  `strip_private_annotations` helper through the stable Python namespace.
- Added source-hashed equal-class and prevalence-crossover BFCL sensitivity
  analyses, an operational semantic-envelope registry in the paper, and audit
  coverage for compute asymmetry, balanced effects, trust-boundary claims, and
  the LLM-use disclosure.

## 0.1.0

- Initial research harness and Action IR.
