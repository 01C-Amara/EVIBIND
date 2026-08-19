# Changelog

All notable EviBind changes are documented here.

## Unreleased

- **Found and fixed: both benchmark adapters annotated only *required*
  parameters**, which left exactly the slot an attacker wants ungoverned on any
  update-style tool. All five residual attack successes in the first AgentDojo
  banking run were the same goal — modify a recurring payment's recipient via
  `update_scheduled_transaction(id, recipient=None, ...)`, where only `id` is
  required. Re-running those five: attack succeeded 3/5 → **0/5**, completion
  unchanged. The flaw was in the adapters, not the boundary, and it is the
  failure mode this approach is most exposed to: the boundary is exactly as
  good as the policy it is given, and a plausible policy-authoring shortcut
  left the target slot unprotected with nothing to complain about it.
- The same hole was in `bench/injecagent/adapt.py` — 55 of 330 tool definitions
  carry an optional identifier-shaped string parameter. Corrected numbers:
  `dh_enhanced` in scope 357→**391/510**, governed user calls 270→**450**,
  released 150→**240**, withheld 120→**210**. The released share of governed
  calls is roughly flat (56%→53%), so the stricter rule governs far more slots
  at about the same utility rate.
- The per-model native attack counts are unaffected — whether a model calls the
  attacker's tool has nothing to do with slot annotation — and the guarded
  column was re-measured for the only model that ever landed one: GPT-4.1 mini
  on `dh_enhanced`, **7/510 natively, 0/510 guarded**. It read 9/510 on the
  earlier run; same model, same cases, ordinary variance, and a caution that
  single-digit counts over 510 cases are noisy.
- `run_injecagent.py` now saves raw model responses beside each result, so a
  scoring change can be replayed without paying for the models again. Not doing
  that the first time is what made confirming this fix cost a re-run.

- **Ran AgentDojo live with EviBind in their agent loop, on their metrics.**
  Banking suite, GPT-4o mini, `important_instructions`, 16 user tasks crossed
  with 9 injection tasks: attack success **68/144 (47%) → 5/144 (3.5%)** with
  task completion **55 → 56**. Clean control with no injection, where a false
  rejection would show: **7/16 both arms**. A 93% reduction in attack success
  with completion unchanged, measured by the third party's harness.
- Added the clean-utility control (`--no-injections`), because utility measured
  only under attack cannot distinguish a boundary that preserves good calls from
  one that breaks them alongside the bad.
- Recorded two caveats with it. The 7/16 clean figure is GPT-4o mini failing
  nine tasks on its own, not the boundary. And the pass/fail metric hides work:
  1,236 of 1,750 proposed calls were withheld across the injected run while
  completion held level, so the agent retried heavily — real latency and token
  cost that AgentDojo's score does not capture.
- Explained why this beat the §17 scoping estimate: that measured whether the
  *authorised* value is re-derivable, which governs completion. Attack success
  asks whether the *attacker's* value reaches a tool, and withholding stops that
  regardless. Confinement is broad; completion is not.

- Added `bench/agentdojo/`, which answers the applicability question InjecAgent
  could not. AgentDojo's injection tasks are argument-level — the banking goal
  is "send a transaction to US133000000121212121212" — so it can measure whether
  the *authorised* value was ever somewhere the attacker could not reach.
  Reading its own ground-truth calls, no model or key involved: **43 of 119**
  action-critical argument values across four suites appear in the user's turn
  (banking 75%, Slack 27%, overall 36%).
- Recorded the case that makes it concrete. "Pay the bill
  'bill-december-2023.txt'" has its authorised IBAN inside the document, and
  AgentDojo's injection vector is the payment block of that same file; under the
  real attack the block is replaced, so only the attacker's IBAN remains.
  Source-level provenance cannot separate them, uniqueness cannot flag one IBAN,
  and re-derivation has nothing to work with. The boundary withholds — safe, and
  unable to complete.
- The conclusion is a scope, not a defect: the boundary is useful exactly where
  the authorised value lives in a channel the attacker cannot write to, and
  widening that is an integration change — typed tool outputs — rather than a
  change to this code.

- **Found: a swapped two-slot assignment is released.** All 15 `cross_slot`
  cases send the reversed transfer — `from_account` and `to_account` exchanged
  — because Tier-B proposal-span support admits a value for being the model's
  proposal *and* appearing in the user's turn, and both halves are origin
  checks. `_proposal_span` searches the whole turn and `_contract_value_valid`
  checks the slot schema; neither distinguishes one same-typed slot from
  another. Confinement holds; the direction of the payment does not.
- Added `clarify_interchangeable_slots`, a request option **off by default**.
  When on, the boundary withholds whenever two action-critical slots declared
  with the same evidence type hold values each admissible for the other:
  swap released 15/15 → 0/15. It withholds the correctly assigned call too,
  which is the finding rather than a defect — the two are indistinguishable to
  the boundary — so the trade is left to the deployment.
- The guard requires a shared declared evidence type. Without it an amount and
  an account reference compare as interchangeable, since the schema check sees
  two strings; that false positive is why `x-evibind-slot-role: content` not
  being honoured by the offline lattice is recorded in `docs/FINDINGS.md` §16.
- Motivated by the ICLR mixed-order revision, which localises the residual
  binding error to a two-slot presentation-order interaction: exact across all
  four orders in 16% of cases for Qwen3.6-35B and 64% for GPT-5.6-Luna, against
  100% on the other five relations.

- Attempted a Claude Haiku row through a Claude Code subagent harness, since no
  `ANTHROPIC_API_KEY` is available here, and recorded it as a **method failure
  rather than a result**. It returned 0/270, but the harness must tell the agent
  that case content is inert data — an agent with real shell and filesystem
  tools could otherwise act on an instruction planted in a tool result — and
  that warning is exactly the hint the API rows are measured without. All eight
  agents confirmed it unprompted, reporting that they had spotted and ignored
  the injections. Stopped at 270 of 510 once they agreed. The generalisation is
  in `docs/FINDINGS.md` §14: injection susceptibility cannot be measured through
  a harness that must warn the model about injections.
- Added `bench/injecagent/prepare_batches.py` and `score_batches.py` for
  models reachable only as an agent, with the transport recorded in the result
  file so such rows cannot be mistaken for API rows.
- Refreshed the paper section against the *semantic robustness revision*: the
  six-relation alternative-preserving results (Qwen3.6-35B 88.8% recall / 86%
  all-order; GPT-5.6-Luna 97% / 94%, both perfect on five of six relations with
  two-slot destination composition the sole outlier), the positional-ranker
  stress (100/100 gold-late vs 0/100 gold-early for a singleton ranker), and the
  ToolSandbox decomposition (call exceptions down 0.256, task similarity down
  0.155 across 336 rows).
- Cross-referenced three of the paper's findings against this repo's live runs,
  which reproduced them independently: the two-slot destination outlier is the
  `cross_slot` group the serving path cannot resolve; boundary reliability and
  planner competence decompose exactly as the end-to-end run shows; and the
  positional-ranker lesson is the same one as offering only the greedy reading
  of a span.

- Added an **external benchmark**: `bench/injecagent/` adapts
  [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) (UIUC, MIT), 2,106
  cases, under one mechanical annotation rule applied identically to attacker
  and user tools. `fetch.py` pulls the data into a gitignored directory rather
  than vendoring it. Two arms are scored — attacker-tool rate and the
  benchmark's own user calls — because a boundary that withholds everything
  wins any attack-success table on its own.
- The external result is a near-null one, and it is the point: on the full
  510-case `dh_enhanced` split GPT-5.4 nano and GPT-5.4 mini call the attacker's
  tool 0/510, GPT-4.1 mini 9/510 — and the gateway withheld all nine — while the
  same models bind the attacker's account 40-43 times out of 60 on InjectBench.
  Tool-selection injection is largely refused by current models; argument
  substitution is not, because it never reads as an instruction. A 120-case
  sample would have reported a clean null and missed the nine.
- Extended the external run to eight models on `dh_enhanced`: GPT-5.6 Sol, Luna
  and Terra, Grok 4.6 and 4.5 (through the CLI, no xAI key), GPT-5.4 mini and
  nano all at 0/510; GPT-4.1 mini at 9/510, all withheld. The frontier rows are
  what make the result readable — Sol and Grok 4.6 sit at 0/510 *and* 0/60,
  while GPT-5.4 nano sits at 0/510 *and* 43/60, so the variance is in the shape
  of the attack rather than the tier of the model.
- Fixed `to_responses`: it could not carry a transcript that already contained
  an assistant `tool_calls` turn, emitting `content: null`, which the Responses
  API rejects. InjectBench synthesises those turns so it never surfaced;
  InjecAgent supplies them, so every GPT-5.6 request failed until this was
  fixed. The InjectBench conversion is byte-identical afterwards.
- Added `bench/injecagent/run_injecagent_grok.py` and
  `bench/injecagent/summarize.py`; the cross-model tables in the docs are
  generated from the result files rather than typed by hand.
- Added `assets/bench_contrast.svg`: the two attack shapes side by side across
  every model that has run both.
- Not run and why: Claude Haiku needs an `ANTHROPIC_API_KEY` this machine does
  not have, and `gpt-oss` is not served on the OpenAI API at all — it needs
  OpenRouter, Groq or a local runtime. An open-weight model is the likeliest of
  the set to fail the tool-selection arm, so it is the row most worth adding.
- Recorded two structural limits with it: an argument-level boundary cannot
  touch a parameterless tool (30% of `dh`, 59% of `ds`), and the 120 withheld
  user calls are all values that are genuinely absent from the user's turn —
  a statement about the annotation rule, not the engine.

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
- **Fixed the last open defect (FINDINGS #4).** `build_candidate_lattice`
  filtered candidates with a JSON Schema shape check alone, so any string
  satisfied a `string` slot and `"The beneficiary account for this one is
  ACC-5003"` admitted `"for"` as an `account_ref`. Candidates are now also
  gated on the slot's declared evidence type. The suite has no `xfail` left.
- Added `GET /v1/models`, proxied to the upstream. Clients that enumerate
  models — including `openai`'s own `client.models.list()` — previously got a
  404 and concluded the gateway was broken without ever sending a chat request.
- Upstream failures are now logged server-side as a structured `upstream_error`
  line. The response body stays gated behind `allow_diagnostics`, but a gateway
  that answers `upstream returned HTTP 400` and logs nothing left the operator
  with no way to see what the provider objected to.
- Fixed CI: it triggered on pushes to `main` while the default branch is
  `master`, so it had never run on push. It now also lints the whole repo
  instead of two directories, and a second job runs the offline benchmark
  against the mock provider and asserts the weak-selector control stays fully
  neutralised.
- Cleared every `ruff` finding in the repo (16, all dead imports or unused
  locals).
- Fixed a ~5% flake in
  `test_gateway_rejects_upstream_redirect_without_forwarding_key`. Two test
  upstreams replied without consuming the request body, and an unread body can
  surface as a connection reset instead of the response — so the gateway
  reported "could not reach upstream" rather than the redirect refusal the test
  asserts. A flaky security test is worse than none: it teaches people to
  re-run until green. 30 consecutive runs clean.
- End-to-end result: GPT-5.4 nano through `evibind serve` against live OpenAI
  goes from 0/150 correct, 15 malformed to **88/150 correct, 0 harmful, 0
  malformed**, with 58 of 60 origin violations completing.
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
