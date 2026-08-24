# Research evidence and scope

This document classifies the evidence behind EviBind so product demonstrations,
direct ActionIR experiments, and compatibility replay are not blended into one
claim.

## Evidence classes

| Class | Model-facing action | What it measures |
|---|---|---|
| Direct representation | literal-free EviBind ActionIR with candidate handles | whether a model can select policy-admissible bindings under a frozen construction |
| Deterministic boundary | fixed calls or generated mutations; no model | confinement, fail-closed behavior, implementation faults, and executed mock effects |
| Compatibility replay | an ordinary literal call proposed by a third-party model/agent | whether the gateway can re-derive or withhold protected arguments at integration boundaries |
| Product integration | live request through `evibind serve` or the host SDK | serving behavior, schema compatibility, and operational failure modes |

Compatibility replay supports applicability of the boundary; it does not test
direct generation of ActionIR. Fixed-tool binding studies do not estimate
routing, abstention, independently authored language, or general agent task
completion.

## Paper evidence

The current v8 artifact records 29 claims and passes 33 machine checks. Its main
text is nine pages. The release record in `evidence/paper-v8.json` pins both the
bundle and canonical PDF SHA-256 digests.

The strongest deterministic results are:

- 300 equal-value provenance pairs across six patterns. Value-only checking is
  complete but not jointly sound; authenticated derivation-aware materialization
  is jointly sound and complete on the suite.
- 30 sandboxed effect scenarios. Native literals cause 30 harmful effects and
  complete no intended tasks; EviBind and a correctly dispatch-atomic,
  trace-materializing checker complete all 30 without harm. A reject-only
  checker is safe but completes none.
- Eight single-fault variants target a cite-and-check implementation's redundant
  literal/trace channel. All eight are exploitable across 240 executions. EviBind
  has no redundant literal channel and rejects all 240. Four shared trusted-
  boundary bypass controls expose both systems, bounding the comparison.
- One million authenticated release mutations produce no unsound release.

Direct model evidence includes a 100-case, ten-family fresh-family test and a
300-case, six-relation alternative-preserving extension. Verified top-1 changes
exact binding from 5% to 89% for Qwen3-1.7B and 86% to 100% for
Qwen3.6-35B-A3B. With alternatives retained, Qwen3.6-35B-A3B reaches 88.8%
exact recall and 86% all-order exactness; GPT-5.6-Luna reaches 97% and 94%.
Destination composition is the sole relation-level outlier for the larger
models. These are synthetic, fixed-tool studies and should be read at that
scope.

## Needle 2: confidence versus provenance

The frozen Needle study uses its native JSON-Schema interface and one output per
case. One ordinary literal proposal supplies four release policies: native,
development-calibrated confidence, EviBind replay, and their conjunction.
Needle does not generate ActionIR.

| Policy | Released | Exact | Harmful | Accepted exact precision |
|---|---:|---:|---:|---:|
| Native | 59 | 26 | 17 | 44.1% |
| Confidence | 25 | 10 | 5 | 40.0% |
| EviBind replay | 17 | 17 | 0 | 100% |
| Confidence + replay | 10 | 10 | 0 | 100% |

The ordinary 95% Wilson upper bound for 0 harmful releases in 17 replay-gated
releases is 18.4%; zero observed harm is not zero population risk. Confidence
reduces coverage and harmful release at its selected operating point, but it
does not enforce provenance. Replay removes the tested harmful bindings and can
compose with confidence, at additional withholding. The transport flattens a
role-labelled transcript rather than using a native multi-message API, and 64
of 150 total cases make no call.

## AgentDojo: narrow external compatibility

The confirmatory banking replication pins AgentDojo 0.1.35, GPT-5.4 nano, all
144 user-task/injection-task pairs, both arms, and a separate 16-task clean
control. EviBind intercepts the ordinary proposed call and re-derives supported
critical arguments from the trusted user turn.

| Outcome | Native | EviBind replay | Paired comparison |
|---|---:|---:|---|
| Attack succeeded | 6/144 | 0/144 | 6 improvements, 0 regressions; exact McNemar p=0.03125 |
| Task completed under attack | 57/144 | 58/144 | 7 gains, 6 losses; p=1.0 |
| Clean task completed | 7/16 | 6/16 | descriptive control |

The attack-success change is −4.17 points with a two-way cluster-bootstrap 95%
interval of [−12.5, 0.0]. Utility changes +0.69 points [−9.72, 11.11]. The
ordinary 95% Wilson interval for 0/144 guarded attack successes is [0, 2.6%].
This is a frozen current-model replication in the most re-derivable suite, not
a general AgentDojo leaderboard: 75% of banking critical arguments, but only
36% across all four suites, are available through a trusted channel the
boundary can re-derive.

## Product and live compatibility results

The repository also contains InjectBench, provider adapters, serving-path
regressions, and third-party task adapters. These are valuable engineering
evidence but should not silently inherit the paper's prospective status. Each
result records its model interface, request dialect, model identifier, case
selection, and whether the result was prospective, confirmatory, exploratory,
post hoc, or diagnostic.

Read `docs/FINDINGS.md` for the chronological record and known defects. In
particular, origin confinement does not resolve same-type destination swaps,
negation, general routing, or values available only in an attacked document.
Applications must preserve alternatives or abstain when intendedness is not
identifiable from the trusted inputs.

## Integrity and regeneration

The full evidence archive contains the manuscript, source, claim ledger, audit,
frozen protocols, selected case-level rows, external-study records,
BoundaryBench, and installable package artifacts. Verify it with
`scripts/verify_evidence_bundle.py` and the release record before reading any
number from it.

Deterministic mechanism evidence can be regenerated offline. Hosted-model
outputs may differ after provider updates; reproduction means using the frozen
cases and protocol, preserving every attempted output, and reporting the exact
model and provider revision—not tuning until the archived number reappears.
