# EviBench Human Study Operations

Status: protocol frozen; the 70-family inventory is mechanically audited;
human license confirmation and recruitment remain pending.

The publication blocker is a data-production program, not a reason to weaken
the outcome gate. The frozen plan uses 70 tool families and 50 cases per
family:

A reproducible coordinator handoff can be built after the package wheel and
inventory audit exist:

```bash
python scripts/build_evibench_human_eval_kit.py \
  --wheel dist/evibind-0.4.0.dev0-py3-none-any.whl \
  --output dist/evibench_human_eval_kit_v1.zip
```

The kit includes role-separated instructions and templates but deliberately
contains no participant registry, license-release marker, human judgment, or
model outcome. Its checksum sidecar and internal manifest make the handoff
auditable without converting placeholders into evidence.

| Split | Families | Cases | Use |
|---|---:|---:|---|
| Train | 10 | 500 | workflow development and runner checks |
| Development | 10 | 500 | codebook pilot and throughput estimation |
| Test | 50 | 2,500 | publication coefficients; never used for tuning |
| Total | 70 | 3,500 | complete human-authored corpus |

Each family contributes 40 English and 10 Spanish requests. The per-family
phenomenon strata in `configs/evibench_human_study_v1.yaml` sum to exactly 50,
so correction, negation, hypothetical, quoted-value, ambiguity,
clarification, derivation, state, untrusted-output, and structured-value cases
cannot disappear during collection.

## Source inventory

Start with real or minimally adapted public tool schemas. The
[APIs.guru OpenAPI Directory](https://github.com/APIs-guru/openapi-directory)
is a useful discovery source, and the
[Official MCP Registry](https://registry.modelcontextprotocol.io/) is a useful
MCP discovery source. Neither discovery source replaces per-artifact review.
Every family inventory row must pin:

- `family`, one family-disjoint `split`, and `source_kind`;
- a stable source locator and revision;
- the source-artifact and normalized-schema SHA-256 digests; and
- the applicable license.

The MCP Registry is in preview, so copy and hash the reviewed schema artifact;
do not depend on mutable registry state. Reject schemas whose provenance,
license, or safe offline adaptation cannot be established.

## Human roles and blinding

Use globally disjoint pseudonymous pools:

- 8 policy engineers;
- at least 12 independent request authors;
- at least 10 annotators; and
- at least 3 adjudicators.

The policy study is a counterbalanced within-participant comparison of manual
authoring and initializer-assisted authoring with independent review on all 70
families and duplicate authoring on 14 families. Policy engineers receive
schemas and the codebook before request authoring. They never receive requests
or model outputs.

Request authors receive role-specific scenario slots. Every case then goes to
two distinct condition-blind annotators. Only disagreements are activated for
an independent adjudicator. The assignment tool emits separate files for the
policy, authoring, and annotation roles; never distribute the complete internal
directory to a participant.

Obtain the applicable ethics/IRB determination before recruitment. Store
consent and compensation records outside the repository, expose only
pseudonymous readiness flags to the assignment tool, and never commit direct
identifiers. For online recruitment, a screened pilot is appropriate:
[Prolific recommends prescreening and a pilot before a full launch](https://researcher-help.prolific.com/en/articles/445165-can-i-screen-participants-within-my-study).
Use a separate technical qualification for policy engineers rather than
treating general crowd participants as policy experts.

## Inputs and deterministic assignments

Participant registry JSONL:

```json
{"participant_id":"request-017","role":"request_author","languages":["en","es"],"consent_recorded":true,"compensation_agreed":true,"training_complete":true}
```

Family inventory JSONL:

```json
{"family":"calendar.create","split":"test","schema_sha256":"...","source_artifact_sha256":"...","source_kind":"openapi","source_locator":"https://example.org/openapi.yaml","source_revision":"v2.1.0","license":"Apache-2.0","license_review_status":"human_confirmed"}
```

Derive and inspect the frozen workload:

```bash
python -m tapbench.cli evibench-study-plan \
  --output work/evibench_powered_v1/study_workload.json
```

After recruitment and schema review, generate deterministic role-specific
assignments:

```bash
python -m tapbench.cli evibench-study-assign \
  --participants work/evibench_powered_v1/participants.jsonl \
  --families work/evibench_powered_v1/families.jsonl \
  --output-dir work/evibench_powered_v1/assignments
```

The command fails on missing staffing, role overlap, training/consent flags,
language coverage, unpinned sources, or split drift. Its manifest hash-pins
every input and output without recording direct identifiers.

Complete policy-authoring records must include timestamps, duration, policy
and normalized-decision digests, standard-registry coverage, custom-resolver
count, validation errors, and both blinding declarations. Independent reviews
must record the reviewed and final policy digests. Freeze the study only after
all 84 authoring tasks and 70 reviews are complete:

```bash
python -m tapbench.cli evibench-policy-study-freeze \
  --policy-tasks work/evibench_powered_v1/assignments/policy_tasks.jsonl \
  --authoring-records work/evibench_powered_v1/policy_authoring_records.jsonl \
  --review-records work/evibench_powered_v1/policy_review_records.jsonl
```

The freeze reports time, registry coverage, custom resolvers, and validation
errors by arm, plus exact normalized-decision agreement on the 14 independently
duplicated families. It also pins the final per-family policy projection.
Readiness cross-checks that projection against the policies later sealed in the
corpus manifest.

## Runs, in order

1. Run the workload and readiness audits. These produce no model outcomes.
2. Run a 120-case pilot across 12 train/development families. Exercise request
   authoring, blind double annotation, disagreement adjudication, export, and
   payment. Do not include pilot judgments in the test analysis.
3. Revise the interface and codebook once, then hash-freeze them.
4. Run policy authoring, full request authoring, 7,000 independent annotation
   judgments, and blinded adjudication.
5. Pass `evibench-powered-freeze`, then require readiness to return zero.
6. Benchmark throughput on development cases without changing the frozen
   protocol. Run the held-out matrix once: 2,500 cases × 7 conditions × 6
   models × 3 seeds = 315,000 one-call cells.
7. Replay and analyze only after all response cells are sealed.
8. Run the separately preregistered 500-case stateful subset: 18,000 sessions
   across two conditions, six models, and three seeds.

Until both the policy-study and corpus freezes pass, a powered model-outcome
run is a protocol violation. A
`--require-run-ready` exit code of 3 is the expected guard behavior, not a
failed experiment.
