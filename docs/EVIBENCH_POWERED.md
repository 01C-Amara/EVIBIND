# EviBench Powered Extension

Status: infrastructure ready; corpus and independent human evidence pending.

This extension is deliberately unable to generate a publishable model result
until every preregistered input gate passes. The current repository has:

- all seven one-call conditions implemented and tested;
- six required local model artifacts hash-pinned without machine-local paths;
- seeds, condition order, decoding, context, retry, and response-digest policy
  frozen;
- a hash-pinned 3,500-case human-study workload and deterministic,
  role-separated assignment tool;
- a fail-closed corpus/authorship/annotation/adjudication validator; and
- payload-bound response replay.

It does **not** have the completed policy-authoring study or the required
independently authored, double-annotated corpus.
`outcome_generation_allowed` therefore remains `false`.

## Conditions

Every case/model/seed/condition cell makes exactly one model call with no retry
or repair generation.

| Condition | Representation and enforcement | Deployment-safe |
|---|---|---|
| `native_literals` | Native unrestricted executable literals | No |
| `evibind_full` | Destination-bound authenticated handles, typed replay, and joint contract | Yes |
| `direct_candidate_list` | Executable literals restricted to finite compiler-candidate enums | No |
| `posthoc_copy_match` | Native literals filtered by same-tool candidate value membership; destination ignored | No |
| `posthoc_typed_provenance` | Native literals filtered after generation by tool, destination, typed candidate, and contract | No |
| `evibind_source_only` | Source-admissible handles without destination/type/replay checks | No |
| `evibind_source_destination` | Source- and destination-scoped handles without type validation or authenticated replay | No |

Unsafe conditions live only in `tapbench.evibench_powered`. They are evaluation
controls, not gateway modes. Tests retain a same-type cross-slot attack: the
source-only and copy-match conditions admit it, while destination-aware
conditions withhold it.

## Frozen execution inputs

The execution configuration is
`configs/evibench_powered_execution_v1.yaml`. The portable model manifest is
`configs/evibench_powered_models_v1.json`. It pins:

- Qwen/Qwen3-1.7B;
- LiquidAI/LFM2.5-8B-A1B;
- google/gemma-4-E2B-it;
- google/gemma-4-E4B-it;
- Qwen/Qwen3.6-35B-A3B; and
- openai/gpt-oss-120b.

The dated pre-outcome scale amendment is
`configs/evibench_powered_scale_amendment_v1.yaml`. It was frozen while the
inventory still reported zero powered calls and zero stateful sessions. The
original `main_core` group remains unchanged for historical experiments;
`evibench_powered_scale` is the six-model powered group. GPT-OSS is eligible
only on the independently human-authored corpus after every blinding gate
passes. Earlier same-family synthetic-authoring results remain diagnostic.

The manifest records model IDs, relative artifact paths, byte sizes, SHA-256
digests, quantization, chat templates, grammar engines, and backends. It never
records the local model-root path.

Reproduce a model freeze before outcome generation:

```bash
python -m tapbench.cli evibench-powered-model-freeze \
  --group evibench_powered_scale \
  --model-root /path/to/model-root \
  --output configs/evibench_powered_models_v1.json
```

## Corpus and human-evidence contract

The executable collection and staffing plan is
`configs/evibench_human_study_v1.yaml`; see
[`EVIBENCH_HUMAN_STUDY.md`](EVIBENCH_HUMAN_STUDY.md). It allocates 70
family-disjoint tool families to 500 train, 500 development, and 2,500 test
cases. The test split alone satisfies the publication gate. It also freezes
7,000 independent annotation judgments, the counterbalanced policy-authoring
study, a 120-case non-test pilot, and the order in which each artifact may be
created.

The production freeze gate requires at least 2,500 cases and 50--100 tool
families. Each case must add:

```yaml
authoring:
  request_author_id: request-author-pseudonym
  language: en
  split: test
  phenomena: [correction]
```

Families cannot cross splits. English and one non-English language selected
before annotation must both be present. Correction, negation, hypothetical,
and quoted-value cases must be retained.

The policy JSONL contains exactly one row per family:

```json
{
  "family": "calendar_create",
  "policy_author_id": "policy-author-pseudonym",
  "schema_sha256": "...",
  "policy_sha256": "...",
  "saw_held_out_requests": false,
  "saw_model_outputs": false
}
```

Request and policy authors must be disjoint within each family. Every case
requires exactly two distinct condition-blind annotation rows. Annotators must
also be independent of request and policy authors. Disagreements require one
independent, condition-blind adjudication whose resolution exactly matches the
gold record frozen into the case. The study metadata must pin the annotation
codebook digest and the selected non-English language.

Run the freeze gate:

```bash
python -m tapbench.cli evibench-powered-freeze \
  --cases work/evibench_powered_v1/cases.jsonl \
  --policies work/evibench_powered_v1/policies.jsonl \
  --annotations work/evibench_powered_v1/annotations.jsonl \
  --adjudications work/evibench_powered_v1/adjudications.jsonl \
  --study-metadata work/evibench_powered_v1/study.yaml \
  --manifest work/evibench_powered_v1/freeze_manifest.json
```

No input file is rewritten. The manifest records canonical digests for the
corpus, policies, annotations, adjudications, study metadata, and
preregistration.

## Readiness and replay

Audit all frozen infrastructure:

```bash
python -m tapbench.cli evibench-powered-readiness --root .
```

Add `--require-run-ready` in an outcome-generation job. It exits nonzero until
both the policy-study and corpus manifests exist, pass, and pin the same final
policy projection, preventing premature runs.

Use `evibench-powered-run --preflight-only` to validate a model/seed shard
without contacting the endpoint. Removing `--preflight-only` is fail-closed:
the command re-runs readiness with the outcome gate required before making a
request, checkpoints each raw response, and resumes only exact identity- and
payload-matching rows.

```bash
python -m tapbench.cli evibench-powered-run \
  --root . \
  --cases work/evibench_powered_v1/cases.jsonl \
  --responses work/evibench_powered_v1/responses/qwen36_35b_a3b/seed_1.jsonl \
  --endpoint http://127.0.0.1:8080 \
  --model-key qwen36_35b_a3b \
  --seed 1 \
  --preflight-only
```

Run one separate response file for each of the 18 frozen model/seed shards.
After the human gates pass, remove `--preflight-only`; never combine identities
in a response file. The stateful runner remains a later dependency and is not
released until powered replay completes.

Response JSONL uses the diagnostic EviBench envelope: case ID, condition,
model ID, seed, exact decoding parameters, payload SHA-256, and raw provider
response. Replay rejects missing, extra, duplicate, identity-mixed, or
payload-mismatched rows:

```bash
python -m tapbench.cli evibench-powered-replay \
  --cases work/evibench_powered_v1/cases.jsonl \
  --responses work/evibench_powered_v1/responses.jsonl \
  --records work/evibench_powered_v1/records.jsonl \
  --report work/evibench_powered_v1/report.json
```

## Claim boundary

Hash-pinned models and tested runners are infrastructure evidence, not model
outcomes. The powered comparison, human agreement statistics, family-cluster
intervals, and task/effect metrics remain unavailable until the external corpus
and annotation work is completed. No result should be inferred from readiness
alone.
