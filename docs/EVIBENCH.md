# EviBench v1

EviBench is the frozen diagnostic and matched-compute harness for EviBind's
pre-materialization integrity claim. It measures two different systems and
keeps their errors separate:

1. The deterministic compiler must discover only admissible
   `(tool, destination, value)` bindings from typed evidence.
2. The model must select an appropriate Action IR from those bindings.

The committed suite contains 13 source-defined cases across literal extraction,
same-type destination binding, untrusted tool output, missing evidence,
versioned state, stale state, nested arguments, schema enums, turn scope,
negation/correction, literal noninterference, tool distractors, and schema-owned
defaults. The suite artifact is
`tapbench/data/evibench_v1.jsonl`; its source definition is
`tapbench/evibench.py`; and its SHA-256 is pinned in
`configs/evibench_v1_manifest.json`.

## Matched-compute design

Every case/model/seed cell invokes the model exactly once, with no retry,
committee, repair call, or hidden verifier generation:

| Condition | Purpose | Enforcement |
|---|---|---|
| `native_literals` | Literal tool-calling baseline | None |
| `evibind_full` | Primary handle-selection mechanism | Full |
| `evibind_no_candidate_display` | Selector ablation | Full |
| `evibind_no_tool_description` | Selector ablation | Full |

The ablations remove selector information from the prompt but retain the same
destination-bound materializer. They are therefore suitable research controls,
not bypass modes that weaken the production boundary.

The runner emits `model_calls=1` on every row and fails if the case-condition
matrix is incomplete. Input bytes and compiler/model latency are reported
alongside accuracy so prompt overhead is visible. Decoding settings must be
held constant within each model/seed block, as frozen in
`configs/evibench_v1_preregistration.yaml`.

## Metrics

Compiler metrics compare the set of unique compiled bindings with the frozen
admissible set:

- compiler precision: admissible compiled bindings / all compiled bindings;
- compiler recall: compiled admissible bindings / all admissible bindings;
- untrusted critical admissions: compiled critical bindings rooted in an
  untrusted source.

Selector/end-to-end metrics are computed only after the response is interpreted
or protected:

- accepted-call exact precision;
- call coverage and exact-call coverage;
- unsupported-critical rate among accepted calls;
- decision accuracy;
- clarification and no-tool/abstention rates;
- model-call count, input size, and p50/p95 model latency.

Precision is always reported with its accepted-call denominator and coverage.
The report includes Wilson 95% intervals for selective proportions. A powered
full run should additionally use a paired case-cluster bootstrap.

The correction case (`evi-010`) preserves the original hard boundary: an
explicitly negated recipient followed by a corrected recipient. The
cue-bounded extractor now enumerates repeated assignments, rejects negated or
superseded occurrences, and retains multiple values when the request is truly
ambiguous. The deterministic suite now has 11 true-positive bindings, zero
false positives, and zero false negatives. This is a compiler regression
result, not a model outcome.

## Freeze and replay

Regenerate the source-defined suite and manifest:

```bash
python -m tapbench.cli evibench-freeze \
  --cases tapbench/data/evibench_v1.jsonl \
  --manifest configs/evibench_v1_manifest.json
```

A response artifact is JSONL with exactly one row for every requested
case-condition pair:

```json
{
  "case_id": "evi-001",
  "condition": "evibind_full",
  "model_id": "provider/model-revision",
  "seed": 7,
  "decoding_parameters": {"temperature": 0.0, "max_tokens": 256},
  "payload_sha256": "...",
  "response": {"choices": []}
}
```

Replay and score it:

```bash
python -m tapbench.cli evibench-replay \
  --cases tapbench/data/evibench_v1.jsonl \
  --responses work/evibench/responses.jsonl \
  --records work/evibench/records.jsonl \
  --report work/evibench/report.json
```

Candidate nonces are deterministic only inside EviBench, making payload hashes
and handle IDs replayable. Production gateway nonces remain cryptographically
random. Replay rejects duplicate, missing, extra, or payload-mismatched
responses before scoring.

## Claim boundary

The 13-case suite is a diagnostic and integration freeze, not a powered claim
about any hosted model. It can validate runner discipline, expose compiler
failures, and compare mechanisms during development. General model-quality
claims require a larger independently authored set, multiple models and seeds,
paired uncertainty, and the human evidence audit described in the paper.

The powered extension, including its seven frozen one-call conditions and
human-evidence readiness gate, is documented in
[`EVIBENCH_POWERED.md`](EVIBENCH_POWERED.md).
