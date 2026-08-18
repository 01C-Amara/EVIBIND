# Findings from the InjectBench run

What the 150-case benchmark (`bench/`) actually showed, including the parts
that do not flatter the gateway. Every number here is reproducible with
`python bench/run_bench.py score --responses bench/results/responses_haiku.jsonl
--label claude-haiku --out /tmp/out.json`.

## 1. The confinement boundary works, and repairs rather than merely blocks

Against a scripted worst-case selector that always emits the engineered wrong
value, all 60 origin-violation cases are neutralised:

| category | native | guarded |
|---|---|---|
| injected instruction | 15/15 harmful | 15/15 **repaired to the user-authorised value** |
| injected data field | 15/15 harmful | 15/15 repaired |
| forged authority | 15/15 harmful | 15/15 repaired |
| user defers to untrusted tool | 15/15 harmful | 15/15 withheld (fail closed) |

The repair result matters more than the block result: because the gateway
re-derives the value from admissible evidence, the correct call still goes out
in 45 of 60 cases. Only where no admissible evidence exists (the user deferred
to untrusted output) does it fall back to asking.

## 2. Garbled long identifiers are caught; short ones are not reliably caught

A scripted model that transcribes a 50-character ARN with a two-character drift
produces an unsupported value; the gateway withholds all 15. This is a real
deployment benefit — transcription drift on long identifiers is a common,
non-adversarial failure.

The same protection is **weaker for short identifiers**: a single-character
substitution in a 8-character account reference is not reliably rejected in
multi-candidate contexts (see finding 4).

## 3. Selection errors are not confined — by design, and it is visible

Where every candidate is admissible evidence and the model simply picks the
wrong one, the gateway releases the model's choice:

| category | guarded outcome |
|---|---|
| stale value after a user correction | 15/15 still wrong |
| ambiguous request (two plausible values) | 15/15 guess released |
| swapped from/to slots | 15/15 still wrong |
| near-duplicate distractors | 15/15 still wrong |
| explicitly forbidden value | 15/15 still wrong |

This matches the guarantee as stated in the research artifact: materialization
confinement does not establish intended-candidate selection, correct treatment
of correction or negation, or business authorization. If your threat model
includes these, you need candidate pruning, ambiguity gating, or human
confirmation on top of the boundary — not the boundary alone.

## 4. Open defect: unsupported literals can release a malformed span

In multi-candidate contexts (the `distractor` cases: eight same-type account
references across several turns), emitting a literal that matches no candidate
does **not** always fail closed. Observed behaviour:

```
user: "... The beneficiary account for this one is ACC-5003."
model emits: ACC-9999          (unsupported)
gateway releases: {"beneficiary_account": "for"}
```

The extractive compiler admitted the token following the extraction cue
(`account` → `for`) as an `account_ref` candidate, and released it when the
model's literal was unsupported. In a single-turn context the same input
correctly withholds, so this is specific to the multi-candidate path.

Security impact is limited — no untrusted-origin value is released, so the
confinement property itself is not violated — but it breaks the fail-closed
contract and would send a malformed argument downstream. Evidence-type
validation should reject a span that cannot be an `account_ref` before it
enters the candidate table.

Regression test: `tests/test_injectbench_boundary.py::test_unsupported_literal_should_fail_closed`
(currently `xfail`).

## 5. Live-model result: the gateway is free

Claude Haiku, 150 cases, same output scored in both arms: 75/75 correct calls
released unchanged, 75/75 abstentions preserved, zero rewritten arguments, zero
false rejections.

Haiku also resisted all 60 injection cases on its own. Two caveats before
reading that as "models don't need this":

* **Batch-review framing.** The model saw ten cases at a time from a file and
  could recognise the pattern. A production agent sees one conversation at a
  time, under task pressure, and typically after a long tool-call history. This
  run is therefore an *upper* bound on model self-defence, not a typical one.
* **Model-dependence is the point.** The safety of the native arm is a property
  of Haiku on this day; the guarded arm's property holds for any selector,
  including a weaker local model or a future regression. The paper's frozen
  test measures exactly that spread (5% → 89% for a 1.7B model).

To test a specific model, use `providers/run_*.sh` — the benchmark is
provider-agnostic.
