# Contributing

Contributions should preserve EviBind's fail-closed contract and the separation
between experimental claims and product behavior.

1. Add focused tests for every release-gate or provider-adapter change.
2. Run `pytest -q` and `python -m tapbench.cli conformance`.
3. Do not weaken certificate checks to raise coverage without a named policy,
   frozen evaluation, and precision/unsupported-action report.
4. Keep provider credentials, gold labels, scorer outputs, and benchmark-only
   fields out of runtime paths.
5. Document whether a result is prospective, held out, exploratory, post hoc, or
   diagnostic-only.
6. Keep public examples provider-free where possible and add a regression test
   for every command shown in the root README or reproducibility guide.
7. Write scientific outputs with deterministic ordering, a schema version,
   complete parameters, and SHA-256 checksums. Separate timing metadata from
   deterministic scientific results.
8. Never include credentials, participant records, proxy-human corpora,
   machine-local paths, generated paper files, or unreviewed model outputs in a
   source or evidence release.

Before opening a pull request, run the offline quickstart and reproduction smoke:

```bash
python examples/minimal_evidence_binding.py
python scripts/reproduce_public_artifact.py \
  --output-dir /tmp/evibind-public-reproduction \
  --fuzz-trials 1000
python -m pytest -q
```

Small pull requests with a clear behavioral contract are easiest to review.
