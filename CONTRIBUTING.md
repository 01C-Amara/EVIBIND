# Contributing

Contributions must preserve EviBind's fail-closed contract and the separation
between product behavior and scientific evidence. Small changes with an explicit
behavioral contract are easiest to review.

## Set up a development checkout

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,research]"
python examples/minimal_evidence_binding.py
python -m pytest -q
python -m ruff check .
```

Windows PowerShell uses `.venv\Scripts\Activate.ps1`.

## Product changes

- Import supported behavior through `evibind`; symbols used only through
  `tapbench` are internal or research-facing unless documented otherwise.
- Add focused tests for every release gate, compiler, materializer, host SDK,
  schema, provider-envelope, nonce, or effect-policy change.
- Do not weaken certificate checks to raise coverage. New fallback behavior
  needs a named policy, a fail-closed default, and explicit unsupported-action
  accounting.
- Keep provider credentials, gold labels, scorer outputs, benchmark-only fields,
  and model responses out of runtime paths.
- Update `docs/PUBLIC_API.md`, `SECURITY.md`, and `CHANGELOG.md` when a public or
  security-relevant contract changes.

Run the product gates:

```bash
python -m pytest -q
python -m tapbench.cli conformance
python -m build
python scripts/audit_release_archives.py dist/*.whl dist/*.tar.gz
```

Install the built wheel into a clean environment and run `evibind --help` plus
the offline example before proposing a release.

## Research changes

- Label every result as prospective, confirmatory, exploratory, post hoc, or
  diagnostic. Compatibility replay must not be described as direct ActionIR
  generation.
- Freeze cases, prompts, catalogs, decoding settings, analysis code, model
  identifiers, and retry policy before inspecting held-out model output.
- Preserve every attempted row. Do not silently retry, regenerate, substitute a
  model, or drop failures to match an archived result.
- Scientific outputs require deterministic ordering, a schema version, complete
  parameters, and SHA-256 digests. Separate timing metadata from deterministic
  comparisons.
- A ranker may reduce the presented admissible set; it does not verify user
  intent. Report coverage, precision, withholding, and unsupported actions
  together.
- Never represent proxy or subagent role-play as independently authored human
  evidence.

Run the offline evidence smoke:

```bash
python scripts/reproduce_public_artifact.py \
  --output-dir reproduced/mechanism-smoke \
  --per-pattern 2 \
  --per-effect-kind 1 \
  --separation-repetitions 2 \
  --fuzz-trials 260
```

## Evidence and release hygiene

- Keep large evidence bundles in release assets, not Git. Add or update a small
  `evidence/*.json` record that pins the archive and canonical paper digests.
- Verify paper bundles with `scripts/verify_evidence_bundle.py` before quoting a
  result or publishing a download URL.
- Never include credentials, participant records, proxy-human corpora,
  machine-local paths, generated paper build files, private rebuttal material,
  or unreviewed model outputs in a source or evidence release.
- Do not overwrite frozen artifacts. Amend them with an explicit chronology and
  new digest.

## Pull requests

Describe the product contract or paper claim affected, the threat-model impact,
the tests and reproduction commands run, and any artifact/version migration.
Security vulnerabilities should follow [SECURITY.md](SECURITY.md), not a public
issue.
