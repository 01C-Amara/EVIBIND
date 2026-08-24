# Reproducing EviBind

This guide separates four reproducibility targets. Do not treat success at one
tier as evidence for a stronger tier.

| Tier | What it establishes | Network or model required |
|---|---|---|
| 0 | The evidence-binding API compiles, filters, materializes, and replays one call | No |
| 1 | OriginBench, CheckerAttack, EffectSuite, and boundary-fuzz mechanism results | No |
| 2 | The manuscript, claim ledger, citations, and archived numerical results agree | No, but the paper evidence bundle is required |
| 3 | Fresh model outputs under the frozen protocols | Yes; model/API access is supplied by the reproducer |

The independently authored human/abstention study is not part of the published
evidence. The archived proxy/subagent role-play corpus is an engineering dry run
and must not be represented as human-subject data.

## Environment

The deterministic suite was last verified with CPython 3.11.15, PyYAML 6.0.1,
NumPy 2.4.6, pandas 3.0.3, and pytest 9.1.1. Python 3.11-3.13 is covered by CI.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/reproducibility.txt
python -m pip install -e . --no-deps
```

Windows PowerShell users activate with `.venv\Scripts\Activate.ps1`. Provider
credentials belong in environment variables or a local `.env` copied from
`.env.example`; never place credentials in commands, configs, model outputs, or
release archives.

## Tier 0: evidence binding in one minute

```bash
python examples/minimal_evidence_binding.py
```

The example presents the same email literal from an authorized user message and
untrusted tool output. The compiler admits only the user derivation, the model
surface contains one destination-bound handle, trusted code materializes the
call, and the certificate replays to the same action. Expected invariants:

- one accepted and one rejected candidate;
- released arguments equal `{"attendee": "alice@example.com"}`; and
- `replay_matches` is `true`.

The deterministic secret and nonce source are example fixtures only. Production
deployments require a protected random secret or the Ed25519 issuer/verifier.

## Tier 1: deterministic mechanism evidence

The public reproduction driver writes content-addressed JSON/JSONL outputs and a
`SHA256SUMS` file. It requires no model, API, or external data.

```bash
python scripts/reproduce_public_artifact.py \
  --output-dir reproduced/mechanism \
  --fuzz-trials 10000
```

Use the paper-scale boundary fuzz when time permits:

```bash
python scripts/reproduce_public_artifact.py \
  --output-dir reproduced/mechanism-full \
  --full
```

The default scientific counts are 300 equal-value pairs (six patterns), 40
checker attacks, 30 executed-effect scenarios, eight redundant-channel fault
variants, and four shared-boundary controls. The quick command changes only the
number of boundary mutations; `--full` restores one million. A passing summary
requires:

- EviBind and dispatch-atomic cite-and-check to be sound and complete on the 300
  provenance pairs;
- value-only and typed value reconstruction to fail joint soundness/completeness;
- zero harm and 30/30 completed authorized effects for EviBind and the
  trace-materializing atomic checker, with 30/30 rejections for the reject-only
  checker;
- zero exploitable CheckerAttack cases for EviBind and the atomic checker; and
- all eight redundant literal/trace fault variants to expose the cite-and-check
  baseline while EviBind fails closed, with all four shared-boundary controls
  exposing both systems; and
- zero unsound releases under every executed mutation.

The standalone BoundaryBench-v1 release asset contains the same generators,
case records, expected analyses, and focused tests. Verify an archive before use:

```bash
sha256sum -c evibind_boundarybench_v1.zip.sha256
unzip evibind_boundarybench_v1.zip
cd BoundaryBench-v1
PYTHONPATH=. python scripts/run_equal_value_benchmark.py \
  --output-dir reproduced/originbench
PYTHONPATH=. python scripts/run_adversarial_boundary.py \
  --output reproduced/checker_and_effects.json
PYTHONPATH=. python scripts/run_implementation_fragility.py \
  --output reproduced/implementation_fragility.json
PYTHONPATH=. python scripts/run_boundary_fuzz.py \
  --trials 10000 \
  --output reproduced/boundary_fuzz.json
python -m pytest -q \
  tests/test_equal_value_benchmark.py \
  tests/test_adversarial_boundary.py \
  tests/test_implementation_fragility.py \
  tests/test_boundary_fuzz.py
```

## Tier 2: paper and archived results

The source repository intentionally excludes generated model outputs and
machine-local `work/` trees. Obtain the versioned paper evidence bundle from the
release page or anonymous review host and keep the downloaded archive unchanged.
The checked-in `evidence/paper-v8.json` record pins its filename, bundle digest,
canonical paper digest, audit count, claim count, and empirical exclusions.

Verify the download before extraction:

```bash
python scripts/verify_evidence_bundle.py \
  EviBind_ICLR_2027_evidence_bundle_20260821_v8.zip \
  --sidecar EviBind_ICLR_2027_evidence_bundle_20260821_v8.zip.sha256 \
  --release-metadata evidence/paper-v8.json
```

The verifier checks path safety, the single-root archive contract, complete
internal-manifest coverage, every member digest, the canonical PDF, and
agreement among the release record, `ARTIFACT.json`, and the paper audit. The
bundle contains the final PDF, canonical sources, selected analyses, frozen
configs, BoundaryBench, claim audit, and installable artifacts.

After verification, from the extracted artifact root whose evidence paths match
`paper/claims.yaml`:

```bash
python tapbench/paper_audit.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/main.tex
python -m pytest -q tests/test_paper_audit.py
```

The machine audit verifies claim-ledger bijection, evidence existence and
integrity, citations, table values, experiment discipline, and the nine-page
main-text boundary. Render every PDF page to images and inspect layout after the
build; textual extraction alone does not detect overlap or clipping.

## Tier 3: model-backed extensions

Model-backed regeneration is necessarily environment-dependent. The public
artifact preserves the cases, catalogs, prompts, ranker, decoding settings,
model identifiers, no-retry policy, and analysis code. Reproducers supply their
own lawful model/API access and must report any provider revision or unavailable
model rather than silently substituting it.

The complete frozen entry points below live in the evidence bundle. The product
checkout includes only the deterministic mechanism drivers and current public
compatibility studies, so absence from the root checkout is not permission to
reconstruct a protocol from prose.

Relevant frozen entry points are:

- `scripts/run_confirmatory_v1.sh`: two-model fresh-family study;
- `scripts/run_confirmatory_gpt_oss_extension_recovery_v4.sh`: GPT-OSS-120B extension;
- `scripts/run_confirmatory_luna_extension_v1.sh`: hosted Luna extension;
- `scripts/run_fresh_family_saturation_luna_v1.sh`: hard-distractor saturation study;
- `scripts/prepare_candidate_position_robustness.py` and
  `scripts/run_candidate_position_robustness.py`: mirrored mention-order and
  deterministic catalog-permutation study; `scripts/analyze_candidate_position_results.py`
  adds the prespecified gold-index stratum under a declared analysis-only amendment;
- `scripts/run_candidate_top2_qwen36.sh` and
  `scripts/run_candidate_top2_robustness.py`: frozen alternative-preserving
  top-2 extension over the same cases, ranker, four catalog orders, and no-retry
  model protocol;
- `scripts/analyze_confirmatory_inference.py` and
  `scripts/analyze_saturation_sweep.py`: response-only analyses; and
- `scripts/run_stateful_mitigation_recovery_v4.sh`: matched stateful intervention;
- `bench/needle_confidence_freeze_v1.json`,
  `bench/needle_confidence_analysis_amendment_v1.json`,
  `bench/run_needle_confidence.py`, and
  `bench/analyze_needle_confidence.py`: pinned Needle 2 dev/test comparison of
  one shared native literal proposal under native, confidence-only, EviBind
  replay-gateway, and combined release, with cluster and Wilson intervals; and
- `bench/agentdojo/confirmatory_banking_v1.json`,
  `bench/agentdojo/chronology_amendment_v1.json`,
  `bench/agentdojo/run_agentdojo.py`, and `bench/agentdojo/analyze.py`:
  current-model AgentDojo replication with per-case rows, trace hashes, exact
  package provenance, suite-selection chronology, clustered comparisons, and
  ordinary finite-sample intervals.

Do not regenerate cases after observing outputs, retry failed generations, tune
on the held-out families, or merge pre-inference transport attempts with model
outputs. Temperature zero does not guarantee bit-exact regeneration from a
hosted API; compare the reported metrics and archived request/response records.

## Claim-to-artifact map

| Paper claim | Public implementation or artifact |
|---|---|
| Equal-value information separation | `tapbench/equal_value_benchmark.py`, OriginBench-300 |
| Cite-and-check and TOCTOU boundary | `tapbench/adversarial_boundary.py`, CheckerAttack-40 |
| Executed effects | `tapbench/adversarial_boundary.py`, EffectSuite-30 |
| Representation-specific fault surface | `tapbench/implementation_fragility.py`, Fragility-12 |
| Mutation robustness | `tapbench/boundary_fuzz.py`, `scripts/run_boundary_fuzz.py` |
| Authenticated materialization | `evibind/core/`, `examples/minimal_evidence_binding.py` |
| Public-key replay | `evibind/core/public_key.py` |
| Admissible top-1 presentation | `tapbench/verified_ranker.py` and prospective ranker analysis |
| Mention/catalog-order robustness | `tapbench/candidate_position_robustness.py` and its prospective protocol |
| Admissible top-2 alternative retention | `tapbench/candidate_top2_robustness.py` and its prospective protocol |
| Fresh-family saturation | `scripts/analyze_saturation_sweep.py` and archived rows |
| Stateful mechanism intervention | stateful analysis and failure-taxonomy artifacts |
| Confidence/provenance complementarity | frozen Needle 2 native-proposal gate study with category-cluster and Wilson analyses |
| Third-party agent external validity | AgentDojo scope analysis and case-level live-run reports |
| Manuscript consistency | `tapbench/paper_audit.py`, `paper/claims.yaml` in the verified evidence bundle |

## Reporting discrepancies

Report the repository revision, Python/package versions, command, full error,
and the affected output checksum. Do not include API keys, private model paths,
participant records, or sensitive prompts in a public issue.
