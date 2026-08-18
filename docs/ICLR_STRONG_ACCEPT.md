# ICLR Evidence Ledger And Post-Verdict Assessment

> **Status (2026-07-30):** This document preserves frozen legacy results used
> as boundary evidence. It is not the canonical manuscript and it is not a
> claim that the work is submission-ready. The post-verdict source is
> **Note.** The paths below live in the research bundle, not in this product repo.
> `paper/main.tex`, its executable claim ledger is
> `paper/claims.yaml`, and the powered study remains
> pending under
> `configs/evibench_powered_extension_preregistration_v1.yaml`.

## Legacy Draft Read

The paper is now substantially stronger than the July 27 draft. It has a formal
conditional guarantee, a prospective selected-model result, a compute-matched
mechanism ablation, a clean three-model BFCL replication, a preregistered
large-model null, and a stateful negative result. The evidence supports a narrow
and defensible claim about selective executable-evidence admission.

No engineering checklist can guarantee a strong accept. These observations
describe the pre-verdict controller and remain useful only with their original
selection, compute, and mixture labels. The canonical rewrite makes no powered
model-quality claim from EviBench v1.
The transparent AI-simulated panel in `docs/ICLR_SYNTHETIC_REVIEW_PANEL.md`
places the paper between weak accept and accept; it is diagnostic, not human evidence.

## Completed Evidence

- R2-F: 512 prospective requests across 16 held-out families; 89/89 released
  calls exact, 69.5% call coverage, and 64.8% safe decision accuracy.
- BFCL: clean frozen replication on Qwen3-1.7B, Qwen3.6-35B-A3B, and
  gpt-oss-120B. The preregistered pooled gains are +18.9, +9.5, and +7.9
  points, but a post-hoc equal-class sensitivity view is +8.8, -5.4, and -7.9
  points. Break-even irrelevance prevalence is 26.2%, 60.0%, and 63.8% across
  the models, with paired-bootstrap intervals. The rejection-driven prevalence
  dependence is now a quantitative deployment condition, not a buried caveat.
- Native BFCL reference: all 2,284 frozen rows completed. Qwen and GPT-OSS are
  diagnostic-only under the original gates; no comparator coefficient is used.
- R2-H: the semantic-closure extension recovers zero calls. Qwen is a clean null;
  GPT-OSS remains diagnostic-only after a frozen length-stop failure.
- Hierarchy: 768 prospective requests and five conditions. Semantic extent adds
  51.5 accepted-precision points at a 16.7-point coverage cost; the full
  controller beats 99.95%-compute-matched best-of by 30.7 safe-decision points.
- Reviewer extensions: a verbatim-span release baseline reaches only 11.8%
  precision with 86/127 unsupported released calls, while all 32 extent policies
  produce a 14-point non-dominated risk--coverage envelope. A scripted
  cooperative second turn recovers 10/20 exact calls, or 31.3% of the extent
  gap; it is explicitly not human evidence.
- Scale reference: clean Liquid-LFM2.5 and Qwen3.6-35B arms preserve 100%
  accepted precision and gain 29.5 and 40.6 safe-decision points, but both fail
  the frozen denominator and coverage gates. This is directional evidence, not
  a successful transfer result.
- ToolSandbox: two clean confirmatory models show a 15.5-point average loss in
  official milestone similarity despite 25.6 points fewer tool-call exceptions.
  A descriptive replay attributes 54.0% of negative-pair milestone loss to
  rejection/clarification categories and 46.0% to released-call exceptions.
  This is an important negative boundary on agent-effectiveness claims.
- Distillation: certificate-verified targets do not improve held-out coverage
  over execution-filtered targets; the failed gate remains in the paper.

## Product And Artifact

- EviBind is an MIT-licensed OpenAI-compatible admission gateway with a stable
  Python API, CLI, strict schema linting, authentication, diagnostics controls,
  documentation, CI, wheel/sdist packaging, and a non-root container.
- The release excludes the 6.4 GB generated experiment tree from source archives
  and audits wheels/sdists for generated paths and machine-local absolute paths.
  The optional PyArrow evaluator remains lazy, so a base clean-wheel install runs
  both CLIs and conformance.
- The current source has a typed derivation IR, a one-call handle-selection
  gateway, immutable clarification state, trust labels, effect authorization,
  and a source-defined EviBench diagnostic. Final test counts belong in the
  release audit rather than this frozen evidence ledger.
- The canonical post-verdict paper has a separate executable audit that checks
  claim/evidence bijection, citations, formal exclusions, suite digests, and
  recomputed diagnostic metrics. The old manuscript audit is retained in
  `docs/RESULTS_AUDIT.md` only as historical provenance.

## Claim To Submit

Under explicit compiler, predicate, and state-authority assumptions, EviBind
prevents release of action-critical literals that lack accepted source, role,
extent, and joint-contract certificates. Empirically, it realizes selective
precision/coverage operating points; it does not establish production safety,
model-agnostic accuracy, or completed-agent-task reliability.

The hierarchy study provides the mechanistic evidence. R2-F provides the
prospective operating point. BFCL provides an external cross-model boundary:
rejection improves the pooled official metric under its irrelevance-heavy mix,
but does not establish balanced capability transfer on the larger models.
R2-H, ToolSandbox, and distillation prevent overclaiming by showing where the
method does not improve capability or task progress.

## Remaining External Work

These items cannot be completed faithfully from the current local environment:

1. Two independent human annotators must complete and adjudicate the prepared
   256-item evidence audit using `docs/EVIDENCE_AUDIT_CODEBOOK.md`. A separate
   Codex AI diagnostic now covers 256/256 rows, but it is not independent human
   evidence; the frozen status remains `pending_double_annotation`, and the
   blank human sheets remain untouched.
2. Hosted-provider smoke tests require real provider credentials. vLLM and
   SGLang live checks require those servers and compatible model/parser setups;
   fixture compatibility is already tested but must not be called live validation.
3. Multi-call planning, streaming, distributed nonce consumption, and
   long-horizon task completion require a later product/research protocol.
   Versioned clarification state is implemented, but does not establish
   long-horizon task utility.

The additional native large-model BFCL reference is complete and all-diagnostic.
Its integrity-only resolution substitutes no later comparator, changes no
generated output, and emits no system contrast. It remains supplementary context,
not a dependency for the paper's current claim.

## Legacy Submission Gate

- Current-controller BFCL discipline and official replay: passed.
- Native BFCL supplementary resolution: passed as `completed_diagnostic_only`.
- R2-H resolution: passed with one disclosed diagnostic exclusion.
- Prospective compute-matched hierarchy: passed.
- Stateful benchmark: completed and reported as a negative result.
- Hash-pinned artifact-to-manuscript audit: passed.
- Clean wheel, both CLIs, conformance, and non-root container health: passed.
- Limitations covering single-call scope, schema authoring, state authority,
  admitted-call versus task-completion accuracy, and external validation: present.
- Significant LLM assistance and the diagnostic-only status of simulated reviews
  are disclosed; no AI labels are represented as human evidence.

The paper should not use "production reliability" or imply that certificate
validity proves correct tool selection. The strongest honest framing is selective
invocation-scoped authorized evidence admission with a measured precision/coverage frontier.

## Current Post-Verdict Gate

- Canonical literal-free title, abstract, and method: complete.
- Materialization-confinement and literal-noninterference theorem with five
  explicit assumptions and exclusions: complete.
- One-call matched-compute runner and 13-case digest-pinned diagnostic:
  complete; diagnostic only.
- Independent policy authoring, double human annotation, and the powered
  multi-model study: pending external work.
- Direct experimental comparisons to related systems: pending compatible
  reference implementations; the paper currently makes only a sourced
  property-level positioning.
- Final release audit: complete and recorded in
  `docs/RELEASE_AUDIT_0.3.0.md`.
