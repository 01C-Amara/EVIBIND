# Results And Submission Audit

Audit date: 2026-07-28

## Verified Artifacts

| Claim | Frozen source | Status |
|---|---|---|
| R2-F safe decision 26.2% to 64.8%; 89/89 released calls exact | `work/r2f/confirmation_v1/analysis/` | verified |
| R2-F call coverage 69.5%; unsupported action-critical 0/512 | same analysis | verified |
| R2-F paired contrast +38.7 points, CI +31.4 to +45.9 | `analysis/report.json` | verified |
| R2-F compute 6.89 versus 1.00 mean calls and 4,223 versus 2,232 mean tokens | timing summaries | verified; not compute matched |
| Distillation best selective result 69.1% safe resolution, 25.0% coverage | `work/verified_distillation/evaluation_v1/analysis/report.json` | verified |
| Certificate-verified minus execution-filtered coverage -17.2 points, CI [-40.6, 0.0] | same report | verified negative |
| Clarification adapter excluded after 53 reasoning-marker failures | report and `queue_resolution.yaml` | verified diagnostic exclusion |
| BFCL combined gains +18.9, +9.5, and +7.9 points | `work/bfcl_current_controller/v2/analysis/report.json` | verified across three models |
| BFCL equal-class changes +8.8, -5.4, and -7.9 points | `work/bfcl_current_controller/v2/analysis/balanced_view.json` | verified post-hoc sensitivity |
| BFCL break-even irrelevance prevalence 26.2%, 60.0%, and 63.8% | same balanced report | verified paired-bootstrap crossover sensitivity |
| BFCL accepted precision 63.5%, 86.3%, and 89.1%; simple coverage 35.7%, 25.6%, and 20.9% | same report | verified tradeoff; all below the 95% gate |
| BFCL certificate replay failures 0 across 115, 73, and 55 calls | per-model certificate summaries | verified |
| Native BFCL reference completed 2,284/2,284 rows; both arms diagnostic-only | `work/bfcl_native_large_models/v1/queue_resolution.yaml` | verified exclusion; no comparator coefficients |
| Native Qwen findings 66 runner errors, 35 length stops, 52 visible markers; GPT-OSS 66, 60, and 1 | same resolution | verified; zero context truncations on both arms |
| R2-H semantic closure gain 0.0 points on clean Qwen; zero recovered calls on both models | `work/r2h/large_model_closure_v1/analysis/report.json` | verified null |
| R2-H GPT-OSS excluded after 8 length-stop rows from 4 shared traces | report and `queue_resolution.yaml` | verified diagnostic exclusion |
| Hierarchy extent precision increment +51.5 points, CI +34.0 to +69.8 | `work/r2h/hierarchy_ablation_v1/analysis.json` | verified prospective |
| Hierarchy coverage change -16.7 points, CI -25.5 to -8.9 | same report | verified tradeoff |
| Full controller minus compute-matched best-of safe decision +30.7 points, CI +26.7 to +34.8 | same report | verified |
| Hierarchy compute use 2,214,906 / 2,216,052 tokens (99.95%) | same report | verified compute match |
| Hierarchy released 63 calls, 62 exact, with 0 certificate replay failures | analysis and certificate summary | verified |
| Verbatim-span release reaches only 11.8% accepted precision; 86/127 released calls remain unsupported | `work/reviewer_extensions_v1/analysis.json` | verified post-hoc baseline |
| Extent sweep has 32 policies, 14 non-dominated points, and 26.5% observed-range normalized AURC | same report and `risk_coverage_frontier.pdf` | verified post-hoc diagnostic |
| Scripted clarification recovers 10/20 exact calls and 5.2 coverage points with 10/10 precision | `work/r2h/clarification_replay_v1/analysis.json` | verified post-hoc cooperative oracle; not human evidence |
| Clean Liquid and Qwen3.6 scale arms gain 29.5 and 40.6 safe-decision points but fail coverage gates | `work/r2g/scale_reference_v1/analysis_amendment_v2/report.json` | verified directional evidence; transfer gate failed |
| ToolSandbox milestone similarity -15.5 points; tool-call exceptions -25.6 points | `work/evibind_toolsandbox_v1/analysis/confirmatory/analysis.json` | verified two-model negative |
| ToolSandbox Qwen and Gemma-E4B progress effects -8.0 and -23.0 points | same report | verified per-model disclosure |
| ToolSandbox loss taxonomy: 77 negative pairs; 54.0% gateway rejection/clarification loss and 46.0% released-call exception loss | `work/reviewer_extensions_v1/analysis.json` | verified descriptive, not causal |
| ToolSandbox Liquid and Gemma-E2B diagnostic exclusions | `work/evibind_toolsandbox_v1/queue_resolution.yaml` | verified |

The R2-F table reports call-task accepted precision for the constrained baseline
(23/98 = 23.5%), while the all-task method summary contains 239 emitted calls.
The manuscript keeps this denominator distinction explicit.

## Machine Audit

Run:

```bash
python scripts/audit_iclr_submission.py \
  --output ../R\&D/tap_iclr_submission/results_audit.json
```
The current `evibind.submission_results_audit.v6` report passes 253/253 checks
across 208 extracted claims. It pins SHA-256 hashes for the pooled and balanced
BFCL reports, R2-H, hierarchy, ToolSandbox, distillation, selected-model,
scale-reference, clarification-replay, reviewer-extension, and frontier-figure
artifacts and checks the corresponding manuscript language and LLM-use
disclosure.

## Build Audit

- Submission source: `R&D/tap_iclr_submission/main.tex`.
- Current PDF: 11 pages total; the numbered conclusion ends on page 8,
  ethics/reproducibility and references begin on page 9, references continue on
  page 10, and the appendix is on page 11.
- Build command: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`.
- Undefined references or citations: none.
- Overfull, underfull, or LaTeX warnings: none in the current log.
- The long `R&D/tap_small_llm_tool_calling.tex` file is an audit archive, not the concise submission source.

## Resolution Status

- BFCL current-controller v2 is complete and clean across all three models. The
  earlier v1 is diagnostic-only.
- R2-H is complete. Qwen is coefficient-eligible and null; GPT-OSS is retained
  diagnostic-only after the frozen length-stop gate.
- The prospective hierarchy study is complete: 3,840 rows, zero generation
  integrity failures, compute audit passed, and 63/63 certificates replayed.
- ToolSandbox is complete with two clean confirmatory models and two transparent
  diagnostic exclusions. Its task-progress result is negative.
- The additional large-model native-tool BFCL reference is complete with 1,142
  rows per model. Qwen and GPT-OSS both failed original integrity gates and are
  diagnostic-only; no native-reference coefficient is promoted to the manuscript.
  A post-generation integrity-only amendment resolved the frozen v1 comparator
  absence without substituting v2, changing generated outputs, or emitting
  comparator contrasts. The terminal outcome is `completed_diagnostic_only`.

## External Boundaries

- Live llama.cpp transport is validated. No hosted-provider credentials are
  available, and vLLM/SGLang are not installed; their status remains protocol
  fixture compatibility rather than live-provider validation.
- The 256-item blinded double-human evidence audit is prepared with
  `docs/EVIDENCE_AUDIT_CODEBOOK.md` but still awaits two genuinely independent
  annotators and adjudication. A separate Codex AI diagnostic is complete for
  256/256 rows with all six axes, 79 `explicit`, 177 `unsupported`, and 36
  ambiguity flags; its 50.0% legacy-key agreement is not an accuracy estimate.
  See `docs/EVIDENCE_AUDIT_AI_DIAGNOSTIC.md`. The human sheets remain untouched,
  and AI-generated labels do not satisfy the independent-human requirement.
- Single-call, non-streaming admission is the evaluated product boundary.
- AI-simulated reviewer perspectives are recorded only in
  `docs/ICLR_SYNTHETIC_REVIEW_PANEL.md`; they are not human evidence.
  Multi-call planning, live-service authorization, and long-horizon completion
  remain out of scope and are named as limitations.

Development or diagnostic outputs must not be promoted into confirmatory causal
claims. Every exclusion above retains the original generated rows and records
its failed frozen gate.
