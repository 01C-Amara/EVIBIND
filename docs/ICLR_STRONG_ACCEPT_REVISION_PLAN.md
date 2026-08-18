# ICLR Post-Verdict Revision Plan

## Objective

Address every actionable part of the verdict without manufacturing evidence.
The canonical contribution is now literal-free, evidence-bound argument
materialization with a conditional noninterference guarantee. Frozen legacy
studies remain boundary evidence; model-quality claims require the separately
preregistered powered extension.

## Post-Verdict Checkpoints

| Checkpoint | Status | Exit evidence |
|---|---|---|
| Typed evidence derivation IR and evidence types | Complete | Product code, schema tests, replay tests |
| One-call literal-free action representation | Complete | Forced handle-selection gateway and matched-call tests |
| Operational modes, adapters, state, trust, and effects | Complete | CLI/adapters, immutable execution graph, exact-manifest confirmation |
| EviBench diagnostic and matched-compute harness | Complete | Source-defined corpus, SHA-256 manifest, payload-bound replay |
| Canonical paper/theory rewrite | Complete | `paper/main.tex`, formal theorem, executable claim audit |
| Powered independently authored evaluation | Pending external work | Frozen protocol exists; corpus, human annotation, and model runs do not |
| Final release audit | Complete | Full tests, conformance, packaging, paper build, identity audit |

## Reviewer Risks And Exit Criteria

| Priority | Reviewer risk | Required resolution | Exit criterion |
|---|---|---|---|
| P0 | BFCL pooled accuracy is dominated by 884 irrelevance versus 258 simple cases | Add a source-hashed, paired, within-class bootstrap sensitivity analysis; report pooled and equal-class views | Abstract, table, discussion, and audit disclose that balanced accuracy is +8.8, -5.4, and -7.9 points |
| P0 | The guarantee assumes semantic soundness and reads as tautological | Recast it as a non-bypass release invariant; define candidates, evidence provenance, invocation-level contracts, critical slots, and the trust boundary | The proposition claims only what the implementation can enforce and no longer writes a joint contract as a unary value predicate |
| P0 | Main comparison hides extra inference | Report mean calls, prompt/completion/total tokens, p95 latency, and distinguish the prospective operating-point study from the compute-matched mechanism study | No headline describes R2-F as a compute-matched causal effect |
| P0 | Safe decision accuracy can be mistaken for task success | Rename the manuscript metric to safe policy-decision rate and keep call, clarification, no-tool, and direct-answer components adjacent | Abstract and conclusion cannot be read as autonomous task-completion claims |
| P1 | Semantic extent is underspecified | Add a worked certificate example, authoring rules, transform restrictions, failure transitions, and an algorithm-level description | A reviewer can implement the interface without reverse-engineering the artifact |
| P1 | Synthetic selected-model evidence limits generality | Foreground selection, authoring debt, mocked effects, and the negative external results; report BFCL expressibility | The high-precision result is explicitly one selected operating point |
| P1 | Statistical reporting is weak for selective policies | Add clustered uncertainty where already frozen, accepted denominators, split sizes, and fixed-coverage/risk interpretation | Every precision claim is paired with coverage and its sampling unit |
| P1 | Related work is too sparse | Cite ToolSandbox, LoRA, pointer selection, executable semantic parsing, and representative tool-use benchmarks | Positioning distinguishes the mechanism from finite selection, semantic parsing, structured decoding, and state-contract systems |
| P2 | One sentence confuses precision with execution | Correct 76.6 points to a 51.5-point exact-execution increase and separately state the 76.5-point precision increase | Arithmetic and labels match Table 3 |

## Execution Tracks

### Track A: Frozen-Evidence Reanalysis

- Completed: added `scripts/analyze_bfcl_balanced_view.py`.
- Completed: tests prove equal class weighting does not collapse to pooled
  prevalence and that bootstrap pairs cases within each category.
- Completed: generated
  `work/bfcl_current_controller/v2/analysis/balanced_view.json` without changing
  any frozen prediction or official score.
- Completed: pinned the sensitivity report in the manuscript claim audit and
  documented its post-hoc status.
- Completed: derived paired-bootstrap break-even irrelevance prevalence at
  26.2%, 60.0%, and 63.8% without changing frozen rows.

### Track B: Method And Theory Rewrite

- Completed: replaced the semantic-soundness reading with a precise non-bypass theorem.
- Completed: separated per-candidate obligations from the invocation-level contract.
- Completed: defined the critical-slot declaration as part of the trusted schema.
- Completed: added one concrete request-to-candidate-to-certificate example.
- Completed: specified every deterministic extent predicate and its non-guarantees
  in the main paper.
- Completed: stated that defaults, transforms, and state references are request-scoped
  authorities, not literal request substrings.

### Track C: Empirical Reframing

- Completed: promoted the compute-matched hierarchy study to the causal mechanism result.
- Completed: presented R2-F as a prospective selected-model operating point with roughly
  1.9x mean token use and 6.89 versus 1.00 model calls.
- Completed: presented BFCL pooled and balanced views together.
- Completed: renamed the aggregate metric in prose while retaining the frozen artifact key.
- Completed: disclosed BFCL envelope expressibility: 487/1,142 cases fully expressible and
  1,172/2,863 required string slots explicitly declared.

### Track D: External Evidence

- Required for a materially stronger external-validity claim: two independent
  annotators and adjudication for the prepared 256-item evidence audit.
- High-value follow-up: an independently authored natural-request benchmark
  with authoring-time, ambiguity, and inter-annotator measurements.
- Optional deployment evidence: hosted-provider and alternate-server checks
  once credentials and compatible vLLM/SGLang deployments exist.
- Out of scope for this submission: multi-call planning, streaming, and
  long-horizon stateful task completion. ToolSandbox already provides the honest
  negative boundary for the current controller.

## Submission Gate

The revised submission is ready for another reviewer-style pass only when:

1. The balanced BFCL report is source-hashed and machine-audited.
2. The full test suite and maintained-source lint pass.
3. The manuscript claim audit has no failed checks.
4. The PDF remains within the ICLR main-text limit with no build warnings.
5. The abstract, conclusion, and title-level framing survive the strongest
   rejection summary: selected synthetic evidence, extra inference, and
   irrelevance-driven external gains are all disclosed rather than buried.
6. Significant LLM assistance is disclosed, and simulated reviews are not
   represented as independent human evidence.

Legacy gate status: all six checks passed for the earlier manuscript. The
canonical paper has a new gate and must not inherit that status. Independent
human annotation and the powered external benchmark remain explicit future
evidence, not hidden submission dependencies.

