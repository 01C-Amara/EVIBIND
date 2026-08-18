# EviBind Research Harness

This repository contains the research harness behind the EviBind tool-calling
paper and its frozen evaluation protocols.

The design intentionally avoids the infeasible full factorial grid:

```text
3 x 3 x 3 x 3 x 2 x 2 = 324 cells/family
324 x 8 families x 5 cases = 12,960 call cases
12,960 x 10 methods x 8 models x 3 seeds = 3,110,400 generations
```

Instead, `configs/hypothesis_subgrids.yaml` defines committed fractional
sub-grids per hypothesis. `tapbench generate --scope full` refuses to run unless
a pilot runtime projection exists.

## Quick Start

```bash
python -m tapbench generate --scope pilot --output work/pilot/cases.jsonl
python -m tapbench score --cases work/pilot/cases.jsonl \
  --predictions tests/fixtures/golden/predictions.jsonl \
  --output /tmp/tap_scores.jsonl
python -m tapbench conformance
python -m pytest
```

The product-path diagnostic suite is frozen separately:

```bash
python -m tapbench.cli evibench-freeze \
  --cases tapbench/data/evibench_v1.jsonl \
  --manifest configs/evibench_v1_manifest.json
```

See `docs/EVIBENCH.md` for its matched-compute conditions, replay artifact
contract, and the required separation between compiler and selector metrics.

## Main Commands

- `tapbench generate`: write synthetic benchmark cases from the fractional grid.
- `tapbench score`: normalize predictions into the Action IR and write versioned
  metrics.
- `tapbench project-runtime`: turn pilot timings into a full-run wall-clock
  projection.
- `tapbench analyze`: export scored rows for `analysis/glmm_fit.R`.
- `tapbench conformance`: prove local grammar controllers reject EOS in
  non-accepting states.
- `tapbench evibench-freeze`: reproduce the digest-pinned EviBench v1 corpus.
- `tapbench evibench-replay`: score a complete payload-bound response matrix.

## Hard Rules Captured In Config

- Hypothesis contrasts never mix backend, quantization, chat template, or output
  format.
- H4 legal-token-mass diagnostics are separate from llama.cpp accuracy GLMMs.
- Specialists are scored only after native-format parsing into the Action IR.
- Retrieval claims are only identified at `N=64`; smaller catalogs report
  recall but do not feed retrieval-help coefficients.
- `GPT-OSS-120B` is teacher/paraphraser only and excluded from evaluated grids.

## External Model Pins

See `configs/model_pins.yaml`. The default main set is Qwen3-1.7B,
LiquidAI/LFM2.5-8B-A1B, Gemma-4-E2B-it, and Gemma-4-E4B-it, with documented
fallbacks and specialist native-format baselines.
