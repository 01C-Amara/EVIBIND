# AI Evidence Audit Diagnostic

Audit date: 2026-07-28

## Status

Codex completed one AI diagnostic pass over all 256 rows in the frozen blinded
sheet. This is not independent human annotation, double annotation,
adjudication, or a substitute for the protocol in
`docs/EVIDENCE_AUDIT_CODEBOOK.md`.

The AI pass was written and hashed before the script opened the legacy system
key. The `human_label` column remains blank on every output row. The original
annotator A, annotator B, and adjudication sheets were not modified.

## Artifacts

- Annotations:
  `work/r2/evidence_audit_ai_diagnostic_v1/annotations.csv`
- Report:
  `work/r2/evidence_audit_ai_diagnostic_v1/report.json`
- Generator:
  `scripts/annotate_evidence_audit_ai.py`
- Annotation SHA-256:
  `8ab513213445c9071b35de9d6ca279db1f9898e522def3555b74fe50c1108115`

## Results

- Rows with all six diagnostic axes: 256/256.
- Overall classes: 79 `explicit`, 177 `unsupported`.
- Rows flagged ambiguous: 36.
- Legacy system-key agreement: 50.0%.

The 50.0% value is diagnostic agreement, not accuracy. The system key is a
legacy controller label rather than ground truth, and this sample is stratified
by that key rather than by deployment prevalence.

Every blind row omits visible slot-role, resolution-type, source-kind,
transform-context, scope-status, and contradiction-status metadata. Therefore
`contract_correct` is `uncertain` on all 256 rows, and no row is promoted to
`normalized` without enough visible transform context. This is a form-quality
finding as well as an annotation result.

## Boundary

The frozen human-audit status remains `pending_double_annotation`. Two genuinely
independent people must still label the sheets without seeing this AI output or
the system key, followed by adjudication under the codebook. The AI class
distribution must not be presented as evidence prevalence or human validation.
