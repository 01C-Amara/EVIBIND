# Evidence Audit Annotation Codebook

## Purpose and independence

This codebook governs the blinded annotation sheets produced by
`scripts/prepare_evidence_audit.py`. Annotators A and B work independently and
must not inspect the controller's diagnostic label, another annotator's sheet,
or aggregate results before submitting their first-pass judgments.

The frozen system key is a legacy diagnostic output, not human ground truth.
Human labels may disagree with it. Do not change a judgment to match the key.

Score all six diagnostic axes before selecting the overall evidence class.
Judge only from the source context, proposed destination slot and value,
declared transform, and visible contract information in the row.

## Axis values

- `yes`: the stated criterion is satisfied by the visible evidence.
- `no`: the stated criterion is violated by the visible evidence.
- `uncertain`: the supplied context, transform, or contract is insufficient to
  decide responsibly.
- `not_applicable`: the criterion genuinely does not apply to this candidate.
  Do not use this value merely because necessary information is missing.

## Diagnostic axes

### `span_support`

Does the cited source span, interpreted in context, support the proposed value
before any declared normalization?

- Use `yes` when the complete value is present in an active source span.
- Use `no` for a partial or boundary-breaking match, such as extracting `42`
  from the opaque identifier `Patient 42`.
- Use `not_applicable` when the candidate makes no span-based claim.
- A matching string elsewhere in the context does not support the slot unless
  it is the cited, operative source.

### `normalization_correct`

Does the declared transform deterministically map the source value to the
proposed value while preserving meaning?

- Use `yes` for a valid declared conversion such as an unambiguous date to its
  canonical ISO representation.
- Use `no` when the transform changes meaning, drops required information, or
  cannot produce the proposed value.
- Use `not_applicable` for a genuine identity mapping.
- Use `uncertain` when a derived value is plausible but the transform or
  necessary context is absent.

### `slot_role_correct`

Does the evidence semantically fill the destination slot?

- Use `yes` when the source presents the value in the role named by the slot.
- Use `no` when the same token appears in a different role. Mere occurrence is
  insufficient.
- Use `uncertain` when the visible schema or discourse does not establish the
  role.

### `scope_correct`

Are entity boundaries, quantifiers, units, and value extent preserved?

- Use `yes` when the evidence covers the complete intended value.
- Use `no` for partial identifiers, omitted units, wrong entity attachment,
  changed quantifiers, or text taken outside the governing clause.
- Use `uncertain` when the supplied context is too short to establish scope.

### `contract_correct`

Does the candidate satisfy the visible destination contract?

Consider type, enum membership, required structure, and any visible cross-field
constraint. Use `uncertain` rather than guessing when the relevant contract is
not supplied. Do not infer application authorization, allowlists, defaults, or
business rules that are not visible in the row.

### `contradiction_correct`

Is it correct to treat the proposed evidence as active and non-contradicted?

- Use `yes` when the evidence is operative and has not been negated,
  superseded, corrected, or confined to a hypothetical or unmet condition.
- Use `no` when the value is mentioned but then rejected, corrected, negated,
  or made inapplicable.
- Use `uncertain` when the discourse does not reveal which statement governs.

## Overall evidence class

Apply the following decision order:

1. `contradicted`: the value is mentioned or otherwise suggested, but the
   operative context negates, supersedes, corrects, or disqualifies it.
2. `explicit`: an active source span directly supports the complete value;
   canonical JSON representation alone does not turn this into normalization.
3. `normalized`: a declared, deterministic, meaning-preserving transform maps
   active source evidence to the proposed value.
4. `inferred_safe`: the visible context and contract uniquely entail the value
   without a matching span or hidden world knowledge.
5. `unsupported`: none of the preceding classes establishes the value.

Use `inferred_safe` sparingly. A likely default, conventional assumption, or
plausible application behavior is not a safe inference unless the visible
contract makes the result unique.

## Recurring boundary cases

- An absent mention of precipitation does not by itself support
  `include_precipitation=true`; absent an explicit visible default, label it
  `unsupported`.
- A derived end time can be `normalized` only when the row declares enough
  source and transform information to reproduce it. Otherwise use
  `unsupported`, with `normalization_correct=uncertain`.
- A number embedded in an opaque identifier is not a standalone numeric value.
- In “not Paris; use Rome,” `Paris` is `contradicted` and `Rome` may be
  `explicit`, depending on the destination role.
- A conditional value is not active evidence unless the visible context
  establishes that its condition holds.

## Adjudication

After both blinded passes are frozen, adjudicate axis disagreements before the
overall class. Record the reason for each resolved disagreement in the
adjudication notes. If the row itself lacks decisive context, preserve
`uncertain` on the affected axis rather than manufacturing a resolution.

Report agreement before adjudication and the final adjudicated class
distribution. The current prepared sample contains no system-key
`inferred_safe` stratum and is stratified for diagnostic coverage rather than
deployment prevalence; do not present its class proportions as prevalence
estimates.
