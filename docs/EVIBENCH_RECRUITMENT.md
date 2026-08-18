# EviBench Recruitment and Inventory Handoff

The schema inventory can be harvested mechanically. The participant registry
cannot: its rows assert real consent, agreed compensation, completed training,
language qualification, and globally disjoint roles.

## Schema inventory

The inventory builder selects distinct APIs with explicit open-license metadata,
limits provider/category concentration, downloads and hash-pins the source
artifacts, and normalizes two to four operations into tool-family schemas.

```bash
python scripts/prepare_evibench_family_inventory.py \
  --directory-index /path/to/apis-guru-list.json \
  --output-dir work/evibench_powered_v1/family_inventory \
  --families-output work/evibench_powered_v1/families.jsonl \
  --recruitment-slots work/evibench_powered_v1/recruitment_slots.jsonl
```

Recompute every cached source digest, normalize each source again, compare the
tool schemas and operation provenance, and verify the split and diversity caps:

```bash
python scripts/audit_evibench_family_inventory.py \
  --families work/evibench_powered_v1/families.jsonl \
  --inventory-dir work/evibench_powered_v1/family_inventory \
  --directory-index /path/to/apis-guru-list.json \
  --output work/evibench_powered_v1/family_inventory/inventory_audit.json
```

The generated `families.jsonl` is technically complete but each row remains
`license_review_status: pending_human_confirmation`. Review every entry in
`family_inventory/license_review.yaml`. After confirming all sources and
licenses, an authorized reviewer must set every family row to
`license_review_status: human_confirmed`, rerun the audit, and only then create
the non-empty marker
`work/evibench_powered_v1/FAMILY_LICENSE_REVIEWED`.

## Participant registry

Recruit exactly one globally disjoint role per pseudonymous participant. The
minimum pools are 8 policy engineers, 12 request authors, 10 annotators, and 3
adjudicators. Request authors, annotators, and adjudicators must be qualified
for both English and Spanish. Policy engineers instead require the technical
policy-authoring qualification.

The final private registry has one JSON object per line:

```json
{"participant_id":"request-017","role":"request_author","languages":["en","es"],"consent_recorded":true,"compensation_agreed":true,"training_complete":true}
```

Do not put names, email addresses, payment identifiers, or consent documents in
the repository or registry. Store those records in the approved research system
and use only its pseudonymous participant ID here. Do not copy the generated
`recruitment_slots.jsonl` to `participants.jsonl`: unfilled slots are not people.

The live queue advances only when the license marker and a valid real
`participants.jsonl` are present. Invalid, duplicate, cross-role, unconsented,
uncompensated, or untrained entries fail closed.
