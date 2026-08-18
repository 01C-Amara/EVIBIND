from __future__ import annotations

from collections import Counter

from tapbench.families import (
    FAMILIES,
)
from tapbench.hierarchy_cases import (
    HIERARCHY_CASE_COUNT,
    generate_hierarchy_cases,
    hierarchy_case_manifest,
)
from tapbench.hierarchy_families import HIERARCHY_FAMILIES
from tapbench.r2b_families import R2B_FAMILIES
from tapbench.r2c_families import R2C_CONFIRM_FAMILIES, R2C_PILOT_FAMILIES
from tapbench.r2d_families import R2D_CONFIRM_FAMILIES
from tapbench.r2e_families import R2E_FAMILIES
from tapbench.r2f_families import R2F_FAMILIES


def test_hierarchy_cases_are_balanced_unique_and_family_disjoint() -> None:
    rows = generate_hierarchy_cases()
    assert len(rows) == HIERARCHY_CASE_COUNT
    assert len({row["case_id"] for row in rows}) == HIERARCHY_CASE_COUNT
    assert len({row["family"] for row in rows}) == 24
    existing = {
        family.name
        for family in (
            *FAMILIES,
            *R2B_FAMILIES,
            *R2C_PILOT_FAMILIES,
            *R2C_CONFIRM_FAMILIES,
            *R2D_CONFIRM_FAMILIES,
            *R2E_FAMILIES,
            *R2F_FAMILIES,
        )
    }
    assert not existing & {definition.spec.name for definition in HIERARCHY_FAMILIES}
    for family in {row["family"] for row in rows}:
        counts = Counter(
            row["task_kind"] for row in rows if row["family"] == family
        )
        assert counts == {
            "call": 8,
            "missing_info": 8,
            "no_tool": 8,
            "direct_answer": 8,
        }


def test_hierarchy_cases_declare_extent_contracts_and_stable_manifest() -> None:
    rows = generate_hierarchy_cases()
    call_rows = [row for row in rows if row["task_kind"] == "call"]
    assert all(row["hierarchy_oracle"]["declared_semantic_envelopes"] for row in call_rows)
    assert {
        row["factors"]["extent_stratum"] for row in rows
    } == {
        "labelled_identity_extent",
        "opaque_identifier",
        "uri",
        "enum_and_cross_field_constraint",
        "bounded_date_or_number_transform",
    }
    for row in call_rows:
        envelopes = set(
            row["hierarchy_oracle"]["declared_semantic_envelopes"].values()
        )
        stratum = row["factors"]["extent_stratum"]
        if stratum == "labelled_identity_extent":
            assert "head_number" in envelopes
        elif stratum == "opaque_identifier":
            assert "opaque_atom" in envelopes
        elif stratum == "uri":
            assert "uri" in envelopes
    first = hierarchy_case_manifest(rows)
    second = hierarchy_case_manifest(generate_hierarchy_cases())
    assert first == second
    assert first["case_count"] == 768


def test_hierarchy_gold_passes_every_certificate_fragment() -> None:
    from tapbench.hierarchy_cases import audit_hierarchy_gold

    report = audit_hierarchy_gold(generate_hierarchy_cases())
    assert report["case_count"] == 768
    assert report["call_count"] == 192
    assert report["failures"] == []
    assert report["passed"]
