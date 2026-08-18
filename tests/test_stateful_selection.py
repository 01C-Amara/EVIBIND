from __future__ import annotations

from tapbench.stateful_selection import (
    CATALOG_VARIANTS,
    canonical_selection_sha256,
    select_stateful_scenarios,
)


def _catalog(*families: str) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for family in families:
        for variant, suffix in CATALOG_VARIANTS.items():
            distractor = {
                "no_distractors": "NO_DISTRACTION_TOOLS",
                "three_distractors": "THREE_DISTRACTION_TOOLS",
                "ten_distractors": "TEN_DISTRACTION_TOOLS",
                "all_tools": "ALL_TOOLS_AVAILABLE",
            }[variant]
            rows[family + suffix] = ["STATE_DEPENDENCY", distractor]
    rows["ordinary"] = ["SINGLE_TOOL_CALL", "NO_DISTRACTION_TOOLS"]
    return rows


def test_selection_is_family_grouped_deterministic_and_excludes_pilot() -> None:
    rows = select_stateful_scenarios(
        _catalog("zeta", "alpha", "pilot"),
        excluded_families=["pilot"],
    )
    assert len(rows) == 8
    assert [row["family"] for row in rows[:4]] == ["alpha"] * 4
    assert [row["variant"] for row in rows[:4]] == list(CATALOG_VARIANTS)
    assert [row["family"] for row in rows[4:]] == ["zeta"] * 4
    assert canonical_selection_sha256(rows) == canonical_selection_sha256(rows)


def test_selection_rejects_incomplete_catalog_family() -> None:
    catalog = _catalog("alpha")
    del catalog["alpha_all_tools"]
    try:
        select_stateful_scenarios(catalog, excluded_families=[])
    except ValueError as exc:
        assert "missing catalog variant" in str(exc)
    else:
        raise AssertionError("incomplete family should fail closed")
