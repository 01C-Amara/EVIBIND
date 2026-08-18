from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


STATEFUL_SELECTION_VERSION = "evibind.toolsandbox_selection.v1"
CATALOG_VARIANTS = {
    "no_distractors": "",
    "three_distractors": "_3_distraction_tools",
    "ten_distractors": "_10_distraction_tools",
    "all_tools": "_all_tools",
}


def select_stateful_scenarios(
    catalog: Mapping[str, Sequence[str]],
    *,
    excluded_families: Sequence[str],
) -> list[dict[str, Any]]:
    excluded = set(excluded_families)
    base_families = sorted(
        name
        for name, categories in catalog.items()
        if "STATE_DEPENDENCY" in categories
        and "NO_DISTRACTION_TOOLS" in categories
        and name not in excluded
    )
    rows: list[dict[str, Any]] = []
    for family in base_families:
        for variant, suffix in CATALOG_VARIANTS.items():
            scenario = family + suffix
            if scenario not in catalog:
                raise ValueError(
                    f"missing catalog variant {variant!r} for family {family!r}"
                )
            categories = sorted(str(value) for value in catalog[scenario])
            if "STATE_DEPENDENCY" not in categories:
                raise ValueError(f"{scenario!r} lost STATE_DEPENDENCY")
            rows.append(
                {
                    "family": family,
                    "variant": variant,
                    "scenario": scenario,
                    "categories": categories,
                }
            )
    return rows


def canonical_selection_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        list(rows),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
