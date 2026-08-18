from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from .extractive_candidates import canonical_slots


SEMANTIC_CLOSURE_VERSION = "tapbench.semantic_closure.v1"
_HEAD_NUMBER_RE = re.compile(r"^([A-Za-z]+)\s+(\d+)$")
_SINGLE_TOKEN_RE = re.compile(r"^[A-Za-z]+$|^\d+$")


def _argument_aliases(tool: dict[str, Any]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for row in canonical_slots(tool):
        canonical = str(row["name"])
        aliases[canonical] = canonical
        aliases[str(row["surface_name"])] = canonical
    return aliases


def _schema_by_slot(tool: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["name"]): row["schema"]
        for row in canonical_slots(tool)
    }


def _unique_head_number_candidate(
    fragment: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not _SINGLE_TOKEN_RE.fullmatch(fragment.strip()):
        return None
    token = fragment.strip().casefold()
    matches: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}
    for row in candidates:
        value = row.get("value")
        source_text = row.get("source_text")
        span = row.get("source_span")
        if (
            not isinstance(value, str)
            or not isinstance(source_text, str)
            or value != source_text
            or row.get("transform") != "identity"
            or not isinstance(span, list)
            or len(span) != 2
        ):
            continue
        parsed = _HEAD_NUMBER_RE.fullmatch(value.strip())
        if parsed is None:
            continue
        head, number = parsed.groups()
        if token not in {head.casefold(), number.casefold()}:
            continue
        key = (value.casefold(), tuple(int(item) for item in span))
        matches[key] = row
    if len(matches) != 1:
        return None
    return next(iter(matches.values()))


def close_unique_head_number_arguments(
    proposal: dict[str, Any],
    *,
    tool: dict[str, Any],
    candidate_table: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Close uniquely identified head-number fragments onto source spans.

    The function has no oracle inputs. It only rewrites a model proposal to an
    existing candidate value, after which ordinary TAP-R certification still
    decides whether the action is executable.
    """
    closed = deepcopy(proposal)
    raw_arguments = closed.get("arguments")
    if closed.get("mode") != "call" or not isinstance(raw_arguments, dict):
        return closed, []

    aliases = _argument_aliases(tool)
    schemas = _schema_by_slot(tool)
    audits: list[dict[str, Any]] = []
    seen: set[str] = set()
    for surface_slot, value in list(raw_arguments.items()):
        canonical = aliases.get(str(surface_slot))
        if canonical is None or canonical in seen:
            continue
        seen.add(canonical)
        schema = schemas.get(canonical, {})
        if str(schema.get("x-tap-semantic-envelope", "")).casefold() != "head_number":
            continue
        if not isinstance(value, str) or _HEAD_NUMBER_RE.fullmatch(value.strip()):
            continue
        candidate = _unique_head_number_candidate(
            value,
            list(candidate_table.get("slots", {}).get(canonical, [])),
        )
        if candidate is None:
            continue
        raw_arguments[surface_slot] = deepcopy(candidate["value"])
        audits.append(
            {
                "slot": canonical,
                "candidate_id": int(candidate["candidate_id"]),
                "source_span": list(candidate["source_span"]),
                "source_text": candidate["source_text"],
                "rule": "unique_head_or_number_fragment",
            }
        )
    return closed, audits
