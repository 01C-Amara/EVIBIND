from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from decimal import Decimal
from typing import Any, Iterable


EXTRACTIVE_CANDIDATE_VERSION = "tapbench.extractive_candidates.v4"
DEFAULT_MAX_SPAN_TOKENS = 10
DEFAULT_MAX_CANDIDATES_PER_SLOT = 48
DEFAULT_MAX_CANDIDATES_PER_ACTION = 384

_TOKEN_RE = re.compile(
    r"[A-Za-z0-9]+(?:[.#@/_:+\-'][A-Za-z0-9]+)*|[^\w\s]",
    re.UNICODE,
)
_NUMBER_RE = re.compile(
    r"(?<![\w.])"
    r"(?P<number>[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)"
    r"(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"(?P<unit>\s*(?:%|[A-Za-z\u00b5\u03bc\u00b0]+"
    r"(?:/[A-Za-z0-9^]+)?(?:\^[+-]?\d+)?))?"
    r"(?=$|[^\w.]|\.(?!\d))",
    re.IGNORECASE,
)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "is", "it", "me", "of", "on", "or", "that",
    "the", "this", "to", "with", "you",
}
_BOUNDARY = " \t\r\n,;:!?\"'"
_EXPRESSION_RE = re.compile(r"[A-Za-z0-9)]\s*\^\s*[A-Za-z0-9(]")
_UNIT_FACTORS = {
    "uf": "1e-6",
    "\u00b5f": "1e-6",
    "\u03bcf": "1e-6",
    "mh": "1e-3",
}
_MAGNITUDE_FACTORS = {
    "thousand": "1e3",
    "million": "1e6",
    "billion": "1e9",
}
_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_NUMBER_WORD_RE = re.compile(
    r"\b(?:" + "|".join(_NUMBER_WORDS) + r")\b",
    re.IGNORECASE,
)
_EXPLICIT_LIST_GAP_RE = re.compile(
    r"^\s*[\]})]?"
    r"\s*(?:,|;|and|or|to)"
    r"\s*(?:[A-Za-z_][A-Za-z0-9_]*\s*=\s*)?"
    r"[\[({]?\s*$",
    re.IGNORECASE,
)
_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
_MONTH_PATTERN = "|".join(
    sorted(_MONTHS, key=len, reverse=True)
)
_DATE_PATTERNS = (
    re.compile(
        rf"(?P<month>{_MONTH_PATTERN})[.\s]+"
        r"(?P<day>\d{1,2})(?:st|nd|rd|th)?"
        r"\s*,?\s*(?P<year>\d{4})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+"
        rf"(?P<month>{_MONTH_PATTERN})\s*,?\s*"
        r"(?P<year>\d{4})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<year>\d{{4}})\s+"
        rf"(?P<month>{_MONTH_PATTERN})\s+"
        r"(?P<day>\d{1,2})(?:st|nd|rd|th)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<year>\d{4})[/-](?P<month>\d{1,2})"
        r"[/-](?P<day>\d{1,2})"
    ),
)



def user_request_text(messages: Iterable[dict[str, Any]]) -> str:
    return "\n".join(
        str(row.get("content", ""))
        for row in messages
        if str(row.get("role", "")).casefold() == "user"
    )


def _tokens(text: str) -> list[dict[str, Any]]:
    return [
        {
            "text": match.group(0),
            "start": match.start(),
            "end": match.end(),
            "norm": match.group(0).casefold(),
        }
        for match in _TOKEN_RE.finditer(text)
    ]


def _schema_object(parameters: dict[str, Any]) -> dict[str, Any]:
    current = parameters
    while isinstance(current, dict):
        properties = current.get("properties")
        if (
            not isinstance(properties, dict)
            or set(properties) != {"payload"}
            or not isinstance(properties.get("payload"), dict)
        ):
            break
        current = properties["payload"]
    return current if isinstance(current, dict) else {}


def canonical_slots(tool: dict[str, Any]) -> list[dict[str, Any]]:
    schema = _schema_object(
        tool.get("parameters", {})
        if isinstance(tool.get("parameters"), dict)
        else {}
    )
    required_surface = {
        str(value) for value in schema.get("required", [])
    }
    slots = []
    for surface, raw in schema.get("properties", {}).items():
        prop = raw if isinstance(raw, dict) else {"type": "string"}
        slots.append(
            {
                "surface_name": str(surface),
                "name": str(prop.get("x-ir-name") or surface),
                "required": str(surface) in required_surface,
                "schema": prop,
            }
        )
    return slots


def _anchor_terms(slot: dict[str, Any]) -> set[str]:
    prop = slot["schema"]
    text = (
        str(slot["surface_name"]).replace("_", " ").replace(".", " ")
        + " "
        + str(slot["name"]).replace("_", " ").replace(".", " ")
        + " "
        + str(prop.get("description", ""))
    )
    return {
        value
        for value in re.findall(r"[a-z0-9]+", text.casefold())
        if len(value) > 1 and value not in _STOPWORDS
    }


def _anchor_positions(tokens: list[dict[str, Any]], anchors: set[str]) -> list[int]:
    return [
        index
        for index, token in enumerate(tokens)
        if token["norm"] in anchors
    ]


def _candidate_score(
    *,
    start_token: int,
    end_token: int,
    value: Any,
    transform: str,
    tokens: list[dict[str, Any]],
    anchors: set[str],
    anchor_positions: list[int],
    type_bonus: float,
) -> float:
    length = end_token - start_token + 1
    candidate_terms = {
        token["norm"]
        for token in tokens[start_token : end_token + 1]
        if re.fullmatch(r"[a-z0-9]+", token["norm"])
    }
    overlap = len(candidate_terms & anchors)
    if anchor_positions:
        distance = min(
            0
            if start_token <= position <= end_token
            else start_token - position
            if position < start_token
            else position - end_token
            for position in anchor_positions
        )
        follows = any(0 < start_token - position <= 4 for position in anchor_positions)
    else:
        distance = len(tokens)
        follows = False
    complete_bonus = min(4.0, math.log2(max(2, len(str(value)) + 1)))
    transform_penalty = 0.0 if transform == "identity" else 0.35
    return (
        type_bonus
        + (5.0 if follows else 0.0)
        + 3.0 / (1.0 + distance)
        + complete_bonus
        - 1.6 * overlap
        - 0.08 * length
        - transform_penalty
    )


def _append(
    rows: list[dict[str, Any]],
    seen: set[tuple[str, int, int, str]],
    *,
    value: Any,
    start: int,
    end: int,
    start_token: int,
    end_token: int,
    source_text: str,
    transform: str,
    schema_type: str,
    score: float,
    component_spans: list[list[int]] | None = None,
) -> None:
    key = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        start,
        end,
        transform,
    )
    if key in seen:
        return
    seen.add(key)
    rows.append(
        {
            "value": value,
            "source_span": [start, end],
            "source_text": source_text,
            "source_kind": "extractive_user_span",
            "transform": transform,
            "schema_type": schema_type,
            "score": score,
            "start_token": start_token,
            "end_token": end_token,
            "component_spans": component_spans or [[start, end]],
        }
    )


def _string_candidates(
    text: str,
    tokens: list[dict[str, Any]],
    slot: dict[str, Any],
    *,
    max_span_tokens: int,
) -> list[dict[str, Any]]:
    anchors = _anchor_terms(slot)
    positions = _anchor_positions(tokens, anchors)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, str]] = set()
    for start_token in range(len(tokens)):
        for end_token in range(
            start_token,
            min(len(tokens), start_token + max_span_tokens),
        ):
            start = tokens[start_token]["start"]
            end = tokens[end_token]["end"]
            raw = text[start:end]
            left = len(raw) - len(raw.lstrip(_BOUNDARY))
            right_text = raw.rstrip(_BOUNDARY)
            if not right_text:
                continue
            right = len(right_text)
            cleaned = raw[left:right]
            clean_start = start + left
            clean_end = start + right
            if not cleaned or not re.search(r"[A-Za-z0-9]", cleaned):
                continue
            clean_token_indices = [
                index
                for index in range(start_token, end_token + 1)
                if (
                    tokens[index]["start"] < clean_end
                    and tokens[index]["end"] > clean_start
                )
            ]
            if not clean_token_indices:
                continue
            clean_start_token = clean_token_indices[0]
            clean_end_token = clean_token_indices[-1]
            terms = re.findall(r"[a-z0-9]+", cleaned.casefold())
            if terms and all(term in _STOPWORDS for term in terms):
                continue
            lexical_parts = re.findall(r"[A-Za-z0-9]+", cleaned)
            named_entity_bonus = (
                7.0
                if lexical_parts
                and len(lexical_parts) <= 5
                and all(
                    part[:1].isupper()
                    for part in lexical_parts
                    if any(char.isalpha() for char in part)
                )
                else 0.0
            )
            score = _candidate_score(
                start_token=clean_start_token,
                end_token=clean_end_token,
                value=cleaned,
                transform="identity",
                tokens=tokens,
                anchors=anchors,
                anchor_positions=positions,
                type_bonus=10.0 + named_entity_bonus,
            )
            _append(
                rows,
                seen,
                value=cleaned,
                start=clean_start,
                end=clean_end,
                start_token=clean_start_token,
                end_token=clean_end_token,
                source_text=cleaned,
                transform="identity",
                schema_type="string",
                score=score,
            )
            if _EXPRESSION_RE.search(cleaned):
                canonical = re.sub(r"\s*\^\s*", "**", cleaned)
                expression_score = _candidate_score(
                    start_token=clean_start_token,
                    end_token=clean_end_token,
                    value=canonical,
                    transform="expression_operator_canonicalization",
                    tokens=tokens,
                    anchors=anchors,
                    anchor_positions=positions,
                    type_bonus=10.5,
                )
                _append(
                    rows,
                    seen,
                    value=canonical,
                    start=clean_start,
                    end=clean_end,
                    start_token=clean_start_token,
                    end_token=clean_end_token,
                    source_text=cleaned,
                    transform="expression_operator_canonicalization",
                    schema_type="string",
                    score=expression_score,
                )
    return rows


def _date_candidates(
    text: str,
    tokens: list[dict[str, Any]],
    slot: dict[str, Any],
) -> list[dict[str, Any]]:
    anchors = _anchor_terms(slot)
    positions = _anchor_positions(tokens, anchors)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, str]] = set()
    matched_spans: set[tuple[int, int]] = set()
    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            if span in matched_spans:
                continue
            matched_spans.add(span)
            month_text = match.group("month")
            month = (
                int(month_text)
                if month_text.isdigit()
                else _MONTHS[month_text.casefold()]
            )
            day = int(match.group("day"))
            year = int(match.group("year"))
            if not (1 <= month <= 12 and 1 <= day <= 31):
                continue
            token_indices = [
                index
                for index, token in enumerate(tokens)
                if (
                    token["start"] < match.end()
                    and token["end"] > match.start()
                )
            ]
            if not token_indices:
                continue
            start_token, end_token = token_indices[0], token_indices[-1]
            values = (
                f"{year:04d}-{month:02d}-{day:02d}",
                f"{month:02d}-{day:02d}-{year:04d}",
                f"{day:02d}-{month:02d}-{year:04d}",
            )
            for value in values:
                score = _candidate_score(
                    start_token=start_token,
                    end_token=end_token,
                    value=value,
                    transform="normalize_iso_date_or_time",
                    tokens=tokens,
                    anchors=anchors,
                    anchor_positions=positions,
                    type_bonus=18.0,
                )
                _append(
                    rows,
                    seen,
                    value=value,
                    start=match.start(),
                    end=match.end(),
                    start_token=start_token,
                    end_token=end_token,
                    source_text=match.group(0),
                    transform="normalize_iso_date_or_time",
                    schema_type="string",
                    score=score,
                )
    return rows


def _word_numeric_candidates(
    text: str,
    tokens: list[dict[str, Any]],
    slot: dict[str, Any],
    *,
    integer: bool,
) -> list[dict[str, Any]]:
    anchors = _anchor_terms(slot)
    positions = _anchor_positions(tokens, anchors)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, str]] = set()
    for match in _NUMBER_WORD_RE.finditer(text):
        value: Any = _NUMBER_WORDS[match.group(0).casefold()]
        if not integer:
            value = float(value)
        token_indices = [
            index
            for index, token in enumerate(tokens)
            if token["start"] < match.end() and token["end"] > match.start()
        ]
        if not token_indices:
            continue
        start_token, end_token = token_indices[0], token_indices[-1]
        score = _candidate_score(
            start_token=start_token,
            end_token=end_token,
            value=value,
            transform="parse_number_word",
            tokens=tokens,
            anchors=anchors,
            anchor_positions=positions,
            type_bonus=19.0,
        )
        _append(
            rows,
            seen,
            value=value,
            start=match.start(),
            end=match.end(),
            start_token=start_token,
            end_token=end_token,
            source_text=match.group(0),
            transform="parse_number_word",
            schema_type="integer" if integer else "number",
            score=score,
        )
    return rows


def _numeric_candidates(
    text: str,
    tokens: list[dict[str, Any]],
    slot: dict[str, Any],
    *,
    integer: bool,
) -> list[dict[str, Any]]:
    anchors = _anchor_terms(slot)
    positions = _anchor_positions(tokens, anchors)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, str]] = set()
    for match in _NUMBER_RE.finditer(text):
        number_text = match.group("number")
        unit_text = (match.group("unit") or "").strip()
        decimal_value = Decimal(number_text.replace(",", ""))
        base_numeric = float(decimal_value)
        numeric: Any = base_numeric
        if integer and decimal_value == decimal_value.to_integral_value():
            numeric = int(decimal_value)
        elif integer:
            continue
        number_start = match.start("number")
        number_end = match.end("number")
        token_indices = [
            index
            for index, token in enumerate(tokens)
            if (
                token["start"] < number_end
                and token["end"] > number_start
            )
        ]
        if not token_indices:
            continue
        start_token, end_token = token_indices[0], token_indices[-1]
        score = _candidate_score(
            start_token=start_token,
            end_token=end_token,
            value=numeric,
            transform="parse_integer_or_decimal",
            tokens=tokens,
            anchors=anchors,
            anchor_positions=positions,
            type_bonus=20.0,
        )
        _append(
            rows,
            seen,
            value=numeric,
            start=number_start,
            end=number_end,
            start_token=start_token,
            end_token=end_token,
            source_text=text[number_start:number_end],
            transform="parse_integer_or_decimal",
            schema_type="integer" if integer else "number",
            score=score,
        )
        if integer or not unit_text:
            continue
        transform = None
        factor = None
        unit_key = unit_text.casefold()
        if unit_key in _UNIT_FACTORS:
            transform = "normalize_si_unit"
            factor = _UNIT_FACTORS[unit_key]
        elif unit_key == "%":
            transform = "normalize_percent_fraction"
            factor = "0.01"
        elif unit_key in _MAGNITUDE_FACTORS:
            transform = "normalize_magnitude_suffix"
            factor = _MAGNITUDE_FACTORS[unit_key]
        if transform is None or factor is None:
            continue
        normalized = float(decimal_value * Decimal(factor))
        unit_end = match.end("unit")
        unit_token_indices = [
            index
            for index, token in enumerate(tokens)
            if token["start"] < unit_end and token["end"] > number_start
        ]
        normalized_start_token = (
            unit_token_indices[0] if unit_token_indices else start_token
        )
        normalized_end_token = (
            unit_token_indices[-1] if unit_token_indices else end_token
        )
        normalized_score = _candidate_score(
            start_token=normalized_start_token,
            end_token=normalized_end_token,
            value=normalized,
            transform=transform,
            tokens=tokens,
            anchors=anchors,
            anchor_positions=positions,
            type_bonus=21.0,
        )
        _append(
            rows,
            seen,
            value=normalized,
            start=number_start,
            end=unit_end,
            start_token=normalized_start_token,
            end_token=normalized_end_token,
            source_text=text[number_start:unit_end],
            transform=transform,
            schema_type="number",
            score=normalized_score,
        )
    rows.extend(
        _word_numeric_candidates(
            text,
            tokens,
            slot,
            integer=integer,
        )
    )
    return rows


def _bounded_candidates(
    rows: list[dict[str, Any]],
    max_candidates: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["score"]),
            int(row["source_span"][0]),
            int(row["source_span"][1]),
            json.dumps(row["value"], sort_keys=True, ensure_ascii=True),
        ),
    )
    unique: list[dict[str, Any]] = []
    seen_values: set[str] = set()
    for row in ordered:
        value_key = json.dumps(
            row["value"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        if value_key in seen_values:
            continue
        seen_values.add(value_key)
        unique.append(row)

    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()

    def take(source: list[dict[str, Any]], predicate, limit: int) -> None:
        for row in source:
            if len(selected) >= max_candidates or limit <= 0:
                return
            if id(row) in selected_ids or not predicate(row):
                continue
            selected.append(row)
            selected_ids.add(id(row))
            limit -= 1

    transformed = sorted(
        unique,
        key=lambda row: (
            int(row["end_token"]) - int(row["start_token"]) + 1,
            -float(row["score"]),
            int(row["source_span"][0]),
            json.dumps(row["value"], sort_keys=True, ensure_ascii=True),
        ),
    )
    take(
        transformed,
        lambda row: row["transform"] != "identity",
        min(8, max_candidates),
    )
    take(
        unique,
        lambda row: int(row["end_token"]) == int(row["start_token"]),
        min(24, max_candidates),
    )
    take(
        unique,
        lambda row: (
            2
            <= int(row["end_token"]) - int(row["start_token"]) + 1
            <= 4
        ),
        min(12, max_candidates),
    )
    take(unique, lambda row: True, max_candidates)
    rank = {id(row): index for index, row in enumerate(ordered)}
    return sorted(selected, key=lambda row: rank[id(row)])


def _is_explicit_list_gap(gap: str) -> bool:
    if _EXPLICIT_LIST_GAP_RE.fullmatch(gap):
        return True
    compact = gap.strip()
    return (
        len(compact) <= 64
        and not re.search(r"[.!?]", compact)
        and bool(re.search(r"\b(?:and|or|to)\b", compact, re.IGNORECASE))
    )


def _array_candidates(
    text: str,
    item_rows: list[dict[str, Any]],
    *,
    max_items: int = 16,
) -> list[dict[str, Any]]:
    rows = [
        {
            **row,
            "value": [row["value"]],
            "schema_type": "array",
            "transform": "split_explicit_list",
            "score": float(row["score"]) - 0.25,
        }
        for row in item_rows
    ]
    def atomic_item(row: dict[str, Any]) -> bool:
        if row.get("schema_type") != "string":
            return True
        terms = re.findall(
            r"[a-z0-9]+",
            str(row.get("source_text", "")).casefold(),
        )
        return bool(
            terms
            and terms[0] not in {"and", "or", "to"}
            and terms[-1] not in {"and", "or", "to"}
        )

    atomic = sorted(
        (
            row
            for row in _bounded_candidates(item_rows, 32)
            if atomic_item(row)
        ),
        key=lambda row: (
            int(row["source_span"][0]),
            int(row["source_span"][1]),
            -float(row["score"]),
        ),
    )
    emitted: set[tuple[str, tuple[tuple[int, int], ...]]] = set()

    def emit(group: list[dict[str, Any]]) -> None:
        value = [row["value"] for row in group]
        spans = tuple(
            (int(row["source_span"][0]), int(row["source_span"][1]))
            for row in group
        )
        key = (
            json.dumps(value, sort_keys=True, ensure_ascii=True),
            spans,
        )
        if key in emitted:
            return
        emitted.add(key)
        start = spans[0][0]
        end = spans[-1][1]
        rows.append(
            {
                "value": value,
                "source_span": [start, end],
                "source_text": text[start:end],
                "source_kind": "extractive_user_span",
                "transform": "split_explicit_list",
                "schema_type": "array",
                "score": (
                    max(float(row["score"]) for row in group)
                    + 2.0 * len(group)
                ),
                "start_token": min(
                    int(row["start_token"]) for row in group
                ),
                "end_token": max(
                    int(row["end_token"]) for row in group
                ),
                "component_spans": [list(span) for span in spans],
            }
        )

    def extend(group: list[dict[str, Any]], last_index: int) -> None:
        if len(group) >= max_items:
            return
        current_end = int(group[-1]["source_span"][1])
        eligible: list[tuple[int, dict[str, Any]]] = []
        for index in range(last_index + 1, len(atomic)):
            candidate = atomic[index]
            candidate_start = int(candidate["source_span"][0])
            if candidate_start < current_end:
                continue
            gap = text[current_end:candidate_start]
            if _is_explicit_list_gap(gap):
                eligible.append((index, candidate))
        if not eligible:
            return
        next_start = min(
            int(candidate["source_span"][0])
            for _, candidate in eligible
        )
        for index, candidate in eligible:
            if int(candidate["source_span"][0]) != next_start:
                continue
            new_group = [*group, candidate]
            emit(new_group)
            extend(new_group, index)

    for index, row in enumerate(atomic):
        extend([row], index)
    return rows


def _boolean_candidates(
    text: str,
    tokens: list[dict[str, Any]],
    slot: dict[str, Any],
) -> list[dict[str, Any]]:
    anchors = _anchor_terms(slot)
    positions = _anchor_positions(tokens, anchors)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, str]] = set()
    lowered = text.casefold()
    for position in positions:
        window_start = max(0, position - 4)
        window_end = min(len(tokens) - 1, position + 5)
        start = tokens[window_start]["start"]
        end = tokens[window_end]["end"]
        phrase = text[start:end]
        before = lowered[max(0, tokens[position]["start"] - 32) : tokens[position]["start"]]
        negative = bool(re.search(r"\b(?:no|not|without|doesn.?t|don.?t|false)\b", before))
        value = not negative
        score = _candidate_score(
            start_token=window_start,
            end_token=window_end,
            value=value,
            transform="explicit_boolean_polarity",
            tokens=tokens,
            anchors=anchors,
            anchor_positions=positions,
            type_bonus=24.0,
        )
        _append(
            rows,
            seen,
            value=value,
            start=start,
            end=end,
            start_token=window_start,
            end_token=window_end,
            source_text=phrase,
            transform="explicit_boolean_polarity",
            schema_type="boolean",
            score=score,
        )
    for match in re.finditer(r"\b(?:true|false|yes|no)\b", lowered):
        value = match.group(0) in {"true", "yes"}
        token_indices = [
            index
            for index, token in enumerate(tokens)
            if token["start"] < match.end() and token["end"] > match.start()
        ]
        if not token_indices:
            continue
        index = token_indices[0]
        score = _candidate_score(
            start_token=index,
            end_token=index,
            value=value,
            transform="explicit_boolean_polarity",
            tokens=tokens,
            anchors=anchors,
            anchor_positions=positions,
            type_bonus=25.0,
        )
        _append(
            rows,
            seen,
            value=value,
            start=match.start(),
            end=match.end(),
            start_token=index,
            end_token=index,
            source_text=text[match.start() : match.end()],
            transform="explicit_boolean_polarity",
            schema_type="boolean",
            score=score,
        )
    return rows


def _enum_candidates(
    text: str,
    tokens: list[dict[str, Any]],
    slot: dict[str, Any],
    enum: list[Any],
) -> list[dict[str, Any]]:
    anchors = _anchor_terms(slot)
    positions = _anchor_positions(tokens, anchors)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, str]] = set()
    lowered = text.casefold()
    for value in enum:
        pieces = [
            piece
            for piece in re.split(r"[\s_-]+", str(value).casefold())
            if piece
        ]
        if not pieces:
            continue
        needle_pattern = r"[\s_-]+".join(
            re.escape(piece) for piece in pieces
        )
        pattern = rf"(?<!\w){needle_pattern}(?!\w)"
        for match in re.finditer(pattern, lowered):
            token_indices = [
                index
                for index, token in enumerate(tokens)
                if token["start"] < match.end() and token["end"] > match.start()
            ]
            if not token_indices:
                continue
            start_token, end_token = token_indices[0], token_indices[-1]
            score = _candidate_score(
                start_token=start_token,
                end_token=end_token,
                value=value,
                transform="casefold_for_enum_comparison",
                tokens=tokens,
                anchors=anchors,
                anchor_positions=positions,
                type_bonus=30.0,
            )
            _append(
                rows,
                seen,
                value=value,
                start=match.start(),
                end=match.end(),
                start_token=start_token,
                end_token=end_token,
                source_text=text[match.start() : match.end()],
                transform="casefold_for_enum_comparison",
                schema_type=str(slot["schema"].get("type", "string")),
                score=score,
            )
    return rows


def enumerate_slot_candidates(
    text: str,
    slot: dict[str, Any],
    *,
    max_span_tokens: int = DEFAULT_MAX_SPAN_TOKENS,
    max_candidates: int = DEFAULT_MAX_CANDIDATES_PER_SLOT,
) -> list[dict[str, Any]]:
    tokens = _tokens(text)
    schema = slot["schema"]
    schema_type = str(schema.get("type") or "any").casefold()
    enum = schema.get("enum") if isinstance(schema.get("enum"), list) else []
    rows: list[dict[str, Any]] = []
    if enum:
        rows.extend(_enum_candidates(text, tokens, slot, enum))
        selected = _bounded_candidates(rows, max_candidates)
        for candidate_id, row in enumerate(selected):
            row["candidate_id"] = candidate_id
            row.pop("start_token", None)
            row.pop("end_token", None)
        return selected
    if schema_type == "integer":
        rows.extend(_numeric_candidates(text, tokens, slot, integer=True))
    elif schema_type == "number":
        rows.extend(_numeric_candidates(text, tokens, slot, integer=False))
    elif schema_type == "boolean":
        rows.extend(_boolean_candidates(text, tokens, slot))
    elif schema_type == "string":
        rows.extend(
            _string_candidates(
                text,
                tokens,
                slot,
                max_span_tokens=max_span_tokens,
            )
        )
        rows.extend(_date_candidates(text, tokens, slot))
    elif schema_type in {"array", "tuple"}:
        item_schema = schema.get("items", {})
        if isinstance(item_schema, list):
            item_schema = (
                item_schema[0] if item_schema else {"type": "string"}
            )
        if not isinstance(item_schema, dict):
            item_schema = {"type": "string"}
        item_slot = {**slot, "schema": item_schema}
        item_type = str(item_schema.get("type") or "any").casefold()
        item_enum = (
            item_schema.get("enum")
            if isinstance(item_schema.get("enum"), list)
            else []
        )
        item_rows: list[dict[str, Any]] = []
        if item_enum:
            item_rows.extend(
                _enum_candidates(text, tokens, item_slot, item_enum)
            )
        elif item_type in {"integer", "number"}:
            item_rows.extend(
                _numeric_candidates(
                    text,
                    tokens,
                    item_slot,
                    integer=item_type == "integer",
                )
            )
        else:
            item_rows.extend(
                _string_candidates(
                    text,
                    tokens,
                    item_slot,
                    max_span_tokens=max_span_tokens,
                )
            )
            if item_type in {"any", ""}:
                item_rows.extend(
                    _numeric_candidates(
                        text,
                        tokens,
                        item_slot,
                        integer=False,
                    )
                )
                item_rows.extend(_boolean_candidates(text, tokens, item_slot))
        rows.extend(_array_candidates(text, item_rows))
    elif schema_type in {"any", ""}:
        rows.extend(
            _string_candidates(
                text,
                tokens,
                slot,
                max_span_tokens=max_span_tokens,
            )
        )
        rows.extend(_date_candidates(text, tokens, slot))
        rows.extend(_numeric_candidates(text, tokens, slot, integer=False))
        rows.extend(_boolean_candidates(text, tokens, slot))
    selected = _bounded_candidates(rows, max_candidates)
    for candidate_id, row in enumerate(selected):
        row["candidate_id"] = candidate_id
        row.pop("start_token", None)
        row.pop("end_token", None)
    return selected


def build_extractive_candidate_table(
    messages: list[dict[str, Any]],
    tool: dict[str, Any],
    *,
    max_span_tokens: int = DEFAULT_MAX_SPAN_TOKENS,
    max_candidates_per_slot: int = DEFAULT_MAX_CANDIDATES_PER_SLOT,
    max_candidates_per_action: int = DEFAULT_MAX_CANDIDATES_PER_ACTION,
    include_optional: bool = False,
) -> dict[str, Any]:
    text = user_request_text(messages)
    canonical = canonical_slots(tool)
    required_slots = [row for row in canonical if row["required"]]
    optional_slots = [row for row in canonical if not row["required"]]
    slots = [
        *required_slots,
        *(optional_slots if include_optional else []),
    ]
    table: dict[str, list[dict[str, Any]]] = {}
    remaining = max_candidates_per_action
    for slot in slots:
        candidates = enumerate_slot_candidates(
            text,
            slot,
            max_span_tokens=max_span_tokens,
            max_candidates=min(max_candidates_per_slot, remaining),
        )
        table[str(slot["name"])] = candidates
        remaining -= len(candidates)
        if remaining <= 0:
            for pending in slots[len(table) :]:
                table[str(pending["name"])] = []
            break
    return {
        "schema_version": EXTRACTIVE_CANDIDATE_VERSION,
        "request_text": text,
        "tool": str(tool.get("canonical_name") or tool.get("name")),
        "slots": table,
        "required_slots": [
            str(row["name"]) for row in required_slots
        ],
        "optional_slots": [
            str(row["name"]) for row in optional_slots
        ] if include_optional else [],
        "candidate_count": sum(len(rows) for rows in table.values()),
        "limits": {
            "max_span_tokens": max_span_tokens,
            "max_candidates_per_slot": max_candidates_per_slot,
            "max_candidates_per_action": max_candidates_per_action,
        },
    }


def candidate_value_recall(
    table: dict[str, Any],
    required_values: dict[str, Any],
) -> dict[str, Any]:
    misses = []
    ranks: dict[str, int] = {}
    for slot, value in required_values.items():
        candidates = table.get("slots", {}).get(slot, [])
        rank = next(
            (
                index + 1
                for index, row in enumerate(candidates)
                if row.get("value") == value
            ),
            None,
        )
        if rank is None:
            misses.append(slot)
        else:
            ranks[slot] = rank
    return {
        "slot_count": len(required_values),
        "recalled_slots": len(required_values) - len(misses),
        "all_required_recalled": not misses,
        "missing_slots": misses,
        "ranks": ranks,
    }
