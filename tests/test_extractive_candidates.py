from __future__ import annotations

from tapbench.extractive_candidates import (
    build_extractive_candidate_table,
    candidate_value_recall,
)


def _tool(properties, required):
    return {
        "name": "example",
        "canonical_name": "example",
        "description": "Example tool",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


def test_extracts_typed_values_with_certified_spans() -> None:
    messages = [
        {
            "role": "user",
            "content": (
                "Find Italian restaurants in New York City rated above 4 "
                "that accept credit cards."
            ),
        }
    ]
    tool = _tool(
        {
            "location": {"type": "string", "description": "city and state"},
            "cuisine": {"type": "string", "description": "type of cuisine"},
            "rating": {"type": "integer", "description": "minimum rating"},
            "accepts_credit_cards": {
                "type": "boolean",
                "description": "accepts credit cards",
            },
        },
        ["location", "cuisine", "rating", "accepts_credit_cards"],
    )
    table = build_extractive_candidate_table(messages, tool)
    values = {
        slot: {str(row["value"]) for row in rows}
        for slot, rows in table["slots"].items()
    }
    assert "New York City" in values["location"]
    assert "Italian" in values["cuisine"]
    assert "4" in values["rating"]
    assert "True" in values["accepts_credit_cards"]
    for rows in table["slots"].values():
        assert all(row["source_span"][0] < row["source_span"][1] for row in rows)


def test_expression_transform_is_span_bounded() -> None:
    messages = [
        {
            "role": "user",
            "content": "Calculate the derivative of the function 3x^2 + 2x - 1.",
        }
    ]
    tool = _tool(
        {"function": {"type": "string", "description": "function expression"}},
        ["function"],
    )
    table = build_extractive_candidate_table(messages, tool)
    match = next(
        row
        for row in table["slots"]["function"]
        if row["value"] == "3x**2 + 2x - 1"
    )
    assert match["transform"] == "expression_operator_canonicalization"
    assert match["source_text"] == "3x^2 + 2x - 1"


def test_candidate_recall_reports_missing_without_invention() -> None:
    messages = [{"role": "user", "content": "Book the table tomorrow."}]
    tool = _tool(
        {
            "date": {"type": "string", "description": "booking date"},
            "party_size": {"type": "integer", "description": "number of diners"},
        },
        ["date", "party_size"],
    )
    table = build_extractive_candidate_table(messages, tool)
    recall = candidate_value_recall(
        table, {"date": "tomorrow", "party_size": 4}
    )
    assert recall["missing_slots"] == ["party_size"]
    assert not recall["all_required_recalled"]


def test_scientific_and_si_unit_candidates_are_span_certified() -> None:
    messages = [
        {
            "role": "user",
            "content": (
                "Use a 100uF capacitor, a 50mH inductor, and set the "
                "threshold to 1.25e-3."
            ),
        }
    ]
    tool = _tool(
        {
            "capacitance": {
                "type": "number",
                "description": "capacitor value in farads",
            },
            "inductance": {
                "type": "number",
                "description": "inductor value in henries",
            },
            "threshold": {"type": "number", "description": "threshold"},
        },
        ["capacitance", "inductance", "threshold"],
    )
    table = build_extractive_candidate_table(messages, tool)
    values = {
        slot: [row["value"] for row in rows]
        for slot, rows in table["slots"].items()
    }
    assert any(abs(value - 1e-4) < 1e-12 for value in values["capacitance"])
    assert any(abs(value - 5e-2) < 1e-12 for value in values["inductance"])
    assert any(abs(value - 1.25e-3) < 1e-12 for value in values["threshold"])
    normalized = [
        row
        for rows in table["slots"].values()
        for row in rows
        if row["transform"] == "normalize_si_unit"
    ]
    assert normalized
    assert all(row["component_spans"] for row in normalized)


def test_explicit_list_requires_and_preserves_component_spans() -> None:
    messages = [
        {
            "role": "user",
            "content": "Use the colors red, green, and blue.",
        }
    ]
    tool = _tool(
        {
            "colors": {
                "type": "array",
                "description": "ordered colors",
                "items": {"type": "string"},
            }
        },
        ["colors"],
    )
    table = build_extractive_candidate_table(messages, tool)
    match = next(
        row
        for row in table["slots"]["colors"]
        if row["value"] == ["red", "green", "blue"]
    )
    assert match["transform"] == "split_explicit_list"
    assert len(match["component_spans"]) == 3
    assert all(start < end for start, end in match["component_spans"])


def test_candidate_cap_preserves_lexical_diversity() -> None:
    messages = [
        {
            "role": "user",
            "content": (
                "Compare steam, vapor, frost, snow, rain, hail, mist, cloud, "
                "river, lake, ocean, ice, and water."
            ),
        }
    ]
    tool = _tool(
        {"substance": {"type": "string", "description": "substance"}},
        ["substance"],
    )
    table = build_extractive_candidate_table(messages, tool)
    values = {row["value"] for row in table["slots"]["substance"]}
    assert "ice" in values
    assert "water" in values


def test_percent_magnitude_number_word_and_attached_unit_candidates() -> None:
    messages = [
        {
            "role": "user",
            "content": (
                "Use 213 million people, a 5% rate, a 298K temperature, "
                "and return the top three results."
            ),
        }
    ]
    tool = _tool(
        {
            "population": {"type": "number", "description": "population"},
            "rate": {"type": "number", "description": "rate as a fraction"},
            "temperature": {
                "type": "integer",
                "description": "temperature magnitude",
            },
            "count": {"type": "integer", "description": "result count"},
        },
        ["population", "rate", "temperature", "count"],
    )
    table = build_extractive_candidate_table(messages, tool)
    values = {
        slot: {row["value"] for row in rows}
        for slot, rows in table["slots"].items()
    }
    assert 213_000_000.0 in values["population"]
    assert 0.05 in values["rate"]
    assert 298 in values["temperature"]
    assert 3 in values["count"]


def test_long_explicit_numeric_list_is_not_truncated() -> None:
    messages = [
        {
            "role": "user",
            "content": "Use the values 85, 90, 88, 92, 86, 89, 91.",
        }
    ]
    tool = _tool(
        {
            "values": {
                "type": "array",
                "description": "ordered values",
                "items": {"type": "integer"},
            }
        },
        ["values"],
    )
    table = build_extractive_candidate_table(messages, tool)
    expected = [85, 90, 88, 92, 86, 89, 91]
    match = next(
        row for row in table["slots"]["values"] if row["value"] == expected
    )
    assert len(match["component_spans"]) == len(expected)


def test_date_and_enum_surface_normalization_remain_span_bounded() -> None:
    messages = [
        {
            "role": "user",
            "content": "Book without joker on March 11th, 2022.",
        }
    ]
    tool = _tool(
        {
            "deck": {
                "type": "string",
                "enum": ["without_joker", "with_joker"],
            },
            "date": {"type": "string", "description": "ISO date"},
        },
        ["deck", "date"],
    )
    table = build_extractive_candidate_table(messages, tool)
    deck = next(
        row
        for row in table["slots"]["deck"]
        if row["value"] == "without_joker"
    )
    date = next(
        row
        for row in table["slots"]["date"]
        if row["value"] == "2022-03-11"
    )
    assert deck["transform"] == "casefold_for_enum_comparison"
    assert date["transform"] == "normalize_iso_date_or_time"
    assert deck["source_text"] == "without joker"
    assert date["source_text"] == "March 11th, 2022"
