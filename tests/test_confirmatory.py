from __future__ import annotations

from tapbench.confirmatory import build_family_cases, normalized_confirmatory_tools


def _tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "send_item",
                "description": "Send an item.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recipient": {"type": "string"},
                        "count": {"type": "integer"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_items",
                "description": "List items.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


def test_confirmatory_cases_are_deterministic_and_critical_only() -> None:
    tools, tool_id, properties = normalized_confirmatory_tools(_tools())
    assert tool_id == "send_item"
    assert properties == ("recipient",)
    assert tools[0]["function"]["parameters"]["required"] == ["recipient"]

    first = build_family_cases(
        family="fresh_family", tools=_tools(), cases_per_family=2
    )
    second = build_family_cases(
        family="fresh_family", tools=_tools(), cases_per_family=2
    )
    assert first == second
    assert all(row["version"] == "evibind.evibench.v1" for row in first)
    assert all(row["authoring"]["split"] == "confirmatory" for row in first)
    assert all(row["expected"]["mode"] == "call" for row in first)
    assert "earlier option" in first[0]["request"]["messages"][0]["content"]
    assert "final choice" in first[0]["request"]["messages"][0]["content"]
