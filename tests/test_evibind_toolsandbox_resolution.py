from __future__ import annotations

from scripts.resolve_evibind_toolsandbox import (
    resolve_study,
    validate_diagnostic_rows,
)


def _row(model: str, scenario: str, condition: str) -> dict:
    return {
        "model_id": model,
        "scenario": scenario,
        "condition": condition,
        "family": "family",
        "first_request_sha256": f"{model}:{scenario}",
        "runner_error": None,
        "thinking_marker_detected": False,
        "length_stops": 0,
    }


def _policy() -> dict:
    return {
        "study_id": "study",
        "expected_scenarios_per_model": 2,
        "resolved_model": {
            "id": "diagnostic",
            "required_findings": {
                "minimum_thinking_markers": 1,
                "minimum_length_stops": 1,
            },
        },
        "confirmatory_models": ["clean-a", "clean-b"],
        "diagnostic_models": ["diagnostic"],
        "claim_boundary": ["diagnostic excluded"],
        "queue_outcome": "completed_with_diagnostic_exclusion",
    }


def _model_rows(model: str) -> list[dict]:
    return [
        _row(model, scenario, condition)
        for scenario in ("s1", "s2")
        for condition in ("native", "evibind")
    ]


def test_validate_diagnostic_rows_accepts_only_documented_failures() -> None:
    rows = _model_rows("diagnostic")
    rows[0]["thinking_marker_detected"] = True
    rows[1]["length_stops"] = 1

    resolution = validate_diagnostic_rows(rows, _policy())

    assert resolution["passed"]
    assert resolution["audit"]["runner_errors"] == 0
    assert resolution["audit"]["thinking_markers"] == 1
    assert resolution["audit"]["length_stops"] == 1


def test_resolve_study_keeps_only_clean_models_confirmatory() -> None:
    rows = _model_rows("clean-a") + _model_rows("clean-b")
    diagnostic = _model_rows("diagnostic")
    diagnostic[0]["thinking_marker_detected"] = True
    diagnostic[1]["length_stops"] = 1
    rows.extend(diagnostic)

    resolution, confirmatory_rows = resolve_study(rows, _policy())

    assert resolution["passed"]
    assert {row["model_id"] for row in confirmatory_rows} == {
        "clean-a",
        "clean-b",
    }


def test_resolve_study_rejects_a_dirty_confirmatory_model() -> None:
    rows = _model_rows("clean-a") + _model_rows("clean-b")
    rows[0]["thinking_marker_detected"] = True
    diagnostic = _model_rows("diagnostic")
    diagnostic[0]["thinking_marker_detected"] = True
    diagnostic[1]["length_stops"] = 1
    rows.extend(diagnostic)

    resolution, _ = resolve_study(rows, _policy())

    assert not resolution["passed"]
    assert not resolution["gates"]["all_confirmatory_models_clean"]
