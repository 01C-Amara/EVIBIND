from __future__ import annotations

from scripts.resolve_evibind_toolsandbox_v2 import resolve_study_v2


def _rows(model: str) -> list[dict]:
    return [
        {
            "model_id": model,
            "scenario": scenario,
            "condition": condition,
            "first_request_sha256": f"{model}:{scenario}",
            "runner_error": None,
            "thinking_marker_detected": False,
            "length_stops": 0,
        }
        for scenario in ("one", "two")
        for condition in ("native", "evibind")
    ]


def _policy() -> dict:
    return {
        "study_id": "study",
        "expected_scenarios_per_model": 2,
        "confirmatory_models": ["clean-a", "clean-b"],
        "diagnostic_models": [
            {
                "id": "thinking",
                "disposition": "diagnostic_only",
                "reason": "thinking",
                "exact_integrity_findings": {
                    "thinking_markers": 1,
                    "length_stops": 1,
                },
            },
            {
                "id": "length",
                "disposition": "diagnostic_only",
                "reason": "length",
                "exact_integrity_findings": {
                    "thinking_markers": 0,
                    "length_stops": 2,
                },
            },
        ],
        "claim_boundary": ["two clean models"],
        "queue_outcome": "completed_with_diagnostic_exclusion",
    }


def test_resolve_study_v2_accepts_two_exact_diagnostic_dispositions() -> None:
    rows = _rows("clean-a") + _rows("clean-b")
    thinking = _rows("thinking")
    thinking[0]["thinking_marker_detected"] = True
    thinking[1]["length_stops"] = 1
    length = _rows("length")
    length[0]["length_stops"] = 2
    rows.extend(thinking + length)

    resolution, confirmatory = resolve_study_v2(rows, _policy())

    assert resolution["passed"]
    assert {row["model_id"] for row in confirmatory} == {"clean-a", "clean-b"}


def test_resolve_study_v2_rejects_unlisted_integrity_findings() -> None:
    rows = _rows("clean-a") + _rows("clean-b")
    thinking = _rows("thinking")
    thinking[0]["thinking_marker_detected"] = True
    thinking[1]["length_stops"] = 1
    length = _rows("length")
    length[0]["length_stops"] = 3
    rows.extend(thinking + length)

    resolution, _ = resolve_study_v2(rows, _policy())

    assert not resolution["passed"]
    assert not resolution["diagnostic_resolutions"]["length"]["passed"]
