from __future__ import annotations

import json

from tapbench.io import read_jsonl, write_jsonl
from tapbench.rotbench import convert_rotbench, score_rotbench


def _record() -> dict:
    tools = [
        {
            "name": "weather",
            "description": "Get weather.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
        {
            "name": "ask_to_user",
            "description": "Ask.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
        {
            "name": "finish",
            "description": "Finish.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
    ]
    return {
        "id": "Turn 1: Weather?",
        "scenario": "RS",
        "conversations": [
            {
                "from": "system",
                "value": "Tools:\n" + json.dumps(tools) + "\n\nLet's Begin!",
            },
            {"from": "user", "value": "Weather in London?\nBegin!\n"},
            {
                "from": "assistant",
                "value": [
                    'Thought:\nAction: weather\nAction Input: {"city": "London"}'
                ],
            },
        ],
    }


def test_rotbench_conversion_and_scoring(tmp_path) -> None:
    source = tmp_path / "source" / "Data" / "First_Turn"
    source.mkdir(parents=True)
    (source / "clean.json").write_text(json.dumps([_record()]))
    cases_path = tmp_path / "cases.jsonl"
    assert convert_rotbench(
        tmp_path / "source",
        cases_path,
        environments=["clean"],
    ) == 1
    case = read_jsonl(cases_path)[0]
    assert [tool["name"] for tool in case["tools"]] == ["weather"]
    predictions_path = tmp_path / "predictions.jsonl"
    write_jsonl(predictions_path, [{
        "case_id": case["case_id"],
        "model_id": "m",
        "method": "tap_r_full",
        "seed": 1,
        "prediction": case["gold_action"],
        "backend": "llama.cpp",
        "quantization": "Q4_K_M",
        "model_artifact": "m.gguf",
        "chat_template": "t",
        "grammar_engine": "gbnf",
        "thinking_mode": "off",
        "thinking_marker_detected": False,
        "runner_error": None,
    }])
    report = score_rotbench(
        cases_path,
        predictions_path,
        tmp_path / "scores.jsonl",
        tmp_path / "report.json",
    )
    assert report["groups"][0]["content_filling"] == 1.0
