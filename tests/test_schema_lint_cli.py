from __future__ import annotations

import json

from tapbench.product_cli import main


def test_lint_schema_cli_reports_contract_digest(tmp_path, capsys) -> None:
    request = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    assert main(["lint-schema", "--request", str(path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["valid"]
    assert report["tool_count"] == 1
    assert len(report["provider_schema_sha256"]) == 64


def test_lint_schema_cli_strict_fails_on_warning(tmp_path) -> None:
    request = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "pay",
                    "parameters": {
                        "type": "object",
                        "properties": {"amount": {"type": "number"}},
                        "required": ["amount"],
                    },
                },
            }
        ]
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    assert main(["lint-schema", "--strict", "--request", str(path)]) == 1
