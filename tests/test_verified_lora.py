from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

from scripts.train_verified_lora import _validate_row
from tapbench.verified_distillation import VERIFIED_DISTILLATION_EXPORT_VERSION


def _row() -> dict:
    return {
        "schema_version": VERIFIED_DISTILLATION_EXPORT_VERSION,
        "example_id": "a" * 64,
        "family": "example",
        "messages": [{"role": "user", "content": "Do the thing."}],
        "tools": [],
        "target": json.dumps({"mode": "no_tool", "payload": {}}),
    }


def test_lora_row_accepts_only_public_training_surface() -> None:
    _validate_row(_row())


def test_lora_row_rejects_hidden_metadata() -> None:
    row = _row()
    row["gold_action"] = {"mode": "call"}
    with pytest.raises(ValueError, match="fields drifted"):
        _validate_row(row)


def test_lora_row_rejects_invalid_action_mode() -> None:
    row = _row()
    row["target"] = json.dumps({"mode": "think", "payload": {}})
    with pytest.raises(ValueError, match="invalid public Action IR"):
        _validate_row(row)
