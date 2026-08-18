from __future__ import annotations

import re
from typing import Any


THINKING_MARKER_PATTERNS = (
    re.compile(r"</?think>", re.IGNORECASE),
    re.compile(r"<\|(?:analysis|thinking|reasoning)\|>", re.IGNORECASE),
    re.compile(r"\b(?:reasoning|thinking)\s*:", re.IGNORECASE),
)


def thinking_metadata(thinking_mode: str | None = None, reasoning_budget: int | None = None) -> dict[str, Any]:
    mode = thinking_mode or "not_applicable"
    return {
        "thinking_mode": mode,
        "reasoning_budget": reasoning_budget,
    }


def text_has_thinking_marker(text: str) -> bool:
    return any(pattern.search(text) for pattern in THINKING_MARKER_PATTERNS)


def prediction_text(prediction: dict[str, Any]) -> str:
    parts: list[str] = []
    metadata = prediction.get("response_metadata")
    if isinstance(metadata, dict):
        for key in ("raw_text", "content", "text"):
            value = metadata.get(key)
            if value is not None:
                parts.append(str(value))
    action = prediction.get("prediction")
    if isinstance(action, dict):
        payload = action.get("payload")
        if isinstance(payload, dict):
            raw_text = payload.get("raw_text")
            if raw_text is not None:
                parts.append(str(raw_text))
    return "\n".join(parts)


def prediction_has_thinking_marker(prediction: dict[str, Any]) -> bool:
    if bool(prediction.get("thinking_marker_detected")):
        return True
    return text_has_thinking_marker(prediction_text(prediction))
