from __future__ import annotations

import numpy as np

from tapbench.supervised_intent_router import hashed_counts, intent_to_tool, rank_tools, train_router


def test_intent_mapping_changes_only_first_separator() -> None:
    assert intent_to_tool("email_sendemail") == "email.sendemail"
    assert intent_to_tool("iot_hue_lightoff") == "iot.hue_lightoff"


def test_hashed_features_are_deterministic_and_unicode_safe() -> None:
    assert hashed_counts("ایمیل من", 1024) == hashed_counts("ایمیل   من", 1024)
    assert hashed_counts("ＡＢＣ", 1024) == hashed_counts("abc", 1024)


def test_router_learns_separable_tiny_intents() -> None:
    rows = [
        {"text": "send an email", "tool": "email.sendemail"},
        {"text": "email Ada now", "tool": "email.sendemail"},
        {"text": "turn the lights off", "tool": "iot.hue_lightoff"},
        {"text": "switch off the lamp", "tool": "iot.hue_lightoff"},
    ]
    model = train_router(rows, dimensions=2048)
    ranking = rank_tools(model, "please send email", k=2)
    assert ranking[0]["tool"] == "email.sendemail"
    assert np.isfinite(ranking[0]["cosine_score"])
