"""The exact McNemar test behind the AgentDojo paired claims."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "agentdojo"))

from paired_stats import mcnemar  # noqa: E402


def test_reproduces_the_frozen_replication() -> None:
    # six improvements, no regressions: the frozen analysis reports 0.03125
    assert mcnemar(6, 0) == 0.03125


def test_the_corrected_rerun() -> None:
    assert mcnemar(5, 0) == 0.0625


def test_balanced_discordance_is_uninformative() -> None:
    assert mcnemar(7, 8) == 1.0
    assert mcnemar(0, 0) == 1.0
