from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_minimal_evidence_binding_example_runs_without_a_provider() -> None:
    completed = subprocess.run(
        [sys.executable, "examples/minimal_evidence_binding.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    result = json.loads(completed.stdout)
    assert result["accepted_candidates"] == 1
    assert result["rejected_candidates"] == 1
    assert result["replay_matches"] is True
    assert result["released_call"]["arguments"] == {
        "attendee": "alice@example.com"
    }
