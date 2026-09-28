"""The routing decision rule and the scoring it relies on."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

from run_routing import ABSTAIN, _load_responses, _route  # noqa: E402

GOOD = (("from_account", "ACC-3"), ("to_account", "ACC-7"))
SWAP = (("from_account", "ACC-7"), ("to_account", "ACC-3"))


def test_majority_is_released() -> None:
    samples = [(GOOD, "correct")] * 3 + [(SWAP, "harmful")] * 2
    assert _route(samples, 0.0) == ("correct", 0.6)


def test_tie_escalates_instead_of_breaking_by_order() -> None:
    # 2-2-1: without the tie rule this released whichever binding came first
    for order in ([GOOD, GOOD, SWAP, SWAP, ABSTAIN], [SWAP, SWAP, GOOD, GOOD, ABSTAIN]):
        samples = [(k, "x") for k in order]
        assert _route(samples, 0.0)[0] is None


def test_abstain_mode_escalates() -> None:
    samples = [(ABSTAIN, "abstain")] * 3 + [(GOOD, "correct")] * 2
    assert _route(samples, 0.0)[0] is None


def test_threshold_escalates_weak_agreement() -> None:
    samples = [(GOOD, "correct")] * 3 + [(SWAP, "harmful")] + [(ABSTAIN, "abstain")]
    assert _route(samples, 0.8)[0] is None
    assert _route(samples, 0.6)[0] == "correct"


def test_samples_are_ordered_by_index_not_completion(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    path.write_text('{"case_id": "c", "sample": 2, "response": null}\n'
                    '{"case_id": "c", "sample": 0, "response": null}\n'
                    '{"case_id": "c", "sample": 1, "response": null}\n', encoding="utf-8")
    assert [r["sample"] for r in _load_responses(path)["c"]] == [0, 1, 2]
