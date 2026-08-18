from __future__ import annotations

from tapbench.supervised_slot_selector import annotation_slots, select_slots


def test_annotation_slot_parser_handles_multiple_values() -> None:
    assert annotation_slots("email [person : Ada] at [time : noon]") == {"person", "time"}


def test_weighted_neighbor_vote_and_required_slot_policy() -> None:
    neighbors = [
        {"score": 0.8, "slots": {"person"}},
        {"score": 0.2, "slots": {"time"}},
    ]
    selected = select_slots(
        neighbors,
        valid_slots={"person", "time", "message"},
        required_slots={"message"},
        k=2,
        vote_threshold=0.5,
    )
    assert selected == ["message", "person"]


def test_zero_similarity_neighbors_fall_back_to_equal_vote() -> None:
    neighbors = [
        {"score": 0.0, "slots": {"person"}},
        {"score": 0.0, "slots": {"person", "time"}},
    ]
    selected = select_slots(
        neighbors,
        valid_slots={"person", "time"},
        required_slots=set(),
        k=2,
        vote_threshold=0.5,
    )
    assert selected == ["person", "time"]
