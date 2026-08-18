from __future__ import annotations

from tapbench.verified_ranker import (
    FEATURE_NAMES,
    LinearCandidateRanker,
    fit_linear_ranker,
)


def test_linear_ranker_learns_assignment_feature_and_round_trips() -> None:
    negative = [0.0] * len(FEATURE_NAMES)
    negative[0] = 1.0
    positive = list(negative)
    positive[1] = 1.0
    examples = [(negative, False)] * 20 + [(positive, True)] * 20

    ranker = fit_linear_ranker(examples, iterations=300)
    restored = LinearCandidateRanker.from_dict(ranker.to_dict())

    assert restored.score(positive) > restored.score(negative)
    assert restored.to_dict() == ranker.to_dict()
