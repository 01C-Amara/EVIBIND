from scripts.analyze_verified_ranker_ablation import _auroc, calibration_report, mask_features
from tapbench.verified_ranker import FEATURE_NAMES


class _FirstFeatureRanker:
    def score(self, values):
        return float(values[0])


def test_mask_features_zeros_only_named_features() -> None:
    row = [float(index + 1) for index in range(len(FEATURE_NAMES))]
    masked = mask_features([(row, True)], {"cue_assignment", "candidate_position"})[0][0]
    assert masked[FEATURE_NAMES.index("cue_assignment")] == 0.0
    assert masked[FEATURE_NAMES.index("candidate_position")] == 0.0
    assert masked[FEATURE_NAMES.index("cue_anywhere")] == row[FEATURE_NAMES.index("cue_anywhere")]


def test_calibration_and_auroc_are_exact_for_perfect_scores() -> None:
    examples = [([0.0], False), ([0.0], False), ([1.0], True), ([1.0], True)]
    report = calibration_report(examples, _FirstFeatureRanker(), bins=10)
    assert report["brier_score"] == 0.0
    assert report["expected_calibration_error_10_bin"] == 0.0
    assert report["auroc"] == 1.0
    assert _auroc([0.5, 0.5], [False, True]) == 0.5
