from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from evibind.core import CandidateTable, EvidenceContext, Span, StateRef
from evibind.core.derivations import canonical_json


VERIFIED_RANKER_VERSION = "evibind.verified_candidate_ranker.v1"
FEATURE_NAMES = (
    "bias",
    "cue_assignment",
    "cue_near_before",
    "cue_anywhere",
    "inverse_cue_distance",
    "after_correction",
    "after_negation",
    "current_user_span",
    "state_reference",
    "schema_value",
    "email_type",
    "registry_type",
    "numeric_type",
    "enum_type",
    "candidate_position",
    "inverse_slot_catalog_size",
)


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _leaf_tokens(destination: str) -> list[str]:
    leaf = destination.rsplit("/", 1)[-1].replace("~1", "/").replace("~0", "~")
    return [
        token.casefold()
        for token in re.findall(r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|\d+", leaf)
        if token
    ] or [leaf.casefold()]


def _span_context(
    context: EvidenceContext,
    candidate: Any,
) -> tuple[str, str, float]:
    derivation = candidate.derivation
    if not isinstance(derivation, Span):
        return "", "", 0.0
    message = context.message(derivation.message_id)
    encoded = message.content.encode("utf-8")
    before = encoded[: derivation.byte_start].decode("utf-8", errors="ignore")
    source = encoded[derivation.byte_start : derivation.byte_end].decode(
        "utf-8", errors="ignore"
    )
    position = derivation.byte_start / max(1, len(encoded))
    return before, source, position


def candidate_features(
    context: EvidenceContext,
    candidate: Any,
    *,
    slot_catalog_size: int,
) -> list[float]:
    destination = candidate.witness.destination_scope
    tokens = _leaf_tokens(destination)
    cue = r"[\s_-]*".join(re.escape(token) for token in tokens)
    before, _source, position = _span_context(context, candidate)
    near = before[-96:]
    assignment = re.search(
        rf"(?<!\w){cue}(?!\w)(?:\s+(?:is|to))?\s*(?:=|:)\s*$",
        near,
        re.IGNORECASE,
    )
    cue_matches = list(re.finditer(cue, before, re.IGNORECASE)) if cue else []
    distance = len(before) - cue_matches[-1].end() if cue_matches else 10_000
    evidence_type = candidate.witness.evidence_type
    derivation = candidate.derivation
    return [
        1.0,
        float(assignment is not None),
        float(bool(re.search(cue, near, re.IGNORECASE))) if cue else 0.0,
        float(bool(cue_matches)),
        1.0 / (1.0 + max(0, distance)),
        float(bool(re.search(r"\b(?:correction|instead|rather)\b", near, re.I))),
        float(bool(re.search(r"\b(?:not|never|without|hypothetical)\b", near, re.I))),
        float(isinstance(derivation, Span)),
        float(isinstance(derivation, StateRef)),
        float(not isinstance(derivation, (Span, StateRef))),
        float(evidence_type == "email_address"),
        float(evidence_type == "opaque_registry_id"),
        float(evidence_type in {"integer", "number", "currency_amount"}),
        float(evidence_type == "schema_enum"),
        position,
        1.0 / max(1, slot_catalog_size),
    ]


@dataclass(frozen=True)
class LinearCandidateRanker:
    weights: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]

    def __post_init__(self) -> None:
        if not (
            len(self.weights)
            == len(self.means)
            == len(self.scales)
            == len(FEATURE_NAMES)
        ):
            raise ValueError("ranker parameter dimensions are invalid")
        if any(scale <= 0 for scale in self.scales):
            raise ValueError("ranker scales must be positive")

    def score(self, features: Sequence[float]) -> float:
        if len(features) != len(self.weights):
            raise ValueError("ranker feature dimensions are invalid")
        logit = sum(
            weight * ((float(value) - mean) / scale)
            for weight, value, mean, scale in zip(
                self.weights, features, self.means, self.scales
            )
        )
        return _sigmoid(logit)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": VERIFIED_RANKER_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "weights": list(self.weights),
            "means": list(self.means),
            "scales": list(self.scales),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LinearCandidateRanker":
        if value.get("version") != VERIFIED_RANKER_VERSION:
            raise ValueError("unsupported verified ranker version")
        if value.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("verified ranker feature manifest changed")
        return cls(
            weights=tuple(float(item) for item in value["weights"]),
            means=tuple(float(item) for item in value["means"]),
            scales=tuple(float(item) for item in value["scales"]),
        )


def fit_linear_ranker(
    examples: Sequence[tuple[Sequence[float], bool]],
    *,
    iterations: int = 500,
    learning_rate: float = 0.08,
    l2: float = 0.01,
) -> LinearCandidateRanker:
    if not examples or not any(label for _features, label in examples):
        raise ValueError("ranker training needs positive and negative examples")
    if all(label for _features, label in examples):
        raise ValueError("ranker training needs positive and negative examples")
    width = len(FEATURE_NAMES)
    raw = [[float(value) for value in features] for features, _label in examples]
    if any(len(row) != width for row in raw):
        raise ValueError("ranker training feature dimensions are invalid")
    means = [sum(row[index] for row in raw) / len(raw) for index in range(width)]
    scales = []
    for index in range(width):
        variance = sum((row[index] - means[index]) ** 2 for row in raw) / len(raw)
        scales.append(max(math.sqrt(variance), 1.0 if index == 0 else 1e-6))
    normalized = [
        [(row[index] - means[index]) / scales[index] for index in range(width)]
        for row in raw
    ]
    positives = sum(bool(label) for _features, label in examples)
    negatives = len(examples) - positives
    positive_weight = len(examples) / (2.0 * positives)
    negative_weight = len(examples) / (2.0 * negatives)
    weights = [0.0] * width
    for step in range(iterations):
        gradient = [0.0] * width
        for row, (_features, label) in zip(normalized, examples):
            prediction = _sigmoid(sum(w * x for w, x in zip(weights, row)))
            sample_weight = positive_weight if label else negative_weight
            error = sample_weight * (prediction - float(label))
            for index, value in enumerate(row):
                gradient[index] += error * value
        rate = learning_rate / math.sqrt(1.0 + step / 50.0)
        for index in range(width):
            gradient[index] = gradient[index] / len(examples) + l2 * weights[index]
            weights[index] -= rate * gradient[index]
    return LinearCandidateRanker(
        weights=tuple(weights),
        means=tuple(means),
        scales=tuple(scales),
    )


def ranker_examples(
    cases: Sequence[Mapping[str, Any]],
) -> list[tuple[list[float], bool]]:
    from .selector_study import (
        _compile_with_proposer,
        _gold_values,
        _protected_critical_destinations,
    )

    output: list[tuple[list[float], bool]] = []
    for case in cases:
        if case["expected"]["mode"] != "call":
            continue
        session = _compile_with_proposer(case, "broad_typed")
        tool_id = str(case["expected"]["tool_id"])
        gold = _gold_values(case)
        protected = _protected_critical_destinations(case)
        for destination in sorted(protected):
            candidates = [
                candidate
                for candidate in session.candidates.candidates.values()
                if candidate.witness.tool_id == tool_id
                and candidate.witness.destination_scope == destination
            ]
            for candidate in candidates:
                output.append(
                    (
                        candidate_features(
                            session.context,
                            candidate,
                            slot_catalog_size=len(candidates),
                        ),
                        destination in gold
                        and canonical_json(candidate.value)
                        == canonical_json(gold[destination]),
                    )
                )
    return output


def _percentile(values: Sequence[int], quantile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def evaluate_ranker(
    cases: Sequence[Mapping[str, Any]],
    ranker: LinearCandidateRanker,
    *,
    ks: Sequence[int] = (1, 4, 8),
) -> dict[str, Any]:
    from .selector_study import (
        _compile_with_proposer,
        _gold_values,
        _protected_critical_destinations,
        _required_destinations,
    )

    if any(k <= 0 for k in ks):
        raise ValueError("ranker K values must be positive")
    call_cases = 0
    slot_total = 0
    results = {
        int(k): {
            "gold_slots_recalled": 0,
            "action_recoverable": 0,
            "selected_candidates": 0,
            "selected_gold_candidates": 0,
            "catalog_sizes": [],
        }
        for k in ks
    }
    full_catalog_action_recoverable = 0
    full_catalog_sizes: list[int] = []
    for case in cases:
        if case["expected"]["mode"] != "call":
            continue
        call_cases += 1
        session = _compile_with_proposer(case, "broad_typed")
        tool_id = str(case["expected"]["tool_id"])
        gold = _gold_values(case)
        required = _required_destinations(case, critical_only=True)
        action_status = {int(k): True for k in ks}
        full_status = True
        for destination in sorted(_protected_critical_destinations(case)):
            if destination not in gold:
                continue
            slot_total += 1
            candidates = [
                candidate
                for candidate in session.candidates.candidates.values()
                if candidate.witness.tool_id == tool_id
                and candidate.witness.destination_scope == destination
            ]
            ranked = sorted(
                candidates,
                key=lambda candidate: (
                    ranker.score(
                        candidate_features(
                            session.context,
                            candidate,
                            slot_catalog_size=len(candidates),
                        )
                    ),
                    candidate.candidate_id,
                ),
                reverse=True,
            )
            full_has_gold = any(
                canonical_json(candidate.value) == canonical_json(gold[destination])
                for candidate in ranked
            )
            if destination in required and not full_has_gold:
                full_status = False
            full_catalog_sizes.append(len(ranked))
            for raw_k in ks:
                k = int(raw_k)
                selected = ranked[:k]
                gold_selected = sum(
                    canonical_json(candidate.value)
                    == canonical_json(gold[destination])
                    for candidate in selected
                )
                results[k]["gold_slots_recalled"] += int(gold_selected > 0)
                results[k]["selected_candidates"] += len(selected)
                results[k]["selected_gold_candidates"] += gold_selected
                results[k]["catalog_sizes"].append(len(selected))
                if destination in required and gold_selected == 0:
                    action_status[k] = False
        full_catalog_action_recoverable += int(full_status)
        for k in results:
            results[k]["action_recoverable"] += int(action_status[k])
    conditions: dict[str, Any] = {}
    for k, row in sorted(results.items()):
        sizes = row.pop("catalog_sizes")
        selected = int(row["selected_candidates"])
        conditions[f"top_{k}"] = {
            **row,
            "slot_recall": row["gold_slots_recalled"] / slot_total if slot_total else None,
            "action_recoverability": (
                row["action_recoverable"] / call_cases if call_cases else None
            ),
            "candidate_precision": (
                row["selected_gold_candidates"] / selected if selected else None
            ),
            "mean_candidates_per_slot": sum(sizes) / len(sizes) if sizes else None,
            "p95_candidates_per_slot": _percentile(sizes, 0.95),
        }
    return {
        "version": VERIFIED_RANKER_VERSION,
        "call_cases": call_cases,
        "protected_gold_slots": slot_total,
        "full_catalog_action_recoverability": (
            full_catalog_action_recoverable / call_cases if call_cases else None
        ),
        "full_catalog_mean_candidates_per_slot": (
            sum(full_catalog_sizes) / len(full_catalog_sizes)
            if full_catalog_sizes
            else None
        ),
        "full_catalog_p95_candidates_per_slot": _percentile(
            full_catalog_sizes, 0.95
        ),
        "conditions": conditions,
    }


def prune_candidate_table(
    table: CandidateTable,
    context: EvidenceContext,
    ranker: LinearCandidateRanker,
    *,
    top_k: int,
) -> CandidateTable:
    if top_k <= 0:
        raise ValueError("verified ranker top_k must be positive")
    groups: dict[tuple[str, str], list[Any]] = {}
    for candidate in table.candidates.values():
        groups.setdefault(
            (
                candidate.witness.tool_id,
                candidate.witness.destination_scope,
            ),
            [],
        ).append(candidate)
    selected: dict[str, Any] = {}
    for candidates in groups.values():
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                ranker.score(
                    candidate_features(
                        context,
                        candidate,
                        slot_catalog_size=len(candidates),
                    )
                ),
                candidate.candidate_id,
            ),
            reverse=True,
        )
        selected.update(
            (candidate.candidate_id, candidate) for candidate in ranked[:top_k]
        )
    return CandidateTable(
        request_digest=table.request_digest,
        candidates=selected,
        rejections=table.rejections,
        policy_epochs=table.policy_epochs,
    )


def write_ranker(path: str | Path, ranker: LinearCandidateRanker) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(ranker.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_ranker(path: str | Path) -> LinearCandidateRanker:
    return LinearCandidateRanker.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )
