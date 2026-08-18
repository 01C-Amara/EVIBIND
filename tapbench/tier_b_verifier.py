from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .io import read_jsonl
from .typed_evidence_programs import ProgramExecution, TypedEvidenceProgram


TIER_B_VERIFIER_VERSION = "tapbench.tier_b_linear_verifier.v2"
FEATURE_NAMES = (
    "role_confidence",
    "execution_valid",
    "execution_risk_complement",
    "slot_cue_near_span",
    "destination_cue",
    "source_cue",
    "negative_scope",
    "hypothetical_scope",
    "superseded",
    "has_versioned_state",
    "is_transform",
    "is_composite",
    "op_parse_date",
    "op_parse_number",
    "op_convert_unit",
    "op_list",
    "op_derive",
)


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) < 1e-9
    if isinstance(left, str) and isinstance(right, str):
        return left.strip().casefold() == right.strip().casefold()
    return left == right


def _primary_span(program: TypedEvidenceProgram) -> tuple[int, int] | None:
    span = program.args.get("span")
    if isinstance(span, (list, tuple)) and len(span) == 2:
        return int(span[0]), int(span[1])
    for value in program.args.values():
        if isinstance(value, TypedEvidenceProgram):
            found = _primary_span(value)
            if found is not None:
                return found
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, TypedEvidenceProgram):
                    found = _primary_span(item)
                    if found is not None:
                        return found
    return None


def verifier_features(
    program: TypedEvidenceProgram,
    execution: ProgramExecution,
    request: str,
    slot: str,
) -> dict[str, float]:
    span = _primary_span(program)
    start = span[0] if span else 0
    end = span[1] if span else 0
    before = request[max(0, start - 48):start].casefold()
    nearby = request[max(0, start - 48):min(len(request), end + 48)].casefold()
    slot_words = [piece.casefold() for piece in slot.replace("-", "_").split("_") if piece]
    cue_hits = sum(word in nearby for word in slot_words)
    slot_cue = cue_hits / max(1, len(slot_words))
    role_confidence = float(program.args.get("role_confidence", 1.0))
    op = program.op
    return {
        "role_confidence": role_confidence,
        "execution_valid": float(execution.valid),
        "execution_risk_complement": max(0.0, 1.0 - execution.risk.upper_bound),
        "slot_cue_near_span": slot_cue,
        "destination_cue": float(any(token in before for token in (" to ", "returning ", "leaving ", "set ", "use "))),
        "source_cue": float(any(token in before for token in (" from ", "old ", "previous ", "was "))),
        "negative_scope": float(any(token in before for token in ("do not ", "don't ", "never ", "without ", "no "))),
        "hypothetical_scope": float(any(token in before for token in ("if ", "would ", "hypothetically ", "suppose "))),
        "superseded": float(bool(program.args.get("superseded"))),
        "has_versioned_state": float(op == "STATE_REF" and program.args.get("version") is not None),
        "is_transform": float(op in {"ENUM", "PARSE_DATE", "PARSE_TIME", "PARSE_NUMBER", "CONVERT_UNIT", "NEGATED_BOOL"}),
        "is_composite": float(op in {"LIST", "DERIVE"}),
        "op_parse_date": float(op == "PARSE_DATE"),
        "op_parse_number": float(op == "PARSE_NUMBER"),
        "op_convert_unit": float(op == "CONVERT_UNIT"),
        "op_list": float(op == "LIST"),
        "op_derive": float(op == "DERIVE"),
    }


def _vector(features: dict[str, float]) -> list[float]:
    return [float(features.get(name, 0.0)) for name in FEATURE_NAMES]


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-min(value, 60.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -60.0))
    return z / (1.0 + z)


def _fit(rows: list[dict[str, Any]], *, epochs: int = 1200, learning_rate: float = 0.08, l2: float = 0.01) -> tuple[float, list[float]]:
    if not rows:
        raise ValueError("Tier-B verifier training corpus is empty")
    labels = [int(row["label"]) for row in rows]
    if len(set(labels)) < 2:
        raise ValueError("Tier-B verifier training requires positive and negative candidates")
    positive_weight = len(labels) / (2.0 * sum(labels))
    negative_weight = len(labels) / (2.0 * (len(labels) - sum(labels)))
    weights = [0.0] * len(FEATURE_NAMES)
    intercept = 0.0
    for epoch in range(epochs):
        grad = [0.0] * len(weights)
        grad_intercept = 0.0
        for row in rows:
            x = row["vector"]
            label = int(row["label"])
            sample_weight = positive_weight if label else negative_weight
            predicted = _sigmoid(intercept + sum(weight * value for weight, value in zip(weights, x, strict=True)))
            error = sample_weight * (predicted - label)
            grad_intercept += error
            for index, value in enumerate(x):
                grad[index] += error * value
        scale = 1.0 / len(rows)
        step = learning_rate / (1.0 + epoch / 500.0)
        intercept -= step * grad_intercept * scale
        for index in range(len(weights)):
            weights[index] -= step * (grad[index] * scale + l2 * weights[index])
    return intercept, weights


def _predict(intercept: float, weights: list[float], vector: list[float]) -> float:
    return _sigmoid(intercept + sum(weight * value for weight, value in zip(weights, vector, strict=True)))


def _select_threshold(rows: list[dict[str, Any]], *, target_precision: float) -> tuple[float, dict[str, Any]]:
    if not rows:
        return 1.0, {"precision": None, "recall": 0.0, "coverage": 0.0, "selected": 0}
    positives = sum(int(row["label"]) for row in rows)
    best: tuple[float, dict[str, Any]] | None = None
    for threshold in sorted({float(row["score"]) for row in rows}, reverse=True):
        selected = [row for row in rows if float(row["score"]) >= threshold]
        tp = sum(int(row["label"]) for row in selected)
        precision = tp / len(selected)
        recall = tp / positives if positives else 0.0
        metrics = {
            "precision": precision,
            "recall": recall,
            "coverage": len(selected) / len(rows),
            "selected": len(selected),
            "true_positive": tp,
        }
        if precision >= target_precision:
            if best is None or metrics["coverage"] > best[1]["coverage"]:
                best = (threshold, metrics)
    if best is None:
        return 1.0, {"precision": None, "recall": 0.0, "coverage": 0.0, "selected": 0}
    return best


@dataclass(frozen=True)
class FrozenTierBVerifier:
    intercept: float
    weights: tuple[float, ...]
    threshold: float
    artifact_sha256: str
    accepted_error_upper_bound: float
    version: str = TIER_B_VERIFIER_VERSION

    @classmethod
    def load(cls, path: str | Path) -> "FrozenTierBVerifier":
        raw = Path(path).read_bytes()
        payload = json.loads(raw)
        if payload.get("schema_version") != TIER_B_VERIFIER_VERSION:
            raise ValueError(f"unsupported Tier-B verifier artifact: {payload.get('schema_version')}")
        if tuple(payload.get("feature_names", [])) != FEATURE_NAMES:
            raise ValueError("Tier-B verifier feature contract changed")
        return cls(
            intercept=float(payload["model"]["intercept"]),
            weights=tuple(float(value) for value in payload["model"]["weights"]),
            threshold=float(payload["operating_point"]["threshold"]),
            accepted_error_upper_bound=float(payload["operating_point"]["accepted_error_upper_bound"]),
            artifact_sha256=hashlib.sha256(raw).hexdigest(),
        )

    def score(
        self,
        program: TypedEvidenceProgram,
        execution: ProgramExecution,
        request: str,
        slot: str,
    ) -> float:
        return _predict(self.intercept, list(self.weights), _vector(verifier_features(program, execution, request, slot)))

    def accepts(
        self,
        program: TypedEvidenceProgram,
        execution: ProgramExecution,
        request: str,
        slot: str,
    ) -> tuple[bool, float]:
        score = self.score(program, execution, request, slot)
        return score >= self.threshold, score


def train_tier_b_verifier(
    corpus: Iterable[dict[str, Any]],
    output: str | Path,
    *,
    target_precision: float = 0.99,
) -> dict[str, Any]:
    rows = [dict(row) for row in corpus]
    families = sorted({str(row["family"]) for row in rows})
    oof: list[dict[str, Any]] = []
    fold_metrics = []
    for heldout in families:
        train = [row for row in rows if str(row["family"]) != heldout]
        test = [row for row in rows if str(row["family"]) == heldout]
        if not train or not test or len({int(row["label"]) for row in train}) < 2:
            continue
        intercept, weights = _fit(train)
        fold_rows = [{**row, "score": _predict(intercept, weights, row["vector"])} for row in test]
        oof.extend(fold_rows)
        fold_metrics.append({
            "heldout_family": heldout,
            "rows": len(test),
            "positives": sum(int(row["label"]) for row in test),
        })
    threshold, operating = _select_threshold(oof, target_precision=target_precision)
    selected = int(operating.get("selected", 0))
    errors = selected - int(operating.get("true_positive", 0))
    if selected <= 0:
        accepted_error_upper_bound = 1.0
    elif errors == 0:
        accepted_error_upper_bound = 1.0 - (0.05 ** (1.0 / selected))
    else:
        observed_error = errors / selected
        accepted_error_upper_bound = min(1.0, observed_error + math.sqrt(math.log(20.0) / (2.0 * selected)))

    intercept, weights = _fit(rows)
    source_digest = hashlib.sha256(
        json.dumps(
            [{key: row[key] for key in ("case_id", "family", "slot", "program_id", "label")} for row in rows],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema_version": TIER_B_VERIFIER_VERSION,
        "status": "frozen",
        "feature_names": list(FEATURE_NAMES),
        "training": {
            "rows": len(rows),
            "positives": sum(int(row["label"]) for row in rows),
            "families": families,
            "family_disjoint_folds": fold_metrics,
            "source_sha256": source_digest,
            "label_used_at_runtime": False,
        },
        "model": {"intercept": intercept, "weights": weights},
        "operating_point": {
            "target_precision": target_precision,
            "threshold": threshold,
            "accepted_error_upper_bound": accepted_error_upper_bound,
            "error_bound_confidence": 0.95,
            "error_bound_method": "exact_zero_failure_one_sided_or_hoeffding",
            "cross_validated": operating,
        },
        "runtime_contract": {
            "allowed_inputs": ["request", "slot", "program", "program_execution"],
            "forbidden_inputs": ["gold_action", "r2a_oracle", "derivable_values", "score_row"],
        },
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def label_program(execution: ProgramExecution, gold_value: Any, unsupported_values: Iterable[Any]) -> int:
    if not execution.valid or any(_same(execution.value, value) for value in unsupported_values):
        return 0
    return int(_same(execution.value, gold_value))
