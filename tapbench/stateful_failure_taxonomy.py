from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence


class StatefulTaxonomyError(ValueError):
    pass


CATEGORIES = (
    "missing_candidate",
    "wrong_candidate_selected",
    "clarification_over_trigger",
    "ambiguity_gate",
    "other_trajectory_divergence",
)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise StatefulTaxonomyError("empty quantile")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _gateway_events(row: Mapping[str, Any]) -> tuple[set[str], bool]:
    errors: set[str] = set()
    clarified = False
    for turn in row.get("turn_records", []):
        metadata = turn.get("response_metadata", {}) if isinstance(turn, Mapping) else {}
        gateway = metadata.get("gateway") if isinstance(metadata, Mapping) else None
        if not isinstance(gateway, Mapping):
            continue
        for choice in gateway.get("choices", []):
            if not isinstance(choice, Mapping):
                continue
            clarified = clarified or choice.get("decision") == "clarify"
            diagnostics = choice.get("diagnostics")
            if not isinstance(diagnostics, Mapping):
                continue
            for history in diagnostics.get("history", []):
                if isinstance(history, Mapping) and history.get("error"):
                    errors.add(str(history["error"]))
    return errors, clarified


def classify_loss(native: Mapping[str, Any], evibind: Mapping[str, Any]) -> str:
    native_exceptions = len(native.get("tool_call_exceptions", []) or [])
    evibind_exceptions = len(evibind.get("tool_call_exceptions", []) or [])
    errors, clarified = _gateway_events(evibind)
    if evibind_exceptions > native_exceptions:
        return "wrong_candidate_selected"
    if errors & {"empty_required_domain", "capability_mismatch"}:
        return "missing_candidate"
    if any("ambig" in error.casefold() for error in errors):
        return "ambiguity_gate"
    if clarified:
        return "clarification_over_trigger"
    if "uncertified_candidate" in errors:
        return "wrong_candidate_selected"
    return "other_trajectory_divergence"


def _pair_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        model = str(row.get("model_id", ""))
        scenario = str(row.get("scenario", ""))
        condition = str(row.get("condition", ""))
        if not model or not scenario or condition not in {"native", "evibind"}:
            raise StatefulTaxonomyError("row lacks model, scenario, or condition")
        key = (model, scenario)
        if condition in grouped[key]:
            raise StatefulTaxonomyError(f"duplicate condition: {key}/{condition}")
        grouped[key][condition] = row
    output = []
    for key, pair in sorted(grouped.items()):
        if set(pair) != {"native", "evibind"}:
            raise StatefulTaxonomyError(f"incomplete pair: {key}")
        native, evibind = pair["native"], pair["evibind"]
        delta = float(evibind.get("similarity", 0.0)) - float(native.get("similarity", 0.0))
        category = classify_loss(native, evibind) if delta < 0 else None
        output.append(
            {
                "model_id": key[0],
                "scenario": key[1],
                "family": str(evibind.get("family", "")),
                "delta_similarity": delta,
                "category": category,
                "native_exceptions": len(native.get("tool_call_exceptions", []) or []),
                "evibind_exceptions": len(evibind.get("tool_call_exceptions", []) or []),
            }
        )
    return output


def _summarize(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(pairs)
    categories = Counter(str(pair["category"]) for pair in pairs if pair.get("category"))
    contributions = {
        category: sum(
            float(pair["delta_similarity"])
            for pair in pairs
            if pair.get("category") == category
        )
        / count
        for category in CATEGORIES
    }
    return {
        "pairs": count,
        "loss_episodes": sum(categories.values()),
        "gain_episodes": sum(float(pair["delta_similarity"]) > 0 for pair in pairs),
        "equal_episodes": sum(float(pair["delta_similarity"]) == 0 for pair in pairs),
        "mean_similarity_delta": sum(float(pair["delta_similarity"]) for pair in pairs) / count,
        "category_counts": {category: categories.get(category, 0) for category in CATEGORIES},
        "category_similarity_contributions": contributions,
    }


def analyze_stateful_failure_taxonomy(
    rows: Iterable[Mapping[str, Any]],
    *,
    replicates: int = 20_000,
    seed: int = 20260814,
) -> dict[str, Any]:
    pairs = _pair_rows(rows)
    families = sorted({str(pair["family"]) for pair in pairs})
    models = sorted({str(pair["model_id"]) for pair in pairs})
    if not families or not models:
        raise StatefulTaxonomyError("no families or models")
    overall = _summarize(pairs)
    by_model = {
        model: _summarize([pair for pair in pairs if pair["model_id"] == model])
        for model in models
    }

    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_family[str(pair["family"])].append(pair)
    rng = random.Random(seed)
    bootstrap_counts: dict[str, list[float]] = {category: [] for category in CATEGORIES}
    bootstrap_contributions: dict[str, list[float]] = {category: [] for category in CATEGORIES}
    for _ in range(replicates):
        sampled = [rng.choice(families) for _ in families]
        replicate_pairs = [pair for family in sampled for pair in by_family[family]]
        summary = _summarize(replicate_pairs)
        for category in CATEGORIES:
            bootstrap_counts[category].append(
                summary["category_counts"][category] / summary["pairs"]
            )
            bootstrap_contributions[category].append(
                summary["category_similarity_contributions"][category]
            )

    intervals = {
        category: {
            "loss_episode_rate_95_ci": [
                _quantile(bootstrap_counts[category], 0.025),
                _quantile(bootstrap_counts[category], 0.975),
            ],
            "similarity_contribution_95_ci": [
                _quantile(bootstrap_contributions[category], 0.025),
                _quantile(bootstrap_contributions[category], 0.975),
            ],
        }
        for category in CATEGORIES
    }
    dominant = min(
        CATEGORIES,
        key=lambda category: overall["category_similarity_contributions"][category],
    )
    oracle_recovery = -overall["category_similarity_contributions"][dominant]
    return {
        "version": "evibind.stateful_failure_taxonomy.v1",
        "families": len(families),
        "models": len(models),
        "overall": overall,
        "by_model": by_model,
        "cluster_bootstrap": {
            "replicates": replicates,
            "seed": seed,
            "intervals": intervals,
        },
        "dominant_category_by_similarity_contribution": dominant,
        "dominant_category_oracle_recovery_upper_bound": oracle_recovery,
        "taxonomy_precedence": [
            "extra tool exception -> wrong candidate",
            "empty domain/capability mismatch -> missing candidate",
            "ambiguity transition -> ambiguity gate",
            "other clarify -> clarification over-trigger",
            "uncertified alternate -> wrong candidate",
            "otherwise -> other trajectory divergence",
        ],
        "rows": pairs,
    }
