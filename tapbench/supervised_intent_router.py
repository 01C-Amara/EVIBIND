from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import unicodedata
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .io import read_jsonl, write_jsonl


ROUTER_VERSION = "tapbench.massive_supervised_intent_router.v1"
RANKING_SCHEMA_VERSION = "tapbench.supervised_tool_ranking.v1"
DEFAULT_DIMENSIONS = 65_536
DEFAULT_K = 8


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def intent_to_tool(intent: str) -> str:
    return intent.replace("_", ".", 1)


def _hash(value: str, dimensions: int) -> int:
    return zlib.crc32(value.encode("utf-8")) % dimensions


def hashed_counts(text: str, dimensions: int = DEFAULT_DIMENSIONS) -> dict[int, int]:
    normalized = normalize_text(text)
    counts: Counter[int] = Counter()
    padded = f" {normalized} "
    for size in (2, 3, 4, 5):
        for index in range(max(0, len(padded) - size + 1)):
            counts[_hash(f"c{size}:{padded[index:index + size]}", dimensions)] += 1
    words = re.findall(r"\w+", normalized, flags=re.UNICODE)
    for word in words:
        counts[_hash(f"w1:{word}", dimensions)] += 1
    for index in range(len(words) - 1):
        counts[_hash(f"w2:{words[index]}\u241f{words[index + 1]}", dimensions)] += 1
    return dict(counts)


def _tfidf_vector(counts: dict[int, int], idf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not counts:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
    indices = np.fromiter(sorted(counts), dtype=np.int64)
    values = np.asarray([1.0 + math.log(counts[int(index)]) for index in indices], dtype=np.float64)
    values *= idf[indices]
    norm = float(np.linalg.norm(values))
    if norm > 0:
        values /= norm
    return indices, values


def _rows(path: str | Path, split: str, supported_tools: set[str]) -> list[dict[str, str]]:
    output = []
    for row in read_jsonl(path):
        if str(row.get("partition")) != split:
            continue
        intent = str(row.get("intent"))
        tool = intent_to_tool(intent)
        if tool not in supported_tools:
            continue
        output.append({"id": str(row.get("id")), "text": str(row.get("utt", "")), "intent": intent, "tool": tool})
    return output


def train_router(
    rows: list[dict[str, str]],
    *,
    dimensions: int = DEFAULT_DIMENSIONS,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("router training split is empty")
    classes = sorted({row["tool"] for row in rows})
    class_index = {name: index for index, name in enumerate(classes)}
    document_frequency = np.zeros(dimensions, dtype=np.int32)
    cached: list[tuple[dict[str, str], dict[int, int]]] = []
    for row in rows:
        counts = hashed_counts(row["text"], dimensions)
        if counts:
            document_frequency[np.fromiter(counts.keys(), dtype=np.int64)] += 1
        cached.append((row, counts))
    idf = np.log((1.0 + len(rows)) / (1.0 + document_frequency.astype(np.float64))) + 1.0
    centroids = np.zeros((len(classes), dimensions), dtype=np.float64)
    class_counts: Counter[str] = Counter()
    for row, counts in cached:
        indices, values = _tfidf_vector(counts, idf)
        centroids[class_index[row["tool"]], indices] += values
        class_counts[row["tool"]] += 1
    norms = np.linalg.norm(centroids, axis=1)
    nonzero = norms > 0
    centroids[nonzero] /= norms[nonzero, None]
    return {
        "schema_version": ROUTER_VERSION,
        "dimensions": dimensions,
        "classes": classes,
        "class_counts": dict(sorted(class_counts.items())),
        "idf": idf.astype(np.float32),
        "centroids": centroids.astype(np.float32),
        "training_rows": len(rows),
    }


def save_router(model: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        schema_version=np.asarray([model["schema_version"]]),
        dimensions=np.asarray([model["dimensions"]], dtype=np.int64),
        classes=np.asarray(model["classes"]),
        class_counts=np.asarray([model["class_counts"][name] for name in model["classes"]], dtype=np.int64),
        idf=model["idf"],
        centroids=model["centroids"],
    )


def load_router(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        classes = [str(value) for value in payload["classes"].tolist()]
        return {
            "schema_version": str(payload["schema_version"][0]),
            "dimensions": int(payload["dimensions"][0]),
            "classes": classes,
            "class_counts": dict(zip(classes, [int(value) for value in payload["class_counts"]], strict=True)),
            "idf": payload["idf"].astype(np.float32),
            "centroids": payload["centroids"].astype(np.float32),
        }


def rank_tools(model: dict[str, Any], text: str, *, k: int = DEFAULT_K) -> list[dict[str, Any]]:
    counts = hashed_counts(text, int(model["dimensions"]))
    indices, values = _tfidf_vector(counts, model["idf"])
    if len(indices):
        scores = model["centroids"][:, indices] @ values.astype(np.float32)
    else:
        scores = np.zeros(len(model["classes"]), dtype=np.float32)
    ordered = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), str(model["classes"][index])))
    return [
        {"rank": rank, "tool": str(model["classes"][index]), "cosine_score": float(scores[index])}
        for rank, index in enumerate(ordered[:k], start=1)
    ]


def _user_text(case: dict[str, Any]) -> str:
    return "\n".join(
        str(row.get("content", ""))
        for row in case.get("messages", [])
        if isinstance(row, dict) and row.get("role") == "user"
    )


def _gold_tool(row: dict[str, Any]) -> str:
    truth = row.get("ground_truth")
    if not isinstance(truth, list) or len(truth) != 1 or not isinstance(truth[0], dict) or len(truth[0]) != 1:
        raise ValueError(f"unsupported gold shape for {row.get('case_id')}")
    return str(next(iter(truth[0])))


def _confidence(ranking: list[dict[str, Any]]) -> float:
    if not ranking:
        return 0.0
    first = float(ranking[0]["cosine_score"])
    second = float(ranking[1]["cosine_score"]) if len(ranking) > 1 else first
    return first - second


def _threshold(rows: list[dict[str, Any]], target_precision: float) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (-float(row["confidence"]), str(row.get("id") or row.get("case_id"))))
    best = None
    accepted = 0
    correct = 0
    index = 0
    while index < len(ordered):
        threshold = float(ordered[index]["confidence"])
        tied = []
        while index < len(ordered) and math.isclose(float(ordered[index]["confidence"]), threshold, abs_tol=1e-15, rel_tol=0.0):
            tied.append(ordered[index])
            index += 1
        accepted += len(tied)
        correct += sum(int(row["top1_correct"]) for row in tied)
        precision = correct / accepted
        if precision >= target_precision:
            best = {
                "threshold": threshold,
                "accepted": accepted,
                "correct": correct,
                "precision": precision,
                "coverage": accepted / len(ordered) if ordered else 0.0,
                "n": len(ordered),
                "target_met": True,
            }
    return best or {
        "threshold": None,
        "accepted": 0,
        "correct": 0,
        "precision": None,
        "coverage": 0.0,
        "n": len(ordered),
        "target_met": False,
    }


def _metrics(rows: list[dict[str, Any]], threshold: float | None = None) -> dict[str, Any]:
    selected = rows if threshold is None else [row for row in rows if float(row["confidence"]) >= threshold]
    top1 = sum(int(row["top1_correct"]) for row in selected)
    recall8 = sum(int(row["gold_rank"] is not None and int(row["gold_rank"]) <= 8) for row in selected)
    return {
        "n": len(rows),
        "accepted": len(selected),
        "coverage": len(selected) / len(rows) if rows else 0.0,
        "top1_correct": top1,
        "top1_accuracy": top1 / len(selected) if selected else None,
        "recall_at_8": recall8 / len(selected) if selected else None,
    }


def _score_labeled_rows(model: dict[str, Any], rows: list[dict[str, str]], language: str) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        ranking = rank_tools(model, row["text"])
        rank_by_tool = {str(item["tool"]): int(item["rank"]) for item in ranking}
        output.append(
            {
                "id": row["id"],
                "language": language,
                "gold_tool": row["tool"],
                "predicted_tool": ranking[0]["tool"],
                "gold_rank": rank_by_tool.get(row["tool"]),
                "top1_correct": ranking[0]["tool"] == row["tool"],
                "confidence": _confidence(ranking),
            }
        )
    return output


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ranking_sha256(ranking: list[dict[str, Any]]) -> str:
    encoded = json.dumps(ranking, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    *,
    source_dir: str | Path,
    cases_path: str | Path,
    gold_path: str | Path,
    output_dir: str | Path,
    languages: tuple[str, ...] = ("en-US", "fa-IR", "ja-JP"),
    dimensions: int = DEFAULT_DIMENSIONS,
    target_precision: float = 0.95,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cases = read_jsonl(cases_path)
    gold_rows = read_jsonl(gold_path)
    gold_by_id = {str(row["case_id"]): _gold_tool(row) for row in gold_rows}
    supported_tools = {
        str(tool.get("canonical_name") or tool.get("name"))
        for case in cases
        for tool in case.get("tools", [])
        if isinstance(tool, dict)
    }
    models: dict[str, dict[str, Any]] = {}
    source_manifest = {}
    dev_scores = []
    for language in languages:
        source_path = Path(source_dir) / f"{language}.jsonl"
        train = _rows(source_path, "train", supported_tools)
        dev = _rows(source_path, "dev", supported_tools)
        model = train_router(train, dimensions=dimensions)
        model_path = output / f"router_{language}.npz"
        save_router(model, model_path)
        models[language] = model
        dev_scores.extend(_score_labeled_rows(model, dev, language))
        source_manifest[language] = {
            "path": str(source_path.resolve()),
            "sha256": _sha256(source_path),
            "train_rows": len(train),
            "dev_rows": len(dev),
            "model_path": str(model_path.resolve()),
            "model_sha256": _sha256(model_path),
            "classes": len(model["classes"]),
        }
    dev_threshold = _threshold(dev_scores, target_precision)
    per_language_thresholds = {
        language: _threshold([row for row in dev_scores if row["language"] == language], target_precision)
        for language in languages
    }

    design_scores = []
    ranking_rows = []
    for case in cases:
        case_id = str(case["case_id"])
        language = str(case.get("metadata", {}).get("language") or case.get("factors", {}).get("language"))
        ranking = rank_tools(models[language], _user_text(case))
        gold_tool = gold_by_id[case_id]
        rank_by_tool = {str(item["tool"]): int(item["rank"]) for item in ranking}
        design_scores.append(
            {
                "case_id": case_id,
                "language": language,
                "gold_tool": gold_tool,
                "predicted_tool": ranking[0]["tool"],
                "gold_rank": rank_by_tool.get(gold_tool),
                "top1_correct": ranking[0]["tool"] == gold_tool,
                "confidence": _confidence(ranking),
            }
        )
        ranking_rows.append(
            {
                "schema_version": RANKING_SCHEMA_VERSION,
                "case_id": case_id,
                "language": language,
                "request_sha256": hashlib.sha256(_user_text(case).encode("utf-8")).hexdigest(),
                "retriever_version": ROUTER_VERSION,
                "retriever_model_id": "hashed_tfidf_nearest_centroid",
                "retriever_revision": "massive_v1.1_train_only",
                "k": DEFAULT_K,
                "ranking": ranking,
                "ranking_sha256": _ranking_sha256(ranking),
            }
        )
    rankings_path = output / "design_rankings.jsonl"
    write_jsonl(rankings_path, ranking_rows)
    _write_csv(output / "official_dev_scores.csv", dev_scores)
    _write_csv(output / "design_scores.csv", design_scores)
    threshold_value = dev_threshold["threshold"]
    report = {
        "schema_version": "tapbench.massive_supervised_intent_router_report.v1",
        "router_version": ROUTER_VERSION,
        "analysis_status": "exploratory_benchmark_supervised_sidecar",
        "confirmation_authorized": False,
        "system_label": "benchmark_supervised_intent_router",
        "supervision_disclosure": "Intent centroids use public MASSIVE v1.1 train labels; this is not an unaided small-LLM condition.",
        "dimensions": dimensions,
        "supported_tool_count": len(supported_tools),
        "sources": source_manifest,
        "threshold_selection": {
            "split": "official_MASSIVE_dev",
            "target_precision": target_precision,
            "global": dev_threshold,
            "per_language_appendix": per_language_thresholds,
        },
        "official_dev": {
            "overall": _metrics(dev_scores),
            "selected_global": _metrics(dev_scores, threshold_value),
            "by_language": {language: _metrics([row for row in dev_scores if row["language"] == language]) for language in languages},
        },
        "design": {
            "split": "development_v7_qa_hybrid_design",
            "overall": _metrics(design_scores),
            "selected_by_dev_global_threshold": _metrics(design_scores, threshold_value),
            "by_language": {language: _metrics([row for row in design_scores if row["language"] == language]) for language in languages},
        },
        "next_stage_gate": {
            "minimum_top1_accuracy": 0.80,
            "minimum_recall_at_8": 0.98,
            "passes": _metrics(design_scores)["top1_accuracy"] >= 0.80 and _metrics(design_scores)["recall_at_8"] >= 0.98,
        },
        "artifacts": {
            "design_rankings": str(rankings_path.resolve()),
            "design_rankings_sha256": _sha256(rankings_path),
            "official_dev_scores": str((output / "official_dev_scores.csv").resolve()),
            "design_scores": str((output / "design_scores.csv").resolve()),
        },
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the explicit MASSIVE-supervised intent router.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    parser.add_argument("--target-precision", type=float, default=0.95)
    args = parser.parse_args()
    report = run_experiment(
        source_dir=args.source_dir,
        cases_path=args.cases,
        gold_path=args.gold,
        output_dir=args.output_dir,
        dimensions=args.dimensions,
        target_precision=args.target_precision,
    )
    print(json.dumps({"design": report["design"], "next_stage_gate": report["next_stage_gate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
