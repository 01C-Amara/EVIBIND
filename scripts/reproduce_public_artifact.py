"""Reproduce the deterministic EviBind mechanism evidence without a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tapbench.adversarial_boundary import (  # noqa: E402
    build_effect_scenarios,
    run_executable_effects,
    run_separation_suite,
)
from tapbench.boundary_fuzz import run_boundary_fuzz  # noqa: E402
from tapbench.equal_value_benchmark import (  # noqa: E402
    build_equal_value_pairs,
    evaluate_equal_value_pairs,
)
from tapbench.implementation_fragility import run_implementation_fragility  # noqa: E402


OUTPUT_NAMES = (
    "originbench_cases.jsonl",
    "originbench_analysis.json",
    "checker_and_effect_analysis.json",
    "implementation_fragility_analysis.json",
    "boundary_fuzz_analysis.json",
    "reproduction_summary.json",
    "SHA256SUMS",
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reproduce(
    output_dir: Path,
    *,
    per_pattern: int = 50,
    per_kind: int = 10,
    separation_repetitions: int = 20,
    fuzz_trials: int = 10_000,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write content-addressed evidence and fail if an invariant regresses."""
    if min(per_pattern, per_kind, separation_repetitions, fuzz_trials) <= 0:
        raise ValueError("all reproduction counts must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    collisions = [name for name in OUTPUT_NAMES if (output_dir / name).exists()]
    if collisions and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing reproduction files: "
            + ", ".join(collisions)
        )

    pairs = build_equal_value_pairs(per_pattern)
    cases_path = output_dir / "originbench_cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(
                json.dumps(pair, sort_keys=True, separators=(",", ":")) + "\n"
            )
    origin_report = evaluate_equal_value_pairs(pairs)
    _write_json(output_dir / "originbench_analysis.json", origin_report)

    checker_effect_report = {
        "executable_effects": run_executable_effects(
            build_effect_scenarios(per_kind)
        ),
        "separation": run_separation_suite(separation_repetitions),
    }
    _write_json(
        output_dir / "checker_and_effect_analysis.json",
        checker_effect_report,
    )

    fragility_report = run_implementation_fragility(
        build_effect_scenarios(per_kind)
    )
    _write_json(
        output_dir / "implementation_fragility_analysis.json",
        fragility_report,
    )

    raw_fuzz = run_boundary_fuzz(fuzz_trials)
    fuzz_report = {
        key: value
        for key, value in raw_fuzz.items()
        if key not in {"elapsed_seconds", "trials_per_second"}
    }
    _write_json(output_dir / "boundary_fuzz_analysis.json", fuzz_report)

    conditions = checker_effect_report["executable_effects"]["conditions"]
    attacks = checker_effect_report["separation"]["attacks"]
    expected_effects = 3 * per_kind
    redundancy = fragility_report["classes"][
        "redundant_literal_trace_coherence"
    ]
    shared = fragility_report["classes"]["shared_trusted_boundary"]
    checks = {
        "origin_pair_count": origin_report["pair_count"] == 6 * per_pattern,
        "origin_evibind_joint": origin_report["methods"]["evibind"][
            "joint_soundness_completeness"
        ]
        == 1.0,
        "origin_value_only_not_joint": origin_report["methods"]["value_only"][
            "joint_soundness_completeness"
        ]
        == 0.0,
        "evibind_effects_safe_and_complete": conditions["evibind"]["harm"]
        == 0
        and conditions["evibind"]["completed"] == expected_effects,
        "trace_checker_effects_safe_and_complete": conditions[
            "trace_materializing_atomic_cite_and_check"
        ]["harm"]
        == 0
        and conditions["trace_materializing_atomic_cite_and_check"]["completed"]
        == expected_effects,
        "reject_only_checker_rejects": conditions[
            "reject_only_atomic_cite_and_check"
        ]["rejected"]
        == expected_effects,
        "native_literals_harm": conditions["native_literals"]["harm"]
        == expected_effects,
        "atomic_checker_blocks_attacks": attacks[
            "normalization_and_citation_gaming"
        ]["atomic_cite_and_check_exploitable"]
        == 0
        and attacks["state_toctou"]["atomic_cite_and_check_exploitable"] == 0,
        "value_only_exposes_attacks": attacks[
            "normalization_and_citation_gaming"
        ]["value_only_exploitable"]
        == separation_repetitions
        and attacks["state_toctou"]["value_only_exploitable"]
        == separation_repetitions,
        "cite_checker_redundancy_faults_exposed": redundancy[
            "trace_materializing_cite_and_check"
        ]["exploitable_variants"]
        == 8
        and redundancy["trace_materializing_cite_and_check"][
            "harmful_executions"
        ]
        == 8 * expected_effects,
        "evibind_redundancy_faults_fail_closed": redundancy["evibind"][
            "exploitable_variants"
        ]
        == 0
        and redundancy["evibind"]["harmful_executions"] == 0
        and redundancy["evibind"]["rejections"] == 8 * expected_effects,
        "shared_tcb_controls_are_symmetric": shared[
            "trace_materializing_cite_and_check"
        ]["exploitable_variants"]
        == 4
        and shared["evibind"]["exploitable_variants"] == 4,
        "boundary_fuzz_has_no_unsound_release": fuzz_report[
            "unsound_releases"
        ]
        == 0
        and fuzz_report["executed_trials"] == fuzz_trials,
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    data_files = [output_dir / name for name in OUTPUT_NAMES[:5]]
    summary = {
        "schema_version": "evibind.public_reproduction.v2",
        "scope": "deterministic mechanism evidence; no model or human data",
        "parameters": {
            "per_pattern": per_pattern,
            "per_effect_kind": per_kind,
            "separation_repetitions": separation_repetitions,
            "fuzz_trials": fuzz_trials,
        },
        "checks": checks,
        "passed": not failures,
        "failures": failures,
        "files": {path.name: _sha256(path) for path in data_files},
    }
    _write_json(output_dir / "reproduction_summary.json", summary)
    checksummed = [*data_files, output_dir / "reproduction_summary.json"]
    (output_dir / "SHA256SUMS").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in checksummed),
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError("reproduction checks failed: " + ", ".join(failures))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-pattern", type=int, default=50)
    parser.add_argument("--per-effect-kind", type=int, default=10)
    parser.add_argument("--separation-repetitions", type=int, default=20)
    parser.add_argument("--fuzz-trials", type=int, default=10_000)
    parser.add_argument(
        "--full",
        action="store_true",
        help="run the paper-scale one-million-trial boundary fuzz",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = reproduce(
        args.output_dir,
        per_pattern=args.per_pattern,
        per_kind=args.per_effect_kind,
        separation_repetitions=args.separation_repetitions,
        fuzz_trials=1_000_000 if args.full else args.fuzz_trials,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
