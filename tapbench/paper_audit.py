from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import yaml


PAPER_AUDIT_VERSION = "evibind.paper_audit.v5"
_CLAIM = re.compile(r"\\claim\{([^}]+)\}")
_CITATION = re.compile(r"\\cite[pt]?\{([^}]+)\}")
_BIB_ENTRY = re.compile(r"@\w+\{([^,\s]+),")
_ALLOWED_STATUSES = {
    "implemented",
    "deterministic_diagnostic",
    "frozen_legacy",
    "frozen_selected_model",
    "frozen_boundary",
    "frozen_negative",
    "development_diagnostic",
    "corrected_replay",
    "mechanism_benchmark",
    "frozen_confirmatory",
    "preregistered_stress_extension",
    "prospective_stress_extension",
    "packaged_artifact",
    "pending",
}


class PaperAuditError(ValueError):
    pass


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise PaperAuditError(f"{path} must contain a mapping")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PaperAuditError(f"{path} must contain an object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise PaperAuditError(f"{path} must contain JSON objects")
            rows.append(value)
    return rows


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _claims(ledger: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = ledger.get("claims")
    if not isinstance(rows, list):
        raise PaperAuditError("paper claim ledger must contain a list")
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("id"), str):
            raise PaperAuditError("invalid paper claim entry")
        claim_id = str(row["id"])
        if claim_id in output:
            raise PaperAuditError(f"duplicate claim: {claim_id}")
        output[claim_id] = row
    return output


def _approx(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= tolerance
    return left == right


def audit_paper(root: str | Path | None = None) -> dict[str, Any]:
    repository = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[1]
    paper = repository / "paper"
    source_paths = [
        paper / "main.tex",
        paper / "formal_guarantee.tex",
        paper / "formal_appendix.tex",
    ]
    failures: list[str] = []
    checks: list[str] = []
    for path in source_paths:
        if not path.is_file():
            failures.append(f"missing_source:{path.relative_to(repository)}")
    if failures:
        raise PaperAuditError("; ".join(failures))
    main = source_paths[0].read_text(encoding="utf-8")
    formal = source_paths[1].read_text(encoding="utf-8")
    appendix = source_paths[2].read_text(encoding="utf-8")
    tex = "\n".join((main, formal, appendix))

    ledger = _read_yaml(paper / "claims.yaml")
    if ledger.get("version") != "evibind.paper_claims.v1":
        failures.append("claim_ledger_version")
    claims = _claims(ledger)
    referenced = set(_CLAIM.findall(tex))
    if referenced != set(claims):
        failures.append(
            "claim_bijection:referenced=" + ",".join(sorted(referenced))
            + ":ledger=" + ",".join(sorted(claims))
        )
    for claim_id, claim in claims.items():
        if claim.get("status") not in _ALLOWED_STATUSES:
            failures.append(f"{claim_id}:invalid_status")
        evidence = claim.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            failures.append(f"{claim_id}:missing_evidence")
        else:
            for raw in evidence:
                if not isinstance(raw, str) or not (repository / raw).is_file():
                    failures.append(f"{claim_id}:missing_path:{raw}")
    checks.append("claim_ledger_is_bijective_and_evidence_backed")

    cited = {
        key.strip()
        for group in _CITATION.findall(tex)
        for key in group.split(",")
        if key.strip()
    }
    bibliography = (paper / "references.bib").read_text(encoding="utf-8")
    missing = cited - set(_BIB_ENTRY.findall(bibliography))
    if missing:
        failures.append("missing_citations:" + ",".join(sorted(missing)))
    checks.append("citations_resolve")

    style_requirements = (
        r"\usepackage{iclr2027_conference,times}",
        r"\author{Anonymous authors}",
        r"\subsection*{AI use statement}",
        r"\subsection*{Ethics statement}",
        r"\subsection*{Reproducibility statement}",
    )
    for phrase in style_requirements:
        if phrase not in main:
            failures.append(f"missing_submission_element:{phrase}")
    if main.find(r"\bibliography{references}") > main.find(r"\appendix"):
        failures.append("appendix_precedes_bibliography")
    for name in ("iclr2027_conference.sty", "iclr2027_conference.bst"):
        if not (paper / name).is_file():
            failures.append(f"missing_official_style:{name}")
    checks.append("iclr2027_structure_and_disclosures_present")

    aux_path = paper / "main.aux"
    log_path = paper / "main.log"
    pdf_path = paper / "main.pdf"
    if not all(path.is_file() for path in (aux_path, log_path, pdf_path)):
        failures.append("canonical_pdf_build_missing")
        main_end_page = None
    else:
        aux = aux_path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"\\newlabel\{main-text-end\}\{\{[^}]*\}\{(\d+)\}", aux)
        main_end_page = int(match.group(1)) if match else None
        if main_end_page is None or main_end_page > 9:
            failures.append(f"main_text_page_limit:{main_end_page}")
        log = log_path.read_text(encoding="utf-8", errors="replace")
        if "Overfull \\hbox" in log or "undefined references" in log.lower():
            failures.append("latex_layout_or_reference_warning")
    checks.append("canonical_pdf_build_and_nine_page_limit_pass")

    formal_requirements = (
        r"\mathrm{Span}(m,b_0,b_1,p)",
        "policy projection; equivalently",
        "recursively",
        "shared verification secret",
        "release certificate",
        "explicitly enumerated",
        "no critical-literal channel",
        "Materialization confinement",
        "does not establish correct tool routing",
        "Ed25519",
    )
    for phrase in formal_requirements:
        if phrase not in formal:
            failures.append(f"missing_formal_boundary:{phrase}")
    if "complete confinement argument" not in appendix.lower():
        failures.append("missing_complete_proof_appendix")
    for phrase in (
        "Value-only post-hoc enforcement cannot recover derivation",
        "cannot be both sound and complete",
    ):
        if phrase not in main:
            failures.append(f"missing_front_loaded_separation:{phrase}")
    checks.append("formal_statement_separation_and_tcb_are_explicit")

    metric_requirements = (
        "Selective coverage (release rate)",
        "expected-call acceptance",
        "exact-binding recall",
        "matched-proposal post-hoc",
        "one invocation per paired case",
        "Model-controlled executable values stop at destination-specific handles",
    )
    for phrase in metric_requirements:
        if phrase not in main:
            failures.append(f"missing_metric_or_figure_phrase:{phrase}")
    if "matched-compute" in main.lower():
        failures.append("inaccurate_matched_compute_label")
    checks.append("metric_names_and_intervention_labels_are_precise")

    for label in (
        "Native literals",
        "Direct candidates",
        "Post-hoc copy",
        "Post-hoc typed",
        "Source-only handles",
        "Source+destination",
        r"Full \EviBind{}",
    ):
        if label not in main:
            failures.append(f"missing_condition:{label}")
    checks.append("all_seven_conditions_are_reported")

    analysis = _read_json(repository / "work/reviewer_revision_analysis.json")
    oracle = analysis.get("actual_compiler_oracle_selector", {})
    oracle_expected = claims["oracle_decomposition"].get("expected", {})
    oracle_observed = {
        "cases": analysis.get("case_count"),
        "families": analysis.get("family_count"),
        "expected_calls": oracle.get("expected_calls"),
        "exact_calls": oracle.get("exact_calls"),
        "exact_call_recall": oracle.get("exact_call_recall"),
        "missing_required_expected_calls": oracle.get("zero_required_expected_call_cases"),
        "ambiguous_expected_calls": oracle.get("ambiguous_expected_call_cases"),
    }
    if set(oracle_expected) != set(oracle_observed) or any(
        not _approx(oracle_expected[key], oracle_observed[key]) for key in oracle_observed
    ):
        failures.append("oracle_decomposition_claim_drift")
    checks.append("oracle_decomposition_is_recomputed_and_pinned")

    deterministic = _read_json(
        repository
        / "work/oracle_selector_phase_v1/compiler_recoverability_deterministic_development.json"
    )
    broad = _read_json(
        repository
        / "work/oracle_selector_phase_v1/compiler_recoverability_broad_typed_development.json"
    )
    mixed_observed = {
        "call_cases": deterministic.get("call_cases"),
        "deterministic_all_leaf_strict": deterministic.get("counts", {}).get(
            "all_leaf_strict"
        ),
        "deterministic_critical_strict": deterministic.get("counts", {}).get(
            "critical_leaf_strict"
        ),
        "deterministic_critical_select": deterministic.get("counts", {}).get(
            "critical_leaf_model_selection"
        ),
        "broad_all_leaf_strict": broad.get("counts", {}).get("all_leaf_strict"),
        "broad_critical_strict": broad.get("counts", {}).get("critical_leaf_strict"),
        "broad_critical_select": broad.get("counts", {}).get(
            "critical_leaf_model_selection"
        ),
    }
    if claims["mixed_compiler"].get("expected") != mixed_observed:
        failures.append("mixed_compiler_claim_drift")
    checks.append("mixed_compiler_ceiling_is_recomputed_and_pinned")

    ranker = _read_json(
        repository / "work/oracle_selector_phase_v1/verified_ranker_development_analysis.json"
    )
    top_one = ranker.get("conditions", {}).get("top_1", {})
    ranker_observed = {
        "call_cases": ranker.get("call_cases"),
        "action_recoverable": top_one.get("action_recoverable"),
        "action_recoverability": top_one.get("action_recoverability"),
        "candidate_precision": top_one.get("candidate_precision"),
        "mean_candidates_per_slot": top_one.get("mean_candidates_per_slot"),
        "p95_candidates_per_slot": top_one.get("p95_candidates_per_slot"),
    }
    if claims["verified_ranker"].get("expected") != ranker_observed:
        failures.append("verified_ranker_claim_drift")
    checks.append("verified_ranker_is_train_split_bound_and_pinned")

    selector = _read_json(
        repository
        / "work/oracle_selector_development_v2/qwen3_1_7b/binding_sweep_analysis.json"
    )
    selector_conditions = selector.get("conditions", {})
    selector_observed = {
        "rows": selector.get("rows"),
        "dynamic_gold_exact": selector_conditions.get(
            "dynamic_enum_binding_only_oracle_0", {}
        ).get("exact_critical_call_recall"),
        "indexed_gold_exact": selector_conditions.get(
            "indexed_tool_binding_only_oracle_0", {}
        ).get("exact_critical_call_recall"),
        "indexed_plus_1_exact": selector_conditions.get(
            "indexed_tool_binding_only_oracle_1", {}
        ).get("exact_critical_call_recall"),
        "indexed_plus_3_exact": selector_conditions.get(
            "indexed_tool_binding_only_oracle_3", {}
        ).get("exact_critical_call_recall"),
        "indexed_plus_7_exact": selector_conditions.get(
            "indexed_tool_binding_only_oracle_7", {}
        ).get("exact_critical_call_recall"),
        "actual_exact": selector_conditions.get(
            "indexed_tool_binding_only_actual", {}
        ).get("exact_critical_call_recall"),
        "verified_top_1_exact": selector_conditions.get(
            "indexed_tool_binding_only_verified_top_1", {}
        ).get("exact_critical_call_recall"),
        "verified_top_1_validity": selector_conditions.get(
            "indexed_tool_binding_only_verified_top_1", {}
        ).get("response_validity"),
    }
    if claims["selector_phase"].get("expected") != selector_observed:
        failures.append("selector_phase_claim_drift")
    checks.append("selector_phase_diagram_is_response_digest_bound_and_pinned")

    freeze = _read_json(
        repository / "work/confirmatory_fresh_family_v1/freeze_manifest.json"
    )
    prereg = _read_json(
        repository / "work/confirmatory_fresh_family_v1/preregistration.json"
    )
    fresh_observed: dict[str, Any] = {
        "cases": freeze.get("counts", {}).get("cases"),
        "families": freeze.get("counts", {}).get("families"),
        "prior_family_overlap": freeze.get("counts", {}).get("prior_family_overlap"),
        "max_output_tokens": prereg.get("max_output_tokens"),
        "retries": prereg.get("retries"),
    }
    if freeze.get("status") != "frozen_before_any_confirmatory_model_output":
        failures.append("confirmatory_not_frozen_before_output")
    confirmatory_result_roots = {
        "qwen3_1_7b": repository
        / "work/confirmatory_fresh_family_v1/results/qwen3_1_7b",
        "qwen36_35b_a3b": repository
        / "work/confirmatory_fresh_family_v1/results/qwen36_35b_a3b",
        "gpt_oss_120b": repository
        / "work/confirmatory_model_extension_recovery_v4/results/gpt_oss_120b",
        "gpt_5_6_luna": repository
        / "work/confirmatory_luna_extension_v3/results/gpt_5_6_luna",
    }
    all_fresh_rows: list[dict[str, Any]] = []
    for model_key, result_root in confirmatory_result_roots.items():
        analysis_row = _read_json(result_root / "confirmatory_analysis.json")
        conditions = analysis_row.get("conditions", {})
        rows = _read_jsonl(result_root / "confirmatory_rows.jsonl")
        all_fresh_rows.extend(rows)
        by_condition = {
            condition: {str(row["case_id"]): row for row in rows if row.get("condition_id") == condition}
            for condition in (
                "indexed_tool_binding_only_actual",
                "indexed_tool_binding_only_verified_top_1",
                "indexed_tool_binding_only_oracle_0",
            )
        }
        actual = by_condition["indexed_tool_binding_only_actual"]
        verified = by_condition["indexed_tool_binding_only_verified_top_1"]
        oracle_rows = by_condition["indexed_tool_binding_only_oracle_0"]
        gains = sum(
            not bool(actual[case_id].get("exact_critical_call"))
            and bool(verified[case_id].get("exact_critical_call"))
            for case_id in actual
        )
        regressions = sum(
            bool(actual[case_id].get("exact_critical_call"))
            and not bool(verified[case_id].get("exact_critical_call"))
            for case_id in actual
        )
        payload_mismatches = sum(
            verified[case_id].get("payload_sha256")
            != oracle_rows[case_id].get("payload_sha256")
            for case_id in verified
        )
        fresh_observed[model_key] = {
            "actual_exact": conditions.get("indexed_tool_binding_only_actual", {}).get(
                "exact_critical_call_recall"
            ),
            "verified_top_1_exact": conditions.get(
                "indexed_tool_binding_only_verified_top_1", {}
            ).get("exact_critical_call_recall"),
            "oracle_exact": conditions.get("indexed_tool_binding_only_oracle_0", {}).get(
                "exact_critical_call_recall"
            ),
            "gains": gains,
            "regressions": regressions,
            "verified_oracle_payload_mismatches": payload_mismatches,
        }
    fresh_observed.update(
        {
            "total_rows": len(all_fresh_rows),
            "length_stops": sum(
                row.get("response", {}).get("choices", [{}])[0].get("finish_reason")
                == "length"
                for row in all_fresh_rows
            ),
            "maximum_completion_tokens": max(
                int(row.get("completion_tokens", 0)) for row in all_fresh_rows
            ),
        }
    )
    if claims["fresh_family_confirmatory"].get("expected") != fresh_observed:
        failures.append("fresh_family_confirmatory_claim_drift")
    checks.append("fresh_family_confirmatory_freeze_and_paired_results_are_pinned")

    saturation_prereg = _read_json(
        repository / "configs/fresh_family_saturation_luna_preregistration_v1.json"
    )
    saturation = _read_json(
        repository
        / "work/fresh_family_saturation_luna_v1/analysis/gpt_5_6_luna_saturation.json"
    )
    saturation_conditions = saturation.get("conditions", {})
    saturation_contrasts = saturation.get("contrasts_vs_gold_only", {})
    plus_3 = saturation_contrasts.get("indexed_tool_binding_only_oracle_3", {})
    plus_7 = saturation_contrasts.get("indexed_tool_binding_only_oracle_7", {})
    saturation_observed = {
        "rows": saturation.get("rows"),
        "cases": saturation_conditions.get(
            "indexed_tool_binding_only_oracle_0", {}
        ).get("rows"),
        "families": saturation.get("families"),
        "length_stops": sum(
            int(row.get("length_stops", 0))
            for row in saturation_conditions.values()
        ),
        "gold_only_exact": saturation_conditions.get(
            "indexed_tool_binding_only_oracle_0", {}
        ).get("exact_rate"),
        "gold_plus_1_exact": saturation_conditions.get(
            "indexed_tool_binding_only_oracle_1", {}
        ).get("exact_rate"),
        "gold_plus_3_exact": saturation_conditions.get(
            "indexed_tool_binding_only_oracle_3", {}
        ).get("exact_rate"),
        "gold_plus_7_exact": saturation_conditions.get(
            "indexed_tool_binding_only_oracle_7", {}
        ).get("exact_rate"),
        "plus_3_regressions": plus_3.get("regressions"),
        "plus_7_regressions": plus_7.get("regressions"),
        "plus_3_family_cluster_ci": plus_3.get("family_cluster_ci95"),
        "plus_7_family_cluster_ci": plus_7.get("family_cluster_ci95"),
        "holm_adjusted_p": plus_3.get("holm_adjusted_p"),
    }
    if saturation_prereg.get("status") != "frozen_before_any_saturation_sweep_api_request":
        failures.append("saturation_not_frozen_before_output")
    if plus_3.get("holm_adjusted_p") != plus_7.get("holm_adjusted_p"):
        failures.append("saturation_holm_adjustment_drift")
    if claims["fresh_family_saturation"].get("expected") != saturation_observed:
        failures.append("fresh_family_saturation_claim_drift")
    checks.append("fresh_family_saturation_freeze_and_stress_frontier_are_pinned")

    position_root = repository / "work/candidate_position_robustness_v1"
    position_protocol = _read_json(position_root / "protocol.json")
    position_preflight = _read_json(position_root / "preflight.json")
    position_analyses = {
        model_key: _read_json(
            position_root / f"results/{model_key}/candidate_position_analysis.json"
        )
        for model_key in ("qwen3_1_7b", "qwen36_35b_a3b")
    }

    def _position_model(model_key: str) -> dict[str, Any]:
        report = position_analyses[model_key]
        groups = report.get("groups", {})
        selection = report.get("actual_catalog_selection", {})
        return {
            "gold_early_actual_recall": [
                groups.get(f"gold_early:actual_{order}", {}).get(
                    "exact_binding_recall"
                )
                for order in ("gold_first", "gold_last", "seeded_a", "seeded_b")
            ],
            "gold_late_actual_recall": [
                groups.get(f"gold_late:actual_{order}", {}).get(
                    "exact_binding_recall"
                )
                for order in ("gold_first", "gold_last", "seeded_a", "seeded_b")
            ],
            "gold_early_top1_recall": groups.get(
                "gold_early:admissible_top1", {}
            ).get("exact_binding_recall"),
            "gold_early_top1_release": groups.get(
                "gold_early:admissible_top1", {}
            ).get("release_rate"),
            "gold_late_top1_recall": groups.get(
                "gold_late:admissible_top1", {}
            ).get("exact_binding_recall"),
            "all_permutations_exact_rate": selection.get(
                "all_permutations_exact_rate"
            ),
            "outcome_consistency_rate": selection.get("outcome_consistency_rate"),
            "first_index_selection_rate": selection.get(
                "first_index_selection_rate"
            ),
        }

    position_observed = {
        "cases": position_preflight.get("cases"),
        "families": position_preflight.get("families"),
        "rows_per_model": position_analyses["qwen3_1_7b"].get("rows"),
        "retries": position_protocol.get("decoding", {}).get("retries"),
        "preflight_top1_gold_late": position_preflight.get(
            "top1_gold_complete", {}
        ).get("gold_late", 0),
        "preflight_top1_gold_early": position_preflight.get(
            "top1_gold_complete", {}
        ).get("gold_early", 0),
        "qwen3_1_7b": _position_model("qwen3_1_7b"),
        "qwen36_35b_a3b": _position_model("qwen36_35b_a3b"),
    }
    if position_analyses["qwen36_35b_a3b"].get("rows") != position_observed[
        "rows_per_model"
    ]:
        failures.append("candidate_position_row_count_mismatch")
    if not position_preflight.get("passed"):
        failures.append("candidate_position_preflight_failed")
    if (
        claims["candidate_position_robustness"].get("expected")
        != position_observed
    ):
        failures.append("candidate_position_robustness_claim_drift")
    checks.append("candidate_position_robustness_is_prospectively_pinned")

    top2_root = repository / "work/candidate_top2_robustness_v1"
    top2_protocol = _read_json(top2_root / "protocol.json")
    top2_analysis = _read_json(
        top2_root
        / "results/qwen36_35b_a3b/candidate_top2_analysis.json"
    )
    top2_rows = _read_jsonl(
        top2_root / "results/qwen36_35b_a3b/candidate_top2_rows.jsonl"
    )
    top2_groups = top2_analysis.get("groups", {})
    top2_selection = top2_analysis.get("selection", {})
    top2_orders = ("gold_first", "gold_last", "seeded_a", "seeded_b")

    def _top2_group(mention: str, order: str) -> Mapping[str, Any]:
        return top2_groups.get(f"{mention}:{order}", {})

    top2_candidate_counts = [
        _top2_group(mention, order).get("mean_catalog_candidates")
        for mention in ("gold_early", "gold_late")
        for order in top2_orders
    ]
    top2_observed = {
        "cases": top2_protocol.get("cases"),
        "families": top2_protocol.get("families"),
        "rows": top2_analysis.get("rows"),
        "retries": top2_protocol.get("decoding", {}).get("retries"),
        "mean_catalog_candidates": (
            top2_candidate_counts[0] if top2_candidate_counts else None
        ),
        "mean_prompt_tokens": (
            sum(int(row["prompt_tokens"]) for row in top2_rows) / len(top2_rows)
            if top2_rows
            else None
        ),
        "mean_input_bytes": (
            sum(int(row["input_bytes"]) for row in top2_rows) / len(top2_rows)
            if top2_rows
            else None
        ),
        "gold_early_catalog_complete": [
            _top2_group("gold_early", order).get("gold_catalog_complete_rate")
            for order in top2_orders
        ],
        "gold_late_catalog_complete": [
            _top2_group("gold_late", order).get("gold_catalog_complete_rate")
            for order in top2_orders
        ],
        "gold_early_recall": [
            _top2_group("gold_early", order).get("exact_binding_recall")
            for order in top2_orders
        ],
        "gold_late_recall": [
            _top2_group("gold_late", order).get("exact_binding_recall")
            for order in top2_orders
        ],
        "gold_late_precision": [
            _top2_group("gold_late", order).get(
                "accepted_exact_binding_precision"
            )
            for order in top2_orders
        ],
        "gold_late_release": [
            _top2_group("gold_late", order).get("release_rate")
            for order in top2_orders
        ],
        "all_permutations_exact_rate": top2_selection.get(
            "all_permutations_exact_rate"
        ),
        "outcome_consistency_rate": top2_selection.get(
            "outcome_consistency_rate"
        ),
        "first_index_selection_rate": top2_selection.get(
            "first_index_selection_rate"
        ),
        "last_index_selection_rate": top2_selection.get(
            "last_index_selection_rate"
        ),
    }
    if len(set(top2_candidate_counts)) != 1:
        failures.append("candidate_top2_catalog_size_drift")
    if not (top2_root / "COMPLETED").is_file():
        failures.append("candidate_top2_completion_marker_missing")
    if claims["candidate_top2_robustness"].get("expected") != top2_observed:
        failures.append("candidate_top2_robustness_claim_drift")
    checks.append("candidate_top2_alternative_retention_is_prospectively_pinned")

    representative: dict[str, Any] = {}
    for model_key in ("qwen3_1_7b", "gemma4_e4b_it", "qwen36_35b_a3b"):
        sweep = _read_json(
            repository
            / f"work/oracle_selector_development_v2/{model_key}/sweep_analysis.json"
        ).get("conditions", {})
        routing = _read_json(
            repository
            / f"work/oracle_selector_development_v2/{model_key}/routing_compare_analysis.json"
        ).get("conditions", {})

        def exact(rows: Mapping[str, Any], condition: str) -> Any:
            return rows.get(condition, {}).get("exact_critical_call_recall")

        representative[model_key] = {
            "gold_dynamic": exact(sweep, "dynamic_enum_binding_only_oracle_0"),
            "gold_indexed": exact(sweep, "indexed_tool_binding_only_oracle_0"),
            "gold_json": exact(sweep, "indexed_json_binding_only_oracle_0"),
            "plus_1_indexed": exact(sweep, "indexed_tool_binding_only_oracle_1"),
            "plus_7_indexed": exact(sweep, "indexed_tool_binding_only_oracle_7"),
            "actual_indexed": exact(sweep, "indexed_tool_binding_only_actual"),
            "full_dynamic": exact(sweep, "dynamic_enum_full_oracle_0"),
            "full_indexed": exact(sweep, "indexed_tool_full_oracle_0"),
            "full_json": exact(sweep, "indexed_json_full_oracle_0"),
            "two_stage_indexed": exact(
                routing, "indexed_tool_two_stage_oracle_0"
            ),
        }
        if model_key == "qwen3_1_7b":
            representative["cases"] = sweep.get(
                "dynamic_enum_full_oracle_0", {}
            ).get("rows")
            representative["call_cases"] = sweep.get(
                "dynamic_enum_binding_only_oracle_0", {}
            ).get("rows")
    if claims["representative_selector"].get("expected") != representative:
        failures.append("representative_selector_claim_drift")
    checks.append("representative_selector_models_and_interfaces_are_pinned")

    equal_value = _read_json(
        repository / "work/equal_value_provenance_v1/equal_value_analysis.json"
    )
    equal_observed = {
        "pairs": equal_value.get("pair_count"),
        "patterns": len(equal_value.get("patterns", [])),
        "value_only_joint": equal_value.get("methods", {})
        .get("value_only", {})
        .get("joint_soundness_completeness"),
        "typed_joint": equal_value.get("methods", {})
        .get("typed_reconstruction", {})
        .get("joint_soundness_completeness"),
        "cite_and_check_joint": equal_value.get("methods", {})
        .get("cite_and_check", {})
        .get("joint_soundness_completeness"),
        "evibind_joint": equal_value.get("methods", {})
        .get("evibind", {})
        .get("joint_soundness_completeness"),
    }
    if claims["equal_value_provenance"].get("expected") != equal_observed:
        failures.append("equal_value_provenance_claim_drift")
    checks.append("equal_value_provenance_pairs_are_pinned")

    adversarial = _read_json(repository / "work/adversarial_boundary_v1/analysis.json")
    attacks = adversarial.get("separation", {}).get("attacks", {})
    normalization = attacks.get("normalization_and_citation_gaming", {})
    toctou = attacks.get("state_toctou", {})
    blocked = adversarial.get("separation", {}).get("blocked_action_cost", {})
    adversarial_observed = {
        "cases_per_attack": adversarial.get("separation", {}).get("cases_per_attack"),
        "normalization_value_only_exploitable": normalization.get("value_only_exploitable"),
        "normalization_atomic_cite_exploitable": normalization.get(
            "atomic_cite_and_check_exploitable"
        ),
        "normalization_evibind_exploitable": normalization.get("evibind_exploitable"),
        "toctou_value_only_exploitable": toctou.get("value_only_exploitable"),
        "toctou_nonatomic_cite_exploitable": toctou.get(
            "authenticated_cite_and_check_exploitable"
        ),
        "toctou_atomic_cite_exploitable": toctou.get(
            "atomic_cite_and_check_exploitable"
        ),
        "toctou_evibind_exploitable": toctou.get("evibind_exploitable"),
        "blocked_cases": blocked.get("cases"),
        "posthoc_model_calls": blocked.get("posthoc_model_calls"),
        "evibind_model_calls": blocked.get("evibind_model_calls"),
    }
    if claims["adversarial_separation"].get("expected") != adversarial_observed:
        failures.append("adversarial_separation_claim_drift")
    checks.append("adversarial_cite_and_check_separation_is_pinned")

    effects = adversarial.get("executable_effects", {})
    effect_conditions = effects.get("conditions", {})
    native_effect = effect_conditions.get("native_literals", {})
    reject_cite_effect = effect_conditions.get(
        "reject_only_atomic_cite_and_check", {}
    )
    trace_cite_effect = effect_conditions.get(
        "trace_materializing_atomic_cite_and_check", {}
    )
    evibind_effect = effect_conditions.get("evibind", {})
    effects_observed = {
        "scenarios": effects.get("scenario_count"),
        "effect_kinds": len(effects.get("effect_kinds", [])),
        "native_harm": native_effect.get("harm"),
        "native_completed": native_effect.get("completed"),
        "reject_only_cite_and_check_harm": reject_cite_effect.get("harm"),
        "reject_only_cite_and_check_completed": reject_cite_effect.get("completed"),
        "reject_only_cite_and_check_rejected": reject_cite_effect.get("rejected"),
        "trace_cite_and_check_harm": trace_cite_effect.get("harm"),
        "trace_cite_and_check_completed": trace_cite_effect.get("completed"),
        "trace_cite_and_check_rejected": trace_cite_effect.get("rejected"),
        "evibind_harm": evibind_effect.get("harm"),
        "evibind_completed": evibind_effect.get("completed"),
        "evibind_rejected": evibind_effect.get("rejected"),
    }
    if claims["executable_effects"].get("expected") != effects_observed:
        failures.append("executable_effects_claim_drift")
    checks.append("sandboxed_executable_effects_are_pinned")

    fragility = _read_json(
        repository / "work/implementation_fragility_v2/analysis.json"
    )
    fragility_classes = fragility.get("classes", {})
    redundancy = fragility_classes.get(
        "redundant_literal_trace_coherence", {}
    )
    cite_redundancy = redundancy.get(
        "trace_materializing_cite_and_check", {}
    )
    evibind_redundancy = redundancy.get("evibind", {})
    shared = fragility_classes.get("shared_trusted_boundary", {})
    cite_shared = shared.get("trace_materializing_cite_and_check", {})
    evibind_shared = shared.get("evibind", {})
    fragility_observed = {
        "scenarios": fragility.get("scenario_count"),
        "effect_kinds": len(fragility.get("effect_kinds", [])),
        "redundancy_variants": cite_redundancy.get("variants"),
        "cite_redundancy_exploitable_variants": cite_redundancy.get(
            "exploitable_variants"
        ),
        "cite_redundancy_harmful_executions": cite_redundancy.get(
            "harmful_executions"
        ),
        "evibind_redundancy_exploitable_variants": evibind_redundancy.get(
            "exploitable_variants"
        ),
        "evibind_redundancy_harmful_executions": evibind_redundancy.get(
            "harmful_executions"
        ),
        "evibind_redundancy_rejections": evibind_redundancy.get("rejections"),
        "shared_variants": cite_shared.get("variants"),
        "cite_shared_exploitable_variants": cite_shared.get(
            "exploitable_variants"
        ),
        "evibind_shared_exploitable_variants": evibind_shared.get(
            "exploitable_variants"
        ),
    }
    if claims["implementation_fragility"].get("expected") != fragility_observed:
        failures.append("implementation_fragility_claim_drift")
    checks.append("representation_specific_fragility_is_prospectively_pinned")

    boundarybench = _read_json(
        repository / "work/boundarybench_v1_package/manifest.json"
    ).get("suites", {})
    boundarybench_observed = {
        "originbench_cases": boundarybench.get("OriginBench-300", {}).get("cases"),
        "originbench_patterns": boundarybench.get("OriginBench-300", {}).get("patterns"),
        "checker_attack_cases": boundarybench.get("CheckerAttack-40", {}).get("cases"),
        "effect_scenarios": boundarybench.get("EffectSuite-30", {}).get("scenarios"),
        "effect_kinds": len(
            boundarybench.get("EffectSuite-30", {}).get("effect_kinds", [])
        ),
        "fragility_scenarios": boundarybench.get("Fragility-12", {}).get(
            "scenarios"
        ),
        "fragility_redundancy_variants": boundarybench.get(
            "Fragility-12", {}
        ).get("redundant_channel_variants"),
        "fragility_shared_controls": boundarybench.get("Fragility-12", {}).get(
            "shared_tcb_controls"
        ),
    }
    if claims["boundarybench_artifact"].get("expected") != boundarybench_observed:
        failures.append("boundarybench_artifact_claim_drift")
    checks.append("boundarybench_versioned_suite_counts_are_pinned")

    fuzz = _read_json(repository / "work/release_boundary_fuzz_v1/analysis.json")
    fuzz_observed = {
        "trials": fuzz.get("executed_trials"),
        "mutation_operators": fuzz.get("mutation_operators"),
        "unsound_releases": fuzz.get("unsound_releases"),
    }
    if claims["boundary_fuzz"].get("expected") != fuzz_observed:
        failures.append("boundary_fuzz_claim_drift")
    checks.append("million_trial_release_boundary_fuzz_is_pinned")

    public_key_observed = {
        "algorithm": "ed25519",
        "public_key_bytes": 32,
        "public_only_verification": (
            "class Ed25519HandleVerifier" in (repository / "evibind/core/public_key.py").read_text(encoding="utf-8")
            and "test_public_key_witness_replays_with_public_key_only"
            in (repository / "tests/test_evibind_core.py").read_text(encoding="utf-8")
        ),
    }
    if claims["public_key_auditability"].get("expected") != public_key_observed:
        failures.append("public_key_auditability_claim_drift")
    checks.append("ed25519_public_only_verification_is_implemented")

    sealed_root = repository / "work/evibench_proxy_20260802/evibench_human_study_handoff/work/simulated_powered_replay_v7_six_model"
    sealed = _read_json(sealed_root / "analysis.json")
    matrix = sealed.get("matrix", {})
    matrix_expected = claims["seven_condition_matrix"].get("expected", {})
    matrix_observed = {
        "cells": matrix.get("production_cells"),
        "cases": matrix.get("case_count"),
        "families": matrix.get("family_count"),
        "conditions": matrix.get("condition_count"),
        "models": matrix.get("model_count"),
        "seeds": matrix.get("seed_count"),
    }
    if matrix_expected != matrix_observed:
        failures.append("seven_condition_matrix_claim_drift")
    checks.append("sealed_matrix_dimensions_are_pinned")

    replay = _read_json(repository / "work/critical_only_scorer_replay_v1/analysis.json")
    replay_models = replay.get("by_model", {})
    qwen_full = replay_models.get("qwen3_1_7b", {}).get("conditions", {}).get(
        "evibind_full", {}
    )
    gpt_full = replay_models.get("gpt_oss_120b_reference", {}).get(
        "conditions", {}
    ).get("evibind_full", {})
    full_observed = {
        "rows": replay.get("rows"),
        "raw_responses_changed": replay.get("raw_responses_changed"),
        "zero_release_models": sum(
            int(row.get("conditions", {}).get("evibind_full", {}).get("accepted_calls", 0))
            == 0
            for row in replay_models.values()
        ),
        "full_path_releases": sum(
            int(row.get("conditions", {}).get("evibind_full", {}).get("accepted_calls", 0))
            for row in replay_models.values()
        ),
        "qwen_critical_precision": qwen_full.get(
            "accepted_call_exact_critical_precision"
        ),
        "qwen_critical_recall": qwen_full.get("exact_critical_call_recall"),
        "qwen_unsupported_ucb95": qwen_full.get("unsupported_critical_rate_ucb95"),
        "gpt_critical_precision": gpt_full.get(
            "accepted_call_exact_critical_precision"
        ),
        "gpt_critical_recall": gpt_full.get("exact_critical_call_recall"),
        "gpt_unsupported_ucb95": gpt_full.get("unsupported_critical_rate_ucb95"),
    }
    full_expected = claims["critical_scorer_replay"].get("expected", {})
    if set(full_expected) != set(full_observed) or any(
        not _approx(full_expected[key], full_observed[key]) for key in full_observed
    ):
        failures.append("critical_scorer_replay_claim_drift")
    checks.append("corrected_critical_scorer_replay_is_pinned")

    parser = sealed.get("parser_recovery_audit", {})
    failure_observed = {
        "length_stops": parser.get("production_length_stops"),
        "raw_length_tool_serializations": parser.get("production_raw_length_tool_serializations"),
        "parser_fallbacks": parser.get("runtime_parser_fallbacks"),
        "nonlimit_parser_fallbacks": parser.get("runtime_nonlimit_parser_fallbacks"),
        "gpt_wire_affected_cells": sealed.get("gpt_orphan_wire_audit", {}).get("affected_gpt_cells"),
    }
    if claims["failure_accounting"].get("expected") != failure_observed:
        failures.append("failure_accounting_claim_drift")
    checks.append("length_parser_and_wire_failures_are_pinned")

    catalog = oracle.get("catalog_candidates", {})
    scaling_observed = {
        "mean_candidates": catalog.get("mean"),
        "median_candidates": catalog.get("median"),
        "p95_candidates": catalog.get("p95"),
        "maximum_candidates": catalog.get("max"),
    }
    if claims["catalog_scaling"].get("expected") != scaling_observed:
        failures.append("catalog_scaling_claim_drift")
    checks.append("catalog_scaling_is_pinned")

    toolsandbox = _read_json(
        repository / "work/evibind_toolsandbox_v1/analysis/confirmatory/analysis.json"
    )
    macro = toolsandbox.get("macro_average", {})
    toolsandbox_observed = {
        "rows": toolsandbox.get("row_count"),
        "models": toolsandbox.get("model_count"),
        "macro_similarity_delta": macro.get("official_similarity", {}).get(
            "delta_evibind_minus_native"
        ),
        "macro_exception_delta": macro.get("tool_call_exception", {}).get(
            "delta_evibind_minus_native"
        ),
        "macro_minefield_delta": macro.get("minefield_activation", {}).get(
            "delta_evibind_minus_native"
        ),
    }
    if claims["toolsandbox_tradeoff"].get("expected") != toolsandbox_observed:
        failures.append("toolsandbox_tradeoff_claim_drift")
    checks.append("prospectively_frozen_stateful_tradeoff_is_pinned")

    test_report = _read_json(repository / "work/reviewer_revision_boundary_tests.json")
    test_count = int(test_report.get("passed", -1))
    expected_tests = claims["boundary_regressions"].get("expected", {}).get("focused_tests_passed")
    if test_count != expected_tests:
        failures.append(f"boundary_test_count:{test_count}!={expected_tests}")
    checks.append("boundary_regression_inventory_is_pinned")

    manifest = _read_json(repository / "work/reviewer_revision_manifest.json")
    for raw, expected in manifest.get("files", {}).items():
        if raw == "frozen_test_cases.jsonl":
            path = repository / "work/evibench_proxy_20260802/evibench_human_study_handoff/work/evibench_powered_v1/test_cases.jsonl"
        elif raw == "sealed_six_model_analysis.json":
            path = sealed_root / "analysis.json"
        else:
            path = repository / raw
        if not path.is_file() or _digest(path) != expected:
            failures.append(f"reproducibility_digest_drift:{raw}")
    checks.append("reproducibility_manifest_digests_match")

    forbidden = (
        "guarantees perfect empirical precision",
        "official BFCL overall improvement",
        "certificate proves intent",
    )
    if any(phrase.lower() in tex.lower() for phrase in forbidden):
        failures.append("known_overclaim_phrase")
    if "TODO" in tex or "TBD" in tex:
        failures.append("unresolved_placeholder")
    checks.append("known_overclaims_and_placeholders_absent")

    report = {
        "version": PAPER_AUDIT_VERSION,
        "passed": not failures,
        "failures": failures,
        "checks": checks,
        "check_count": len(checks),
        "claim_count": len(claims),
        "citation_count": len(cited),
        "main_text_end_page": main_end_page,
        "oracle_decomposition": oracle_observed,
        "mixed_compiler": mixed_observed,
        "verified_ranker": ranker_observed,
        "selector_phase": selector_observed,
        "fresh_family_confirmatory": fresh_observed,
        "fresh_family_saturation": saturation_observed,
        "candidate_position_robustness": position_observed,
        "candidate_top2_robustness": top2_observed,
        "representative_selector": representative,
        "equal_value_provenance": equal_observed,
        "adversarial_separation": adversarial_observed,
        "executable_effects": effects_observed,
        "implementation_fragility": fragility_observed,
        "boundarybench_artifact": boundarybench_observed,
        "boundary_fuzz": fuzz_observed,
        "public_key_auditability": public_key_observed,
        "matrix": matrix_observed,
        "full_path": full_observed,
        "failure_accounting": failure_observed,
        "catalog_scaling": scaling_observed,
        "toolsandbox_tradeoff": toolsandbox_observed,
        "boundary_test_inventory": test_count,
    }
    if failures:
        raise PaperAuditError("; ".join(failures))
    return report


def main() -> int:
    print(json.dumps(audit_paper(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
