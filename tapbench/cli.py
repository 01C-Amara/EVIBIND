from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze import export_scores_csv, run_lme4
from .alpha import fragmentation_stats
from .artifacts import write_artifact_manifest
from .bfcl import convert_bfcl_cases, score_bfcl_files
from .bfcl_official import evaluate_bfcl_official
from .bestof import run_best_of_n
from .calibration import evaluate_calibrator
from .committee import evaluate_committee_files
from .config import DEFAULT_RUNTIME_PROJECTION, load_experiment_config
from .discipline import assert_coefficient_discipline, coefficient_discipline_failures
from .deployable_resolution import resolve_deployable_files
from .evidence_audit import build_evidence_audit, score_evidence_audit
from .fc_rewardbench import evaluate_fc_rewardbench
from .evidence_contract import build_candidate_lattice, build_pointer_action_schema
from .evibench import (
    MATCHED_COMPUTE_CONDITIONS,
    run_replay_files,
    write_frozen_suite,
)
from .evibench_freeze import write_powered_freeze_manifest
from .evibench_models import write_model_artifact_manifest
from .evibench_powered import POWERED_CONDITIONS, run_powered_replay_files
from .evibench_powered_runner import collect_powered_responses
from .evibench_readiness import audit_powered_readiness
from .evibench_study import write_policy_study_freeze
from .evibench_study import write_study_assignments, write_study_workload
from .external_runner import run_external_cases
from .resolution import diagnose_files
from .tapr import resolve_files
from .tapr_calibration import calibrate_files as calibrate_tap_r_files
from .generator import generate_cases
from .grammar import assert_conformance, run_conformance
from .io import read_jsonl, write_jsonl
from .retrieval import evaluate_retrieval
from .rotbench import convert_rotbench, score_rotbench
from .r1_report import write_r1_report
from .r1_uncertainty import write_r1_cluster_bootstrap
from .r2_cases import write_r2a_cases
from .r2_eval import evaluate_r2a_components, train_tier_b_from_cases
from .runner import run_cases
from .r2_model_runner import run_r2a_model_conditions
from .r2_model_report import write_r2a_model_report
from .r2_analysis import run_r2_lme4, write_r2_full_analysis
from .r2b_analysis import run_r2b_lme4, write_r2b_full_analysis
from .r2b import (
    R2B_CONDITIONS,
    run_r2b_model_conditions,
    score_r2b_files,
    write_r2b_cases,
    write_r2b_runtime_projection,
)
from .r2c import (
    R2C_CONDITIONS,
    run_r2c_model_conditions,
    score_r2c_files,
    write_r2c_cases,
)
from .runtime import write_runtime_projection
from .resolver_eval import evaluate_resolver
from .runtime_audit import write_runtime_dependency_audit
from .scoring import score_files
from .summarize import write_summary_tables



def _cmd_best_of_n(args: argparse.Namespace) -> int:
    manifest = run_best_of_n(
        args.cases,
        args.output,
        args.timings,
        args.manifest,
        endpoint=args.endpoint,
        n=args.n,
        model_id=args.model_id,
        model_artifact=args.model_artifact,
        quantization=args.quantization,
        chat_template=args.chat_template,
        grammar_engine=args.grammar_engine,
        thinking_mode=args.thinking_mode,
        reasoning_budget=args.reasoning_budget,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )
    print(json.dumps({"generation_count": manifest["generation_count"], "n": manifest["n"]}, sort_keys=True))
    return 0


def _cmd_bfcl_prepare(args: argparse.Namespace) -> int:
    categories = [item.strip() for item in args.categories.split(",") if item.strip()]
    count = convert_bfcl_cases(
        args.source_root,
        args.output,
        categories=categories,
        limit_per_category=args.limit_per_category,
        grid_id=args.grid_id,
        manifest_path=args.manifest,
    )
    print(json.dumps({"case_count": count, "output": args.output, "manifest": args.manifest}, sort_keys=True))
    return 0


def _cmd_bfcl_score(args: argparse.Namespace) -> int:
    count = score_bfcl_files(
        args.cases,
        args.predictions,
        args.output,
        slot_errors_path=args.slot_errors,
        summary_path=args.summary,
    )
    print(json.dumps({"score_rows": count, "output": args.output, "slot_errors": args.slot_errors, "summary": args.summary}, sort_keys=True))
    return 0



def _cmd_bfcl_official(args: argparse.Namespace) -> int:
    report = evaluate_bfcl_official(
        args.cases,
        args.predictions,
        args.bfcl_root,
        args.output_dir,
        source_commit=args.source_commit,
    )
    print(json.dumps({
        "case_count": report["case_count"],
        "prediction_count": report["prediction_count"],
        "groups": len(report["groups"]),
        "output_dir": args.output_dir,
    }, sort_keys=True))
    return 0


def _cmd_bfcl_committee(args: argparse.Namespace) -> int:
    risk_caps = [float(item.strip()) for item in args.risk_caps.split(",") if item.strip()]
    fixed_thresholds = [int(item.strip()) for item in args.fixed_thresholds.split(",") if item.strip()]
    report = evaluate_committee_files(
        args.cases,
        args.predictions,
        args.output_dir,
        risk_caps=risk_caps,
        fixed_thresholds=fixed_thresholds,
        split_modulus=args.split_modulus,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    print(json.dumps({
        "output_dir": args.output_dir,
        "operating_points": [
            {
                "policy_id": row["policy_id"],
                "vote_threshold": row["selected_vote_threshold"],
                "heldout": row["heldout"],
            }
            for row in report["operating_points"]
        ],
    }, sort_keys=True))
    return 0



def _cmd_rotbench_prepare(args: argparse.Namespace) -> int:
    environments = [
        item.strip() for item in args.environments.split(",") if item.strip()
    ]
    count = convert_rotbench(
        args.source_root,
        args.output,
        environments=environments,
        limit_per_environment=args.limit_per_environment,
    )
    print(json.dumps({"case_count": count, "output": args.output}, sort_keys=True))
    return 0


def _cmd_rotbench_score(args: argparse.Namespace) -> int:
    report = score_rotbench(
        args.cases,
        args.predictions,
        args.output,
        args.report,
    )
    print(json.dumps({
        "case_count": report["case_count"],
        "prediction_count": report["prediction_count"],
        "groups": len(report["groups"]),
        "report": args.report,
    }, sort_keys=True))
    return 0


def _cmd_r1_cluster_bootstrap(args: argparse.Namespace) -> int:
    report = write_r1_cluster_bootstrap(
        args.initial_scores,
        args.tapr_scores,
        args.output,
        replicates=args.replicates,
        seed=args.seed,
    )
    print(json.dumps({
        "output": args.output,
        "contrasts": {
            name: row["point_estimate"]
            for name, row in report["contrasts"].items()
        },
    }, sort_keys=True))
    return 0

def _cmd_runtime_audit(args: argparse.Namespace) -> int:
    report = write_runtime_dependency_audit(args.output)
    print(json.dumps({
        "legacy_deployable": report["legacy_r1_oracle_path"]["deployable_ready"],
        "evidence_bounded_deployable": report["evidence_bounded_path"]["deployable_ready"],
        "output": args.output,
    }, sort_keys=True))
    return 0

def _cmd_evibench_freeze(args: argparse.Namespace) -> int:
    manifest = write_frozen_suite(args.cases, args.manifest)
    print(json.dumps(manifest, sort_keys=True))
    return 0


def _cmd_evibench_replay(args: argparse.Namespace) -> int:
    conditions = tuple(
        value.strip() for value in args.conditions.split(",") if value.strip()
    )
    report = run_replay_files(
        args.cases,
        args.responses,
        args.records,
        args.report,
        conditions=conditions,
    )
    print(
        json.dumps(
            {
                "row_count": report["row_count"],
                "conditions": sorted(report["conditions"]),
                "report": args.report,
            },
            sort_keys=True,
        )
    )
    return 0


def _cmd_evibench_powered_freeze(args: argparse.Namespace) -> int:
    manifest = write_powered_freeze_manifest(
        cases_path=args.cases,
        policies_path=args.policies,
        annotations_path=args.annotations,
        adjudications_path=args.adjudications,
        study_metadata_path=args.study_metadata,
        preregistration_path=args.preregistration,
        manifest_path=args.manifest,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


def _cmd_evibench_powered_replay(args: argparse.Namespace) -> int:
    conditions = tuple(
        value.strip() for value in args.conditions.split(",") if value.strip()
    )
    report = run_powered_replay_files(
        args.cases,
        args.responses,
        args.records,
        args.report,
        conditions=conditions,
    )
    print(
        json.dumps(
            {
                "row_count": report["row_count"],
                "conditions": sorted(report["conditions"]),
                "report": args.report,
            },
            sort_keys=True,
        )
    )
    return 0


def _cmd_evibench_powered_run(args: argparse.Namespace) -> int:
    conditions = tuple(
        value.strip() for value in args.conditions.split(",") if value.strip()
    )
    api_key = None
    if args.api_key_env:
        import os

        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise ValueError(f"API key environment variable is unset: {args.api_key_env}")
    report = collect_powered_responses(
        root=args.root,
        cases_path=args.cases,
        output_path=args.responses,
        endpoint=args.endpoint,
        model_key=args.model_key,
        seed=args.seed,
        conditions=conditions,
        preflight_only=args.preflight_only,
        api_key=api_key,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


def _cmd_evibench_powered_model_freeze(args: argparse.Namespace) -> int:
    manifest = write_model_artifact_manifest(
        catalog_path=args.catalog,
        group_name=args.group,
        model_root=args.model_root,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "model_count": len(manifest["artifacts"]),
                "projection_sha256": manifest["projection_sha256"],
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


def _cmd_evibench_powered_readiness(args: argparse.Namespace) -> int:
    report = audit_powered_readiness(args.root)
    print(json.dumps(report, sort_keys=True))
    if not report["infrastructure_passed"]:
        return 2
    if args.require_run_ready and not report["outcome_generation_allowed"]:
        return 3
    return 0


def _cmd_evibench_study_plan(args: argparse.Namespace) -> int:
    report = write_study_workload(args.protocol, args.output)
    print(
        json.dumps(
            {
                "total_cases": report["corpus"]["total_cases"],
                "publication_cases": report["corpus"]["publication_cases"],
                "annotation_judgments": report["human_work"][
                    "annotation_judgments"
                ],
                "publication_model_calls": report["powered_compute"][
                    "publication_model_calls"
                ],
                "next_run": report["next_run"],
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


def _cmd_evibench_study_assign(args: argparse.Namespace) -> int:
    manifest = write_study_assignments(
        protocol_path=args.protocol,
        participants_path=args.participants,
        families_path=args.families,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "participants": manifest["counts"]["participants"],
                "families": manifest["counts"]["families"],
                "authoring_slots": manifest["counts"]["authoring_slots"],
                "annotation_judgments": manifest["counts"][
                    "annotation_judgments"
                ],
                "output_dir": args.output_dir,
            },
            sort_keys=True,
        )
    )
    return 0


def _cmd_evibench_policy_study_freeze(args: argparse.Namespace) -> int:
    report = write_policy_study_freeze(
        protocol_path=args.protocol,
        policy_tasks_path=args.policy_tasks,
        authoring_records_path=args.authoring_records,
        review_records_path=args.review_records,
        manifest_path=args.manifest,
    )
    print(
        json.dumps(
            {
                "families": report["counts"]["families"],
                "authoring_records": report["counts"]["authoring_records"],
                "review_records": report["counts"]["review_records"],
                "final_policy_projection_sha256": report[
                    "final_policy_projection_sha256"
                ],
                "manifest": args.manifest,
            },
            sort_keys=True,
        )
    )
    return 0


def _cmd_r2_generate(args: argparse.Namespace) -> int:
    count = write_r2a_cases(args.output, scope=args.scope)
    print(json.dumps({"case_count": count, "scope": args.scope, "output": args.output}, sort_keys=True))
    return 0


def _cmd_r2_train_verifier(args: argparse.Namespace) -> int:
    report = train_tier_b_from_cases(
        args.cases,
        args.output,
        target_precision=args.target_precision,
    )
    print(json.dumps({
        "rows": report["training"]["rows"],
        "threshold": report["operating_point"]["threshold"],
        "cross_validated": report["operating_point"]["cross_validated"],
        "output": args.output,
    }, sort_keys=True))
    return 0


def _cmd_r2_evaluate(args: argparse.Namespace) -> int:
    report = evaluate_r2a_components(
        args.cases,
        args.rows,
        args.report,
        preregistration_path=args.preregistration,
        tier_b_verifier_path=args.tier_b_verifier,
    )
    passed = bool(report["release_decision"]["passed"])
    print(json.dumps({
        "case_count": report["case_count"],
        "passed": passed,
        "gates": report["release_decision"]["gates"],
        "report": args.report,
    }, sort_keys=True))
    return 0 if passed or not args.require_pass else 2





def _cmd_r2_run(args: argparse.Namespace) -> int:
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    manifest = run_r2a_model_conditions(
        args.cases,
        args.output,
        args.timings,
        args.manifest,
        endpoint=args.endpoint,
        model_id=args.model_id,
        model_key=args.model_key,
        model_artifact=args.model_artifact,
        chat_template=args.chat_template,
        tier_b_verifier_path=args.tier_b_verifier,
        conditions=conditions,
        seeds=seeds,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_generations=args.max_generations,
    )
    print(json.dumps({
        "generation_count": manifest["generation_count"],
        "runner_errors": manifest["runner_errors"],
        "model_id": manifest["model_id"],
    }, sort_keys=True))
    return 0 if manifest["runner_errors"] == 0 else 3


def _cmd_r2_model_report(args: argparse.Namespace) -> int:
    report = write_r2a_model_report(
        args.scores,
        args.timings,
        args.predictions,
        args.discipline_failures,
        args.cases,
        args.output,
        expected_model_count=args.expected_models,
        expected_condition_count=args.expected_conditions,
        expected_seed_count=args.expected_seeds,
        context_window=args.context_window,
    )
    passed = bool(report["release_decision"]["passed"])
    print(json.dumps({
        "passed": passed,
        "gates": report["release_decision"]["gates"],
        "output": args.output,
    }, sort_keys=True))
    return 0 if passed or not args.require_pass else 2


def _cmd_r2_analyze_full(args: argparse.Namespace) -> int:
    report = write_r2_full_analysis(
        args.scores,
        args.slot_errors,
        args.timings,
        args.component_report,
        args.tier_b_verifier,
        args.release_report,
        args.output_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    glmm_output = None
    if args.run_r:
        glmm_output = str(run_r2_lme4(Path(args.output_dir) / "scores.csv", args.output_dir))
    print(json.dumps({
        "analysis_version": report["schema_version"],
        "score_rows": report["integrity"]["score_rows"],
        "output_dir": args.output_dir,
        "glmm_output": glmm_output,
    }, sort_keys=True))
    return 0



def _cmd_r2b_analyze_full(args: argparse.Namespace) -> int:
    report = write_r2b_full_analysis(
        args.scores,
        args.slot_errors,
        args.timings,
        args.release_report,
        args.output_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    glmm_output = None
    if args.run_r:
        glmm_output = str(
            run_r2b_lme4(Path(args.output_dir) / "scores.csv", args.output_dir)
        )
    print(json.dumps({
        "analysis_version": report["schema_version"],
        "score_rows": report["integrity"]["score_rows"],
        "production_gate_passed": report["production_gate_passed"],
        "output_dir": args.output_dir,
        "glmm_output": glmm_output,
    }, sort_keys=True))
    return 0


def _cmd_r2b_generate(args: argparse.Namespace) -> int:
    count = write_r2b_cases(args.output, scope=args.scope)
    print(json.dumps({"case_count": count, "scope": args.scope, "output": args.output}, sort_keys=True))
    return 0


def _cmd_r2b_run(args: argparse.Namespace) -> int:
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    manifest = run_r2b_model_conditions(
        args.cases,
        args.output,
        args.timings,
        args.manifest,
        endpoint=args.endpoint,
        model_id=args.model_id,
        model_key=args.model_key,
        model_artifact=args.model_artifact,
        chat_template=args.chat_template,
        tier_b_verifier_path=args.tier_b_verifier,
        conditions=conditions,
        seeds=seeds,
        max_tokens=args.max_tokens,
        max_generations=args.max_generations,
    )
    print(json.dumps({
        "generation_count": manifest["generation_count"],
        "actual_model_calls": manifest["actual_model_calls"],
        "runner_errors": manifest["runner_errors"],
        "model_id": manifest["model_id"],
    }, sort_keys=True))
    return 0 if manifest["runner_errors"] == 0 else 3


def _cmd_r2b_score(args: argparse.Namespace) -> int:
    report = score_r2b_files(
        args.cases,
        args.predictions,
        args.output,
        args.slot_errors,
        args.report,
        expected_model_count=args.expected_models,
        expected_condition_count=args.expected_conditions,
        expected_seed_count=args.expected_seeds,
    )
    passed = bool(report["release_decision"]["passed"])
    print(json.dumps({
        "score_count": report["score_count"],
        "passed": passed,
        "gates": report["release_decision"]["gates"],
        "report": args.report,
    }, sort_keys=True))
    return 0 if passed or not args.require_pass else 2



def _cmd_r2c_generate(args: argparse.Namespace) -> int:
    count = write_r2c_cases(args.output, scope=args.scope)
    print(json.dumps({
        "case_count": count,
        "scope": args.scope,
        "output": args.output,
    }, sort_keys=True))
    return 0


def _cmd_r2c_run(args: argparse.Namespace) -> int:
    conditions = [
        item.strip() for item in args.conditions.split(",") if item.strip()
    ]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    manifest = run_r2c_model_conditions(
        args.cases,
        args.output,
        args.timings,
        args.manifest,
        endpoint=args.endpoint,
        model_id=args.model_id,
        model_key=args.model_key,
        model_artifact=args.model_artifact,
        chat_template=args.chat_template,
        tier_b_verifier_path=args.tier_b_verifier,
        conditions=conditions,
        seeds=seeds,
        max_tokens=args.max_tokens,
        max_generations=args.max_generations,
    )
    print(json.dumps({
        "generation_count": manifest["generation_count"],
        "actual_model_calls": manifest["actual_model_calls"],
        "runner_errors": manifest["runner_errors"],
        "model_id": manifest["model_id"],
    }, sort_keys=True))
    return 0 if manifest["runner_errors"] == 0 else 3


def _cmd_r2c_score(args: argparse.Namespace) -> int:
    report = score_r2c_files(
        args.cases,
        args.predictions,
        args.output,
        args.slot_errors,
        args.report,
        expected_model_count=args.expected_models,
        expected_condition_count=args.expected_conditions,
        expected_seed_count=args.expected_seeds,
    )
    passed = bool(report["release_decision"]["passed"])
    print(json.dumps({
        "score_count": report["score_count"],
        "passed": passed,
        "gates": report["release_decision"]["gates"],
        "report": args.report,
    }, sort_keys=True))
    return 0 if passed or not args.require_pass else 2


def _cmd_r2b_project_runtime(args: argparse.Namespace) -> int:
    report = write_r2b_runtime_projection(
        args.timings,
        args.output,
        full_case_count=args.full_cases,
        full_seed_count=args.full_seeds,
    )
    print(json.dumps({
        "timing_rows": report["timing_rows"],
        "projected_full_p95_hours_serial": report["projected_full_p95_hours_serial"],
        "output": args.output,
    }, sort_keys=True))
    return 0


def _cmd_tap_r_deployable(args: argparse.Namespace) -> int:
    count = resolve_deployable_files(
        args.cases,
        args.predictions,
        args.output,
        diagnostics_path=args.diagnostics,
        reference_date=args.reference_date,
        timezone=args.timezone,
        candidate_seed=args.candidate_seed,
        budget=args.repair_budget,
        source_method=args.source_method,
        output_method=args.output_method,
        evidence_mode=args.evidence_mode,
        tier_b_verifier_artifact=args.tier_b_verifier,
    )
    print(json.dumps({"rows": count, "output": args.output, "diagnostics": args.diagnostics}, sort_keys=True))
    return 0

def _cmd_resolver_evaluate(args: argparse.Namespace) -> int:
    report = evaluate_resolver(
        args.cases,
        args.output,
        reference_date=args.reference_date,
        timezone=args.timezone,
        candidate_seed=args.candidate_seed,
        max_cases=args.max_cases,
    )
    print(json.dumps(report["summary"], sort_keys=True))
    return 0

def _cmd_evidence_build(args: argparse.Namespace) -> int:
    rows = []
    context = {"reference_date": args.reference_date, "timezone": args.timezone}
    for index, case in enumerate(read_jsonl(args.cases)):
        if args.max_cases is not None and index >= args.max_cases:
            break
        lattice = build_candidate_lattice(
            case.get("messages", []),
            case.get("tools", []),
            dialogue_state={},
            reference_context=context,
            candidate_seed=args.candidate_seed,
        )
        rows.append({"case_id": case["case_id"], "lattice": lattice, "pointer_schema": build_pointer_action_schema(lattice)})
    write_jsonl(args.output, rows)
    print(json.dumps({"cases": len(rows), "output": args.output}, sort_keys=True))
    return 0


def _cmd_evidence_audit_build(args: argparse.Namespace) -> int:
    report = build_evidence_audit(
        args.cases,
        args.ledger,
        args.blind,
        args.key,
        args.manifest,
        per_label=args.per_label,
        seed=args.seed,
        blind_b_path=args.blind_b,
        adjudication_path=args.adjudication,
    )
    print(json.dumps({"selected_total": report["selected_total"], "status": report["status"]}, sort_keys=True))
    return 0


def _cmd_evidence_audit_score(args: argparse.Namespace) -> int:
    report = score_evidence_audit(
        args.blind,
        args.key,
        args.output,
        blind_b_path=args.blind_b,
        adjudication_path=args.adjudication,
    )
    print(json.dumps({"labeled_rows": report["labeled_rows"], "coverage": report["coverage"], "status": report["status"]}, sort_keys=True))
    return 0


def _cmd_combine_jsonl(args: argparse.Namespace) -> int:
    inputs = [item.strip() for item in args.inputs.split(",") if item.strip()]
    rows = []
    for path in inputs:
        rows.extend(read_jsonl(path))
    write_jsonl(args.output, rows)
    print(json.dumps({"inputs": len(inputs), "rows": len(rows), "output": args.output}, sort_keys=True))
    return 0



def _cmd_fc_rewardbench(args: argparse.Namespace) -> int:
    report = evaluate_fc_rewardbench(args.arrow, args.output_dir)
    print(json.dumps({
        "pair_count": report["pair_count"],
        "strict_pair_accuracy": report["strict_pair_accuracy"],
        "tie_adjusted_pair_accuracy": report["tie_adjusted_pair_accuracy"],
        "output_dir": args.output_dir,
    }, sort_keys=True))
    return 0



def _cmd_external_run(args: argparse.Namespace) -> int:
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    manifest = run_external_cases(
        args.cases,
        args.output,
        args.timings,
        args.manifest,
        endpoint=args.endpoint,
        model_id=args.model_id,
        model_key=args.model_key,
        model_artifact=args.model_artifact,
        chat_template=args.chat_template,
        methods=methods,
        seeds=seeds,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_generations=args.max_generations,
    )
    print(json.dumps({
        "generation_count": manifest["generation_count"],
        "runner_errors": manifest["runner_errors"],
        "thinking_markers": manifest["thinking_markers"],
        "context_truncations": manifest["context_truncations"],
    }, sort_keys=True))
    return 0 if manifest["runner_errors"] == 0 else 3


def _cmd_generate(args: argparse.Namespace) -> int:
    grid_ids = args.grids.split(",") if args.grids else None
    count = generate_cases(
        scope=args.scope,
        output=args.output,
        grid_ids=grid_ids,
        runtime_projection_path=args.runtime_projection,
        artifact_manifest_path=args.artifact_manifest,
    )
    print(f"wrote {count} cases to {args.output}")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    count = score_files(args.cases, args.predictions, args.output, slot_errors_path=args.slot_errors)
    print(f"wrote {count} score rows to {args.output}")
    if args.slot_errors:
        print(f"wrote slot-error rows to {args.slot_errors}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    manifest = run_cases(
        cases_path=args.cases,
        output_path=args.output,
        timings_path=args.timings,
        manifest_path=args.manifest,
        backend=args.backend,
        endpoint=args.endpoint,
        methods=args.methods,
        models=args.models,
        seeds=args.seeds,
        max_generations=args.max_generations,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        model_artifact=args.model_artifact,
        quantization=args.quantization,
        chat_template=args.chat_template,
        grammar_engine=args.grammar_engine,
        diagnostics_path=args.diagnostics,
        thinking_mode=args.thinking_mode,
        reasoning_budget=args.reasoning_budget,
    )
    print(json.dumps({"generation_count": manifest["generation_count"], "dry_run": manifest["dry_run"]}, sort_keys=True))
    return 0


def _cmd_diagnose(args: argparse.Namespace) -> int:
    counts = diagnose_files(args.cases, args.predictions, args.validator_errors, args.evidence_ledger)
    print(json.dumps(counts, sort_keys=True))
    return 0


def _cmd_tap_r_resolve(args: argparse.Namespace) -> int:
    result = resolve_files(
        args.cases,
        args.predictions,
        args.output,
        args.iterations,
        args.scores,
        args.summary,
        repair_budget=args.repair_budget,
        source_method=args.source_method,
        output_method=args.output_method,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "summary"}, sort_keys=True))
    return 0


def _cmd_tap_r_calibrate(args: argparse.Namespace) -> int:
    result = calibrate_tap_r_files(
        args.cases,
        args.predictions,
        args.output,
        args.calibration_csv,
        args.scores,
        args.summary,
        args.report,
        target_precision=args.target_precision,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def _cmd_conformance(args: argparse.Namespace) -> int:
    rows = run_conformance()
    if args.output:
        write_jsonl(args.output, rows)
    failures = [row for row in rows if not row["passed"]]
    print(json.dumps({"checks": len(rows), "failures": len(failures)}, sort_keys=True))
    assert_conformance()
    return 0


def _cmd_project_runtime(args: argparse.Namespace) -> int:
    cfg = load_experiment_config()
    projection = write_runtime_projection(cfg.subgrids, args.timings, args.output)
    print(json.dumps({"planned_full_generations": projection["planned_full_generations"], "projected_full_p95_hours": projection["projected_full_p95_hours"]}, sort_keys=True))
    return 0


def _cmd_retrieval(args: argparse.Namespace) -> int:
    cases = read_jsonl(args.cases)
    rows = evaluate_retrieval(cases, k=args.k, arm=args.arm)
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} retrieval rows to {args.output}")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    report = evaluate_calibrator(args.scores, args.output_dir, target_precision=args.target_precision)
    print(json.dumps({"global_threshold": report["global_threshold"], "calibrator_version": report["calibrator_version"]}, sort_keys=True))
    return 0


def _cmd_r1_report(args: argparse.Namespace) -> int:
    report = write_r1_report(
        args.initial_scores,
        args.bestof_scores,
        args.tapr_summary,
        args.calibrated_summary,
        args.initial_timings,
        args.bestof_timings,
        args.tapr_iterations,
        args.output_json,
        args.output_csv,
    )
    print(json.dumps({"case_count": report["design"]["case_count"], "gates": report["gates_vs_one_pass_full_tap"]}, sort_keys=True))
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    payload = write_summary_tables(args.scores, args.output_dir, slot_errors_path=args.slot_errors)
    print(json.dumps({"n_scores": payload["n_scores"], "n_slot_errors": payload["n_slot_errors"], "output_dir": args.output_dir}, sort_keys=True))
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    csv_path = export_scores_csv(args.scores, output_dir)
    print(f"wrote {csv_path}")
    if args.run_r:
        try:
            coeffs = run_lme4(args.scores, output_dir)
        except RuntimeError as exc:
            print(json.dumps({"error": str(exc)}, sort_keys=True))
            return 1
        print(f"wrote {coeffs}")
    return 0


def _cmd_validate_run(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.scores)
    failures = coefficient_discipline_failures(rows)
    if args.output:
        write_jsonl(args.output, failures)
    print(json.dumps({"failures": len(failures)}, sort_keys=True))
    assert_coefficient_discipline(rows)
    return 0


def _cmd_artifact_manifest(args: argparse.Namespace) -> int:
    cfg = load_experiment_config()
    manifest = write_artifact_manifest(cfg.models, args.output, required_quantization=args.required_quantization)
    print(json.dumps({"main_coefficients_ready": manifest["main_coefficients_ready"], "eligible_model_count": manifest["eligible_model_count"]}, sort_keys=True))
    return 0


def _cmd_alpha_proxy(args: argparse.Namespace) -> int:
    report = fragmentation_stats(read_jsonl(args.cases))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote alpha proxy report to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tapbench")
    sub = parser.add_subparsers(dest="command", required=True)


    best_of = sub.add_parser("best-of-n", help="sample and inference-safely rerank n llama-server candidates per case")
    best_of.add_argument("--cases", required=True)
    best_of.add_argument("--output", required=True)
    best_of.add_argument("--timings", required=True)
    best_of.add_argument("--manifest", required=True)
    best_of.add_argument("--endpoint", default="http://127.0.0.1:8080")
    best_of.add_argument("--n", type=int, required=True)
    best_of.add_argument("--model-id", required=True)
    best_of.add_argument("--model-artifact", required=True)
    best_of.add_argument("--quantization", required=True)
    best_of.add_argument("--chat-template", required=True)
    best_of.add_argument("--grammar-engine", required=True)
    best_of.add_argument("--thinking-mode", choices=["off", "budget_128", "not_applicable"], default="off")
    best_of.add_argument("--reasoning-budget", type=int, default=0)
    best_of.add_argument("--max-tokens", type=int, default=256)
    best_of.add_argument("--temperature", type=float, default=0.2)
    best_of.add_argument("--seed", type=int, default=1)
    best_of.set_defaults(func=_cmd_best_of_n)

    bfcl_prepare = sub.add_parser("bfcl-prepare", help="convert frozen BFCL v4 data into external-anchor TAP cases")
    bfcl_prepare.add_argument("--source-root", required=True, help="BFCL data directory containing BFCL_v4_*.json files")
    bfcl_prepare.add_argument("--output", required=True)
    bfcl_prepare.add_argument("--manifest", required=True)
    bfcl_prepare.add_argument("--categories", default="irrelevance,simple_python")
    bfcl_prepare.add_argument("--limit-per-category", type=int, default=100)
    bfcl_prepare.add_argument("--grid-id", default="BFCL_v4_external_anchor")
    bfcl_prepare.set_defaults(func=_cmd_bfcl_prepare)

    bfcl_score = sub.add_parser("bfcl-score", help="score BFCL external-anchor predictions")
    bfcl_score.add_argument("--cases", required=True)
    bfcl_score.add_argument("--predictions", required=True)
    bfcl_score.add_argument("--output", required=True)
    bfcl_score.add_argument("--slot-errors", required=True)
    bfcl_score.add_argument("--summary", required=True)
    bfcl_score.set_defaults(func=_cmd_bfcl_score)

    bfcl_official = sub.add_parser("bfcl-official-evaluate", help="replay Action IR through the pinned official BFCL AST checker")
    bfcl_official.add_argument("--cases", required=True)
    bfcl_official.add_argument("--predictions", required=True)
    bfcl_official.add_argument("--bfcl-root", required=True)
    bfcl_official.add_argument("--output-dir", required=True)
    bfcl_official.add_argument("--source-commit", required=True)
    bfcl_official.set_defaults(func=_cmd_bfcl_official)

    bfcl_committee = sub.add_parser("bfcl-committee", help="evaluate a held-out TAP-R agreement committee")
    bfcl_committee.add_argument("--cases", required=True)
    bfcl_committee.add_argument("--predictions", required=True)
    bfcl_committee.add_argument("--output-dir", required=True)
    bfcl_committee.add_argument("--risk-caps", default="0.05,0.10")
    bfcl_committee.add_argument("--fixed-thresholds", default="6")
    bfcl_committee.add_argument("--split-modulus", type=int, default=5)
    bfcl_committee.add_argument("--bootstrap-replicates", type=int, default=20000)
    bfcl_committee.set_defaults(func=_cmd_bfcl_committee)

    rotbench_prepare = sub.add_parser("rotbench-prepare", help="convert pinned RoTBench first-turn noise conditions to Action IR cases")
    rotbench_prepare.add_argument("--source-root", required=True)
    rotbench_prepare.add_argument("--output", required=True)
    rotbench_prepare.add_argument("--environments", default="clean,slight,medium,heavy,union")
    rotbench_prepare.add_argument("--limit-per-environment", type=int)
    rotbench_prepare.set_defaults(func=_cmd_rotbench_prepare)

    rotbench_score = sub.add_parser("rotbench-score", help="score RoTBench tool selection, parameter identification, and content filling")
    rotbench_score.add_argument("--cases", required=True)
    rotbench_score.add_argument("--predictions", required=True)
    rotbench_score.add_argument("--output", required=True)
    rotbench_score.add_argument("--report", required=True)
    rotbench_score.set_defaults(func=_cmd_rotbench_score)

    r1_uncertainty = sub.add_parser("r1-cluster-bootstrap", help="paired case/model cluster bootstrap for R1 contrasts")
    r1_uncertainty.add_argument("--initial-scores", required=True)
    r1_uncertainty.add_argument("--tapr-scores", required=True)
    r1_uncertainty.add_argument("--output", required=True)
    r1_uncertainty.add_argument("--replicates", type=int, default=20000)
    r1_uncertainty.add_argument("--seed", type=int, default=20260710)
    r1_uncertainty.set_defaults(func=_cmd_r1_cluster_bootstrap)

    runtime_audit = sub.add_parser("audit-runtime-inputs", help="trace forbidden oracle fields in legacy and deployable resolution paths")
    runtime_audit.add_argument("--output", required=True)
    runtime_audit.set_defaults(func=_cmd_runtime_audit)

    evibench_freeze = sub.add_parser(
        "evibench-freeze",
        help="write the source-defined frozen EviBench v1 suite and manifest",
    )
    evibench_freeze.add_argument("--cases", required=True)
    evibench_freeze.add_argument("--manifest", required=True)
    evibench_freeze.set_defaults(func=_cmd_evibench_freeze)

    evibench_replay = sub.add_parser(
        "evibench-replay",
        help="score payload-bound one-call EviBench response artifacts",
    )
    evibench_replay.add_argument("--cases", required=True)
    evibench_replay.add_argument("--responses", required=True)
    evibench_replay.add_argument("--records", required=True)
    evibench_replay.add_argument("--report", required=True)
    evibench_replay.add_argument(
        "--conditions",
        default=",".join(MATCHED_COMPUTE_CONDITIONS),
    )
    evibench_replay.set_defaults(func=_cmd_evibench_replay)

    powered_freeze = sub.add_parser(
        "evibench-powered-freeze",
        help="validate and hash-pin the powered corpus and human evidence",
    )
    powered_freeze.add_argument("--cases", required=True)
    powered_freeze.add_argument("--policies", required=True)
    powered_freeze.add_argument("--annotations", required=True)
    powered_freeze.add_argument("--adjudications", required=True)
    powered_freeze.add_argument("--study-metadata", required=True)
    powered_freeze.add_argument(
        "--preregistration",
        default=(
            "configs/"
            "evibench_powered_extension_preregistration_v1.yaml"
        ),
    )
    powered_freeze.add_argument("--manifest", required=True)
    powered_freeze.set_defaults(func=_cmd_evibench_powered_freeze)

    powered_models = sub.add_parser(
        "evibench-powered-model-freeze",
        help="hash-pin required model artifacts without storing local paths",
    )
    powered_models.add_argument(
        "--catalog",
        default="tapbench/data/model_catalog.yaml",
    )
    powered_models.add_argument("--group", default="main_core")
    powered_models.add_argument("--model-root", required=True)
    powered_models.add_argument("--output", required=True)
    powered_models.set_defaults(func=_cmd_evibench_powered_model_freeze)

    powered_readiness = sub.add_parser(
        "evibench-powered-readiness",
        help="audit frozen infrastructure and block premature outcome runs",
    )
    powered_readiness.add_argument("--root", default=".")
    powered_readiness.add_argument("--require-run-ready", action="store_true")
    powered_readiness.set_defaults(func=_cmd_evibench_powered_readiness)

    powered_run = sub.add_parser(
        "evibench-powered-run",
        help="collect a fail-closed, resumable powered-study model/seed shard",
    )
    powered_run.add_argument("--root", default=".")
    powered_run.add_argument("--cases", required=True)
    powered_run.add_argument("--responses", required=True)
    powered_run.add_argument("--endpoint", default="http://127.0.0.1:8080")
    powered_run.add_argument("--model-key", required=True)
    powered_run.add_argument("--seed", required=True, type=int)
    powered_run.add_argument(
        "--conditions",
        default=",".join(POWERED_CONDITIONS),
    )
    powered_run.add_argument("--api-key-env")
    powered_run.add_argument("--preflight-only", action="store_true")
    powered_run.set_defaults(func=_cmd_evibench_powered_run)

    study_plan = sub.add_parser(
        "evibench-study-plan",
        help="validate and derive the frozen powered human-study workload",
    )
    study_plan.add_argument(
        "--protocol",
        default="configs/evibench_human_study_v1.yaml",
    )
    study_plan.add_argument("--output", required=True)
    study_plan.set_defaults(func=_cmd_evibench_study_plan)

    study_assign = sub.add_parser(
        "evibench-study-assign",
        help="validate staffing and emit blind role-specific study assignments",
    )
    study_assign.add_argument(
        "--protocol",
        default="configs/evibench_human_study_v1.yaml",
    )
    study_assign.add_argument("--participants", required=True)
    study_assign.add_argument("--families", required=True)
    study_assign.add_argument("--output-dir", required=True)
    study_assign.set_defaults(func=_cmd_evibench_study_assign)

    policy_study_freeze = sub.add_parser(
        "evibench-policy-study-freeze",
        help="validate and hash-freeze the completed policy-authoring study",
    )
    policy_study_freeze.add_argument(
        "--protocol",
        default="configs/evibench_human_study_v1.yaml",
    )
    policy_study_freeze.add_argument("--policy-tasks", required=True)
    policy_study_freeze.add_argument("--authoring-records", required=True)
    policy_study_freeze.add_argument("--review-records", required=True)
    policy_study_freeze.add_argument(
        "--manifest",
        default="work/evibench_powered_v1/policy_study_manifest.json",
    )
    policy_study_freeze.set_defaults(
        func=_cmd_evibench_policy_study_freeze
    )

    powered_replay = sub.add_parser(
        "evibench-powered-replay",
        help="score the seven payload-bound powered-study conditions",
    )
    powered_replay.add_argument("--cases", required=True)
    powered_replay.add_argument("--responses", required=True)
    powered_replay.add_argument("--records", required=True)
    powered_replay.add_argument("--report", required=True)
    powered_replay.add_argument(
        "--conditions",
        default=",".join(POWERED_CONDITIONS),
    )
    powered_replay.set_defaults(func=_cmd_evibench_powered_replay)

    r2_generate = sub.add_parser("r2-generate", help="generate the balanced R2-A component suite")
    r2_generate.add_argument("--scope", choices=["smoke", "pilot", "full"], required=True)
    r2_generate.add_argument("--output", required=True)
    r2_generate.set_defaults(func=_cmd_r2_generate)

    r2_train = sub.add_parser("r2-train-verifier", help="fit and freeze the family-disjoint Tier-B verifier")
    r2_train.add_argument("--cases", required=True)
    r2_train.add_argument("--output", required=True)
    r2_train.add_argument("--target-precision", type=float, default=0.99)
    r2_train.set_defaults(func=_cmd_r2_train_verifier)

    r2_evaluate = sub.add_parser("r2-evaluate", help="evaluate R2-A evidence components and preregistered release gates")
    r2_evaluate.add_argument("--cases", required=True)
    r2_evaluate.add_argument("--rows", required=True)
    r2_evaluate.add_argument("--report", required=True)
    r2_evaluate.add_argument("--preregistration", default="configs/r2_tep_preregistration.yaml")
    r2_evaluate.add_argument("--tier-b-verifier")
    r2_evaluate.add_argument("--require-pass", action="store_true")
    r2_evaluate.set_defaults(func=_cmd_r2_evaluate)


    r2_run = sub.add_parser("r2-run", help="run literal and pointer R2-A conditions through one llama.cpp model")
    r2_run.add_argument("--cases", required=True)
    r2_run.add_argument("--output", required=True)
    r2_run.add_argument("--timings", required=True)
    r2_run.add_argument("--manifest", required=True)
    r2_run.add_argument("--endpoint", default="http://127.0.0.1:18121")
    r2_run.add_argument("--model-id", required=True)
    r2_run.add_argument("--model-key", required=True)
    r2_run.add_argument("--model-artifact", required=True)
    r2_run.add_argument("--chat-template", required=True)
    r2_run.add_argument("--tier-b-verifier", required=True)
    r2_run.add_argument("--conditions", default="r2_literal_generation,r2_pointer_unrestricted,r2_pointer_tep_tier_ab")
    r2_run.add_argument("--seeds", default="1")
    r2_run.add_argument("--max-tokens", type=int, default=256)
    r2_run.add_argument("--temperature", type=float, default=0.0)
    r2_run.add_argument("--max-generations", type=int)
    r2_run.set_defaults(func=_cmd_r2_run)


    r2_report = sub.add_parser("r2-model-report", help="write the R2-A model integrity and directional release report")
    r2_report.add_argument("--cases", required=True)
    r2_report.add_argument("--scores", required=True)
    r2_report.add_argument("--timings", required=True)
    r2_report.add_argument("--predictions", required=True)
    r2_report.add_argument("--discipline-failures", required=True)
    r2_report.add_argument("--output", required=True)
    r2_report.add_argument("--expected-models", type=int, required=True)
    r2_report.add_argument("--expected-conditions", type=int, required=True)
    r2_report.add_argument("--expected-seeds", type=int, required=True)
    r2_report.add_argument("--context-window", type=int, default=8192)
    r2_report.add_argument("--require-pass", action="store_true")
    r2_report.set_defaults(func=_cmd_r2_model_report)

    r2_analysis = sub.add_parser("r2-analyze-full", help="analyze a release-gated R2-A full run")
    r2_analysis.add_argument("--scores", required=True)
    r2_analysis.add_argument("--slot-errors", required=True)
    r2_analysis.add_argument("--timings", required=True)
    r2_analysis.add_argument("--component-report", required=True)
    r2_analysis.add_argument("--tier-b-verifier", required=True)
    r2_analysis.add_argument("--release-report", required=True)
    r2_analysis.add_argument("--output-dir", required=True)
    r2_analysis.add_argument("--bootstrap-replicates", type=int, default=10000)
    r2_analysis.add_argument("--bootstrap-seed", type=int, default=20260713)
    r2_analysis.add_argument("--run-r", action="store_true")
    r2_analysis.set_defaults(func=_cmd_r2_analyze_full)

    r2b_analysis = sub.add_parser(
        "r2b-analyze-full",
        help="analyze a release-gated R2-B full run",
    )
    r2b_analysis.add_argument("--scores", required=True)
    r2b_analysis.add_argument("--slot-errors", required=True)
    r2b_analysis.add_argument("--timings", required=True)
    r2b_analysis.add_argument("--release-report", required=True)
    r2b_analysis.add_argument("--output-dir", required=True)
    r2b_analysis.add_argument("--bootstrap-replicates", type=int, default=10000)
    r2b_analysis.add_argument("--bootstrap-seed", type=int, default=20260715)
    r2b_analysis.add_argument("--run-r", action="store_true")
    r2b_analysis.set_defaults(func=_cmd_r2b_analyze_full)

    r2b_generate = sub.add_parser("r2b-generate", help="generate the held-out R2-B deployable open-world suite")
    r2b_generate.add_argument("--scope", choices=["smoke", "pilot", "full"], required=True)
    r2b_generate.add_argument("--output", required=True)
    r2b_generate.set_defaults(func=_cmd_r2b_generate)

    r2b_run = sub.add_parser("r2b-run", help="run preregistered R2-B controls and TAP-R variants through one llama.cpp model")
    r2b_run.add_argument("--cases", required=True)
    r2b_run.add_argument("--output", required=True)
    r2b_run.add_argument("--timings", required=True)
    r2b_run.add_argument("--manifest", required=True)
    r2b_run.add_argument("--endpoint", default="http://127.0.0.1:18121")
    r2b_run.add_argument("--model-id", required=True)
    r2b_run.add_argument("--model-key", required=True)
    r2b_run.add_argument("--model-artifact", required=True)
    r2b_run.add_argument("--chat-template", required=True)
    r2b_run.add_argument("--tier-b-verifier", required=True)
    r2b_run.add_argument("--conditions", default=",".join(R2B_CONDITIONS))
    r2b_run.add_argument("--seeds", default="1")
    r2b_run.add_argument("--max-tokens", type=int, default=384)
    r2b_run.add_argument("--max-generations", type=int)
    r2b_run.set_defaults(func=_cmd_r2b_run)

    r2b_score = sub.add_parser("r2b-score", help="score R2-B and enforce its pilot/full release gates")
    r2b_score.add_argument("--cases", required=True)
    r2b_score.add_argument("--predictions", required=True)
    r2b_score.add_argument("--output", required=True)
    r2b_score.add_argument("--slot-errors", required=True)
    r2b_score.add_argument("--report", required=True)
    r2b_score.add_argument("--expected-models", type=int)
    r2b_score.add_argument("--expected-conditions", type=int)
    r2b_score.add_argument("--expected-seeds", type=int)
    r2b_score.add_argument("--require-pass", action="store_true")
    r2b_score.set_defaults(func=_cmd_r2b_score)

    r2b_runtime = sub.add_parser("r2b-project-runtime", help="project serial full-run p95 wall time from the R2-B pilot")
    r2b_runtime.add_argument("--timings", required=True)
    r2b_runtime.add_argument("--output", required=True)
    r2b_runtime.add_argument("--full-cases", type=int, default=256)
    r2b_runtime.add_argument("--full-seeds", type=int, default=3)
    r2b_runtime.set_defaults(func=_cmd_r2b_project_runtime)

    r2c_generate = sub.add_parser(
        "r2c-generate",
        help="generate engineering or untouched R2-C effect-first cases",
    )
    r2c_generate.add_argument(
        "--scope",
        choices=["smoke", "pilot", "confirmation"],
        required=True,
    )
    r2c_generate.add_argument("--output", required=True)
    r2c_generate.set_defaults(func=_cmd_r2c_generate)

    r2c_run = sub.add_parser(
        "r2c-run",
        help="run R2-C baselines and effect-first locked variants",
    )
    r2c_run.add_argument("--cases", required=True)
    r2c_run.add_argument("--output", required=True)
    r2c_run.add_argument("--timings", required=True)
    r2c_run.add_argument("--manifest", required=True)
    r2c_run.add_argument("--endpoint", default="http://127.0.0.1:18121")
    r2c_run.add_argument("--model-id", required=True)
    r2c_run.add_argument("--model-key", required=True)
    r2c_run.add_argument("--model-artifact", required=True)
    r2c_run.add_argument("--chat-template", required=True)
    r2c_run.add_argument("--tier-b-verifier", required=True)
    r2c_run.add_argument("--conditions", default=",".join(R2C_CONDITIONS))
    r2c_run.add_argument("--seeds", default="1")
    r2c_run.add_argument("--max-tokens", type=int, default=128)
    r2c_run.add_argument("--max-generations", type=int)
    r2c_run.set_defaults(func=_cmd_r2c_run)

    r2c_score = sub.add_parser(
        "r2c-score",
        help="score R2-C and enforce release discipline",
    )
    r2c_score.add_argument("--cases", required=True)
    r2c_score.add_argument("--predictions", required=True)
    r2c_score.add_argument("--output", required=True)
    r2c_score.add_argument("--slot-errors", required=True)
    r2c_score.add_argument("--report", required=True)
    r2c_score.add_argument("--expected-models", type=int)
    r2c_score.add_argument("--expected-conditions", type=int)
    r2c_score.add_argument("--expected-seeds", type=int)
    r2c_score.add_argument("--require-pass", action="store_true")
    r2c_score.set_defaults(func=_cmd_r2c_score)

    deployable = sub.add_parser("tap-r-deployable", help="resolve one-pass predictions through non-oracle evidence certificates")
    deployable.add_argument("--cases", required=True)
    deployable.add_argument("--predictions", required=True)
    deployable.add_argument("--output", required=True)
    deployable.add_argument("--diagnostics", required=True)
    deployable.add_argument("--reference-date", default="2026-07-10")
    deployable.add_argument("--timezone", default="Europe/London")
    deployable.add_argument("--candidate-seed", type=int, default=17)
    deployable.add_argument("--repair-budget", type=int, default=2)
    deployable.add_argument("--source-method", default="full_tap_b2")
    deployable.add_argument("--output-method", default="tap_r_deployable")
    deployable.add_argument(
        "--evidence-mode",
        choices=["deterministic", "proposal_span_hybrid", "typed_programs", "typed_program_hybrid", "typed_programs_tier_ab", "typed_program_hybrid_tier_ab"],
        default="deterministic",
    )
    deployable.add_argument("--tier-b-verifier")
    deployable.set_defaults(func=_cmd_tap_r_deployable)

    resolver_eval = sub.add_parser("resolver-evaluate", help="score deployable candidate certificates with gold used offline only")
    resolver_eval.add_argument("--cases", required=True)
    resolver_eval.add_argument("--output", required=True)
    resolver_eval.add_argument("--reference-date", default="2026-07-10")
    resolver_eval.add_argument("--timezone", default="Europe/London")
    resolver_eval.add_argument("--candidate-seed", type=int, default=17)
    resolver_eval.add_argument("--max-cases", type=int)
    resolver_eval.set_defaults(func=_cmd_resolver_evaluate)

    evidence_lattice = sub.add_parser("evidence-build", help="build deployable provenance candidates and pointer domains")
    evidence_lattice.add_argument("--cases", required=True)
    evidence_lattice.add_argument("--output", required=True)
    evidence_lattice.add_argument("--reference-date", default="2026-07-10")
    evidence_lattice.add_argument("--timezone", default="Europe/London")
    evidence_lattice.add_argument("--candidate-seed", type=int, default=0)
    evidence_lattice.add_argument("--max-cases", type=int)
    evidence_lattice.set_defaults(func=_cmd_evidence_build)

    evidence_build = sub.add_parser("evidence-audit-build", help="build a blinded stratified slot-evidence annotation set")
    evidence_build.add_argument("--cases", required=True)
    evidence_build.add_argument("--ledger", required=True)
    evidence_build.add_argument("--blind", required=True, help="annotator A sheet")
    evidence_build.add_argument("--blind-b", help="independent annotator B sheet")
    evidence_build.add_argument("--adjudication", help="blank adjudication sheet")
    evidence_build.add_argument("--key", required=True)
    evidence_build.add_argument("--manifest", required=True)
    evidence_build.add_argument("--per-label", type=int, default=64)
    evidence_build.add_argument("--seed", type=int, default=17)
    evidence_build.set_defaults(func=_cmd_evidence_audit_build)

    evidence_score = sub.add_parser("evidence-audit-score", help="score completed blinded slot-evidence annotations")
    evidence_score.add_argument("--blind", required=True, help="completed annotator A sheet")
    evidence_score.add_argument("--blind-b", help="completed annotator B sheet")
    evidence_score.add_argument("--adjudication", help="completed adjudication sheet")
    evidence_score.add_argument("--key", required=True)
    evidence_score.add_argument("--output", required=True)
    evidence_score.set_defaults(func=_cmd_evidence_audit_score)

    combine = sub.add_parser("combine-jsonl", help="combine ordered JSONL artifacts without changing rows")
    combine.add_argument("--inputs", required=True, help="comma-separated input paths")
    combine.add_argument("--output", required=True)
    combine.set_defaults(func=_cmd_combine_jsonl)

    fc_reward = sub.add_parser("fc-rewardbench", help="rank FC-RewardBench preference pairs with the deployable TAP-R controller")
    fc_reward.add_argument("--arrow", required=True)
    fc_reward.add_argument("--output-dir", required=True)
    fc_reward.set_defaults(func=_cmd_fc_rewardbench)

    external_run = sub.add_parser("external-run", help="run arbitrary external-anchor Action IR cases through one llama.cpp artifact")
    external_run.add_argument("--cases", required=True)
    external_run.add_argument("--output", required=True)
    external_run.add_argument("--timings", required=True)
    external_run.add_argument("--manifest", required=True)
    external_run.add_argument("--endpoint", default="http://127.0.0.1:18141")
    external_run.add_argument("--model-id", required=True)
    external_run.add_argument("--model-key", required=True)
    external_run.add_argument("--model-artifact", required=True)
    external_run.add_argument("--chat-template", required=True)
    external_run.add_argument("--methods", default="prompt_few_shot,full_tap_b2")
    external_run.add_argument("--seeds", default="1")
    external_run.add_argument("--max-tokens", type=int, default=384)
    external_run.add_argument("--temperature", type=float, default=0.0)
    external_run.add_argument("--max-generations", type=int)
    external_run.set_defaults(func=_cmd_external_run)

    generate = sub.add_parser("generate", help="generate pilot or full synthetic cases")
    generate.add_argument("--scope", choices=["pilot", "full"], required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--grids", help="comma-separated grid ids")
    generate.add_argument("--runtime-projection", default=str(DEFAULT_RUNTIME_PROJECTION))
    generate.add_argument("--artifact-manifest", default="work/main_coefficients/artifact_manifest.yaml")
    generate.set_defaults(func=_cmd_generate)

    score = sub.add_parser("score", help="score predictions against cases")
    score.add_argument("--cases", required=True)
    score.add_argument("--predictions", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--slot-errors", help="optional JSONL output for slot-level failure rows")
    score.set_defaults(func=_cmd_score)

    run = sub.add_parser("run", help="produce predictions.jsonl from generated cases")
    run.add_argument("--cases", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--timings", required=True)
    run.add_argument("--manifest", required=True)
    run.add_argument("--backend", choices=["oracle", "llama-server", "hf-xgrammar", "hf-native"], default="oracle")
    run.add_argument("--endpoint", default="http://127.0.0.1:8080")
    run.add_argument("--methods", help="comma-separated method names")
    run.add_argument("--models", help="comma-separated model keys or model ids")
    run.add_argument("--seeds", default="1")
    run.add_argument("--max-generations", type=int)
    run.add_argument("--max-tokens", type=int, default=256)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--model-artifact", help="override recorded model_artifact for this run")
    run.add_argument("--quantization", help="override recorded quantization for this run")
    run.add_argument("--chat-template", help="override recorded chat_template for this run")
    run.add_argument("--grammar-engine", help="override recorded grammar_engine for this run")
    run.add_argument("--diagnostics", help="optional JSONL output for backend diagnostic metrics")
    run.add_argument("--thinking-mode", choices=["off", "budget_128", "not_applicable"], default="off")
    run.add_argument("--reasoning-budget", type=int)
    run.set_defaults(func=_cmd_run)

    diagnose = sub.add_parser("diagnose", help="emit TAP-R typed validator errors and slot evidence ledger")
    diagnose.add_argument("--cases", required=True)
    diagnose.add_argument("--predictions", required=True)
    diagnose.add_argument("--validator-errors", required=True)
    diagnose.add_argument("--evidence-ledger", required=True)
    diagnose.set_defaults(func=_cmd_diagnose)

    tap_r = sub.add_parser("tap-r-resolve", help="run bounded inference-safe TAP-R transitions over one-pass predictions")
    tap_r.add_argument("--cases", required=True)
    tap_r.add_argument("--predictions", required=True)
    tap_r.add_argument("--output", required=True)
    tap_r.add_argument("--iterations", required=True)
    tap_r.add_argument("--scores", required=True)
    tap_r.add_argument("--summary", required=True)
    tap_r.add_argument("--repair-budget", type=int, default=2)
    tap_r.add_argument("--source-method", default="full_tap_b2")
    tap_r.add_argument("--output-method", default="tap_r_no_calibrator")
    tap_r.set_defaults(func=_cmd_tap_r_resolve)

    tap_r_cal = sub.add_parser("tap-r-calibrate", help="apply family-disjoint TAP-R accept/clarify/escalate calibration")
    tap_r_cal.add_argument("--cases", required=True)
    tap_r_cal.add_argument("--predictions", required=True)
    tap_r_cal.add_argument("--output", required=True)
    tap_r_cal.add_argument("--calibration-csv", required=True)
    tap_r_cal.add_argument("--scores", required=True)
    tap_r_cal.add_argument("--summary", required=True)
    tap_r_cal.add_argument("--report", required=True)
    tap_r_cal.add_argument("--target-precision", type=float, default=0.95)
    tap_r_cal.set_defaults(func=_cmd_tap_r_calibrate)

    conformance = sub.add_parser("conformance", help="run EOS masking conformance checks")
    conformance.add_argument("--output")
    conformance.set_defaults(func=_cmd_conformance)

    runtime = sub.add_parser("project-runtime", help="write pilot runtime projection")
    runtime.add_argument("--timings", required=True)
    runtime.add_argument("--output", default=str(DEFAULT_RUNTIME_PROJECTION))
    runtime.set_defaults(func=_cmd_project_runtime)

    retrieval = sub.add_parser("retrieval", help="measure recall@k for a retriever arm")
    retrieval.add_argument("--cases", required=True)
    retrieval.add_argument("--output", required=True)
    retrieval.add_argument("--arm", choices=["none", "tfidf_char", "cheap_embedding"], default="tfidf_char")
    retrieval.add_argument("--k", type=int, default=8)
    retrieval.set_defaults(func=_cmd_retrieval)

    calibrate = sub.add_parser("calibrate", help="train/evaluate family-disjoint lightweight calibrator")
    calibrate.add_argument("--scores", required=True)
    calibrate.add_argument("--output-dir", required=True)
    calibrate.add_argument("--target-precision", type=float, default=0.95)
    calibrate.set_defaults(func=_cmd_calibrate)

    r1_report = sub.add_parser("r1-report", help="write the R1 matched-control table and provisional gate report")
    r1_report.add_argument("--initial-scores", required=True)
    r1_report.add_argument("--bestof-scores", required=True)
    r1_report.add_argument("--tapr-summary", required=True)
    r1_report.add_argument("--calibrated-summary", required=True)
    r1_report.add_argument("--initial-timings", required=True)
    r1_report.add_argument("--bestof-timings", required=True)
    r1_report.add_argument("--tapr-iterations", required=True)
    r1_report.add_argument("--output-json", required=True)
    r1_report.add_argument("--output-csv", required=True)
    r1_report.set_defaults(func=_cmd_r1_report)

    summarize = sub.add_parser("summarize", help="write paper-ready aggregate tables from scores and slot errors")
    summarize.add_argument("--scores", required=True)
    summarize.add_argument("--slot-errors")
    summarize.add_argument("--output-dir", required=True)
    summarize.set_defaults(func=_cmd_summarize)

    analyze = sub.add_parser("analyze", help="export scores for R/lme4 analysis")
    analyze.add_argument("--scores", required=True)
    analyze.add_argument("--output-dir", required=True)
    analyze.add_argument("--run-r", action="store_true")
    analyze.set_defaults(func=_cmd_analyze)

    validate = sub.add_parser("validate-run", help="fail if score rows mix backend or artifact discipline")
    validate.add_argument("--scores", required=True)
    validate.add_argument("--output")
    validate.set_defaults(func=_cmd_validate_run)

    artifacts = sub.add_parser("artifact-manifest", help="audit main-coefficient GGUF artifact availability and quantization")
    artifacts.add_argument("--output", default="work/main_coefficients/artifact_manifest.yaml")
    artifacts.add_argument("--required-quantization", default="Q4_K_M")
    artifacts.set_defaults(func=_cmd_artifact_manifest)

    alpha = sub.add_parser("alpha-proxy", help="compute predeclared tokenizer-fragmentation alpha proxy")
    alpha.add_argument("--cases", required=True)
    alpha.add_argument("--output", required=True)
    alpha.set_defaults(func=_cmd_alpha_proxy)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
