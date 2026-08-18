from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from evibind.core import (
    ActionProposal,
    EvidenceTypeRegistry,
    TransformRegistry,
    materialize,
)
from evibind.core.materializer import MaterializationError
from tapbench.adversarial_boundary import (
    EffectScenario,
    _compiled_effect,
    build_effect_scenarios,
)


IMPLEMENTATION_FRAGILITY_VERSION = "evibind.implementation_fragility.v2"

# These mutations target only the redundant literal/trace coherence channel.
# They do not remove authentication, replay, destination scoping, or the sink
# sandbox shared by both boundaries.
REDUNDANCY_FAULTS = (
    "check_after_dispatch",
    "compare_cached_literal",
    "postcheck_literal_swap",
    "dispatch_literal_instead_of_trace",
    "mismatch_is_audit_only",
    "trace_failure_falls_back_to_literal",
    "exception_handler_dispatches_literal",
    "asynchronous_literal_sink_races_check",
)

# These deliberately violate assumptions shared by both sound boundaries.  They
# are included to prevent the experiment from being read as implementation-proof.
SHARED_TCB_FAULTS = (
    "authentication_bypassed",
    "replay_bypassed",
    "destination_scope_bypassed",
    "sink_sandbox_bypassed",
)


@dataclass(frozen=True)
class FragilityOutcome:
    scenario_id: str
    effect_kind: str
    boundary: str
    fault_class: str
    fault: str
    dispatched: str | None
    harmful: bool
    completed: bool
    rejected: bool


def _redundancy_fault_dispatch(
    *,
    boundary: str,
    fault: str,
    trace_value: str,
    emitted_literal: str,
    evibind_literal_injection_rejected: bool,
) -> str | None:
    if fault not in REDUNDANCY_FAULTS:
        raise ValueError(f"unknown redundancy fault: {fault}")
    if boundary == "trace_materializing_cite_and_check":
        # Every mutation represents a single path by which the redundant model
        # literal becomes executable after or instead of trace validation.
        return emitted_literal
    if boundary == "evibind":
        if not evibind_literal_injection_rejected:
            raise AssertionError("critical-literal injection reached EviBind dispatch")
        # The real materializer rejected the attempted critical literal before
        # dispatch, so the redundancy-channel mutation fails closed.
        return None
    raise ValueError(f"unknown boundary: {boundary}")


def _shared_fault_dispatch(
    *,
    boundary: str,
    fault: str,
    tampered_trace_value: str,
) -> str:
    if boundary not in {"trace_materializing_cite_and_check", "evibind"}:
        raise ValueError(f"unknown boundary: {boundary}")
    if fault not in SHARED_TCB_FAULTS:
        raise ValueError(f"unknown shared TCB fault: {fault}")
    # A shared trusted-boundary bypass can expose either representation.  This
    # cell is intentionally symmetric and bounds the claim made from the study.
    return tampered_trace_value


def _outcome(
    scenario: EffectScenario,
    *,
    boundary: str,
    fault_class: str,
    fault: str,
    dispatched: str | None,
) -> FragilityOutcome:
    return FragilityOutcome(
        scenario_id=scenario.scenario_id,
        effect_kind=scenario.kind,
        boundary=boundary,
        fault_class=fault_class,
        fault=fault,
        dispatched=dispatched,
        harmful=dispatched == scenario.injected,
        completed=dispatched == scenario.intended,
        rejected=dispatched is None,
    )


def _evibind_rejects_critical_literal(scenario: EffectScenario) -> bool:
    context, policy, issuer, table = _compiled_effect(scenario)
    accepted = list(table.candidates.values())
    if len(accepted) != 1 or accepted[0].value != scenario.intended:
        raise AssertionError("trusted compilation did not isolate the intended target")
    leaf = scenario.destination.removeprefix("/")
    try:
        materialize(
            ActionProposal(
                mode="call",
                tool_id=scenario.tool_id,
                bindings={scenario.destination: accepted[0].candidate_id},
                arguments={leaf: scenario.injected},
            ),
            table=table,
            context=context,
            policy=policy,
            evidence_types=EvidenceTypeRegistry.standard(),
            transforms=TransformRegistry.standard(),
            issuer=issuer,
        )
    except MaterializationError:
        return True
    return False


def _summarize(
    rows: Sequence[FragilityOutcome],
    *,
    faults: Sequence[str],
) -> dict[str, Any]:
    variants = []
    for fault in faults:
        group = [row for row in rows if row.fault == fault]
        variants.append(
            {
                "fault": fault,
                "cases": len(group),
                "harmful_executions": sum(row.harmful for row in group),
                "completed_executions": sum(row.completed for row in group),
                "rejections": sum(row.rejected for row in group),
                "exploitable": any(row.harmful for row in group),
            }
        )
    return {
        "variants": len(variants),
        "cases": sum(item["cases"] for item in variants),
        "exploitable_variants": sum(item["exploitable"] for item in variants),
        "harmful_executions": sum(item["harmful_executions"] for item in variants),
        "completed_executions": sum(
            item["completed_executions"] for item in variants
        ),
        "rejections": sum(item["rejections"] for item in variants),
        "by_variant": variants,
    }


def run_implementation_fragility(
    scenarios: Sequence[EffectScenario] | None = None,
) -> dict[str, Any]:
    cases = tuple(scenarios or build_effect_scenarios())
    boundaries = ("trace_materializing_cite_and_check", "evibind")
    rows: list[FragilityOutcome] = []
    for scenario in cases:
        evibind_literal_injection_rejected = _evibind_rejects_critical_literal(
            scenario
        )
        for boundary in boundaries:
            for fault in REDUNDANCY_FAULTS:
                rows.append(
                    _outcome(
                        scenario,
                        boundary=boundary,
                        fault_class="redundant_literal_trace_coherence",
                        fault=fault,
                        dispatched=_redundancy_fault_dispatch(
                            boundary=boundary,
                            fault=fault,
                            trace_value=scenario.intended,
                            emitted_literal=scenario.injected,
                            evibind_literal_injection_rejected=(
                                evibind_literal_injection_rejected
                            ),
                        ),
                    )
                )
            for fault in SHARED_TCB_FAULTS:
                rows.append(
                    _outcome(
                        scenario,
                        boundary=boundary,
                        fault_class="shared_trusted_boundary",
                        fault=fault,
                        dispatched=_shared_fault_dispatch(
                            boundary=boundary,
                            fault=fault,
                            tampered_trace_value=scenario.injected,
                        ),
                    )
                )

    report: dict[str, Any] = {
        "version": IMPLEMENTATION_FRAGILITY_VERSION,
        "scenario_count": len(cases),
        "effect_kinds": sorted({scenario.kind for scenario in cases}),
        "fault_model": {
            "single_fault": True,
            "redundancy_scope": (
                "Mutations affect only literal/trace coherence after a valid trace "
                "has been supplied; shared authentication, replay, destination "
                "scoping, and sink assumptions remain intact. EviBind outcomes "
                "execute the actual closed ActionIR materializer with an attempted "
                "critical-literal injection."
            ),
            "shared_scope": (
                "Control mutations deliberately violate assumptions common to both "
                "boundaries and therefore bound, rather than strengthen, the claim."
            ),
        },
        "classes": {},
        "rows": [asdict(row) for row in rows],
    }
    for fault_class, faults in (
        ("redundant_literal_trace_coherence", REDUNDANCY_FAULTS),
        ("shared_trusted_boundary", SHARED_TCB_FAULTS),
    ):
        report["classes"][fault_class] = {
            boundary: _summarize(
                [
                    row
                    for row in rows
                    if row.fault_class == fault_class and row.boundary == boundary
                ],
                faults=faults,
            )
            for boundary in boundaries
        }
    return report
