from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .families import FAMILIES, FamilySpec
from .io import write_jsonl


R2A_CASE_SCHEMA_VERSION = "tapbench.r2a_case.v1"
R2A_GRID_ID = "R2A_component_evaluation"
R2A_REFERENCE_DATE = date(2026, 7, 10)
R2A_STRATA = (
    "normalization",
    "negation",
    "state",
    "default",
    "unit",
    "list",
    "derivation",
    "same_type_role_counterfactual",
)


def _resolved_weekday(phrase: str) -> str:
    cleaned = phrase.casefold()
    if cleaned == "today":
        return R2A_REFERENCE_DATE.isoformat()
    if cleaned == "tomorrow":
        return (R2A_REFERENCE_DATE + timedelta(days=1)).isoformat()
    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    name = cleaned.removeprefix("next ")
    target = weekdays[name]
    delta = (target - R2A_REFERENCE_DATE.weekday()) % 7
    if cleaned.startswith("next ") or delta == 0:
        delta = delta or 7
    return (R2A_REFERENCE_DATE + timedelta(days=delta)).isoformat()


def _property(
    slot: str,
    json_type: str,
    *,
    role: str = "control",
    resolution_type: str = "normalizable",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "type": json_type,
        "description": f"Canonical action argument {slot}.",
        "x-ir-name": slot,
        "x-tap-slot-role": role,
        "x-tap-resolution-type": resolution_type,
        "x-tap-criticality": "high" if role in {"control", "identifier"} else "moderate",
        **extra,
    }


def _stratum_spec(family: FamilySpec, variant: int) -> dict[str, Any]:
    operation = family.call_tool.replace("_", " ")
    relative_dates = ("today", "tomorrow", "Monday", "next Tuesday", "Wednesday", "next Thursday", "Saturday", "next Sunday")
    if variant >= len(relative_dates):
        raise ValueError(f"invalid R2-A variant: {variant}")

    if variant % 5 == 0:
        negation_slot, negation = "notify_guests", "do not notify guests"
    elif variant % 5 == 1:
        negation_slot, negation = "notify_guests", "don't notify guests"
    elif variant % 5 == 2:
        negation_slot, negation = "notify_guests", "never notify guests"
    elif variant % 5 == 3:
        negation_slot, negation = "guest_notifications", "without guest notifications"
    else:
        negation_slot, negation = "notifications", "no notifications"

    units = (
        ("two", "kilometers", 2000.0, 2.0),
        ("five", "km", 5000.0, 5.0),
        ("300", "centimeters", 3.0, 300.0),
        ("4", "kilometers", 4000.0, 4.0),
        ("1", "kilometer", 1000.0, 1.0),
        ("twenty", "centimeters", 0.2, 20.0),
        ("250", "centimeters", 2.5, 250.0),
        ("3", "km", 3000.0, 3.0),
    )
    lists = (
        ["Alice", "Bob"],
        ["Alice", "Bob", "Carol"],
        ["Devon", "Emery", "Finley"],
        ["Grace", "Harper"],
        ["Indigo", "Jules", "Kai"],
        ["Lane", "Morgan"],
        ["Noor", "Oakley", "Parker"],
        ["Quinn", "River"],
    )
    intervals = (
        ("09:00", "10:30", 90),
        ("10:15", "11:00", 45),
        ("13:00", "15:30", 150),
        ("08:30", "09:00", 30),
        ("14:20", "16:00", 100),
        ("17:00", "17:45", 45),
        ("06:15", "08:15", 120),
        ("11:10", "12:40", 90),
    )
    correction_pairs = (
        ("Monday", "Tuesday"),
        ("today", "tomorrow"),
        ("Tuesday", "Wednesday"),
        ("2026-07-15", "2026-07-16"),
        ("Thursday", "Friday"),
        ("2026-07-18", "2026-07-19"),
        ("Saturday", "Sunday"),
        ("2026-07-20", "2026-07-21"),
    )

    specs: dict[str, dict[str, Any]] = {}
    phrase = relative_dates[variant]
    specs["normalization"] = {
        "slot": "date",
        "property": _property("date", "string"),
        "request": f"For the {operation} operation, set the date to {phrase}.",
        "gold": _resolved_weekday(phrase),
        "expected_ops": ["PARSE_DATE"],
        "unsupported_values": [],
    }
    specs["negation"] = {
        "slot": negation_slot,
        "property": _property(negation_slot, "boolean", resolution_type="enumerated"),
        "request": f"Run the {operation} operation, but {negation}.",
        "gold": False,
        "expected_ops": ["NEGATED_BOOL"],
        "unsupported_values": [True],
    }
    state_value = f"{family.name.upper()}-STATE-{variant + 1:03d}"
    specs["state"] = {
        "slot": "account_id",
        "property": _property("account_id", "string", role="identifier", resolution_type="referential"),
        "request": f"Run the {operation} operation using the currently verified account.",
        "gold": state_value,
        "expected_ops": ["STATE_REF"],
        "unsupported_values": [f"{family.name.upper()}-STALE-{variant + 1:03d}"],
        "dialogue_state": {
            "account_id": {"value": state_value, "version": variant + 2},
            "stale_account_id": {"value": f"{family.name.upper()}-STALE-{variant + 1:03d}", "version": 1},
        },
    }
    default_value = ("en-GB", "en-US", "fr-FR", "de-DE", "es-ES", "it-IT", "nl-NL", "sv-SE")[variant]
    specs["default"] = {
        "slot": "locale",
        "property": _property(
            "locale",
            "string",
            role="defaultable",
            resolution_type="defaultable",
            default=default_value,
        ),
        "request": f"Run the {operation} operation using its declared locale default.",
        "gold": default_value,
        "expected_ops": ["SCHEMA_DEFAULT"],
        "unsupported_values": [],
    }
    unit_number, source_unit, converted, raw_number = units[variant]
    specs["unit"] = {
        "slot": "distance_m",
        "property": _property("distance_m", "number", **{"x-tap-unit": "m"}),
        "request": f"For the {operation} operation, set the travel distance to {unit_number} {source_unit}.",
        "gold": converted,
        "expected_ops": ["CONVERT_UNIT"],
        "unsupported_values": [raw_number],
    }
    list_values = lists[variant]
    specs["list"] = {
        "slot": "recipients",
        "property": _property(
            "recipients",
            "array",
            **{"x-tap-list-cue": "recipients", "items": {"type": "string"}},
        ),
        "request": f"For the {operation} operation, set recipients to {', '.join(list_values[:-1])} and {list_values[-1]}.",
        "gold": list_values,
        "expected_ops": ["LIST"],
        "unsupported_values": [list(reversed(list_values))],
    }
    start, end, duration = intervals[variant]
    specs["derivation"] = {
        "slot": "duration_minutes",
        "property": _property(
            "duration_minutes",
            "integer",
            role="derived",
            **{"x-tap-derive": {"op": "duration_minutes"}},
        ),
        "request": f"For the {operation} operation, use a window from {start} to {end}.",
        "gold": duration,
        "expected_ops": ["DERIVE"],
        "unsupported_values": [int(start.split(":")[0]), int(end.split(":")[0])],
    }
    source, destination = correction_pairs[variant]
    gold_destination = destination if destination.startswith("2026-") else _resolved_weekday(destination)
    unsupported_source = source if source.startswith("2026-") else _resolved_weekday(source)
    specs["same_type_role_counterfactual"] = {
        "slot": "date",
        "property": _property("date", "string"),
        "request": f"Move the {family.name} date from {source} to {destination}.",
        "gold": gold_destination,
        "expected_ops": ["PARSE_DATE"],
        "unsupported_values": [unsupported_source],
        "superseded_values": [unsupported_source],
    }
    return specs


def _tool_catalog(family: FamilySpec, slot: str, prop: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    names = (family.call_tool, *family.distractor_tools[:3])
    tools = []
    aliases = {}
    for index, name in enumerate(names):
        aliases[name] = name
        tools.append({
            "name": name,
            "canonical_name": name,
            "description": (
                f"Use for explicit {family.name} requests that ask to {name.replace('_', ' ')}."
                if index == 0
                else f"Different {family.name} operation; do not use for {family.call_tool.replace('_', ' ')} requests."
            ),
            "parameters": {
                "type": "object",
                "properties": {slot: deepcopy(prop)},
                "required": [slot],
                "additionalProperties": False,
            },
        })
    return tools, aliases


def _case(family: FamilySpec, family_index: int, stratum: str, stratum_index: int, variant: int, scope: str) -> dict[str, Any]:
    spec = _stratum_spec(family, variant)[stratum]
    slot = str(spec["slot"])
    tools, aliases = _tool_catalog(family, slot, spec["property"])
    reference_context = {
        "reference_date": R2A_REFERENCE_DATE.isoformat(),
        "timezone": "Europe/London",
        "action_risk_budget": 0.05,
    }
    oracle = {
        "operator_stratum": stratum,
        "target_slot": slot,
        "gold_value": spec["gold"],
        "expected_program_ops": list(spec["expected_ops"]),
        "unsupported_values": list(spec.get("unsupported_values", [])),
        "superseded_values": list(spec.get("superseded_values", [])),
        "family_index": family_index,
        "stratum_index": stratum_index,
        "variant": variant,
    }
    return {
        "schema_version": R2A_CASE_SCHEMA_VERSION,
        "case_id": f"r2a_{family.name}_{stratum}_{variant:02d}",
        "hypothesis_grid_id": R2A_GRID_ID,
        "hypothesis": "R2A",
        "split": scope,
        "family": family.name,
        "task_kind": "call",
        "factors": {
            "N": 4,
            "q": 1,
            "d": 1,
            "e": 1,
            "sigma": "high",
            "alpha": "aligned",
            "operator_stratum": stratum,
            "variant": variant,
            "repair_budget": 2,
        },
        "messages": [
            {
                "role": "system",
                "content": "Return exactly one Action IR object. Never use quoted, negated, superseded, hypothetical, or stale values as active arguments.",
            },
            {"role": "user", "content": spec["request"]},
        ],
        "tools": tools,
        "tool_aliases": aliases,
        "argument_aliases": {slot: slot},
        "dialogue_state": deepcopy(spec.get("dialogue_state", {})),
        "reference_context": reference_context,
        "gold_action": {
            "mode": "call",
            "tool": family.call_tool,
            "arguments": {slot: spec["gold"]},
            "payload": {},
        },
        "derivable_values": {slot: spec["gold"]},
        "r2a_oracle": oracle,
        "metadata": {
            "backend_namespace": "llama_cpp_q4km_r2a",
            "coefficient_backend": "llama.cpp",
            "model_group": "main_core",
            "quantization": "Q4_K_M",
            "chat_template_regime": "per_model_frozen",
            "grammar_engine": "gbnf",
            "thinking_mode": "off",
            "reasoning_budget": 0,
            "repair_budget": 2,
            "runtime_allowed_fields": ["messages", "tools", "tool_aliases", "argument_aliases", "dialogue_state", "reference_context"],
            "offline_only_fields": ["gold_action", "derivable_values", "r2a_oracle", "task_kind"],
        },
    }


def generate_r2a_cases(*, scope: str = "pilot") -> list[dict[str, Any]]:
    if scope not in {"smoke", "pilot", "full"}:
        raise ValueError("R2-A scope must be smoke, pilot, or full")
    rows = [
        _case(family, family_index, stratum, stratum_index, variant, scope)
        for variant in range(8)
        for family_index, family in enumerate(FAMILIES)
        for stratum_index, stratum in enumerate(R2A_STRATA)
    ]
    if scope == "full":
        return rows
    if scope == "pilot":
        return [row for row in rows if int(row["factors"]["variant"]) < 2]
    return [
        row
        for row in rows
        if int(row["factors"]["variant"]) == 0
        and int(row["r2a_oracle"]["family_index"]) in {
            int(row["r2a_oracle"]["stratum_index"]) % len(FAMILIES),
            (int(row["r2a_oracle"]["stratum_index"]) + 4) % len(FAMILIES),
        }
    ]


def write_r2a_cases(output: str | Path, *, scope: str) -> int:
    return write_jsonl(output, generate_r2a_cases(scope=scope))
