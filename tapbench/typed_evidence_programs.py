from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Iterable, Mapping


TEP_VERSION = "tapbench.typed_evidence_program.v3"
HYPERGRAPH_VERSION = "tapbench.evidence_hypergraph.v1"
MAX_PROGRAM_DEPTH = 3
ALLOWED_OPERATORS = {
    "COPY",
    "ENUM",
    "PARSE_DATE",
    "PARSE_TIME",
    "PARSE_NUMBER",
    "CONVERT_UNIT",
    "NEGATED_BOOL",
    "STATE_REF",
    "SCHEMA_DEFAULT",
    "LIST",
    "DERIVE",
    "CONTRACT_CONST",
}

_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
_CURRENCY_SYMBOLS = {
    "$": "USD",
    "£": "GBP",
    "€": "EUR",
    "¥": "JPY",
}
_UNIT_FACTORS = {
    "m": ("length", 1.0),
    "meter": ("length", 1.0),
    "meters": ("length", 1.0),
    "km": ("length", 1000.0),
    "kilometer": ("length", 1000.0),
    "kilometers": ("length", 1000.0),
    "cm": ("length", 0.01),
    "centimeter": ("length", 0.01),
    "centimeters": ("length", 0.01),
    "s": ("duration", 1.0),
    "second": ("duration", 1.0),
    "seconds": ("duration", 1.0),
    "min": ("duration", 60.0),
    "minute": ("duration", 60.0),
    "minutes": ("duration", 60.0),
    "h": ("duration", 3600.0),
    "hour": ("duration", 3600.0),
    "hours": ("duration", 3600.0),
}


@dataclass(frozen=True)
class TypedEvidenceProgram:
    op: str
    args: Mapping[str, Any] = field(default_factory=dict)
    output_type: str = "any"
    tier: str = "A"

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "args": _serialize(self.args),
            "output_type": self.output_type,
            "tier": self.tier,
            "tep_version": TEP_VERSION,
        }

    @property
    def program_id(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ProgramRisk:
    evidence: float = 0.0
    role: float = 0.0
    transform: float = 0.0
    contract: float = 0.0
    state_staleness: float = 0.0

    @property
    def upper_bound(self) -> float:
        return min(1.0, sum(asdict(self).values()))

    def to_dict(self) -> dict[str, float]:
        return {**asdict(self), "upper_bound": self.upper_bound}


@dataclass(frozen=True)
class ProgramExecution:
    program_id: str
    value: Any
    output_type: str
    trace: tuple[dict[str, Any], ...]
    risk: ProgramRisk
    accepted_tier: str
    valid: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["risk"] = self.risk.to_dict()
        row["trace"] = list(self.trace)
        row["execution_version"] = TEP_VERSION
        return row


def _serialize(value: Any) -> Any:
    if isinstance(value, TypedEvidenceProgram):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def _children(program: TypedEvidenceProgram) -> Iterable[TypedEvidenceProgram]:
    def visit(value: Any) -> Iterable[TypedEvidenceProgram]:
        if isinstance(value, TypedEvidenceProgram):
            yield value
        elif isinstance(value, Mapping):
            for item in value.values():
                yield from visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from visit(item)

    yield from visit(program.args)


def program_depth(program: TypedEvidenceProgram) -> int:
    children = list(_children(program))
    return 1 if not children else 1 + max(program_depth(child) for child in children)


def validate_program(program: TypedEvidenceProgram, *, max_depth: int = MAX_PROGRAM_DEPTH) -> None:
    if program.op not in ALLOWED_OPERATORS:
        raise ValueError(f"operator is not allowed: {program.op}")
    if program.tier not in {"A", "B", "C"}:
        raise ValueError(f"invalid acceptance tier: {program.tier}")
    if program_depth(program) > max_depth:
        raise ValueError(f"program depth exceeds {max_depth}")
    for child in _children(program):
        validate_program(child, max_depth=max_depth)


def _span_text(program: TypedEvidenceProgram, request: str) -> tuple[str, tuple[int, int]]:
    span = program.args.get("span")
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        raise ValueError(f"{program.op} requires a [start, end] span")
    start, end = int(span[0]), int(span[1])
    if start < 0 or end < start or end > len(request):
        raise ValueError(f"invalid request span: {(start, end)}")
    return request[start:end], (start, end)


def _reference_date(context: Mapping[str, Any]) -> date:
    raw = context.get("reference_date")
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if isinstance(raw, str):
        return date.fromisoformat(raw)
    raise ValueError("PARSE_DATE requires reference_context.reference_date")


def _parse_date(raw: str, context: Mapping[str, Any]) -> str:
    cleaned = raw.strip().lower()
    try:
        return date.fromisoformat(cleaned).isoformat()
    except ValueError:
        pass
    reference = _reference_date(context)
    if cleaned == "today":
        return reference.isoformat()
    if cleaned == "tomorrow":
        return (reference + timedelta(days=1)).isoformat()
    match = re.fullmatch(r"(?:(next)\s+)?([a-z]+)", cleaned)
    if match and match.group(2) in _WEEKDAYS:
        target = _WEEKDAYS[match.group(2)]
        delta = (target - reference.weekday()) % 7
        if match.group(1) or delta == 0:
            delta = delta or 7
        return (reference + timedelta(days=delta)).isoformat()
    raise ValueError(f"unsupported date expression: {raw!r}")


def _parse_time(raw: str) -> str:
    cleaned = raw.strip().lower().replace(".", "")
    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", cleaned)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        meridiem = match.group(3)
        if meridiem:
            if not 1 <= hour <= 12:
                raise ValueError("12-hour time is out of range")
            hour = hour % 12 + (12 if meridiem == "pm" else 0)
        if hour > 23 or minute > 59:
            raise ValueError("time is out of range")
        return f"{hour:02d}:{minute:02d}"
    words = "|".join(_NUMBER_WORDS)
    match = re.fullmatch(rf"half past ({words})", cleaned)
    if match and 0 <= _NUMBER_WORDS[match.group(1)] <= 23:
        return f"{_NUMBER_WORDS[match.group(1)]:02d}:30"
    raise ValueError(f"unsupported time expression: {raw!r}")


def _parse_number(raw: str) -> int | float:
    cleaned = raw.strip().lower().replace(",", "")
    magnitude = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\s*k", cleaned)
    if magnitude:
        value = float(magnitude.group(1)) * 1000
        return int(value) if value.is_integer() else value
    try:
        value = float(cleaned)
        return int(value) if value.is_integer() else value
    except ValueError:
        pass
    tokens = [token for token in re.split(r"[\s-]+", cleaned) if token and token != "and"]
    if not tokens or any(token not in _NUMBER_WORDS and token not in {"hundred", "thousand"} for token in tokens):
        raise ValueError(f"unsupported number expression: {raw!r}")
    total = current = 0
    for token in tokens:
        if token in _NUMBER_WORDS:
            current += _NUMBER_WORDS[token]
        elif token == "hundred":
            current = max(1, current) * 100
        elif token == "thousand":
            total += max(1, current) * 1000
            current = 0
    return total + current


def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "null"
    return "any"


def _type_valid(value: Any, expected: str) -> bool:
    actual = _json_type(value)
    return expected == "any" or actual == expected or (expected == "number" and actual == "integer")


def _risk_for(program: TypedEvidenceProgram, *, context: Mapping[str, Any]) -> ProgramRisk:
    semantic = bool(program.args.get("semantic", False))
    role_confidence = float(program.args.get("role_confidence", 1.0))
    state_stale = bool(program.args.get("state_stale", False))
    transform_risk = 0.002 if program.op in {"COPY", "ENUM", "NEGATED_BOOL", "STATE_REF", "SCHEMA_DEFAULT", "CONTRACT_CONST"} else 0.005
    totals = {
        "evidence": 0.02 if semantic else 0.001,
        "role": max(0.0, 1.0 - role_confidence),
        "transform": transform_risk,
        "contract": 0.001,
        "state_staleness": 0.1 if state_stale else 0.0,
    }
    for child in _children(program):
        child_risk = asdict(_risk_for(child, context=context))
        for factor, value in child_risk.items():
            totals[factor] = min(1.0, totals[factor] + float(value))
    return ProgramRisk(**totals)


def execute_program(
    program: TypedEvidenceProgram,
    request: str,
    *,
    reference_context: Mapping[str, Any] | None = None,
    dialogue_state: Mapping[str, Any] | None = None,
    schema_defaults: Mapping[str, Any] | None = None,
    contract_constants: Mapping[str, Any] | None = None,
) -> ProgramExecution:
    """Execute a bounded TEP. This evaluator never evaluates generated source code."""
    context = reference_context or {}
    state = dialogue_state or {}
    defaults = schema_defaults or {}
    constants = contract_constants or {}
    try:
        validate_program(program)
        trace: list[dict[str, Any]] = []

        def run(node: TypedEvidenceProgram) -> Any:
            child_values: list[Any] = []
            if node.op in {"COPY", "PARSE_DATE", "PARSE_TIME", "PARSE_NUMBER", "NEGATED_BOOL"}:
                raw, span = _span_text(node, request)
                if node.op == "COPY":
                    value = raw
                elif node.op == "PARSE_DATE":
                    value = _parse_date(raw, context)
                elif node.op == "PARSE_TIME":
                    value = _parse_time(raw)
                elif node.op == "PARSE_NUMBER":
                    value = _parse_number(raw)
                else:
                    lowered = raw.lower()
                    if not re.search(r"\b(?:do not|don't|never|without|no)\b", lowered):
                        raise ValueError("NEGATED_BOOL requires explicit negative evidence")
                    value = False
                trace.append({"op": node.op, "span": list(span), "source_text": raw, "value": value})
                return value
            if node.op == "ENUM":
                raw, span = _span_text(node, request)
                aliases = {str(key).casefold(): value for key, value in dict(node.args.get("aliases", {})).items()}
                value = aliases.get(raw.strip().casefold(), node.args.get("value"))
                if value is None:
                    raise ValueError(f"ENUM has no mapping for {raw!r}")
                trace.append({"op": node.op, "span": list(span), "source_text": raw, "value": value})
                return value
            if node.op == "STATE_REF":
                key = str(node.args["key"])
                expected_version = node.args.get("version")
                item = state.get(key)
                if isinstance(item, list):
                    item = item[-1] if item else None
                if isinstance(item, Mapping):
                    if expected_version is not None and item.get("version") != expected_version:
                        raise ValueError(f"state version mismatch for {key}")
                    value = item.get("value")
                    actual_version = item.get("version")
                else:
                    value, actual_version = item, None
                if value is None:
                    raise ValueError(f"missing state reference: {key}")
                trace.append({"op": node.op, "key": key, "version": actual_version, "value": value})
                return value
            if node.op in {"SCHEMA_DEFAULT", "CONTRACT_CONST"}:
                key_name = "default_id" if node.op == "SCHEMA_DEFAULT" else "id"
                key = str(node.args[key_name])
                source = defaults if node.op == "SCHEMA_DEFAULT" else constants
                if key not in source:
                    raise ValueError(f"unknown {node.op} identifier: {key}")
                value = source[key]
                trace.append({"op": node.op, key_name: key, "value": value})
                return value
            if node.op == "LIST":
                programs = node.args.get("programs", [])
                if not isinstance(programs, (list, tuple)):
                    raise ValueError("LIST requires programs")
                value = [run(child) if isinstance(child, TypedEvidenceProgram) else child for child in programs]
                trace.append({"op": node.op, "value": value})
                return value
            if node.op == "CONVERT_UNIT":
                child = node.args.get("input")
                value = run(child) if isinstance(child, TypedEvidenceProgram) else node.args.get("value")
                source_unit = str(node.args["source_unit"]).lower()
                target_unit = str(node.args["target_unit"]).lower()
                source_kind, source_factor = _UNIT_FACTORS[source_unit]
                target_kind, target_factor = _UNIT_FACTORS[target_unit]
                if source_kind != target_kind:
                    raise ValueError("unit dimensions do not match")
                value = float(value) * source_factor / target_factor
                value = int(value) if value.is_integer() else value
                trace.append({"op": node.op, "source_unit": source_unit, "target_unit": target_unit, "value": value})
                return value
            if node.op == "DERIVE":
                inputs = node.args.get("inputs", [])
                child_values = [run(child) if isinstance(child, TypedEvidenceProgram) else child for child in inputs]
                derivation = str(node.args.get("derivation"))
                if derivation == "sum":
                    value = sum(child_values)
                elif derivation == "difference":
                    value = child_values[0] - child_values[1]
                elif derivation == "duration_minutes":
                    start = datetime.strptime(str(child_values[0]), "%H:%M")
                    end = datetime.strptime(str(child_values[1]), "%H:%M")
                    value = int((end - start).total_seconds() / 60)
                else:
                    raise ValueError(f"unsupported derivation: {derivation}")
                trace.append({"op": node.op, "derivation": derivation, "inputs": child_values, "value": value})
                return value
            raise ValueError(f"unimplemented operator: {node.op}")

        value = run(program)
        if not _type_valid(value, program.output_type):
            raise ValueError(f"program produced {_json_type(value)}, expected {program.output_type}")
        risk = _risk_for(program, context=context)
        accepted_tier = program.tier if program.tier != "A" or risk.upper_bound <= 0.02 else "B"
        return ProgramExecution(program.program_id, value, program.output_type, tuple(trace), risk, accepted_tier, True)
    except (KeyError, TypeError, ValueError) as exc:
        return ProgramExecution(program.program_id, None, program.output_type, tuple(), ProgramRisk(contract=1.0), "C", False, str(exc))


def slot_risk_budget(role: str, *, criticality: str | None = None) -> float:
    if criticality == "high" or role in {"control", "identifier"}:
        return 0.02
    if role == "defaultable":
        return 0.03
    return 0.05


def compose_action_risk(executions: Iterable[ProgramExecution]) -> dict[str, Any]:
    rows = list(executions)
    bound = min(1.0, sum(row.risk.upper_bound for row in rows if row.valid))
    return {
        "composition": "union_bound",
        "program_count": len(rows),
        "action_risk_upper_bound": bound,
        "program_risks": {row.program_id: row.risk.to_dict() for row in rows},
    }


def build_evidence_hypergraph(
    *,
    request: str,
    tool: str,
    slot_programs: Mapping[str, Iterable[TypedEvidenceProgram]],
    executions: Mapping[str, ProgramExecution],
) -> dict[str, Any]:
    evidence_nodes: dict[str, dict[str, Any]] = {}
    slot_nodes = [{"id": f"slot:{tool}:{slot}", "slot": slot} for slot in sorted(slot_programs)]
    edges = []
    for slot, programs in slot_programs.items():
        for program in programs:
            execution = executions.get(program.program_id)
            sources = []
            if execution is not None:
                for step in execution.trace:
                    span = step.get("span")
                    if span is None:
                        continue
                    node_id = f"span:{span[0]}:{span[1]}"
                    evidence_nodes[node_id] = {
                        "id": node_id,
                        "span": span,
                        "text": request[int(span[0]):int(span[1])],
                    }
                    sources.append(node_id)
            edges.append({
                "id": f"program:{program.program_id}",
                "sources": sources,
                "target": f"slot:{tool}:{slot}",
                "program": program.to_dict(),
                "execution": execution.to_dict() if execution is not None else None,
            })
    return {
        "schema_version": HYPERGRAPH_VERSION,
        "request_sha256": hashlib.sha256(request.encode("utf-8")).hexdigest(),
        "tool": tool,
        "evidence_nodes": sorted(evidence_nodes.values(), key=lambda row: row["id"]),
        "slot_nodes": slot_nodes,
        "program_hyperedges": edges,
    }


def _slot_cues(slot: str) -> str:
    return r"[\s_-]+".join(re.escape(piece) for piece in slot.split("_") if piece)


def compile_slot_programs(
    request: str,
    slot: str,
    prop: Mapping[str, Any],
    *,
    role: str,
    reference_context: Mapping[str, Any] | None = None,
    dialogue_state: Mapping[str, Any] | None = None,
) -> list[TypedEvidenceProgram]:
    """Compile a conservative Tier-A/Tier-B candidate set for one schema slot."""
    programs: list[TypedEvidenceProgram] = []
    expected = str(prop.get("type", "string"))
    enum = prop.get("enum") if isinstance(prop.get("enum"), list) else []
    aliases = dict(prop.get("x-tap-enum-aliases", {})) if isinstance(prop.get("x-tap-enum-aliases"), Mapping) else {}
    for value in enum:
        aliases.setdefault(str(value), value)
    for alias in sorted(aliases, key=len, reverse=True):
        for match in re.finditer(rf"(?<!\w){re.escape(alias)}(?!\w)", request, re.IGNORECASE):
            programs.append(TypedEvidenceProgram("ENUM", {"span": match.span(), "aliases": aliases}, expected))

    if expected == "string" and "currency" in slot.casefold():
        for symbol, currency_code in _CURRENCY_SYMBOLS.items():
            for match in re.finditer(re.escape(symbol), request):
                programs.append(TypedEvidenceProgram(
                    "ENUM",
                    {"span": match.span(), "aliases": {symbol: currency_code}, "role_confidence": 1.0},
                    "string",
                ))

    if expected == "array":
        cue = _slot_cues(str(prop.get("x-tap-list-cue") or slot))
        list_pattern = re.compile(
            rf"\b{cue}\b(?:\s+(?:to|are|is)|\s*:)\s*([^.;\n]+)",
            re.IGNORECASE,
        )
        for match in list_pattern.finditer(request):
            raw = match.group(1).strip()
            parts = [part.strip() for part in re.split(r"\s*,\s*|\s+and\s+", raw) if part.strip()]
            if len(parts) < 2:
                continue
            children = []
            cursor = match.start(1)
            for part in parts:
                start = request.casefold().find(part.casefold(), cursor, match.end(1))
                if start < 0:
                    children = []
                    break
                children.append(TypedEvidenceProgram("COPY", {"span": (start, start + len(part))}, "string"))
                cursor = start + len(part)
            if children:
                programs.append(TypedEvidenceProgram("LIST", {"programs": children}, "array"))

    cue = _slot_cues(slot)
    if expected == "boolean":
        negative = re.compile(rf"\b(?:do not|don't|never)\s+{cue}\b|\bwithout\s+{cue}\b|\bno\s+{cue}\b", re.IGNORECASE)
        for match in negative.finditer(request):
            programs.append(TypedEvidenceProgram("NEGATED_BOOL", {"span": match.span(), "role_confidence": 1.0}, "boolean"))

    if expected in {"integer", "number"}:
        number_pattern = r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:\s*[kK])?(?![\w.])|\b(?:" + "|".join(_NUMBER_WORDS) + r")(?:[ -](?:" + "|".join(_NUMBER_WORDS) + r"|hundred|thousand))*\b"
        for match in re.finditer(number_pattern, request, re.IGNORECASE):
            nearby = request[max(0, match.start() - 32):min(len(request), match.end() + 32)].lower()
            prefix = request[max(0, match.start() - 2):match.start()].strip()
            currency_amount = slot.casefold() in {"amount", "price", "total", "value"} and any(
                prefix.endswith(symbol) for symbol in _CURRENCY_SYMBOLS
            )
            confidence = 1.0 if re.search(cue, nearby, re.IGNORECASE) or currency_amount else 0.85
            tier = "A" if confidence == 1.0 else "B"
            base = TypedEvidenceProgram("PARSE_NUMBER", {"span": match.span(), "role_confidence": confidence}, expected, tier)
            target_unit = prop.get("x-tap-unit")
            unit_match = re.match(r"\s*([A-Za-z]+)", request[match.end():])
            if target_unit and unit_match and unit_match.group(1).lower() in _UNIT_FACTORS and str(target_unit).lower() in _UNIT_FACTORS:
                programs.append(TypedEvidenceProgram("CONVERT_UNIT", {"input": base, "source_unit": unit_match.group(1), "target_unit": target_unit, "role_confidence": confidence}, expected, tier))
            else:
                programs.append(base)
    derivation = prop.get("x-tap-derive")
    if isinstance(derivation, Mapping) and derivation.get("op") == "duration_minutes":
        time_pattern = r"\b\d{1,2}:\d{2}(?:\s*(?:am|pm))?\b|\bhalf past (?:" + "|".join(_NUMBER_WORDS) + r")\b"
        matches = list(re.finditer(time_pattern, request, re.IGNORECASE))
        if len(matches) >= 2:
            start_match, end_match = matches[-2:]
            inputs = [
                TypedEvidenceProgram("PARSE_TIME", {"span": start_match.span(), "role_confidence": 1.0}, "string"),
                TypedEvidenceProgram("PARSE_TIME", {"span": end_match.span(), "role_confidence": 1.0}, "string"),
            ]
            programs.append(TypedEvidenceProgram(
                "DERIVE", {"derivation": "duration_minutes", "inputs": inputs, "role_confidence": 1.0}, expected,
            ))


    if role == "control" and (slot == "date" or slot.endswith("_date")):
        date_pattern = r"\b\d{4}-\d{2}-\d{2}\b|\b(?:today|tomorrow|(?:(?:next)\s+)?(?:" + "|".join(_WEEKDAYS) + r"))\b"
        matches = list(re.finditer(date_pattern, request, re.IGNORECASE))
        correction = re.search(r"\bfrom\s+([^,.]+?)\s+to\s+([^,.]+?)(?:[,.]|$)", request, re.IGNORECASE)
        date_role_cues = {
            "depart_date": r"\bleaving\s*$",
            "return_date": r"\breturning\s*$",
            "modified_after": r"\bmodified\s+after\s*$",
        }
        for match in matches:
            nearby = request[max(0, match.start() - 28):match.start()].lower()
            if slot == "date":
                role_match = len(matches) == 1 or bool(re.search(r"\b(?:on|to)\s*$", nearby))
            elif slot in date_role_cues:
                role_match = bool(re.search(date_role_cues[slot], nearby))
            else:
                role_match = bool(re.search(rf"\b{_slot_cues(slot)}\s*$", nearby))
            confidence = 1.0 if role_match else 0.82
            tier = "A" if confidence == 1.0 else "B"
            args: dict[str, Any] = {"span": match.span(), "role_confidence": confidence}
            if correction and correction.start(1) <= match.start() < correction.end(1):
                args["superseded"] = True
                tier = "C"
            programs.append(TypedEvidenceProgram("PARSE_DATE", args, "string", tier))

    if role == "control" and (slot == "time" or slot.endswith("_time")):
        time_pattern = r"\b\d{1,2}:\d{2}(?:\s*(?:am|pm))?\b|\bhalf past (?:" + "|".join(_NUMBER_WORDS) + r")\b"
        time_role_cues = {
            "start_time": r"\bfrom\s*$",
            "end_time": r"\bto\s*$",
            "send_time": r"\bat\s*$",
            "time": r"\bat\s*$",
            "hour": r"\bat\s*$",
        }
        time_matches = list(re.finditer(time_pattern, request, re.IGNORECASE))
        for match in time_matches:
            nearby = request[max(0, match.start() - 24):match.start()].lower()
            cue_pattern = time_role_cues.get(slot, rf"\b{_slot_cues(slot)}\s*$")
            role_match = bool(re.search(cue_pattern, nearby))
            confidence = 1.0 if role_match else 0.82
            programs.append(TypedEvidenceProgram("PARSE_TIME", {"span": match.span(), "role_confidence": confidence}, "string", "A" if confidence == 1.0 else "B"))

    state = dialogue_state or {}
    if slot in state:
        item = state[slot][-1] if isinstance(state[slot], list) and state[slot] else state[slot]
        version = item.get("version") if isinstance(item, Mapping) else None
        programs.append(TypedEvidenceProgram("STATE_REF", {"key": slot, "version": version}, expected))
    if "default" in prop:
        programs.append(TypedEvidenceProgram("SCHEMA_DEFAULT", {"default_id": slot}, expected))
    constant_id = prop.get("x-tap-constant-id")
    constants = prop.get("x-tap-contract-constants")
    if constant_id is not None and isinstance(constants, Mapping) and constant_id in constants:
        programs.append(TypedEvidenceProgram("CONTRACT_CONST", {"id": str(constant_id)}, expected))

    unique: dict[str, TypedEvidenceProgram] = {}
    for program in programs:
        unique[program.program_id] = program
    return list(unique.values())
