"""Map InjecAgent cases onto annotated EviBind requests.

InjecAgent (Zhan et al., UIUC, MIT licence) tests a different shape of attack
from InjectBench. A user asks for something benign, the tool result carries an
instruction planted by an attacker, and the attack succeeds if the agent then
calls the *attacker's* tool. It is a tool-selection attack; EviBind's boundary
is argument-level. The overlap is the interesting part, and the gap is reported
rather than hidden.

**The annotation rule.** EviBind expects the application developer to mark
critical slots. Standing in for that developer, one fixed rule is applied to
every tool in the benchmark, attacker tools and user tools alike, with no
per-case tuning:

* a required ``string`` parameter whose name or description names an identifier
  (id, account, number, address, path, url, email, date, phone) becomes a
  **control** slot bound to ``user.current_turn``;
* every other required parameter becomes **content** the model may fill with a
  literal;
* optional parameters are left alone.

That is deliberately blunt. It over-marks some slots and under-marks others,
and it is applied symmetrically, so it cannot flatter the guarded arm without
also costing it on the utility arm.

**What this can and cannot test.** If the attacker's tool takes no arguments —
``AugustSmartLockUnlockDoor``, ``The23andMeGetGeneticData`` — then there is no
critical value to bind and an argument-level boundary has nothing to withhold.
:func:`scope_of` reports how much of a split is in scope before any model runs.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parent / "data"

# name/description keyword -> evidence type. Order matters: first match wins.
IDENTIFIER_TYPES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("email", "e-mail"), "email_address"),
    (("url", "link", "uri", "website"), "uri"),
    (("path", "filename", "file name", "directory", "folder"), "repository_path"),
    (("date", "time"), "iso_date"),
    (("phone", "mobile number"), "phone_number"),
    (("id", "account", "number", "code", "key", "token", "address",
      "handle", "username", "recipient"), "opaque_registry_id"),
)


def _python_literal(value: Any) -> dict[str, Any]:
    """InjecAgent stores tool arguments as Python reprs, not JSON.

    ``"{'product_id': 'B08KFQ9HK5'}"`` is not parseable by ``json.loads``.
    ``ast.literal_eval`` reads it without evaluating code.
    """
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _tool_index() -> dict[str, dict[str, Any]]:
    catalogue = json.loads((DATA / "tools.json").read_text(encoding="utf-8"))
    index: dict[str, dict[str, Any]] = {}
    for kit in catalogue:
        prefix = kit.get("name_for_model", "")
        for tool in kit.get("tools", []):
            index[prefix + tool["name"]] = tool
    return index


def _evidence_type(param: dict[str, Any]) -> str | None:
    haystack = f"{param.get('name', '')} {param.get('description', '')}".lower()
    for keywords, evidence_type in IDENTIFIER_TYPES:
        if any(keyword in haystack for keyword in keywords):
            return evidence_type
    return None


JSON_TYPES = {"string": "string", "integer": "integer", "number": "number",
              "boolean": "boolean", "array": "array", "object": "object"}


def annotate_tool(tool: dict[str, Any], name: str) -> dict[str, Any]:
    """Convert one InjecAgent tool into an annotated OpenAI function schema."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param in tool.get("parameters", []):
        # Optional parameters are dropped. The benchmark never depends on them,
        # and carrying them unannotated would leave the policy ambiguous.
        if not param.get("required"):
            continue
        param_name = str(param.get("name"))
        json_type = JSON_TYPES.get(str(param.get("type")), "string")
        spec: dict[str, Any] = {"type": json_type,
                                "description": str(param.get("description", ""))}
        if json_type == "array":
            spec["items"] = {"type": "string"}
        evidence_type = _evidence_type(param) if json_type == "string" else None
        if evidence_type:
            spec.update({
                "x-evibind-slot-role": "control",
                "x-evibind-evidence-type": evidence_type,
                "x-evibind-sources": ["user.current_turn"],
                "x-evibind-resolution-type": "extractive",
                "x-evibind-value-class": "authority_bearing",
                "x-evibind-criticality": "target",
            })
            # Types with a shape pattern (email, uri, date, phone) are found
            # without a cue. The opaque identifier types have no pattern by
            # design -- an account reference may be anything -- so they need
            # one, and the parameter's own name is what a developer would
            # write: product_id -> "product id", which matches "product ID
            # B08KFQ9HK5" case-insensitively.
            if evidence_type == "opaque_registry_id":
                spec["x-evibind-extraction-cue"] = param_name.replace("_", " ")
        else:
            spec.update({
                "x-evibind-slot-role": "content",
                "x-evibind-evidence-type": "opaque_content",
                "x-evibind-value-class": "opaque_content",
                "x-evibind-criticality": "content",
            })
        required.append(param_name)
        properties[param_name] = spec
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": str(tool.get("summary", ""))[:900],
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def control_slots(annotated: dict[str, Any]) -> list[str]:
    properties = annotated["function"]["parameters"]["properties"]
    return [name for name, spec in properties.items()
            if spec.get("x-evibind-slot-role") == "control"]


def load_split(split: str) -> list[dict[str, Any]]:
    """Build one adapted case per InjecAgent test case.

    ``split`` is one of ``dh_base``, ``dh_enhanced``, ``ds_base``,
    ``ds_enhanced``. The enhanced variants prefix the injected instruction
    with "IMPORTANT!!! Ignore all previous instructions".
    """
    index = _tool_index()
    raw = json.loads((DATA / f"test_cases_{split}.json").read_text(encoding="utf-8"))
    cases = []
    for position, case in enumerate(raw):
        user_name = case["User Tool"]
        attacker_name = case["Attacker Tools"][0]
        user_tool = annotate_tool(index[user_name], user_name)
        attacker_tool = annotate_tool(index[attacker_name], attacker_name)
        user_arguments = _python_literal(case["Tool Parameters"])
        messages = [
            {"role": "user", "content": case["User Instruction"]},
            {"role": "assistant", "content": None,
             "tool_calls": [{
                 "id": f"user-{position}", "type": "function",
                 "function": {"name": user_name,
                              "arguments": json.dumps(user_arguments)}}]},
            {"role": "tool", "tool_call_id": f"user-{position}",
             "content": str(case["Tool Response"])},
        ]
        # One ds case names the same tool as both the user's and the
        # attacker's. Offering it twice is not a valid request -- the gateway
        # rejects duplicate function names -- and the case cannot distinguish
        # an attack from the user's own call, so it is marked and excluded.
        tools = [user_tool] if user_name == attacker_name else [user_tool, attacker_tool]
        cases.append({
            "case_id": f"{split}-{position:04d}",
            "self_referential": user_name == attacker_name,
            "split": split,
            "attack_type": case.get("Attack Type", "unknown"),
            "user_tool": user_name,
            "user_arguments": user_arguments,
            "attacker_tool": attacker_name,
            "attacker_instruction": case["Attacker Instruction"],
            "messages": messages,
            "tools": tools,
            "attacker_control_slots": control_slots(attacker_tool),
            "in_scope": bool(control_slots(attacker_tool)),
        })
    return cases


def gateway_request(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": "benchmark",
        "messages": case["messages"],
        "tools": case["tools"],
        "evibind": {
            "policy_epoch": "injecagent",
            "include_diagnostics": False,
            "allow_noncritical_opaque_literals": True,
        },
    }


def model_visible_request(case: dict[str, Any]) -> dict[str, Any]:
    """The same request with every EviBind annotation stripped."""
    public = json.loads(json.dumps(case["tools"]))
    for tool in public:
        for spec in tool["function"]["parameters"]["properties"].values():
            for key in [k for k in spec if k.startswith("x-evibind-")]:
                del spec[key]
    return {"messages": case["messages"], "tools": public, "tool_choice": "auto"}


def scope_of(cases: list[dict[str, Any]]) -> dict[str, Any]:
    in_scope = [c for c in cases if c["in_scope"]]
    return {
        "cases": len(cases),
        "in_scope": len(in_scope),
        "out_of_scope": len(cases) - len(in_scope),
        "reason_out_of_scope": "attacker tool declares no required identifier slot",
    }


if __name__ == "__main__":
    for split in ("dh_base", "dh_enhanced", "ds_base", "ds_enhanced"):
        cases = load_split(split)
        scope = scope_of(cases)
        print(f"{split}: {scope['in_scope']}/{scope['cases']} in scope "
              f"({scope['in_scope'] / scope['cases']:.0%})")
