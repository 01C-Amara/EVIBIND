from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping


def _typed(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def json_contract_accepts(value: Any, schema: Any) -> bool:
    """Validate the deterministic JSON Schema subset used by tool contracts."""
    if not isinstance(schema, Mapping):
        return True
    if "const" in schema and value != schema["const"]:
        return False
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        return False

    all_of = schema.get("allOf")
    if isinstance(all_of, list) and not all(
        json_contract_accepts(value, branch) for branch in all_of
    ):
        return False
    any_of = schema.get("anyOf")
    if isinstance(any_of, list) and not any(
        json_contract_accepts(value, branch) for branch in any_of
    ):
        return False
    one_of = schema.get("oneOf")
    if (
        isinstance(one_of, list)
        and sum(json_contract_accepts(value, branch) for branch in one_of) != 1
    ):
        return False
    negated = schema.get("not")
    if isinstance(negated, Mapping) and json_contract_accepts(
        value,
        negated,
    ):
        return False

    expected = schema.get("type")
    if isinstance(expected, list):
        return any(
            json_contract_accepts(value, {**schema, "type": item})
            for item in expected
            if isinstance(item, str)
        )
    if isinstance(expected, str) and not _typed(value, expected):
        return False

    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            properties = {}
        required = schema.get("required", [])
        if not isinstance(required, list) or any(
            item not in value for item in required
        ):
            return False
        if schema.get("additionalProperties") is False and any(
            key not in properties for key in value
        ):
            return False
        if not all(
            json_contract_accepts(item, properties.get(key, {}))
            for key, item in value.items()
        ):
            return False
        if len(value) < int(schema.get("minProperties", 0)):
            return False
        maximum = schema.get("maxProperties")
        if isinstance(maximum, int) and len(value) > maximum:
            return False
        dependent = schema.get("dependentRequired", {})
        if isinstance(dependent, Mapping):
            for key, dependencies in dependent.items():
                if (
                    key in value
                    and isinstance(dependencies, list)
                    and any(item not in value for item in dependencies)
                ):
                    return False

    if isinstance(value, list):
        item_schema = schema.get("items", {})
        if not all(json_contract_accepts(item, item_schema) for item in value):
            return False
        if len(value) < int(schema.get("minItems", 0)):
            return False
        maximum = schema.get("maxItems")
        if isinstance(maximum, int) and len(value) > maximum:
            return False
        if schema.get("uniqueItems") is True:
            canonical = [_canonical(item) for item in value]
            if len(canonical) != len(set(canonical)):
                return False

    if isinstance(value, str):
        if len(value) < int(schema.get("minLength", 0)):
            return False
        maximum = schema.get("maxLength")
        if isinstance(maximum, int) and len(value) > maximum:
            return False
        pattern = schema.get("pattern")
        if isinstance(pattern, str):
            try:
                if re.search(pattern, value) is None:
                    return False
            except re.error:
                return False

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            return False
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)) and value < minimum:
            return False
        maximum = schema.get("maximum")
        if isinstance(maximum, (int, float)) and value > maximum:
            return False
        exclusive_minimum = schema.get("exclusiveMinimum")
        if isinstance(exclusive_minimum, (int, float)) and value <= exclusive_minimum:
            return False
        exclusive_maximum = schema.get("exclusiveMaximum")
        if isinstance(exclusive_maximum, (int, float)) and value >= exclusive_maximum:
            return False
        multiple = schema.get("multipleOf")
        if isinstance(multiple, (int, float)):
            if multiple <= 0:
                return False
            quotient = value / multiple
            if not math.isclose(
                quotient,
                round(quotient),
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                return False

    return True
