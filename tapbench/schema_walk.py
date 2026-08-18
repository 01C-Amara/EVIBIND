from __future__ import annotations

from typing import Any, Iterator, Mapping


def validate_required_properties(
    schema: Mapping[str, Any],
) -> tuple[bool, tuple[str, ...]]:
    """Validate a JSON Schema object's required declaration."""
    required = schema.get("required", [])
    if not isinstance(required, list) or not all(
        isinstance(value, str) for value in required
    ):
        return False, ()
    properties = schema.get("properties")
    known = set(properties) if isinstance(properties, Mapping) else set()
    return True, tuple(sorted(set(required) - known))


def iter_schema_properties(
    properties: Mapping[str, Any],
    *,
    base_path: str,
) -> Iterator[tuple[str, str, Any, bool]]:
    """Yield nested object containers and leaf parameter schemas."""
    for raw_slot, raw_schema in properties.items():
        slot = str(raw_slot)
        path = f"{base_path}.{slot}"
        nested = (
            raw_schema.get("properties")
            if isinstance(raw_schema, Mapping) and raw_schema.get("type") == "object"
            else None
        )
        is_container = isinstance(nested, Mapping)
        yield slot, path, raw_schema, is_container
        if is_container:
            yield from iter_schema_properties(
                nested,
                base_path=path + ".properties",
            )
