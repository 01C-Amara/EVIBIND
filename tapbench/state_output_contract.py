from __future__ import annotations

import ast
import json
from collections import defaultdict
from typing import Any, Iterable, Mapping


STATE_OUTPUT_CONTRACT_VERSION = "evibind.state_output_contract.v1"

# These aliases are public tool-output contracts, not values recovered from
# benchmark labels.  They let later tools bind a scalar POSIX timestamp that
# was returned by an earlier tool to the conventional timestamp slot names.
SCALAR_OUTPUT_ALIASES: dict[str, tuple[str, ...]] = {
    "get_current_timestamp": ("timestamp", "timestamp_0", "timestamp_1"),
    "datetime_info_to_timestamp": ("timestamp", "timestamp_0", "timestamp_1"),
    "shift_timestamp": ("timestamp", "timestamp_0", "timestamp_1"),
    "search_holiday": ("timestamp", "timestamp_0", "timestamp_1"),
    "unit_conversion": ("amount",),
    "calculate_lat_lon_distance": ("distance",),
}


def _parse_observation(content: str) -> Any:
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return ast.literal_eval(content)
    except (SyntaxError, ValueError):
        return None


def build_trusted_dialogue_state(
    observations: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build versioned state from successful execution-environment outputs.

    An observation has ``content``, ``version``, ``function_name`` and the
    boolean ``failed``.  Nested structured values keep their explicit keys.
    Top-level scalar values are exposed only through the fixed public alias
    contract above; arbitrary type-compatible aliasing is intentionally
    forbidden.
    """

    state: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def add_structured(value: Any, version: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, (str, int, float, bool)) or item is None:
                    state[str(key)].append({"value": item, "version": version})
                else:
                    add_structured(item, version)
        elif isinstance(value, list):
            for item in value:
                add_structured(item, version)

    for index, observation in enumerate(observations):
        if bool(observation.get("failed")):
            continue
        parsed = _parse_observation(str(observation.get("content") or ""))
        if parsed is None:
            continue
        version = str(observation.get("version") or f"tool_observation_{index}")
        add_structured(parsed, version)
        if isinstance(parsed, (str, int, float, bool)):
            function_name = str(observation.get("function_name") or "")
            for alias in SCALAR_OUTPUT_ALIASES.get(function_name, ()):
                state[alias].append({"value": parsed, "version": version})
    return dict(state)
