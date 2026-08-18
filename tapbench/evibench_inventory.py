from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from urllib.parse import quote, urlsplit, urlunsplit
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .evibench_freeze import artifact_sha256
from .io import write_jsonl, write_yaml


INVENTORY_VERSION = "evibind.evibench_family_inventory.v1"
_HTTP_METHODS = ("get", "post", "put", "patch", "delete")
_OPEN_LICENSE_TOKENS = (
    "apache",
    "mit",
    "bsd",
    "cc0",
    "cc-by",
    "creative commons attribution",
    "public domain",
    "open government",
    "open licence",
    "gnu lesser",
)
_SCHEMA_KEYS = frozenset(
    {
        "type",
        "description",
        "enum",
        "const",
        "default",
        "format",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
        "properties",
        "required",
        "items",
        "additionalProperties",
        "oneOf",
        "anyOf",
        "allOf",
        "nullable",
    }
)


class InventoryError(ValueError):
    pass


@dataclass(frozen=True)
class DirectoryCandidate:
    api_name: str
    version: str
    provider: str
    category: str
    source_url: str
    origin_url: str
    license_name: str
    license_url: str | None
    rank: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _slug(value: str, *, maximum: int = 48) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return (normalized or "api")[:maximum]


def _explicit_open_license(value: str) -> bool:
    folded = value.casefold()
    return any(token in folded for token in _OPEN_LICENSE_TOKENS)


def _directory_candidates(
    directory: Mapping[str, Any],
    *,
    seed: int,
) -> list[DirectoryCandidate]:
    candidates: list[DirectoryCandidate] = []
    for api_name, api_row in directory.items():
        if not isinstance(api_name, str) or not isinstance(api_row, Mapping):
            continue
        version = api_row.get("preferred")
        versions = api_row.get("versions")
        if not isinstance(version, str) or not isinstance(versions, Mapping):
            continue
        raw = versions.get(version)
        if not isinstance(raw, Mapping):
            continue
        info = raw.get("info")
        if not isinstance(info, Mapping):
            continue
        license_row = info.get("license")
        if not isinstance(license_row, Mapping):
            continue
        license_name = license_row.get("name")
        if not isinstance(license_name, str) or not _explicit_open_license(
            license_name
        ):
            continue
        source_url = raw.get("swaggerUrl")
        if not isinstance(source_url, str) or not source_url.startswith("https://"):
            continue
        categories = info.get("x-apisguru-categories")
        category = (
            str(categories[0])
            if isinstance(categories, list) and categories
            else "uncategorized"
        )
        origins = info.get("x-origin")
        origin_url = source_url
        if isinstance(origins, list):
            for origin in origins:
                if isinstance(origin, Mapping) and isinstance(origin.get("url"), str):
                    origin_url = str(origin["url"])
                    break
        license_url = license_row.get("url")
        candidates.append(
            DirectoryCandidate(
                api_name=api_name,
                version=version,
                provider=api_name.split(":", 1)[0],
                category=category,
                source_url=source_url,
                origin_url=origin_url,
                license_name=license_name,
                license_url=(license_url if isinstance(license_url, str) else None),
                rank=hashlib.sha256(f"{seed}:{api_name}".encode()).hexdigest(),
            )
        )
    return candidates


def select_diverse_candidates(
    directory: Mapping[str, Any],
    *,
    seed: int,
    maximum: int = 100,
    maximum_per_category: int = 8,
    maximum_per_provider: int = 4,
) -> list[DirectoryCandidate]:
    pool = _directory_candidates(directory, seed=seed)
    selected: list[DirectoryCandidate] = []
    category_counts: Counter[str] = Counter()
    provider_counts: Counter[str] = Counter()
    while pool and len(selected) < maximum:
        eligible = [
            row
            for row in pool
            if category_counts[row.category] < maximum_per_category
            and provider_counts[row.provider] < maximum_per_provider
        ]
        if not eligible:
            break
        chosen = min(
            eligible,
            key=lambda row: (
                category_counts[row.category],
                provider_counts[row.provider],
                row.rank,
                row.api_name,
            ),
        )
        selected.append(chosen)
        category_counts[chosen.category] += 1
        provider_counts[chosen.provider] += 1
        pool.remove(chosen)
    return selected


def _pointer(root: Mapping[str, Any], reference: str) -> Any:
    if not reference.startswith("#/"):
        raise InventoryError("only document-local references are supported")
    current: Any = root
    for part in reference[2:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or key not in current:
            raise InventoryError(f"unresolved schema reference: {reference}")
        current = current[key]
    return current


def _clean_schema(
    value: Any,
    root: Mapping[str, Any],
    *,
    depth: int = 0,
    seen: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if depth > 10 or not isinstance(value, Mapping):
        return {}
    reference = value.get("$ref")
    if isinstance(reference, str):
        if reference in seen:
            return {"type": "object"}
        resolved = _pointer(root, reference)
        merged = dict(resolved) if isinstance(resolved, Mapping) else {}
        merged.update({key: item for key, item in value.items() if key != "$ref"})
        return _clean_schema(
            merged,
            root,
            depth=depth + 1,
            seen=seen | {reference},
        )
    cleaned: dict[str, Any] = {}
    for key, item in value.items():
        if key not in _SCHEMA_KEYS:
            continue
        if key == "description" and isinstance(item, str):
            cleaned[key] = item[:600]
        elif key == "properties" and isinstance(item, Mapping):
            cleaned[key] = {
                str(name): _clean_schema(
                    schema,
                    root,
                    depth=depth + 1,
                    seen=seen,
                )
                for name, schema in item.items()
                if isinstance(name, str)
            }
        elif key == "items":
            cleaned[key] = _clean_schema(
                item,
                root,
                depth=depth + 1,
                seen=seen,
            )
        elif key in {"oneOf", "anyOf", "allOf"} and isinstance(item, list):
            cleaned[key] = [
                _clean_schema(
                    child,
                    root,
                    depth=depth + 1,
                    seen=seen,
                )
                for child in item[:8]
            ]
        elif key == "additionalProperties" and isinstance(item, Mapping):
            cleaned[key] = _clean_schema(
                item,
                root,
                depth=depth + 1,
                seen=seen,
            )
        elif key == "enum" and isinstance(item, list):
            cleaned[key] = item[:100]
        elif isinstance(item, (str, int, float, bool, list)) or item is None:
            cleaned[key] = item
    if not cleaned:
        cleaned["type"] = "string"
    return cleaned


def _parameter_schema(
    parameter: Mapping[str, Any],
    root: Mapping[str, Any],
) -> dict[str, Any]:
    schema = parameter.get("schema")
    if isinstance(schema, Mapping):
        return _clean_schema(schema, root)
    inline = {key: value for key, value in parameter.items() if key in _SCHEMA_KEYS}
    return _clean_schema(inline or {"type": "string"}, root)


def _unique_property_name(name: str, properties: Mapping[str, Any]) -> str:
    candidate = name
    index = 2
    while candidate in properties:
        candidate = f"{name}_{index}"
        index += 1
    return candidate


def _operation_tool(
    spec: Mapping[str, Any],
    path: str,
    method: str,
    path_row: Mapping[str, Any],
    operation: Mapping[str, Any],
    *,
    used_names: set[str],
) -> tuple[dict[str, Any], dict[str, str]]:
    raw_name = operation.get("operationId")
    if not isinstance(raw_name, str) or not raw_name:
        raw_name = f"{method}_{path}"
    base_name = _slug(raw_name, maximum=56)
    name = base_name
    suffix = 2
    while name in used_names:
        name = f"{base_name[:52]}_{suffix}"
        suffix += 1
    used_names.add(name)

    properties: dict[str, Any] = {}
    required: list[str] = []
    raw_parameters: list[Any] = []
    for source in (path_row.get("parameters"), operation.get("parameters")):
        if isinstance(source, list):
            raw_parameters.extend(source)
    for raw_parameter in raw_parameters:
        if not isinstance(raw_parameter, Mapping):
            continue
        parameter = raw_parameter
        reference = raw_parameter.get("$ref")
        if isinstance(reference, str):
            resolved = _pointer(spec, reference)
            if not isinstance(resolved, Mapping):
                continue
            parameter = resolved
        location = parameter.get("in")
        raw_parameter_name = parameter.get("name", "argument")
        if not isinstance(raw_parameter_name, str):
            continue
        if location == "header" and raw_parameter_name.casefold() in {
            "authorization",
            "x-api-key",
            "api-key",
        }:
            continue
        property_name = "body" if location == "body" else raw_parameter_name
        property_name = _unique_property_name(property_name, properties)
        properties[property_name] = _parameter_schema(parameter, spec)
        if parameter.get("required") is True:
            required.append(property_name)

    request_body = operation.get("requestBody")
    if isinstance(request_body, Mapping):
        reference = request_body.get("$ref")
        if isinstance(reference, str):
            resolved = _pointer(spec, reference)
            if isinstance(resolved, Mapping):
                request_body = resolved
        content = request_body.get("content")
        if isinstance(content, Mapping) and content:
            media = content.get("application/json")
            if not isinstance(media, Mapping):
                media = next(
                    (
                        item
                        for key, item in content.items()
                        if isinstance(key, str)
                        and key.endswith("+json")
                        and isinstance(item, Mapping)
                    ),
                    None,
                )
            if isinstance(media, Mapping) and isinstance(media.get("schema"), Mapping):
                property_name = _unique_property_name("body", properties)
                properties[property_name] = _clean_schema(media["schema"], spec)
                if request_body.get("required") is True:
                    required.append(property_name)

    description = operation.get("description") or operation.get("summary") or ""
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        parameters["required"] = sorted(set(required))
    tool = {
        "type": "function",
        "function": {
            "name": name,
            "description": str(description)[:600],
            "parameters": parameters,
        },
    }
    return tool, {"method": method.upper(), "path": path, "operation_id": raw_name}


def normalized_tools(
    spec: Mapping[str, Any],
    *,
    maximum_tools: int = 4,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    paths = spec.get("paths")
    if not isinstance(paths, Mapping):
        raise InventoryError("OpenAPI document has no paths mapping")
    operations: list[tuple[str, str, Mapping[str, Any], Mapping[str, Any]]] = []
    for path, path_row in sorted(paths.items()):
        if not isinstance(path, str) or not isinstance(path_row, Mapping):
            continue
        for method in _HTTP_METHODS:
            operation = path_row.get(method)
            if isinstance(operation, Mapping):
                operations.append((path, method, path_row, operation))
    operations.sort(
        key=lambda row: (
            str(row[3].get("operationId", "")),
            row[0],
            row[1],
        )
    )
    if len(operations) < 2:
        raise InventoryError("tool family needs at least two operations")
    tools: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    used_names: set[str] = set()
    for path, method, path_row, operation in operations[:maximum_tools]:
        tool, source = _operation_tool(
            spec,
            path,
            method,
            path_row,
            operation,
            used_names=used_names,
        )
        tools.append(tool)
        sources.append(source)
    return tools, sources


def fetch_openapi(url: str, *, maximum_bytes: int = 8_000_000) -> bytes:
    parts = urlsplit(url)
    safe_url = urlunsplit(
        (parts.scheme, parts.netloc, quote(parts.path, safe="/:@"), parts.query, "")
    )
    request = urllib.request.Request(
        safe_url,
        headers={"User-Agent": "EviBind-EviBench-Inventory/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        value = response.read(maximum_bytes + 1)
    if len(value) > maximum_bytes:
        raise InventoryError(f"source artifact exceeds {maximum_bytes} bytes")
    return value


def _split_map(
    families: Sequence[str],
    *,
    seed: int,
    split_counts: Mapping[str, int],
) -> dict[str, str]:
    if sum(split_counts.values()) != len(families):
        raise InventoryError("split counts do not match selected families")
    ordered = sorted(
        families,
        key=lambda family: hashlib.sha256(
            f"{seed}:split:{family}".encode()
        ).hexdigest(),
    )
    result: dict[str, str] = {}
    offset = 0
    for split in ("train", "development", "test"):
        count = int(split_counts.get(split, 0))
        for family in ordered[offset : offset + count]:
            result[family] = split
        offset += count
    return result


def build_family_inventory(
    *,
    directory_index_path: str | Path,
    output_dir: str | Path,
    families_output_path: str | Path,
    count: int = 70,
    seed: int = 20260731,
    fetcher: Callable[[str], bytes] = fetch_openapi,
    split_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    if count < 1:
        raise InventoryError("count must be positive")
    if split_counts is None:
        if count < 21:
            raise InventoryError(
                "production split defaults require at least 21 families"
            )
        split_counts = {"train": 10, "development": 10, "test": count - 20}
    index_path = Path(directory_index_path)
    index_bytes = index_path.read_bytes()
    directory = json.loads(index_bytes)
    if not isinstance(directory, Mapping):
        raise InventoryError("directory index must be a mapping")
    candidates = select_diverse_candidates(
        directory,
        seed=seed,
        maximum=max(100, count + 30),
    )
    target = Path(output_dir)
    source_dir = target / "sources"
    schema_dir = target / "schemas"
    source_dir.mkdir(parents=True, exist_ok=True)
    schema_dir.mkdir(parents=True, exist_ok=True)
    index_sha256 = _sha256_bytes(index_bytes)
    accepted: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    seen_source_digests: set[str] = set()
    for candidate in candidates:
        if len(accepted) >= count:
            break
        try:
            raw = fetcher(candidate.source_url)
            source_digest = _sha256_bytes(raw)
            if source_digest in seen_source_digests:
                failures["duplicate_source_artifact"] += 1
                continue
            spec = json.loads(raw)
            if not isinstance(spec, Mapping):
                raise InventoryError("source is not a JSON mapping")
            tools, operations = normalized_tools(spec)
        except (InventoryError, json.JSONDecodeError, OSError, ValueError) as exc:
            failures[type(exc).__name__] += 1
            continue
        family_hash = hashlib.sha256(
            f"{candidate.api_name}:{candidate.version}".encode()
        ).hexdigest()[:10]
        family = f"openapi_{_slug(candidate.api_name)}_{family_hash}"
        source_name = f"{family}.json"
        schema_name = f"{family}.json"
        (source_dir / source_name).write_bytes(raw)
        schema_artifact = {
            "version": "evibind.evibench_normalized_tool_family.v1",
            "family": family,
            "tools": tools,
            "operation_sources": operations,
        }
        (schema_dir / schema_name).write_text(
            json.dumps(schema_artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        accepted.append(
            {
                "family": family,
                "api_name": candidate.api_name,
                "provider": candidate.provider,
                "category": candidate.category,
                "schema_sha256": artifact_sha256([tools]),
                "source_artifact_sha256": source_digest,
                "source_kind": "openapi",
                "source_locator": candidate.source_url,
                "source_origin_locator": candidate.origin_url,
                "source_revision": candidate.version,
                "license": candidate.license_name,
                "license_url": candidate.license_url,
                "license_evidence": "openapi_info_license_explicit",
                "license_review_status": "pending_human_confirmation",
                "source_index_sha256": index_sha256,
                "operation_count": len(tools),
                "source_artifact": f"sources/{source_name}",
                "schema_artifact": f"schemas/{schema_name}",
            }
        )
        seen_source_digests.add(source_digest)
    if len(accepted) != count:
        raise InventoryError(
            f"only {len(accepted)} of {count} requested families were usable"
        )
    splits = _split_map(
        [str(row["family"]) for row in accepted],
        seed=seed,
        split_counts=split_counts,
    )
    for row in accepted:
        row["split"] = splits[str(row["family"])]
    accepted.sort(key=lambda row: str(row["family"]))
    write_jsonl(families_output_path, accepted)
    review_rows = [
        {
            "family": row["family"],
            "api_name": row["api_name"],
            "source_locator": row["source_locator"],
            "source_origin_locator": row["source_origin_locator"],
            "license": row["license"],
            "license_url": row["license_url"],
            "review_status": "pending_human_confirmation",
        }
        for row in accepted
    ]
    write_yaml(
        target / "license_review.yaml",
        {
            "version": "evibind.evibench_family_license_review.v1",
            "status": "pending_human_confirmation",
            "instructions": (
                "Confirm each source and license, then set every row to approved "
                "and create FAMILY_LICENSE_REVIEWED."
            ),
            "families": review_rows,
        },
    )
    manifest = {
        "version": INVENTORY_VERSION,
        "status": "built_license_review_pending",
        "directory_index_sha256": index_sha256,
        "selection_seed": seed,
        "counts": {
            "families": len(accepted),
            "providers": len({str(row["provider"]) for row in accepted}),
            "categories": len({str(row["category"]) for row in accepted}),
            "licenses": len({str(row["license"]) for row in accepted}),
            "splits": dict(Counter(str(row["split"]) for row in accepted)),
        },
        "coverage": {
            "categories": dict(Counter(str(row["category"]) for row in accepted)),
            "providers": dict(Counter(str(row["provider"]) for row in accepted)),
            "licenses": dict(Counter(str(row["license"]) for row in accepted)),
        },
        "rejected_candidates": dict(sorted(failures.items())),
        "families_sha256": artifact_sha256(accepted),
        "license_review_complete": False,
    }
    (target / "inventory_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def write_recruitment_slots(
    *,
    output_path: str | Path,
) -> int:
    role_counts = {
        "policy_author": 8,
        "request_author": 12,
        "annotator": 10,
        "adjudicator": 3,
    }
    rows = []
    for role, count in role_counts.items():
        for index in range(1, count + 1):
            rows.append(
                {
                    "recruitment_slot_id": f"{role}-{index:02d}",
                    "role": role,
                    "required_languages": (
                        [] if role == "policy_author" else ["en", "es"]
                    ),
                    "status": "unfilled_external_recruitment",
                    "participant_id": None,
                }
            )
    return write_jsonl(output_path, rows)
