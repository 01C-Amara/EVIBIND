from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .evibench_freeze import artifact_sha256
from .evibench_inventory import InventoryError, normalized_tools
from .io import read_jsonl


INVENTORY_AUDIT_VERSION = "evibind.evibench_family_inventory_audit.v1"


def _contained_file(root: Path, relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise InventoryError(f"{label} must be a non-empty relative path")
    root_resolved = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise InventoryError(f"{label} escapes the inventory directory") from exc
    if not candidate.is_file():
        raise InventoryError(f"{label} is missing: {relative}")
    return candidate


def _load_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise InventoryError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise InventoryError(f"{label} must be a JSON mapping: {path}")
    return value


def audit_family_inventory(
    *,
    families_path: str | Path,
    inventory_dir: str | Path,
    expected_count: int,
    expected_split_counts: Mapping[str, int],
    manifest_path: str | Path | None = None,
    directory_index_path: str | Path | None = None,
    maximum_per_category: int = 8,
    maximum_per_provider: int = 4,
) -> dict[str, Any]:
    """Recompute every source, normalized schema, and inventory digest."""
    families = read_jsonl(families_path)
    root = Path(inventory_dir)
    if len(families) != expected_count:
        raise InventoryError(
            f"expected {expected_count} families, found {len(families)}"
        )

    family_ids = [row.get("family") for row in families]
    if any(not isinstance(family, str) or not family for family in family_ids):
        raise InventoryError("every family must have a non-empty string identifier")
    if len(set(family_ids)) != len(family_ids):
        raise InventoryError("family identifiers must be unique")

    actual_splits = Counter(str(row.get("split")) for row in families)
    if dict(actual_splits) != dict(expected_split_counts):
        raise InventoryError(
            f"split drift: expected {dict(expected_split_counts)}, "
            f"found {dict(actual_splits)}"
        )

    categories = Counter(str(row.get("category")) for row in families)
    providers = Counter(str(row.get("provider")) for row in families)
    if categories and max(categories.values()) > maximum_per_category:
        raise InventoryError("category concentration exceeds the configured cap")
    if providers and max(providers.values()) > maximum_per_provider:
        raise InventoryError("provider concentration exceeds the configured cap")

    source_digests: set[str] = set()
    index_digests: set[str] = set()
    review_statuses: Counter[str] = Counter()
    operation_counts: Counter[int] = Counter()
    for row in families:
        family = str(row["family"])
        source_path = _contained_file(
            root,
            row.get("source_artifact"),
            label=f"{family}.source_artifact",
        )
        schema_path = _contained_file(
            root,
            row.get("schema_artifact"),
            label=f"{family}.schema_artifact",
        )

        source_bytes = source_path.read_bytes()
        source_digest = hashlib.sha256(source_bytes).hexdigest()
        if source_digest != row.get("source_artifact_sha256"):
            raise InventoryError(f"source digest mismatch for {family}")
        if source_digest in source_digests:
            raise InventoryError(f"duplicate source artifact for {family}")
        source_digests.add(source_digest)

        source_spec = _load_mapping(source_path, label=f"{family} source")
        derived_tools, derived_operations = normalized_tools(source_spec)
        schema = _load_mapping(schema_path, label=f"{family} normalized schema")
        if schema.get("family") != family:
            raise InventoryError(f"normalized schema family mismatch for {family}")
        if schema.get("version") != "evibind.evibench_normalized_tool_family.v1":
            raise InventoryError(f"normalized schema version mismatch for {family}")
        if schema.get("tools") != derived_tools:
            raise InventoryError(f"normalized tools do not reproduce for {family}")
        if schema.get("operation_sources") != derived_operations:
            raise InventoryError(
                f"operation provenance does not reproduce for {family}"
            )
        if not 2 <= len(derived_tools) <= 4:
            raise InventoryError(f"operation count is outside [2, 4] for {family}")
        if row.get("operation_count") != len(derived_tools):
            raise InventoryError(f"operation count mismatch for {family}")
        if row.get("schema_sha256") != artifact_sha256([derived_tools]):
            raise InventoryError(f"schema digest mismatch for {family}")

        index_digest = row.get("source_index_sha256")
        if not isinstance(index_digest, str) or len(index_digest) != 64:
            raise InventoryError(f"invalid source index digest for {family}")
        index_digests.add(index_digest)
        review_status = str(row.get("license_review_status"))
        if review_status not in {"pending_human_confirmation", "human_confirmed"}:
            raise InventoryError(f"invalid license review status for {family}")
        review_statuses[review_status] += 1
        operation_counts[len(derived_tools)] += 1

    if len(index_digests) != 1:
        raise InventoryError("families do not share one source index digest")
    index_digest = next(iter(index_digests))
    if directory_index_path is not None:
        actual_index_digest = hashlib.sha256(
            Path(directory_index_path).read_bytes()
        ).hexdigest()
        if actual_index_digest != index_digest:
            raise InventoryError("directory index digest mismatch")

    manifest_file = (
        Path(manifest_path)
        if manifest_path is not None
        else root / "inventory_manifest.json"
    )
    manifest = _load_mapping(manifest_file, label="inventory manifest")
    if manifest.get("families_sha256") != artifact_sha256(families):
        raise InventoryError("inventory manifest family digest mismatch")
    if manifest.get("directory_index_sha256") != index_digest:
        raise InventoryError("inventory manifest index digest mismatch")
    manifest_counts = manifest.get("counts")
    if not isinstance(manifest_counts, Mapping):
        raise InventoryError("inventory manifest counts are missing")
    if manifest_counts.get("families") != expected_count:
        raise InventoryError("inventory manifest family count mismatch")
    if manifest_counts.get("splits") != dict(actual_splits):
        raise InventoryError("inventory manifest split counts mismatch")

    return {
        "version": INVENTORY_AUDIT_VERSION,
        "passed": True,
        "families_sha256": artifact_sha256(families),
        "directory_index_sha256": index_digest,
        "counts": {
            "families": len(families),
            "sources": len(source_digests),
            "schemas": len(families),
            "providers": len(providers),
            "categories": len(categories),
            "splits": dict(actual_splits),
            "license_review_statuses": dict(review_statuses),
            "operation_counts": {
                str(key): value for key, value in sorted(operation_counts.items())
            },
        },
    }
