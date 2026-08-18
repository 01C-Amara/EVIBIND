from __future__ import annotations

from dataclasses import dataclass

from .families import FamilySpec
from .r2f_families import R2F_FAMILIES


@dataclass(frozen=True)
class HierarchyFamily:
    spec: FamilySpec
    source_family: str
    extent_stratum: str


_NAMES = (
    "outpatient_dispensing",
    "secure_zone_credentials",
    "contractor_settlements",
    "aquaculture_inspections",
    "vessel_certifications",
    "fellowship_disbursements",
    "archive_transfers",
    "broadband_plan_migrations",
    "tissue_shipments",
    "festival_catering",
    "zoning_applications",
    "appliance_warranty_claims",
    "depot_replenishments",
    "dataset_submissions",
    "wind_interconnections",
    "disaster_response_shifts",
    "donor_material_releases",
    "datacenter_access_badges",
    "royalty_payments",
    "forestry_inspections",
    "drone_registrations",
    "research_award_disbursements",
    "collection_relocations",
    "cloud_subscription_changes",
)

_STRATA = (
    "labelled_identity_extent",
    "opaque_identifier",
    "uri",
    "enum_and_cross_field_constraint",
    "bounded_date_or_number_transform",
)

_SOURCE_INDEXES = {
    "labelled_identity_extent": (0, 1, 2, 3, 4),
    "opaque_identifier": (0, 1, 2, 6, 7),
    "uri": (13,),
    "enum_and_cross_field_constraint": (5, 8, 9, 10, 15),
    "bounded_date_or_number_transform": (2, 3, 4, 12),
}


def _clone(source: FamilySpec, name: str) -> FamilySpec:
    operation = name.removesuffix("s")
    return FamilySpec(
        name=name,
        call_tool=f"execute_{operation}",
        distractor_tools=(
            f"search_{name}",
            f"cancel_{operation}",
            f"inspect_{operation}",
        ),
        required_slots=source.required_slots,
        optional_slots=source.optional_slots,
        enum_slot=source.enum_slot,
        enum_values=source.enum_values,
        request_template=source.request_template,
        missing_slot=source.missing_slot,
        no_tool_request=source.no_tool_request,
    )


def _source_for(index: int) -> FamilySpec:
    stratum = _STRATA[index % len(_STRATA)]
    indexes = _SOURCE_INDEXES[stratum]
    return R2F_FAMILIES[indexes[(index // len(_STRATA)) % len(indexes)]]


HIERARCHY_FAMILIES: tuple[HierarchyFamily, ...] = tuple(
    HierarchyFamily(
        spec=_clone(_source_for(index), name),
        source_family=_source_for(index).name,
        extent_stratum=_STRATA[index % len(_STRATA)],
    )
    for index, name in enumerate(_NAMES)
)
