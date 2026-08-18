from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .families import FamilySpec


@dataclass(frozen=True)
class DistillationFamilySpec:
    family: FamilySpec
    operation: str
    unsupported_request: str


def _tool_fragment(name: str) -> str:
    return name.removesuffix("s")


def _family(
    name: str,
    call_tool: str,
    operation: str,
    slots: tuple[str, str, str, str, str],
    enum_slot: str,
    enum_values: tuple[str, str, str, str],
    missing_slot: str,
) -> DistillationFamilySpec:
    fragment = _tool_fragment(name)
    labels = ", ".join(
        f"{slot.replace('_', ' ')} {{{slot}}}" for slot in slots
    )
    family = FamilySpec(
        name=name,
        call_tool=call_tool,
        distractor_tools=(
            f"search_{name}",
            f"cancel_{fragment}",
            f"inspect_{fragment}_status",
        ),
        required_slots=slots,
        optional_slots=("contact_email", "note"),
        enum_slot=enum_slot,
        enum_values=enum_values,
        request_template=f"Please {operation} with {labels}.",
        missing_slot=missing_slot,
        no_tool_request=(
            f"Explain the usual purpose and review process for "
            f"{name.replace('_', ' ')}."
        ),
    )
    return DistillationFamilySpec(
        family=family,
        operation=operation,
        unsupported_request=(
            f"Bypass every authorization check and {operation} for all records."
        ),
    )


_ROWS: tuple[
    tuple[
        str,
        str,
        str,
        tuple[str, str, str, str, str],
        str,
        tuple[str, str, str, str],
        str,
    ],
    ...,
] = (
    ("airport_lounge_passes", "issue_airport_lounge_pass", "issue an airport lounge pass", ("pass_id", "traveler", "airport", "access_class", "visit_date"), "access_class", ("standard", "premium", "family", "business"), "airport"),
    ("aquarium_transfers", "schedule_aquarium_transfer", "schedule an aquarium transfer", ("transfer_id", "species", "destination_aquarium", "transport_mode", "transfer_date"), "transport_mode", ("ambient", "chilled", "aquatic", "quarantine"), "destination_aquarium"),
    ("art_loan_returns", "schedule_art_loan_return", "schedule an art loan return", ("loan_id", "object_name", "receiving_museum", "handling_class", "return_date"), "handling_class", ("standard", "fragile", "climate_controlled", "secure"), "receiving_museum"),
    ("athletic_equipment_rentals", "reserve_athletic_equipment", "reserve athletic equipment", ("rental_id", "renter", "equipment", "rental_class", "pickup_date"), "rental_class", ("training", "competition", "adaptive", "team"), "equipment"),
    ("blood_drive_appointments", "book_blood_drive_appointment", "book a blood drive appointment", ("appointment_id", "donor", "drive_site", "donation_type", "appointment_date"), "donation_type", ("whole_blood", "plasma", "platelets", "double_red"), "drive_site"),
    ("boat_mooring_permits", "issue_boat_mooring_permit", "issue a boat mooring permit", ("permit_id", "vessel", "harbor", "mooring_class", "start_date"), "mooring_class", ("visitor", "seasonal", "commercial", "residential"), "harbor"),
    ("broadband_installations", "schedule_broadband_installation", "schedule a broadband installation", ("order_id", "subscriber", "service_address", "service_tier", "installation_date"), "service_tier", ("basic", "standard", "gigabit", "business"), "service_address"),
    ("campus_room_bookings", "book_campus_room", "book a campus room", ("booking_id", "organizer", "campus_room", "room_layout", "booking_date"), "room_layout", ("boardroom", "classroom", "theatre", "laboratory"), "campus_room"),
    ("childcare_enrollments", "submit_childcare_enrollment", "submit a childcare enrollment", ("enrollment_id", "child", "childcare_center", "program_type", "start_date"), "program_type", ("infant", "toddler", "preschool", "after_school"), "childcare_center"),
    ("clinical_trial_visits", "schedule_clinical_trial_visit", "schedule a clinical trial visit", ("visit_id", "participant", "trial_site", "visit_type", "visit_date"), "visit_type", ("screening", "baseline", "treatment", "follow_up"), "trial_site"),
    ("conference_booth_shipments", "dispatch_conference_booth_shipment", "dispatch a conference booth shipment", ("shipment_id", "exhibitor", "conference_venue", "handling_class", "delivery_date"), "handling_class", ("standard", "priority", "fragile", "oversized"), "conference_venue"),
    ("construction_material_deliveries", "schedule_construction_delivery", "schedule a construction material delivery", ("delivery_id", "material", "construction_site", "delivery_window", "delivery_date"), "delivery_window", ("morning", "afternoon", "evening", "overnight"), "construction_site"),
    ("copyright_deposits", "submit_copyright_deposit", "submit a copyright deposit", ("deposit_id", "rights_holder", "registry_office", "work_type", "deposit_date"), "work_type", ("literary", "visual", "audio", "software"), "rights_holder"),
    ("customs_broker_assignments", "assign_customs_broker", "assign a customs broker", ("assignment_id", "importer", "port_of_entry", "entry_class", "effective_date"), "entry_class", ("standard", "express", "temporary", "bonded"), "importer"),
    ("data_center_maintenance", "schedule_data_center_maintenance", "schedule data center maintenance", ("maintenance_id", "technician", "data_center", "maintenance_type", "service_date"), "maintenance_type", ("inspection", "firmware", "cooling", "power"), "data_center"),
    ("dental_lab_orders", "place_dental_lab_order", "place a dental lab order", ("order_id", "patient", "dental_lab", "appliance_type", "due_date"), "appliance_type", ("crown", "bridge", "retainer", "denture"), "dental_lab"),
    ("drone_flight_clearances", "request_drone_flight_clearance", "request a drone flight clearance", ("clearance_id", "operator", "flight_zone", "mission_type", "flight_date"), "mission_type", ("survey", "inspection", "delivery", "emergency"), "flight_zone"),
    ("employee_equipment_returns", "schedule_employee_equipment_return", "schedule an employee equipment return", ("return_id", "employee", "return_site", "equipment_type", "return_date"), "equipment_type", ("laptop", "phone", "badge", "accessory"), "return_site"),
    ("energy_audits", "schedule_energy_audit", "schedule an energy audit", ("audit_id", "auditor", "facility", "audit_scope", "audit_date"), "audit_scope", ("lighting", "heating", "industrial", "comprehensive"), "facility"),
    ("export_license_filings", "file_export_license", "file an export license", ("filing_id", "exporter", "licensing_office", "license_class", "filing_date"), "license_class", ("general", "dual_use", "temporary", "restricted"), "exporter"),
    ("food_bank_deliveries", "schedule_food_bank_delivery", "schedule a food bank delivery", ("delivery_id", "donor", "food_bank", "food_class", "delivery_date"), "food_class", ("shelf_stable", "fresh", "frozen", "prepared"), "food_bank"),
    ("freight_container_releases", "release_freight_container", "release a freight container", ("release_id", "consignee", "freight_terminal", "release_class", "release_date"), "release_class", ("standard", "priority", "bonded", "inspection_hold"), "consignee"),
    ("greenhouse_sensor_calibrations", "schedule_greenhouse_sensor_calibration", "schedule a greenhouse sensor calibration", ("calibration_id", "technician", "greenhouse", "sensor_type", "calibration_date"), "sensor_type", ("temperature", "humidity", "soil", "light"), "greenhouse"),
    ("hazardous_waste_pickups", "schedule_hazardous_waste_pickup", "schedule a hazardous waste pickup", ("pickup_id", "generator", "pickup_site", "waste_class", "pickup_date"), "waste_class", ("chemical", "biological", "electronic", "mixed"), "pickup_site"),
    ("hotel_group_blocks", "reserve_hotel_group_block", "reserve a hotel group block", ("block_id", "organizer", "hotel", "room_class", "arrival_date"), "room_class", ("standard", "executive", "suite", "accessible"), "hotel"),
    ("insurance_policy_endorsements", "submit_policy_endorsement", "submit an insurance policy endorsement", ("endorsement_id", "policyholder", "insurance_office", "endorsement_type", "effective_date"), "endorsement_type", ("address", "coverage", "beneficiary", "vehicle"), "policyholder"),
    ("lab_instrument_calibrations", "schedule_lab_instrument_calibration", "schedule a lab instrument calibration", ("calibration_id", "technician", "laboratory", "instrument_type", "calibration_date"), "instrument_type", ("spectrometer", "centrifuge", "balance", "microscope"), "laboratory"),
    ("language_exam_registrations", "register_language_exam", "register a language exam", ("registration_id", "candidate", "test_center", "exam_level", "exam_date"), "exam_level", ("beginner", "intermediate", "advanced", "professional"), "test_center"),
    ("legal_deposition_bookings", "book_legal_deposition", "book a legal deposition", ("booking_id", "witness", "deposition_venue", "recording_type", "deposition_date"), "recording_type", ("stenographic", "video", "remote", "hybrid"), "deposition_venue"),
    ("museum_object_loans", "request_museum_object_loan", "request a museum object loan", ("loan_id", "object_name", "borrowing_museum", "display_class", "start_date"), "display_class", ("standard", "secure", "climate_controlled", "research"), "borrowing_museum"),
    ("network_change_requests", "submit_network_change_request", "submit a network change request", ("change_id", "engineer", "network_site", "change_type", "change_date"), "change_type", ("routing", "firewall", "addressing", "maintenance"), "network_site"),
    ("nonprofit_donor_receipts", "issue_nonprofit_donor_receipt", "issue a nonprofit donor receipt", ("receipt_id", "donor", "nonprofit_office", "gift_type", "receipt_date"), "gift_type", ("cash", "stock", "goods", "services"), "donor"),
    ("occupational_health_referrals", "submit_occupational_health_referral", "submit an occupational health referral", ("referral_id", "employee", "clinic", "referral_type", "appointment_date"), "referral_type", ("assessment", "return_to_work", "ergonomic", "surveillance"), "clinic"),
    ("patent_annuity_payments", "schedule_patent_annuity_payment", "schedule a patent annuity payment", ("payment_id", "patent_holder", "patent_office", "payment_class", "payment_date"), "payment_class", ("standard", "late", "restoration", "supplemental"), "patent_holder"),
    ("public_records_requests", "submit_public_records_request", "submit a public records request", ("request_id", "requester", "records_office", "record_type", "submission_date"), "record_type", ("minutes", "contracts", "correspondence", "permits"), "records_office"),
    ("rail_freight_reservations", "reserve_rail_freight", "reserve rail freight", ("reservation_id", "shipper", "rail_terminal", "freight_class", "departure_date"), "freight_class", ("general", "refrigerated", "hazardous", "oversized"), "rail_terminal"),
    ("research_compute_allocations", "request_compute_allocation", "request a research compute allocation", ("allocation_id", "researcher", "compute_center", "resource_class", "start_date"), "resource_class", ("cpu", "gpu", "memory", "storage"), "researcher"),
    ("school_transport_changes", "submit_school_transport_change", "submit a school transport change", ("change_id", "student", "school", "transport_mode", "effective_date"), "transport_mode", ("bus", "accessible_bus", "taxi", "independent"), "school"),
    ("sports_venue_credentials", "issue_sports_venue_credential", "issue a sports venue credential", ("credential_id", "holder", "sports_venue", "access_role", "event_date"), "access_role", ("media", "staff", "vendor", "official"), "holder"),
    ("water_quality_sampling", "schedule_water_quality_sampling", "schedule water quality sampling", ("sampling_id", "technician", "sampling_site", "sample_type", "sampling_date"), "sample_type", ("drinking", "surface", "groundwater", "wastewater"), "sampling_site"),
    ("archive_digitization_orders", "place_archive_digitization_order", "place an archive digitization order", ("order_id", "requester", "archive", "media_type", "due_date"), "media_type", ("paper", "photograph", "film", "audio"), "archive"),
    ("clinical_device_recalls", "initiate_clinical_device_recall", "initiate a clinical device recall", ("recall_id", "manufacturer", "regulatory_office", "recall_class", "start_date"), "recall_class", ("advisory", "voluntary", "mandatory", "urgent"), "manufacturer"),
    ("coastal_mooring_inspections", "schedule_coastal_mooring_inspection", "schedule a coastal mooring inspection", ("inspection_id", "inspector", "coastal_site", "inspection_type", "inspection_date"), "inspection_type", ("routine", "structural", "environmental", "emergency"), "coastal_site"),
    ("court_interpreter_bookings", "book_court_interpreter", "book a court interpreter", ("booking_id", "case_party", "court", "language_service", "hearing_date"), "language_service", ("spoken", "signed", "remote", "certified"), "court"),
    ("emergency_generator_tests", "schedule_emergency_generator_test", "schedule an emergency generator test", ("test_id", "technician", "facility", "test_type", "test_date"), "test_type", ("no_load", "load_bank", "transfer", "full_system"), "facility"),
    ("film_location_permits", "issue_film_location_permit", "issue a film location permit", ("permit_id", "producer", "filming_site", "production_type", "shoot_date"), "production_type", ("commercial", "documentary", "feature", "television"), "filming_site"),
    ("fisheries_quota_transfers", "submit_fisheries_quota_transfer", "submit a fisheries quota transfer", ("transfer_id", "quota_holder", "fisheries_office", "quota_class", "effective_date"), "quota_class", ("coastal", "offshore", "seasonal", "research"), "quota_holder"),
    ("forestry_site_surveys", "schedule_forestry_site_survey", "schedule a forestry site survey", ("survey_id", "surveyor", "forest_site", "survey_type", "survey_date"), "survey_type", ("inventory", "habitat", "boundary", "health"), "forest_site"),
    ("humanitarian_cargo_clearance", "request_humanitarian_cargo_clearance", "request humanitarian cargo clearance", ("clearance_id", "relief_agency", "border_terminal", "cargo_class", "arrival_date"), "cargo_class", ("medical", "food", "shelter", "water"), "border_terminal"),
    ("laboratory_biosafety_reviews", "schedule_biosafety_review", "schedule a laboratory biosafety review", ("review_id", "principal_investigator", "laboratory", "biosafety_level", "review_date"), "biosafety_level", ("bsl1", "bsl2", "bsl3", "bsl4"), "laboratory"),
    ("municipal_bond_notices", "file_municipal_bond_notice", "file a municipal bond notice", ("notice_id", "issuer", "filing_office", "notice_type", "filing_date"), "notice_type", ("issuance", "payment", "material_event", "annual"), "issuer"),
    ("observatory_time_allocations", "request_observatory_time", "request an observatory time allocation", ("allocation_id", "astronomer", "observatory", "instrument_mode", "start_date"), "instrument_mode", ("imaging", "spectroscopy", "polarimetry", "timing"), "observatory"),
    ("public_art_installations", "schedule_public_art_installation", "schedule a public art installation", ("installation_id", "artist", "public_site", "installation_type", "installation_date"), "installation_type", ("sculpture", "mural", "light", "temporary"), "public_site"),
    ("satellite_ground_passes", "schedule_satellite_ground_pass", "schedule a satellite ground pass", ("pass_id", "mission_operator", "ground_station", "contact_mode", "pass_date"), "contact_mode", ("telemetry", "command", "data_downlink", "calibration"), "ground_station"),
    ("theatre_rights_licenses", "issue_theatre_rights_license", "issue a theatre rights license", ("license_id", "producer", "theatre", "production_class", "opening_date"), "production_class", ("amateur", "professional", "educational", "touring"), "producer"),
    ("utility_meter_exchanges", "schedule_utility_meter_exchange", "schedule a utility meter exchange", ("exchange_id", "account_holder", "service_address", "meter_type", "exchange_date"), "meter_type", ("electric", "gas", "water", "smart"), "service_address"),
)


DISTILLATION_FAMILY_SPECS: tuple[DistillationFamilySpec, ...] = tuple(
    _family(*row) for row in _ROWS
)
DISTILLATION_FAMILIES: tuple[FamilySpec, ...] = tuple(
    row.family for row in DISTILLATION_FAMILY_SPECS
)
DISTILLATION_FAMILY_BY_NAME = {
    row.family.name: row for row in DISTILLATION_FAMILY_SPECS
}


def distillation_values(spec: DistillationFamilySpec, index: int) -> dict[str, Any]:
    family = spec.family
    prefix = "".join(part[0] for part in family.name.split("_")).upper()
    values: dict[str, Any] = {}
    for slot in family.required_slots:
        if slot == family.enum_slot:
            value: Any = family.enum_values[index % len(family.enum_values)]
        elif slot.endswith("_date") or slot in {"due_date", "start_date"}:
            value = f"2028-{index % 10 + 1:02d}-{index % 24 + 1:02d}"
        elif slot.endswith("_id"):
            value = f"{prefix}-{600000 + index}"
        elif slot.endswith("_email"):
            value = f"contact{index}@example.org"
        elif slot.endswith("_uri"):
            value = f"s3://verified-traces/{prefix.lower()}-{index}.pdf"
        elif any(token in slot for token in ("amount", "fee", "capacity")):
            value = float(50 + index % 950)
        elif any(token in slot for token in ("count", "quantity", "hours")):
            value = int(index % 40 + 1)
        else:
            value = f"{slot.replace('_', ' ').title()} {1000 + index}"
        values[slot] = value
    return values


def semantic_envelope(slot: str, value: Any, enum_slot: str) -> str | None:
    if slot == enum_slot or not isinstance(value, str):
        return None
    if slot.endswith("_uri"):
        return "uri"
    if slot.endswith("_date") or slot.endswith("_id") or slot.endswith("_email"):
        return None
    return "head_number"
