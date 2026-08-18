from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FamilySpec:
    name: str
    call_tool: str
    distractor_tools: tuple[str, ...]
    required_slots: tuple[str, ...]
    optional_slots: tuple[str, ...]
    enum_slot: str
    enum_values: tuple[str, ...]
    request_template: str
    missing_slot: str
    no_tool_request: str


FAMILIES: tuple[FamilySpec, ...] = (
    FamilySpec(
        name="calendar",
        call_tool="create_calendar_event",
        distractor_tools=("search_calendar_events", "update_calendar_event", "cancel_calendar_event"),
        required_slots=("title", "date", "start_time", "end_time", "calendar"),
        optional_slots=("location", "attendees"),
        enum_slot="calendar",
        enum_values=("work", "personal", "team", "travel", "family", "school", "ops", "sales", "support", "planning", "private", "shared"),
        request_template="Create a {calendar} calendar event titled {title} on {date} from {start_time} to {end_time}.",
        missing_slot="date",
        no_tool_request="Summarize the latest astronomy news without using tools.",
    ),
    FamilySpec(
        name="email",
        call_tool="send_email",
        distractor_tools=("search_email", "archive_email", "schedule_email"),
        required_slots=("recipient", "subject", "body", "priority", "send_time"),
        optional_slots=("cc", "bcc"),
        enum_slot="priority",
        enum_values=("low", "normal", "high", "urgent", "follow_up", "newsletter", "receipt", "legal", "sales", "support", "internal", "external"),
        request_template="Send {recipient} a {priority} email with subject {subject} at {send_time}: {body}.",
        missing_slot="recipient",
        no_tool_request="Tell me whether email was invented before the web.",
    ),
    FamilySpec(
        name="weather",
        call_tool="get_weather_forecast",
        distractor_tools=("get_weather_history", "set_weather_alert", "compare_weather"),
        required_slots=("location", "date", "unit", "detail_level", "include_precipitation"),
        optional_slots=("hour", "language"),
        enum_slot="unit",
        enum_values=("celsius", "fahrenheit", "kelvin", "metric", "imperial", "wind_mph", "wind_kph", "rain_mm", "rain_in", "uv_index", "humidity", "pressure"),
        request_template="Get the {detail_level} weather forecast for {location} on {date} in {unit}.",
        missing_slot="location",
        no_tool_request="Explain why seasons happen.",
    ),
    FamilySpec(
        name="shopping",
        call_tool="add_product_to_cart",
        distractor_tools=("search_products", "remove_product_from_cart", "apply_coupon"),
        required_slots=("product", "quantity", "size", "color", "shipping_speed"),
        optional_slots=("gift_wrap", "store"),
        enum_slot="shipping_speed",
        enum_values=("standard", "expedited", "overnight", "pickup", "same_day", "two_day", "economy", "scheduled", "locker", "drone", "freight", "international"),
        request_template="Add {quantity} {color} {product} in size {size} to my cart with {shipping_speed} shipping.",
        missing_slot="size",
        no_tool_request="Compare cotton and linen for summer clothing.",
    ),
    FamilySpec(
        name="file_search",
        call_tool="search_files",
        distractor_tools=("open_file", "delete_file", "share_file"),
        required_slots=("query", "folder", "file_type", "modified_after", "owner"),
        optional_slots=("limit", "sort_by"),
        enum_slot="file_type",
        enum_values=("pdf", "docx", "txt", "csv", "xlsx", "pptx", "image", "video", "audio", "archive", "code", "markdown"),
        request_template="Search {folder} for {file_type} files owned by {owner} about {query} modified after {modified_after}.",
        missing_slot="query",
        no_tool_request="What are good habits for naming files?",
    ),
    FamilySpec(
        name="database",
        call_tool="query_customer_database",
        distractor_tools=("update_customer_record", "export_customer_database", "delete_customer_record"),
        required_slots=("table", "filter_field", "filter_value", "aggregation", "limit"),
        optional_slots=("order_by", "include_inactive"),
        enum_slot="aggregation",
        enum_values=("count", "sum", "average", "min", "max", "median", "distinct", "group_by", "none", "latest", "earliest", "percentile"),
        request_template="Query {table} where {filter_field} is {filter_value}, use {aggregation}, and limit to {limit} rows.",
        missing_slot="filter_value",
        no_tool_request="Explain what a database index is.",
    ),
    FamilySpec(
        name="support",
        call_tool="create_support_ticket",
        distractor_tools=("search_support_tickets", "close_support_ticket", "escalate_support_ticket"),
        required_slots=("customer_id", "issue", "severity", "product", "channel"),
        optional_slots=("attachment", "preferred_contact_time"),
        enum_slot="severity",
        enum_values=("low", "medium", "high", "critical", "billing", "security", "outage", "bug", "feature", "account", "performance", "data_loss"),
        request_template="Create a {severity} support ticket for customer {customer_id} about {product}: {issue} via {channel}.",
        missing_slot="customer_id",
        no_tool_request="What makes a good support ticket description?",
    ),
    FamilySpec(
        name="travel",
        call_tool="book_travel_itinerary",
        distractor_tools=("search_flights", "cancel_booking", "check_visa_requirements"),
        required_slots=("origin", "destination", "depart_date", "return_date", "cabin"),
        optional_slots=("hotel_needed", "traveler_count"),
        enum_slot="cabin",
        enum_values=("economy", "premium_economy", "business", "first", "sleeper", "coach", "standard", "flex", "basic", "comfort", "suite", "accessible"),
        request_template="Book a {cabin} trip from {origin} to {destination}, leaving {depart_date} and returning {return_date}.",
        missing_slot="return_date",
        no_tool_request="List common causes of flight delays.",
    ),
)


def get_family(name: str) -> FamilySpec:
    # Keep held-out confirmation families out of the main generation tuple while
    # allowing shared validation and slot-error code to score them.
    from .r2b_families import R2B_FAMILIES
    from .r2c_families import R2C_CONFIRM_FAMILIES, R2C_PILOT_FAMILIES
    from .r2d_families import R2D_CONFIRM_FAMILIES
    from .r2e_families import R2E_FAMILIES
    from .r2f_families import R2F_FAMILIES
    from .distillation_families import DISTILLATION_FAMILIES

    for family in (
        *FAMILIES,
        *R2B_FAMILIES,
        *R2C_PILOT_FAMILIES,
        *R2C_CONFIRM_FAMILIES,
        *R2D_CONFIRM_FAMILIES,
        *R2E_FAMILIES,
        *R2F_FAMILIES,
        *DISTILLATION_FAMILIES,
    ):
        if family.name == name:
            return family
    raise KeyError(f"unknown family: {name}")


def family_names() -> list[str]:
    return [family.name for family in FAMILIES]


def deterministic_values(family: FamilySpec, index: int) -> dict[str, Any]:
    return {
        "title": f"Team sync {index}",
        "date": f"2026-07-{(index % 20) + 9:02d}",
        "start_time": f"{9 + (index % 6):02d}:00",
        "end_time": f"{9 + (index % 6):02d}:30",
        "calendar": family.enum_values[index % len(family.enum_values)],
        "recipient": f"user{index}@example.com",
        "subject": f"Status update {index}",
        "body": f"Please review item {index}",
        "priority": family.enum_values[index % len(family.enum_values)],
        "send_time": f"2026-07-{(index % 20) + 9:02d}T15:00:00",
        "location": f"City {index % 7}",
        "unit": family.enum_values[index % len(family.enum_values)],
        "detail_level": "daily",
        "include_precipitation": True,
        "product": f"shirt {index}",
        "quantity": (index % 4) + 1,
        "size": ["XS", "S", "M", "L", "XL"][index % 5],
        "color": ["blue", "green", "black", "white"][index % 4],
        "shipping_speed": family.enum_values[index % len(family.enum_values)],
        "query": f"budget plan {index}",
        "folder": "/Team/Planning",
        "file_type": family.enum_values[index % len(family.enum_values)],
        "modified_after": f"2026-06-{(index % 25) + 1:02d}",
        "owner": f"owner{index % 5}",
        "table": "customers",
        "filter_field": "region",
        "filter_value": f"region-{index % 4}",
        "aggregation": family.enum_values[index % len(family.enum_values)],
        "limit": 25 + index,
        "customer_id": f"CUST-{1000 + index}",
        "issue": f"login failure {index}",
        "severity": family.enum_values[index % len(family.enum_values)],
        "channel": "email",
        "origin": "London",
        "destination": f"Destination {index % 6}",
        "depart_date": f"2026-08-{(index % 20) + 1:02d}",
        "return_date": f"2026-08-{(index % 20) + 8:02d}",
        "cabin": family.enum_values[index % len(family.enum_values)],
    }
