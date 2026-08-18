from tapbench.state_output_contract import build_trusted_dialogue_state


def test_structured_and_scalar_outputs_are_versioned() -> None:
    state = build_trusted_dialogue_state(
        [
            {
                "content": "1785194835.0",
                "version": "clock-1",
                "function_name": "get_current_timestamp",
                "failed": False,
            },
            {
                "content": "{'latitude': 51.5, 'longitude': -0.1}",
                "version": "location-1",
                "function_name": "get_current_location",
                "failed": False,
            },
        ]
    )
    assert state["timestamp_0"] == [
        {"value": 1785194835.0, "version": "clock-1"}
    ]
    assert state["timestamp_1"] == [
        {"value": 1785194835.0, "version": "clock-1"}
    ]
    assert state["latitude"] == [{"value": 51.5, "version": "location-1"}]


def test_failures_and_uncontracted_scalars_do_not_enter_state() -> None:
    state = build_trusted_dialogue_state(
        [
            {
                "content": "ConnectionError: Wifi is not enabled",
                "version": "failed-1",
                "function_name": "search_holiday",
                "failed": True,
            },
            {
                "content": "False",
                "version": "status-1",
                "function_name": "get_wifi_status",
                "failed": False,
            },
        ]
    )
    assert state == {}
