from tapbench.native_tool_runtime import (
    native_messages,
    normalize_native_message,
    openai_native_tools,
)


def _case() -> dict:
    return {
        "messages": [
            {
                "role": "system",
                "content": "Return exactly one Action IR JSON object. Do not call a tool unless relevant.",
            },
            {"role": "user", "content": "Look up London."},
        ],
        "tools": [
            {
                "name": "weather.lookup",
                "description": "Look up weather.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "x-tap-semantic-envelope": "free_text",
                        }
                    },
                    "required": ["city"],
                },
            }
        ],
    }


def test_native_tools_strip_tap_annotations() -> None:
    tools = openai_native_tools(_case())
    city = tools[0]["function"]["parameters"]["properties"]["city"]
    assert city == {"type": "string"}


def test_native_messages_remove_action_ir_instruction() -> None:
    qwen = native_messages(_case(), model_profile="qwen")
    gpt = native_messages(_case(), model_profile="gpt_oss")
    assert qwen[0]["role"] == "system"
    assert gpt[0]["role"] == "developer"
    assert gpt[0]["content"].startswith("Reasoning: medium.")
    assert all("Action IR" not in row["content"] for row in qwen)


def test_normalizes_native_tool_call() -> None:
    action, metadata = normalize_native_message(
        {
            "content": None,
            "reasoning_content": "The weather function applies.",
            "tool_calls": [
                {
                    "function": {
                        "name": "weather.lookup",
                        "arguments": '{"city":"London"}',
                    }
                }
            ],
        }
    )
    assert action == {
        "mode": "call",
        "tool": "weather.lookup",
        "arguments": {"city": "London"},
        "payload": {},
    }
    assert metadata["native_tool_call_count"] == 1


def test_normalizes_no_native_call_to_no_tool() -> None:
    action, metadata = normalize_native_message(
        {"content": "No listed function is relevant."}
    )
    assert action["mode"] == "no_tool"
    assert metadata["native_tool_call_count"] == 0
