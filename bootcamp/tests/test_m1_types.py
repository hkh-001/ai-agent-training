"""M1 验收测试 A：消息数据结构与 wire 序列化。

先读测试，再写实现。测试就是你和「下游使用方」之间的合同。
运行：uv run pytest bootcamp/tests/test_m1_types.py -v
"""

import json

import pytest

from bootcamp.common.types import (
    ChatMessage,
    ToolCall,
    assistant,
    messages_to_wire,
    system,
    tool_result,
    user,
)


def test_user_message_wire_omits_optional_fields() -> None:
    wire = user("你好").to_wire()
    assert wire == {"role": "user", "content": "你好"}


def test_system_message_roundtrip() -> None:
    msg = system("你是一名电商导购，只能基于工具结果回答。")
    assert ChatMessage.from_wire(msg.to_wire()) == msg


def test_tool_message_requires_tool_call_id() -> None:
    with pytest.raises(ValueError, match="tool_call_id"):
        ChatMessage(role="tool", content="ok").validate()


def test_tool_message_cannot_carry_tool_calls() -> None:
    msg = ChatMessage(
        role="tool",
        content="ok",
        tool_call_id="call_1",
        tool_calls=(ToolCall("call_1", "get_stock", {}),),
    )
    with pytest.raises(ValueError, match="tool_calls"):
        msg.validate()


def test_user_message_rejects_blank_content() -> None:
    with pytest.raises(ValueError, match="content"):
        ChatMessage(role="user", content="   ").validate()


def test_unknown_role_rejected() -> None:
    with pytest.raises(ValueError, match="role"):
        ChatMessage(role="critic", content="x").validate()


def test_assistant_with_tool_calls_may_have_none_content() -> None:
    msg = assistant(tool_calls=[ToolCall("call_1", "get_stock", {"sku": "A1"})])
    msg.validate()
    assert msg.content is None


def test_duplicate_tool_call_ids_rejected() -> None:
    msg = assistant(
        tool_calls=[
            ToolCall("call_1", "get_stock", {"sku": "A1"}),
            ToolCall("call_1", "get_price", {"sku": "A1"}),
        ]
    )
    with pytest.raises(ValueError, match="唯一"):
        msg.validate()


def test_assistant_tool_calls_serialize_arguments_as_json_string() -> None:
    wire = assistant(
        tool_calls=[ToolCall("call_1", "get_stock", {"sku": "A1", "qty": 2})]
    ).to_wire()

    assert "role" in wire and wire["role"] == "assistant"
    assert "content" not in wire  # None 不输出该键
    call = wire["tool_calls"][0]
    assert call["id"] == "call_1"
    assert call["type"] == "function"
    assert call["function"]["name"] == "get_stock"
    assert isinstance(call["function"]["arguments"], str)
    assert json.loads(call["function"]["arguments"]) == {"sku": "A1", "qty": 2}


def test_tool_message_wire_contains_tool_call_id() -> None:
    wire = tool_result("call_1", '{"available": 3}').to_wire()
    assert wire == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"available": 3}',
    }


def test_from_wire_parses_arguments_json() -> None:
    msg = ChatMessage.from_wire(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_stock",
                        "arguments": '{"sku": "A1"}',
                    },
                }
            ],
        }
    )
    assert msg.content is None
    assert msg.tool_calls[0].name == "get_stock"
    assert msg.tool_calls[0].arguments == {"sku": "A1"}


def test_from_wire_rejects_invalid_json_arguments() -> None:
    bad = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_stock", "arguments": "{sku: A1"},
            }
        ],
    }
    with pytest.raises(ValueError, match="arguments"):
        ChatMessage.from_wire(bad)


def test_from_wire_rejects_unknown_role() -> None:
    with pytest.raises(ValueError, match="role"):
        ChatMessage.from_wire({"role": "critic", "content": "x"})


def test_messages_to_wire_keeps_order() -> None:
    wire = messages_to_wire([system("s"), user("u"), assistant("a")])
    assert [w["role"] for w in wire] == ["system", "user", "assistant"]
