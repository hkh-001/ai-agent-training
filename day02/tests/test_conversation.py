"""消息构造、会话顺序和裁剪测试。"""

import pytest

from day02.examples.conversation import (
    Conversation,
    build_message,
    trim_to_recent_rounds,
)


def test_build_message_strips_content() -> None:
    assert build_message("user", "  你好  ") == {
        "role": "user",
        "content": "你好",
    }


def test_build_message_rejects_empty_content() -> None:
    with pytest.raises(ValueError, match="不能为空"):
        build_message("assistant", "   ")


def test_conversation_preserves_role_order() -> None:
    conversation = Conversation("系统规则", max_rounds=2)
    conversation.add_user("问题")
    conversation.add_assistant("回答")

    assert [message["role"] for message in conversation.messages] == [
        "system",
        "user",
        "assistant",
    ]


def test_trim_keeps_system_and_recent_complete_rounds() -> None:
    messages = [
        build_message("system", "规则"),
        build_message("user", "第一问"),
        build_message("assistant", "第一答"),
        build_message("user", "第二问"),
        build_message("assistant", "第二答"),
        build_message("user", "第三问"),
    ]

    trimmed = trim_to_recent_rounds(messages, max_rounds=2)

    assert [message["content"] for message in trimmed] == [
        "规则",
        "第二问",
        "第二答",
        "第三问",
    ]


def test_messages_property_protects_internal_state() -> None:
    conversation = Conversation("系统规则")
    copied_messages = conversation.messages
    copied_messages[0]["content"] = "篡改"

    assert conversation.messages[0]["content"] == "系统规则"

