"""Day 2 实践参考实现测试。"""

from day02.exercises.solution import NicknameAssistant


def test_reset_only_keeps_system_message() -> None:
    assistant = NicknameAssistant()
    assistant.reply("我叫小林")

    assistant.reset()

    assert len(assistant.messages) == 1
    assert assistant.messages[0]["role"] == "system"


def test_reply_records_user_and_assistant_messages() -> None:
    assistant = NicknameAssistant()

    assistant.reply("你好")

    assert [message["role"] for message in assistant.messages[-2:]] == [
        "user",
        "assistant",
    ]
