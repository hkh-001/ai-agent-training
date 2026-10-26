"""Mock 多轮行为测试。"""

from day02.examples.conversation import Conversation
from day02.examples.mock_chat_model import MockChatModel


def ask(conversation: Conversation, model: MockChatModel, content: str) -> str:
    conversation.add_user(content)
    answer = model.generate(conversation.messages)
    conversation.add_assistant(answer)
    return answer


def test_mock_remembers_nickname_while_message_is_present() -> None:
    conversation = Conversation("规则", max_rounds=3)
    model = MockChatModel()

    ask(conversation, model, "我叫小林")
    answer = ask(conversation, model, "我叫什么？")

    assert "小林" in answer


def test_mock_forgets_nickname_after_round_is_trimmed() -> None:
    conversation = Conversation("规则", max_rounds=3)
    model = MockChatModel()

    ask(conversation, model, "我叫小林")
    ask(conversation, model, "第一条")
    ask(conversation, model, "第二条")
    answer = ask(conversation, model, "我叫什么？")

    assert answer == "当前保留的消息历史中没有你的名字。"

