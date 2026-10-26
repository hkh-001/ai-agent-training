"""M1 验收测试 B：Mock 模型的可断言性。

Agent 开发 80% 的时间在排查「到底发给了模型什么」，所以 Mock 必须能把
每次请求原样记下来。这部分测试在检查你的 Mock 是否「老实」。
运行：uv run pytest bootcamp/tests/test_m1_llm.py -v
"""

import pytest

from bootcamp.common.llm import ChatModel, MockChatModel, ModelResponse, Usage
from bootcamp.common.types import ToolCall, assistant, user


def test_mock_returns_scripted_responses_in_order() -> None:
    mock = MockChatModel([assistant("第一句"), assistant("第二句")])

    assert mock.invoke([user("hi")]).message.content == "第一句"
    assert mock.invoke([user("hi")]).message.content == "第二句"


def test_mock_wraps_bare_message_with_zero_usage() -> None:
    mock = MockChatModel([assistant("ok")], model="mock-1")

    resp = mock.invoke([user("hi")])

    assert isinstance(resp, ModelResponse)
    assert resp.usage == Usage()
    assert resp.usage.total_tokens == 0
    assert resp.model == "mock-1"


def test_mock_passes_through_model_response() -> None:
    scripted = ModelResponse(
        message=assistant("ok"), usage=Usage(10, 5), model="mock-2"
    )
    mock = MockChatModel([scripted])

    assert mock.invoke([user("hi")]) == scripted


def test_mock_records_each_call_as_snapshot() -> None:
    mock = MockChatModel([assistant("ok")])
    schema = [{"type": "function", "function": {"name": "get_stock"}}]

    history = [user("A1 还有货吗")]
    mock.invoke(history, tools=schema, temperature=0.0)
    history.append(assistant("我不该影响记录"))

    assert len(mock.calls) == 1
    call = mock.calls[0]
    assert call.messages == (user("A1 还有货吗"),)  # 快照，不受后续 list 修改影响
    assert call.tools == tuple(schema)
    assert call.temperature == 0.0


def test_mock_records_tool_call_round() -> None:
    tool_msg = assistant(tool_calls=[ToolCall("call_1", "get_stock", {"sku": "A1"})])
    mock = MockChatModel([tool_msg])

    mock.invoke([user("A1 库存")])

    assert mock.calls[0].messages[0].role == "user"
    assert mock.remaining == 0


def test_mock_raises_when_script_exhausted() -> None:
    mock = MockChatModel([assistant("only one")])
    mock.invoke([user("hi")])

    with pytest.raises(RuntimeError, match="exhausted"):
        mock.invoke([user("hi")])


def test_mock_satisfies_chat_model_protocol() -> None:
    assert isinstance(MockChatModel([]), ChatModel)
