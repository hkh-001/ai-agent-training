"""真实模型适配器的兼容性测试（完全离线，不需要网络/Key）。

这些测试来自一次真实联调（DeepSeek）的观察，用来防止"Mock 能过、真机不能用"：

1. 真实返回里 `content` 是**空字符串** `""`（不是 None），且带 tool_calls；
2. 真实 tool_call id 形如 `call_00_21AUeXViUzVDMeAh7okU5289`（不透明，不能重生成）；
3. 某些环境的 NO_PROXY 含 `[::1]`，会让 httpx 直接抛 Invalid port；
4. 没有 Key 时应当给出清晰错误，而不是 KeyError。
"""


import pytest

from bootcamp.common.llm_openai import (
    ModelCallError,
    OpenAICompatChatModel,
    _build_http_client,
)
from bootcamp.common.types import ChatMessage

# 取自真实响应（已脱敏：只保留结构）
REAL_TOOL_CALL_PAYLOAD = {
    "role": "assistant",
    "content": "",                     # ← 真实是空字符串，不是 None
    "tool_calls": [
        {
            "id": "call_00_21AUeXViUzVDMeAh7okU5289",
            "type": "function",
            "function": {"name": "get_stock", "arguments": '{"sku": "A100"}'},
        }
    ],
    "refusal": None,
    "annotations": [],
}


def test_real_payload_parses() -> None:
    message = ChatMessage.from_wire(REAL_TOOL_CALL_PAYLOAD)

    assert message.role == "assistant"
    assert message.content == ""
    assert message.tool_calls[0].name == "get_stock"
    assert message.tool_calls[0].arguments == {"sku": "A100"}
    assert message.tool_calls[0].id == "call_00_21AUeXViUzVDMeAh7okU5289"


def test_empty_string_content_with_tool_calls_is_valid() -> None:
    """空字符串 content 必须合法（真实 API 就是这么返回的）。"""
    message = ChatMessage.from_wire(REAL_TOOL_CALL_PAYLOAD)

    message.validate()  # 不抛异常即通过


def test_missing_api_key_raises_model_call_error(monkeypatch) -> None:
    for name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ModelCallError, match="API Key"):
        OpenAICompatChatModel()


def test_http_client_tolerates_ipv6_no_proxy(monkeypatch) -> None:
    """NO_PROXY 里含 [::1] 时，httpx 解析会崩；我们要能兜住。"""
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1,::1,[::1]")
    monkeypatch.delenv("HTTPS_PROXY", raising=False)

    client = _build_http_client(5.0)

    assert client is not None
    assert client.trust_env is False     # 关键：不再读 NO_PROXY
    client.close()
