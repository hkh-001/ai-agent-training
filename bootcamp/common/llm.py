"""M1：模型调用边界——ChatModel 协议 + 可脚本化的 Mock。

为什么要先定义协议：
Agent Loop 只依赖 `ChatModel.invoke(...)` 这一个方法，因此真实模型、
Mock 模型、未来的自训练模型可以互换，Agent 代码一行不改。
测试里全用 Mock，保证「离线、确定性、可断言」。
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from bootcamp.common.types import ChatMessage


@dataclass(frozen=True)
class Usage:
    """一次调用的 token 消耗。Agent 的成本意识从这里开始。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class ModelResponse:
    """模型返回：一条 assistant 消息 + 用量 + 元信息。"""

    message: ChatMessage
    usage: Usage = field(default_factory=Usage)
    model: str = "unknown"
    latency_ms: float = 0.0
    raw: dict[str, Any] | None = None


@dataclass(frozen=True)
class ModelCall:
    """Mock 记录下来的「这一次请求到底发了什么」，用于断言与轨迹回放。"""

    messages: tuple[ChatMessage, ...]
    tools: tuple[dict[str, Any], ...] = ()
    temperature: float | None = None


@runtime_checkable
class ChatModel(Protocol):
    """所有模型适配器的统一接口。Agent Loop 只认这个协议。"""

    def invoke(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict[str, Any]] = (),
        temperature: float | None = None,
    ) -> ModelResponse: ...


class MockChatModel:
    """按脚本依次吐出的假模型。

    用法::

        mock = MockChatModel([assistant(tool_calls=[ToolCall("c1", "get_stock", {"sku": "A1"})])])
        resp = mock.invoke([user("A1 还有货吗")])

    契约（也是测试的断言点）：
    - 每次 invoke 都往 self.calls 追加一条 ModelCall（messages/tools 转成 tuple 快照）
    - 按顺序 pop(0) 脚本；脚本项可以是 ChatMessage 或 ModelResponse
      * ChatMessage → 包装成 ModelResponse（usage 全 0，model=self.model）
      * ModelResponse → 原样返回
    - 脚本耗尽时抛 RuntimeError，信息含 "exhausted"
    - remaining 属性返回剩余脚本条数
    """

    def __init__(
        self,
        script: Sequence[ChatMessage | ModelResponse],
        *,
        model: str = "mock-model",
    ) -> None:
        self._script: list[ChatMessage | ModelResponse] = list(script)
        self.model = model
        self.calls: list[ModelCall] = []

    @property
    def remaining(self) -> int:
        """剩余脚本条数。"""
        return len(self._script)

    def invoke(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict[str, Any]] = (),
        temperature: float | None = None,
    ) -> ModelResponse:
        # 1) 记录这次"到底发了什么"（快照，不是引用）
        #    messages 里是 frozen dataclass，浅拷贝即可；
        #    tools 里是可变 dict，必须深拷贝，否则调用方改 schema 会污染记录
        self.calls.append(
            ModelCall(
                messages=tuple(messages),
                tools=tuple(copy.deepcopy(tools)),
                temperature=temperature,
            )
        )

        # 2) 台词用完了就报错
        if not self._script:
            raise RuntimeError("MockChatModel script exhausted")

        # 3) 取下一句台词（注意：从头部取，保证顺序）
        item = self._script.pop(0)

        # 4) 如果是裸消息，包装成 ModelResponse；如果已经是 ModelResponse，原样返回
        if isinstance(item, ChatMessage):
            return ModelResponse(message=item, model=self.model)

        # 5) 不是 ChatMessage 就是 ModelResponse
        return item
