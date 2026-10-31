"""真实模型适配器：OpenAI 兼容协议（DeepSeek / OpenAI / 本地 vLLM 均可）。

为什么单独一个文件，而不是塞进 llm.py：
    llm.py 是"零依赖"的（只有标准库），这样 200 多个测试不需要装 openai、
    也不依赖网络。真实 SDK 只在这一层出现，其余代码只认识 ChatModel 协议。

它做的三件事（正好是 M1 的两个转换点）::

    内部对象 ──to_wire()──▶ dict ──openai SDK──▶ JSON ──▶ 服务端
    内部对象 ◀──from_wire()── dict ◀──SDK── JSON ◀── 服务端

注意：`from_wire` 走的是和 Mock 完全相同的解析路径——这正是 M1 分层设计
的价值：换模型不需要改任何上层代码。
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Any

from bootcamp.common.llm import ModelResponse, Usage
from bootcamp.common.types import ChatMessage


def _build_http_client(timeout_s: float):
    """构造 httpx 客户端。

    为什么要显式传 proxy 并关掉 trust_env：
    某些环境的 NO_PROXY 写成 `...,[::1]`，httpx 把它当 URL 解析会直接抛
    `Invalid port: ':1]'`。显式传代理 + trust_env=False 可以绕开这个坑。
    """
    try:
        import httpx
    except ImportError:  # pragma: no cover
        return None

    proxy = (
        os.getenv("HTTPS_PROXY")
        or os.getenv("https_proxy")
        or os.getenv("ALL_PROXY")
        or os.getenv("all_proxy")
    )
    kwargs: dict[str, Any] = {"timeout": timeout_s, "trust_env": False}
    if proxy:
        kwargs["proxy"] = proxy
    try:
        return httpx.Client(**kwargs)
    except Exception:  # noqa: BLE001 - 代理配置异常时退回默认
        return None


class ModelCallError(RuntimeError):
    """真实模型调用失败（网络、鉴权、限流、服务端错误）。"""


class OpenAICompatChatModel:
    """把 OpenAI 兼容的 chat.completions 适配成项目的 ChatModel 协议。"""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout_s: float = 60.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - 环境问题
            raise ModelCallError("需要安装 openai 包：uv add openai") from exc

        self.model = model or os.getenv("DEEPSEEK_MODEL") or "deepseek-chat"
        key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ModelCallError("缺少 API Key：请在 .env 里设置 DEEPSEEK_API_KEY")

        self._client = OpenAI(
            api_key=key,
            base_url=base_url or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
            timeout=timeout_s,
            http_client=_build_http_client(timeout_s),
        )
        self.temperature = temperature
        self.calls: list[dict[str, Any]] = []

    def invoke(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict[str, Any]] = (),
        temperature: float | None = None,
    ) -> ModelResponse:
        payload = [message.to_wire() for message in messages]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": payload,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if tools:
            kwargs["tools"] = list(tools)

        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise ModelCallError(
                f"模型调用失败: {type(exc).__name__}: {exc}"
            ) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000

        self.calls.append({"messages": len(messages), "tools": len(tools)})

        if not response.choices:
            raise ModelCallError("模型没有返回任何 choice")

        raw = response.choices[0].message.model_dump()
        message = ChatMessage.from_wire(raw)          # ← 与 Mock 共用同一条解析路径

        usage = Usage(
            prompt_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
        )
        return ModelResponse(
            message=message,
            usage=usage,
            model=getattr(response, "model", self.model),
            latency_ms=elapsed_ms,
            raw={"finish_reason": response.choices[0].finish_reason},
        )
