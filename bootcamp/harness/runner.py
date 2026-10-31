"""M6：最小 Harness —— 一次运行的执行器 + 记录仪 + 预算官。

它把 M5 的 `run_agent` 包起来，额外提供四件事：

    配置     RunConfig 统一收纳参数（可复现、可对比）
    观测     trace 事件流（六类事件、连续序号、成对 start/end）
    成本     token 累加 + 美元估算
    韧性     模型调用异常时按策略重试（并记录重试次数）

设计上的一个关键决定：**不重写循环**。
M6 通过 `on_event` 钩子挂进 M5，而不是复制一份循环逻辑——
否则"循环改一处、harness 改一处"，两边迟早不一致。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from bootcamp.common.config import RunConfig
from bootcamp.common.llm import ChatModel, ModelResponse, Usage
from bootcamp.common.loop import AgentRun, run_agent
from bootcamp.common.telemetry import (
    TraceEvent,
    Tracer,
    configure_logging,
    estimate_cost,
    log_run_summary,
)
from bootcamp.common.tools import ToolRegistry
from bootcamp.common.types import ChatMessage


class HarnessError(RuntimeError):
    """Harness 层面的失败（如模型调用重试耗尽）。"""


@dataclass(frozen=True)
class RunResult:
    """一次 harness 运行的完整产出：轨迹 + 事件 + 成本 + 耗时。"""

    run: AgentRun
    trace: tuple[TraceEvent, ...]
    usage: Usage
    cost_usd: float
    elapsed_ms: float
    retries: int
    # 物理请求次数 = 逻辑步数 + 重试次数。
    # 区分 attempts 和 steps 很重要：成本按 attempts 算，进度按 steps 看。
    attempts: int = 0


class _RetryingModel:
    """模型调用重试装饰器（只处理异常，不处理"回答得不好"）。

    为什么重试放在模型边界，而不是循环里：
    网络抖动是**调用级**问题，重试一次就够；而"回答不好"是**语义级**问题，
    重试会有副作用（重复扣库存），必须交给模型自己改（M4/M5）。
    """

    def __init__(self, inner: ChatModel, retries: int, on_retry) -> None:
        self._inner = inner
        self._retries = retries
        self._on_retry = on_retry

    def invoke(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict[str, Any]] = (),
        temperature: float | None = None,
    ) -> ModelResponse:
        last_error: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                return self._inner.invoke(
                    messages, tools=tools, temperature=temperature
                )
            except Exception as exc:  # noqa: BLE001 - 网络类异常种类多，统一兜住
                last_error = exc
                if attempt < self._retries:
                    self._on_retry(attempt + 1, exc)
                    time.sleep(min(0.02 * (2**attempt), 0.2))  # 轻量退避
        raise HarnessError(
            f"模型调用失败（已重试 {self._retries} 次）: {type(last_error).__name__}"
        ) from last_error


class Runner:
    """一次性执行器：`Runner.run(messages)` → `RunResult`。"""

    def __init__(
        self,
        model: ChatModel,
        registry: ToolRegistry,
        config: RunConfig | None = None,
    ) -> None:
        self.config = config or RunConfig()
        self.model = model
        self.registry = registry
        self._tracer = Tracer()
        self._retries = 0

    @property
    def trace(self) -> tuple[TraceEvent, ...]:
        """当前累积的 trace（即使 run() 抛异常也能取到现场）。"""
        return tuple(self._tracer.events)

    def _on_event(self, kind: str, data: dict[str, Any]) -> None:
        self._tracer.emit(kind, **data)

    def _on_retry(self, attempt: int, exc: Exception) -> None:
        self._retries += 1
        self._tracer.emit(
            "model_retry",
            attempt=attempt,
            error=type(exc).__name__,
        )

    def run(
        self, messages: Sequence[ChatMessage], *, tool_gate=None
    ) -> RunResult:
        configure_logging(self.config.log_level)
        self._tracer = Tracer()
        self._retries = 0

        wrapped = _RetryingModel(
            self.model, self.config.model_retries, self._on_retry
        )
        started = time.perf_counter()
        try:
            run = run_agent(
                wrapped,
                self.registry,
                messages,
                config=self.config.loop,
                on_event=self._on_event,
                tool_gate=tool_gate,
            )
        except Exception as exc:
            # 失败也要留下现场：trace 里有 run_start / model_start / run_error，
            # 一眼就能看出死在哪一步——这正是"成对事件"的价值。
            self._tracer.emit(
                "run_error", error=type(exc).__name__, message=str(exc)[:100]
            )
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000

        usage = self._aggregate_usage()
        cost = estimate_cost(
            usage,
            price_prompt_per_1k=self.config.price_prompt_per_1k,
            price_completion_per_1k=self.config.price_completion_per_1k,
        )
        log_run_summary(
            run_name=self.config.run_name,
            stop_reason=run.stop_reason.value,
            steps=run.steps,
            tool_calls=len(run.tool_results),
            usage=usage,
            cost_usd=cost,
            elapsed_ms=elapsed_ms,
            retries=self._retries,
        )

        return RunResult(
            run=run,
            trace=tuple(self._tracer.events),
            usage=usage,
            cost_usd=cost,
            elapsed_ms=elapsed_ms,
            retries=self._retries,
            attempts=run.steps + self._retries,
        )

    def _aggregate_usage(self) -> Usage:
        """从 trace 的 model_end 事件累加 token（不在别处重复记账）。"""
        prompt = completion = 0
        for event in self._tracer.events:
            if event.kind == "model_end":
                prompt += int(event.data.get("prompt_tokens", 0))
                completion += int(event.data.get("completion_tokens", 0))
        return Usage(prompt_tokens=prompt, completion_tokens=completion)
