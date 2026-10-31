"""M5：Agent Loop —— 把"一次调用"变成"会自己循环的 Agent"。

一句话：Agent = 一个**有终止条件的 while 循环**，循环体是"问模型 → 执行工具 → 回填结果"。

四个出口（这是本模块的全部价值）::

    final             模型给出最终答案（没有 tool_calls）
    max_steps         步数用尽（防止模型无限调工具）
    tool_error_budget 连续工具失败过多（防止模型对着同一个错误死磕）
    budget_exceeded   上下文超预算（防止请求超窗 / 成本失控）

为什么必须有多个出口：
    只靠模型"自觉停下"是不可靠的。真实项目里，模型会因为工具报错而反复重试、
    因为描述含糊而在两个工具之间来回跳。**终止条件是宿主的责任，不是模型的。**

为什么错误计数要"连续"而不是"总数"：
    一次成功说明流程还在推进。如果按总数算，一个跑了 20 步但中间成功 18 次的任务
    会被误杀；按连续算，只杀"卡住"的情况。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from bootcamp.common.context import check_invariants, estimate_tokens
from bootcamp.common.dispatch import ToolResult, dispatch
from bootcamp.common.llm import ChatModel
from bootcamp.common.tools import ToolRegistry
from bootcamp.common.types import ChatMessage

EventHook = Callable[[str, dict[str, Any]], None]
# 执行前的闸门：返回 ToolResult 表示"不要执行，用这个结果"；
# 返回 None 表示放行。（M11 的人工审批就挂在这里。）
ToolGate = Callable[[Any], Any]


def _emit(hook: EventHook | None, kind: str, **data: Any) -> None:
    """安全地发出一个观测事件（没有钩子时是空操作）。"""
    if hook is not None:
        hook(kind, data)


class StopReason(str, Enum):
    """循环为什么停下来。这既是日志字段，也是评测指标。"""

    FINAL = "final"
    MAX_STEPS = "max_steps"
    TOOL_ERROR_BUDGET = "tool_error_budget"
    BUDGET_EXCEEDED = "budget_exceeded"


@dataclass(frozen=True)
class LoopConfig:
    """一次运行的安全边界。全部有默认值，便于测试时按需覆盖。"""

    max_steps: int = 5
    max_consecutive_tool_errors: int = 3
    max_context_tokens: int = 4000
    tool_timeout_s: float = 5.0


@dataclass(frozen=True)
class AgentRun:
    """一次运行的完整结果（可直接落盘成 trajectory）。"""

    messages: tuple[ChatMessage, ...]
    stop_reason: StopReason
    steps: int
    tool_results: tuple[ToolResult, ...]
    final_answer: str | None
    context_tokens: int


def run_agent(
    model: ChatModel,
    registry: ToolRegistry,
    messages: Sequence[ChatMessage],
    *,
    config: LoopConfig | None = None,
    on_event: EventHook | None = None,
    tool_gate: ToolGate | None = None,
) -> AgentRun:
    """跑一次 Agent。**保证一定终止**，并在退出时返回完整轨迹。

    `on_event` 是可选的观测钩子（M6 的 Tracer 会挂在这里）。
    用"钩子注入"而不是重写循环，是为了让 M5 的核心逻辑保持单一版本。
    """
    cfg = config or LoopConfig()
    history = list(messages)

    # 开跑前先自检：带着坏历史进场，后面每一步都是错的
    check_invariants(history)
    _emit(on_event, "run_start", context_tokens=estimate_tokens(history))

    tool_results: list[ToolResult] = []
    consecutive_errors = 0
    steps = 0
    tools_schema = registry.to_openai_tools()

    while steps < cfg.max_steps:
        # 出口 4：上下文预算（在调用模型前检查，省一次请求）
        if estimate_tokens(history) > cfg.max_context_tokens:
            return _finish(history, StopReason.BUDGET_EXCEEDED, steps, tool_results, on_event)

        _emit(on_event, "model_start", step=steps + 1)
        reply = model.invoke(history, tools=tools_schema)
        steps += 1
        _emit(
            on_event,
            "model_end",
            step=steps,
            prompt_tokens=reply.usage.prompt_tokens,
            completion_tokens=reply.usage.completion_tokens,
            has_tool_calls=bool(reply.message.tool_calls),
        )
        history.append(reply.message)

        # 出口 1：没有工具调用 = 最终答案
        if not reply.message.tool_calls:
            return _finish(history, StopReason.FINAL, steps, tool_results, on_event)

        # 行动：串行执行并回填。**必须保证每个 tool_call 都有一条 tool 消息**，
        # 否则轨迹违反协议（孤儿 tool_call → 下一次请求 400）。
        # 所以错误预算触发后，不再执行剩余调用，但要把它们记成"已取消"结果。
        budget_hit = False
        for call in reply.message.tool_calls:
            if budget_hit:
                skipped = _cancelled_result(call)
                tool_results.append(skipped)
                history.append(skipped.to_message())
                _emit(
                    on_event,
                    "tool_cancelled",
                    tool_call_id=call.id,
                    name=call.name,
                    reason="reached_error_budget",
                )
                continue

            gated = tool_gate(call) if tool_gate is not None else None
            if gated is not None:
                # 闸门拦下了：不执行，直接用闸门给的结果（如"已拒绝"/"待审批"）
                result = gated
                _emit(on_event, "tool_gated", tool_call_id=call.id, name=call.name,
                      ok=result.ok, cancelled=result.cancelled)
            else:
                _emit(on_event, "tool_start", tool_call_id=call.id, name=call.name)
                result = dispatch(call, registry, timeout_s=cfg.tool_timeout_s)
            tool_results.append(result)
            history.append(result.to_message())
            if gated is None:
                _emit(
                    on_event,
                    "tool_end",
                    tool_call_id=result.tool_call_id,
                    name=call.name,
                    ok=result.ok,
                    retryable=result.retryable,
                    latency_ms=round(result.latency_ms, 2),
                )

            consecutive_errors = 0 if result.ok else consecutive_errors + 1
            # 出口 3：连续失败过多。不要立刻 return——先把本批剩下的补成取消结果
            if consecutive_errors >= cfg.max_consecutive_tool_errors:
                budget_hit = True

        if budget_hit:
            return _finish(
                history, StopReason.TOOL_ERROR_BUDGET, steps, tool_results, on_event
            )

    # 出口 2：步数用尽
    return _finish(history, StopReason.MAX_STEPS, steps, tool_results, on_event)


def _cancelled_result(call) -> ToolResult:
    """为一个"没有再执行"的调用生成结构化结果。

    注意语义：它**不是失败**，也**不是成功**，而是"宿主主动放弃执行"。
    必须与真正执行过的结果区分开，否则轨迹会骗人（也骗训练数据）。
    """
    import json

    return ToolResult(
        tool_call_id=call.id,
        ok=False,
        content=json.dumps(
            {
                "ok": False,
                "error": "未执行：已达到连续失败预算，本次调用被取消",
                "retryable": False,
                "cancelled": True,
            },
            ensure_ascii=False,
        ),
        error="未执行：已达到连续失败预算",
        retryable=False,
        cancelled=True,
    )


def _finish(
    history: list[ChatMessage],
    reason: StopReason,
    steps: int,
    tool_results: list[ToolResult],
    on_event: EventHook | None = None,
) -> AgentRun:
    """统一收尾：抽出最终答案、估算上下文，保证轨迹可直接落盘。"""
    final_answer: str | None = None
    if reason is StopReason.FINAL and history and history[-1].role == "assistant":
        final_answer = history[-1].content

    _emit(on_event, "run_end", stop_reason=reason.value, steps=steps)

    return AgentRun(
        messages=tuple(history),
        stop_reason=reason,
        steps=steps,
        tool_results=tuple(tool_results),
        final_answer=final_answer,
        context_tokens=estimate_tokens(history),
    )
