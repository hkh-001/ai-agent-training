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

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from bootcamp.common.context import check_invariants, estimate_tokens
from bootcamp.common.dispatch import ToolResult, dispatch_many
from bootcamp.common.llm import ChatModel
from bootcamp.common.tools import ToolRegistry
from bootcamp.common.types import ChatMessage


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
) -> AgentRun:
    """跑一次 Agent。**保证一定终止**，并在退出时返回完整轨迹。"""
    cfg = config or LoopConfig()
    history = list(messages)

    # 开跑前先自检：带着坏历史进场，后面每一步都是错的
    check_invariants(history)

    tool_results: list[ToolResult] = []
    consecutive_errors = 0
    steps = 0
    tools_schema = registry.to_openai_tools()

    while steps < cfg.max_steps:
        # 出口 4：上下文预算（在调用模型前检查，省一次请求）
        if estimate_tokens(history) > cfg.max_context_tokens:
            return _finish(history, StopReason.BUDGET_EXCEEDED, steps, tool_results)

        reply = model.invoke(history, tools=tools_schema)
        steps += 1
        history.append(reply.message)

        # 出口 1：没有工具调用 = 最终答案
        if not reply.message.tool_calls:
            return _finish(history, StopReason.FINAL, steps, tool_results)

        # 行动：执行并回填（一条 call 对应一条 tool 消息）
        for result in dispatch_many(
            reply.message.tool_calls, registry, timeout_s=cfg.tool_timeout_s
        ):
            tool_results.append(result)
            history.append(result.to_message())

            consecutive_errors = 0 if result.ok else consecutive_errors + 1
            # 出口 3：连续失败过多，再试也是浪费预算
            if consecutive_errors >= cfg.max_consecutive_tool_errors:
                return _finish(
                    history, StopReason.TOOL_ERROR_BUDGET, steps, tool_results
                )

    # 出口 2：步数用尽
    return _finish(history, StopReason.MAX_STEPS, steps, tool_results)


def _finish(
    history: list[ChatMessage],
    reason: StopReason,
    steps: int,
    tool_results: list[ToolResult],
) -> AgentRun:
    """统一收尾：抽出最终答案、估算上下文，保证轨迹可直接落盘。"""
    final_answer: str | None = None
    if reason is StopReason.FINAL and history and history[-1].role == "assistant":
        final_answer = history[-1].content

    return AgentRun(
        messages=tuple(history),
        stop_reason=reason,
        steps=steps,
        tool_results=tuple(tool_results),
        final_answer=final_answer,
        context_tokens=estimate_tokens(history),
    )
