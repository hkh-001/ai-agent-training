"""M5 验收测试：四个出口 + 轨迹正确性。

运行：uv run pytest bootcamp/tests/test_m5_loop.py -q
"""

import pytest

from bootcamp.common.context import check_invariants
from bootcamp.common.dispatch import ToolBusinessError
from bootcamp.common.llm import MockChatModel
from bootcamp.common.loop import LoopConfig, StopReason, run_agent
from bootcamp.common.tools import ToolRegistry
from bootcamp.common.types import ToolCall, assistant, system, user

# ------------------------------------------------------------------ 测试用工具


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()

    @registry.register
    def get_stock(sku: str) -> dict:
        """查询库存。

        Args:
            sku: 商品编码
        """
        return {"sku": sku, "available": 5}

    @registry.register
    def reserve_stock(sku: str, qty: int) -> dict:
        """预留库存。

        Args:
            sku: 商品编码
            qty: 件数
        """
        if qty > 5:
            raise ToolBusinessError("库存不足：当前可用 5 件")
        return {"sku": sku, "reserved": qty}

    return registry


def start() -> list:
    return [system("你是电商客服"), user("A1 还有货吗")]


# ------------------------------------------------------------------ 出口 1：final


def test_final_on_first_reply() -> None:
    model = MockChatModel([assistant("你好，有什么可以帮你？")])

    run = run_agent(model, make_registry(), start())

    assert run.stop_reason is StopReason.FINAL
    assert run.steps == 1
    assert run.final_answer == "你好，有什么可以帮你？"
    assert run.tool_results == ()


def test_tool_then_final() -> None:
    model = MockChatModel(
        [
            assistant(tool_calls=[ToolCall("c1", "get_stock", {"sku": "A1"})]),
            assistant("A1 还有 5 件。"),
        ]
    )

    run = run_agent(model, make_registry(), start())

    assert run.stop_reason is StopReason.FINAL
    assert run.steps == 2
    assert run.final_answer == "A1 还有 5 件。"
    assert [m.role for m in run.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    check_invariants(run.messages)  # 轨迹必须合法（可直接用于训练数据）
    assert len(run.tool_results) == 1
    assert run.tool_results[0].ok is True


# ------------------------------------------------------------------ 出口 2：max_steps


def test_max_steps_stops_loop() -> None:
    call = ToolCall("c1", "get_stock", {"sku": "A1"})
    model = MockChatModel([assistant(tool_calls=[call]) for _ in range(10)])

    run = run_agent(model, make_registry(), start(), config=LoopConfig(max_steps=3))

    assert run.stop_reason is StopReason.MAX_STEPS
    assert run.steps == 3
    assert run.final_answer is None
    assert len(run.tool_results) == 3


# ------------------------------------------------------------------ 出口 3：连续失败


def test_tool_error_budget_stops_loop() -> None:
    # 每次都缺必填参数 sku → 每次都是可重试失败
    bad_call = ToolCall("c1", "get_stock", {})
    model = MockChatModel([assistant(tool_calls=[bad_call]) for _ in range(10)])

    run = run_agent(
        model,
        make_registry(),
        start(),
        config=LoopConfig(max_steps=10, max_consecutive_tool_errors=2),
    )

    assert run.stop_reason is StopReason.TOOL_ERROR_BUDGET
    assert run.steps == 2
    assert all(not r.ok for r in run.tool_results)


def test_error_counter_resets_after_success() -> None:
    model = MockChatModel(
        [
            assistant(tool_calls=[ToolCall("c1", "get_stock", {})]),          # 失败
            assistant(tool_calls=[ToolCall("c2", "get_stock", {"sku": "A1"})]),  # 成功
            assistant(tool_calls=[ToolCall("c3", "get_stock", {})]),          # 失败
            assistant("好的，A1 有 5 件。"),
        ]
    )

    run = run_agent(
        model,
        make_registry(),
        start(),
        config=LoopConfig(max_consecutive_tool_errors=2),
    )

    assert run.stop_reason is StopReason.FINAL  # 没有连续 2 次失败，不该被误杀
    assert run.steps == 4


# ------------------------------------------------------------------ 出口 4：预算


def test_budget_exceeded_before_first_call() -> None:
    model = MockChatModel([assistant("永远不会被调用到")])

    run = run_agent(
        model,
        make_registry(),
        start(),
        config=LoopConfig(max_context_tokens=1),
    )

    assert run.stop_reason is StopReason.BUDGET_EXCEEDED
    assert run.steps == 0
    assert model.calls == []


# ------------------------------------------------------------------ 轨迹正确性


def test_multiple_tool_calls_in_one_turn() -> None:
    model = MockChatModel(
        [
            assistant(
                tool_calls=[
                    ToolCall("c1", "get_stock", {"sku": "A1"}),
                    ToolCall("c2", "reserve_stock", {"sku": "A1", "qty": 99}),
                ]
            ),
            assistant("A1 最多只能预留 5 件。"),
        ]
    )

    run = run_agent(model, make_registry(), start())

    tool_messages = [m for m in run.messages if m.role == "tool"]
    assert [m.tool_call_id for m in tool_messages] == ["c1", "c2"]
    assert [r.ok for r in run.tool_results] == [True, False]
    assert run.stop_reason is StopReason.FINAL


def test_model_receives_tool_schema_every_turn() -> None:
    model = MockChatModel(
        [
            assistant(tool_calls=[ToolCall("c1", "get_stock", {"sku": "A1"})]),
            assistant("A1 还有 5 件。"),
        ]
    )

    run_agent(model, make_registry(), start())

    for call in model.calls:
        names = [tool["function"]["name"] for tool in call.tools]
        assert names == ["get_stock", "reserve_stock"]


def test_tool_call_ids_match_assistant_messages() -> None:
    model = MockChatModel(
        [
            assistant(tool_calls=[ToolCall("c1", "get_stock", {"sku": "A1"})]),
            assistant("A1 还有 5 件。"),
        ]
    )

    run = run_agent(model, make_registry(), start())

    pending = []
    for msg in run.messages:
        if msg.role == "assistant" and msg.tool_calls:
            pending = [c.id for c in msg.tool_calls]
        elif msg.role == "tool":
            assert msg.tool_call_id in pending
            pending.remove(msg.tool_call_id)
    assert pending == []


def test_invalid_start_history_rejected() -> None:
    """带着孤儿 tool 消息进场 → 立刻拒绝，不要浪费模型调用。"""
    from bootcamp.common.types import tool_result

    model = MockChatModel([assistant("不该跑到这里")])

    with pytest.raises(Exception, match="tool"):
        run_agent(model, make_registry(), [user("hi"), tool_result("c9", "{}")])
