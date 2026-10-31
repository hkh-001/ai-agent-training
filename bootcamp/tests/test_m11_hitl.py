"""M11 验收测试：审批策略、放行、拒绝、fail-closed。

运行：uv run pytest bootcamp/tests/test_m11_hitl.py -q

四条路径必须都能复现：
    ① 小额写操作 → 自动放行
    ② 大额写操作 + 批准 → 真的执行
    ③ 大额写操作 + 拒绝 → 库存不变（不是"失败"，是"未执行"）
    ④ 没有审批人 → 默认拒绝（fail-closed）
"""

import json

from bootcamp.common.llm import MockChatModel
from bootcamp.common.loop import LoopConfig, run_agent
from bootcamp.common.types import ToolCall, assistant, system, user
from bootcamp.day2.m8_tools import build_store, register_tools
from bootcamp.day2.m11_hitl import (
    ApprovalGate,
    ApprovalPolicy,
    Decision,
    make_scripted_approver,
)


def setup(approver=None, policy: ApprovalPolicy | None = None):
    store = build_store()
    registry = register_tools(store)
    gate = ApprovalGate(policy=policy or ApprovalPolicy(), approver=approver)
    return store, registry, gate


def run_with_gate(store, registry, gate, calls: list[ToolCall]):
    """跑一轮：模型发 calls → 闸门判断 → 执行或拒绝 → 模型收尾。"""
    model = MockChatModel([assistant(tool_calls=calls), assistant("处理完毕。")])
    return run_agent(
        model,
        registry,
        [system("客服"), user("下单")],
        config=LoopConfig(max_steps=3),
        tool_gate=gate.guard,
    )


# ------------------------------------------------------------------ 策略


def test_read_only_never_needs_approval() -> None:
    policy = ApprovalPolicy()

    needs, reason = policy.evaluate(ToolCall("c1", "get_stock", {"sku": "A100"}))

    assert needs is False
    assert "只读" in reason


def test_small_write_is_auto_approved() -> None:
    policy = ApprovalPolicy(auto_approve_qty=5)

    needs, _ = policy.evaluate(ToolCall("c1", "reserve_stock", {"sku": "A100", "qty": 2}))

    assert needs is False


def test_large_write_needs_approval() -> None:
    policy = ApprovalPolicy(auto_approve_qty=5)

    needs, reason = policy.evaluate(
        ToolCall("c1", "reserve_stock", {"sku": "A100", "qty": 20})
    )

    assert needs is True
    assert "写操作" in reason


# ------------------------------------------------------------------ 四条路径


def test_approved_write_executes() -> None:
    store, registry, gate = setup(make_scripted_approver({"c1": Decision.APPROVED}))

    run = run_with_gate(store, registry, gate, [
        ToolCall("c1", "reserve_stock", {"sku": "A100", "qty": 20})
    ])

    assert store.inventory.get("A100").available == 2      # 真的扣了
    assert run.tool_results[0].ok is True
    assert gate.records[0].decision is Decision.APPROVED


def test_rejected_write_does_not_touch_state() -> None:
    store, registry, gate = setup(make_scripted_approver({"c1": Decision.REJECTED}))

    run = run_with_gate(store, registry, gate, [
        ToolCall("c1", "reserve_stock", {"sku": "A100", "qty": 20})
    ])

    assert store.inventory.get("A100").available == 22     # 库存没变
    result = run.tool_results[0]
    assert result.cancelled is True                        # 是"未执行"，不是"失败"
    assert result.retryable is False
    payload = json.loads(result.content)
    assert payload["needs_human"] is True


def test_no_approver_fails_closed() -> None:
    store, registry, gate = setup(approver=None)

    run = run_with_gate(store, registry, gate, [
        ToolCall("c1", "reserve_stock", {"sku": "A100", "qty": 20})
    ])

    assert store.inventory.get("A100").available == 22     # 没有被执行
    assert gate.records[0].decision is Decision.NO_APPROVER
    assert "没有审批" in (run.tool_results[0].error or "")


def test_approver_crash_fails_closed() -> None:
    def broken(_request):
        raise RuntimeError("审批系统挂了")

    store, registry, gate = setup(broken)

    run_with_gate(store, registry, gate, [
        ToolCall("c1", "reserve_stock", {"sku": "A100", "qty": 20})
    ])

    assert store.inventory.get("A100").available == 22
    assert gate.records[0].decision is Decision.NO_APPROVER


# ------------------------------------------------------------------ 与其他机制的配合


def test_small_write_bypasses_gate_entirely() -> None:
    store, registry, gate = setup(make_scripted_approver({}))

    run_with_gate(store, registry, gate, [
        ToolCall("c1", "reserve_stock", {"sku": "A100", "qty": 2})
    ])

    assert store.inventory.get("A100").available == 20
    assert gate.records == ()                              # 没触发审批


def test_mixed_batch_only_gates_the_risky_one() -> None:
    store, registry, gate = setup(
        make_scripted_approver({"c2": Decision.REJECTED})
    )

    run = run_with_gate(store, registry, gate, [
        ToolCall("c1", "reserve_stock", {"sku": "A100", "qty": 2}),    # 自动放行
        ToolCall("c2", "reserve_stock", {"sku": "A100", "qty": 20}),   # 被拒
    ])

    assert store.inventory.get("A100").available == 20     # 只扣了第一笔
    assert [r.cancelled for r in run.tool_results] == [False, True]
    assert len(gate.records) == 1


def test_gate_events_are_emitted() -> None:
    """被闸门拦下的调用应该有 tool_gated 事件，而不是 tool_start/tool_end。"""
    _store, registry, gate = setup(make_scripted_approver({"c1": Decision.REJECTED}))
    model = MockChatModel([assistant(tool_calls=[
        ToolCall("c1", "reserve_stock", {"sku": "A100", "qty": 20})
    ]), assistant("需要您确认。")])

    events: list[str] = []
    run_agent(
        model,
        registry,
        [system("s"), user("u")],
        config=LoopConfig(max_steps=3),
        on_event=lambda kind, _data: events.append(kind),
        tool_gate=gate.guard,
    )

    assert "tool_gated" in events
    assert "tool_start" not in events          # 没有真的执行
