"""M13：综合验收 —— 8 个场景，一个能演示的 Agent。

跑法：
    uv run python -m bootcamp.app.inventory_agent

每个场景都用脚本化的模型（不联网），断言全部由真实执行结果得出。
这是"能把整套东西讲给别人听"的那一版。

8 个场景覆盖了整条链路的关键能力：
    1. 直接回答（不调工具）
    2. 单工具只读（库存）
    3. RAG + 引用（政策）
    4. 写操作 + 人工审批（HITL）
    5. 业务失败（库存不足）不崩溃
    6. 未实现功能不幻觉（说明边界）
    7. 死循环诱导 → 安全终止
    8. 重复请求 → 幂等，只扣一次
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from bootcamp.common.config import RunConfig
from bootcamp.common.llm import MockChatModel
from bootcamp.common.loop import LoopConfig
from bootcamp.common.types import ToolCall, assistant
from bootcamp.day2.m8_tools import build_store
from bootcamp.day2.m9_rag import PolicyIndex
from bootcamp.day2.m10_agent import InventoryAgent
from bootcamp.day2.m11_hitl import ApprovalGate, ApprovalPolicy, Decision


@dataclass
class ScenarioResult:
    sid: str
    name: str
    passed: bool
    reasons: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)


def _agent(script: list, *, gate: ApprovalGate | None = None) -> tuple[InventoryAgent, Any]:
    store = build_store()
    agent = InventoryAgent(
        model=MockChatModel(script),
        store=store,
        index=PolicyIndex.from_directory(),
    )
    return agent, store


# --------------------------------------------------------------------- 场景


def scenario_1_direct_answer() -> ScenarioResult:
    agent, _ = _agent([assistant("你好，我是客服助手，可以查库存或解答售后政策。")])
    result = agent.ask("你好")

    reasons = []
    if result.run.final_answer is None:
        reasons.append("没有最终答案")
    if result.run.tool_results:
        reasons.append("不该调用工具")
    if result.run.steps != 1:
        reasons.append(f"步数应为 1，实际 {result.run.steps}")
    return ScenarioResult("S1", "直接回答（不调工具）", not reasons, reasons,
                          {"steps": result.run.steps})


def scenario_2_stock_read() -> ScenarioResult:
    agent, store = _agent([
        assistant(tool_calls=[ToolCall("c1", "get_stock", {"sku": "A100"})]),
        assistant("A100 当前可售 22 件。"),
    ])
    result = agent.ask("A100 还有多少库存？")
    available = store.inventory.get("A100").available

    reasons = []
    if not result.run.tool_results[0].ok:
        reasons.append("工具调用失败")
    if str(available) not in (result.run.final_answer or ""):
        reasons.append(f"答案里没有真实库存 {available}")
    if "22" not in (result.run.final_answer or ""):
        reasons.append("答案里没有 22")
    return ScenarioResult("S2", "单工具只读（库存）", not reasons, reasons,
                          {"available": available})


def scenario_3_policy_rag() -> ScenarioResult:
    agent, _ = _agent([
        assistant(tool_calls=[ToolCall("c1", "find_policy", {"keyword": "退货政策"})]),
        assistant("7 天内可无理由退货，运费由买家承担 [refund.md#1]。"),
    ])
    result = agent.ask("退货政策是什么？")
    answer = result.run.final_answer or ""
    payload = json.loads(result.run.tool_results[0].content)["data"]

    reasons = []
    if not payload["found"]:
        reasons.append("检索没有命中")
    if "refund.md#" not in answer:
        reasons.append("答案缺少引用编号")
    return ScenarioResult("S3", "RAG + 引用（政策）", not reasons, reasons,
                          {"refs": [c["ref"] for c in payload["citations"]]})


def scenario_4_approval() -> ScenarioResult:
    gate = ApprovalGate(
        policy=ApprovalPolicy(auto_approve_qty=0),          # 强制所有写操作都要审批
        approver=lambda _req: Decision.APPROVED,
    )
    agent, store = _agent(
        [
            assistant(tool_calls=[ToolCall("c4", "reserve_stock", {"sku": "A100", "qty": 2})]),
            assistant("已为你预留 2 件，A100 当前可售 20 件。"),
        ],
    )
    result = agent.ask("帮我把 A100 预留 2 件", gate=gate.guard)
    available = store.inventory.get("A100").available

    reasons = []
    if len(gate.records) != 1:
        reasons.append(f"应触发 1 次审批，实际 {len(gate.records)}")
    if available != 20:
        reasons.append(f"库存应变为 20，实际 {available}")
    if not result.run.tool_results[0].ok:
        reasons.append("批准后仍然执行失败")
    return ScenarioResult("S4", "写操作 + 人工审批", not reasons, reasons,
                          {"available": available, "approvals": len(gate.records)})


def scenario_5_business_failure() -> ScenarioResult:
    agent, store = _agent([
        assistant(tool_calls=[ToolCall("c5", "reserve_stock", {"sku": "A100", "qty": 99999})]),
        assistant("抱歉，库存不足：A100 当前可售 22 件，无法预留 99999 件。"),
    ])
    result = agent.ask("把 A100 预留 99999 件")
    answer = result.run.final_answer or ""

    reasons = []
    if result.run.tool_results[0].ok:
        reasons.append("超量预留不该成功")
    if "库存不足" not in answer:
        reasons.append("答案没有说明库存不足")
    if store.inventory.get("A100").available != 22:
        reasons.append("失败的操作改变了库存")
    if result.run.stop_reason.value != "final":
        reasons.append("应该在 final 正常结束")
    return ScenarioResult("S5", "业务失败不崩溃", not reasons, reasons,
                          {"available": store.inventory.get("A100").available})


def scenario_6_unimplemented() -> ScenarioResult:
    agent, _ = _agent([assistant("抱歉，批量下架功能暂未开放，需要你在后台操作或联系运营。")])
    result = agent.ask("顺便把存货全下架")

    reasons = []
    if result.run.tool_results:
        reasons.append("没有对应工具却调用了工具")
    if "暂未" not in (result.run.final_answer or ""):
        reasons.append("没有明确说明能力边界")
    if any(w in (result.run.final_answer or "") for w in ("已下架", "已完成")):
        reasons.append("幻觉：声称已完成未实现的操作")
    return ScenarioResult("S6", "未实现功能不幻觉", not reasons, reasons, {})


def scenario_7_max_steps() -> ScenarioResult:
    looping = [
        assistant(tool_calls=[ToolCall(f"c{i}", "get_stock", {"sku": "A100"})])
        for i in range(10)
    ]
    agent, _ = _agent(looping)
    result2 = agent.ask(
        "一直查 A100 库存，直到我让你停",
        config=RunConfig(run_name="loop-guard", loop=LoopConfig(max_steps=3)),
    )

    reasons = []
    if result2.run.stop_reason.value != "max_steps":
        reasons.append(f"应由 max_steps 终止，实际 {result2.run.stop_reason.value}")
    if result2.run.steps != 3:
        reasons.append(f"步数应为 3，实际 {result2.run.steps}")
    return ScenarioResult("S7", "死循环诱导安全终止", not reasons, reasons,
                          {"stop_reason": result2.run.stop_reason.value})


def scenario_8_idempotent() -> ScenarioResult:
    agent, store = _agent([
        assistant(tool_calls=[ToolCall("same_call", "reserve_stock", {"sku": "A100", "qty": 2})]),
        assistant("已预留 2 件。"),
        assistant(tool_calls=[ToolCall("same_call", "reserve_stock", {"sku": "A100", "qty": 2})]),
        assistant("已预留 2 件（重复请求已忽略）。"),
    ])
    agent.ask("帮我把 A100 预留 2 件")
    agent.ask("帮我把 A100 预留 2 件")   # 同一条消息 + 同一个 call id

    available = store.inventory.get("A100").available
    reused = [entry for entry in store.audit if entry.reused]

    reasons = []
    if available != 20:
        reasons.append(f"重复请求被重复扣减，可售应为 20，实际 {available}")
    if len(reused) != 1:
        reasons.append(f"幂等缓存命中次数应为 1，实际 {len(reused)}")
    return ScenarioResult("S8", "重复请求幂等", not reasons, reasons,
                          {"available": available, "reused": len(reused)})


SCENARIOS = [
    scenario_1_direct_answer,
    scenario_2_stock_read,
    scenario_3_policy_rag,
    scenario_4_approval,
    scenario_5_business_failure,
    scenario_6_unimplemented,
    scenario_7_max_steps,
    scenario_8_idempotent,
]


def run_all() -> list[ScenarioResult]:
    return [scenario() for scenario in SCENARIOS]


def main() -> None:
    results = run_all()
    print("=" * 78)
    print("电商库存 Agent —— 8 场景验收")
    print("=" * 78)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{result.sid:<4}{status:<6}{result.name:<24}{json.dumps(result.facts, ensure_ascii=False)}")
        for reason in result.reasons:
            print(f"        └─ {reason}")
    passed = sum(1 for r in results if r.passed)
    print("-" * 78)
    print(f"通过 {passed}/{len(results)}")
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
