"""M11：Human-in-the-loop（人工审批门禁）。

为什么需要它
------------
M4 做的是"错误降级"，M8 做的是"重复幂等"，但它们都拦不住这类事故：

    用户："A100 那个还有吗？"
    模型：reserve_stock(A100, qty=1)      ← 用户只是问，没有要预留

参数完全合法、业务完全允许、执行完全成功——**技术上全对，语义上全错**。

所以第三层防线是：**不可逆的动作，执行前先问人。**

本模块的四个要素
----------------
    ApprovalPolicy  规则：哪些动作必须审批（写操作 / 数量超阈值）
    ApprovalRequest 审批单：谁、要做什么、为什么
    Decision        裁决：APPROVED / REJECTED / NO_APPROVER
    ApprovalGate    闸门：执行前拦截，改走"审批"，拿到决定再决定执不执行

两条重要设计
------------
1. **fail-closed**：没有审批人 → 默认拒绝。宁可少做一笔，不可多做一笔。
2. **拒绝 ≠ 失败**：拒绝生成的是 `cancelled` 结果（M5 学过：既不是成功也不是失败），
   模型看到它会去解释"需要您确认"，而不是把它当成工具错误反复重试。

能力边界（诚实说明）
--------------------
本项目实现的是**同步审批**：闸门阻塞等待审批人返回决定。
生产级的"异步挂起 / 跨进程恢复"（interrupt + resume + 线程状态持久化）
需要状态机与持久化，不在本训练范围内。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from bootcamp.common.dispatch import ToolResult
from bootcamp.common.types import ToolCall, tool_result


class Decision(str, Enum):
    """审批结果。注意 NO_APPROVER 是"没人能批"，不是"批了"。"""

    APPROVED = "approved"
    REJECTED = "rejected"
    NO_APPROVER = "no_approver"


@dataclass(frozen=True)
class ApprovalRequest:
    """一张审批单。给人看的信息要足够做判断。"""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    reason: str
    created_at: float


@dataclass(frozen=True)
class ApprovalRecord:
    """审批留痕（审计）。"""

    request: ApprovalRequest
    decision: Decision
    note: str = ""


Approver = Callable[[ApprovalRequest], Decision]


@dataclass(frozen=True)
class ApprovalPolicy:
    """哪些动作必须人工审批。

    write_tools 里的工具默认都要审批；但允许"小额自动放行"，
    否则每次查库存后的小操作都要人点一下，体验无法接受。
    """

    write_tools: frozenset[str] = frozenset({"reserve_stock", "release_stock"})
    auto_approve_qty: int = 5

    def evaluate(self, call: ToolCall) -> tuple[bool, str]:
        """返回 (是否需要审批, 理由)。"""
        if call.name not in self.write_tools:
            return False, "只读操作"

        qty = call.arguments.get("qty")
        if isinstance(qty, int) and not isinstance(qty, bool) and qty <= self.auto_approve_qty:
            return False, f"数量 {qty} 未超过自动放行阈值 {self.auto_approve_qty}"

        return True, f"{call.name} 是写操作（qty={qty}），需要人工确认"


@dataclass
class ApprovalGate:
    """执行前闸门。返回值符合 loop 的 tool_gate 协议：

    - 返回 None            → 放行，正常执行
    - 返回 ToolResult      → 不执行，用这个结果
    """

    policy: ApprovalPolicy = field(default_factory=ApprovalPolicy)
    approver: Approver | None = None
    _records: list[ApprovalRecord] = field(default_factory=list)

    @property
    def records(self) -> tuple[ApprovalRecord, ...]:
        return tuple(self._records)

    def guard(self, call: ToolCall) -> ToolResult | None:
        needs, reason = self.policy.evaluate(call)
        if not needs:
            return None

        request = ApprovalRequest(
            tool_call_id=call.id,
            tool_name=call.name,
            arguments=dict(call.arguments),
            reason=reason,
            created_at=time.time(),
        )

        if self.approver is None:
            decision, note = Decision.NO_APPROVER, "没有可用的审批人，按 fail-closed 拒绝"
        else:
            try:
                decision = self.approver(request)
            except Exception as exc:  # noqa: BLE001 - 审批人崩溃也必须 fail-closed
                decision, note = Decision.NO_APPROVER, f"审批人异常: {type(exc).__name__}"
            else:
                note = ""

        self._records.append(ApprovalRecord(request=request, decision=decision, note=note))

        if decision is Decision.APPROVED:
            return None  # 放行

        # 拒绝 / 无审批人 → 不执行，返回"已取消"结果
        return ToolResult(
            tool_call_id=call.id,
            ok=False,
            content=_rejection_content(call, decision, reason),
            error=f"未执行：{_decision_text(decision)}",
            retryable=False,
            cancelled=True,
        )

    def to_message(self, result: ToolResult) -> Any:
        """方便测试：把闸门产生的结果转成 tool 消息。"""
        return tool_result(result.tool_call_id, result.content)


def _decision_text(decision: Decision) -> str:
    return {
        Decision.APPROVED: "已批准",
        Decision.REJECTED: "人工拒绝了该操作",
        Decision.NO_APPROVER: "没有审批人，已按安全策略拒绝",
    }[decision]


def _rejection_content(call: ToolCall, decision: Decision, reason: str) -> str:
    import json

    return json.dumps(
        {
            "ok": False,
            "error": (
                f"操作未执行：{_decision_text(decision)}"
                f"（{call.name}，原因：{reason}）"
            ),
            "retryable": False,
            "cancelled": True,
            "needs_human": True,
        },
        ensure_ascii=False,
    )


def make_scripted_approver(decisions: Mapping[str, Decision]) -> Approver:
    """测试/演示用：按 tool_call_id 预先给定决策。"""

    def approver(request: ApprovalRequest) -> Decision:
        return decisions.get(request.tool_call_id, Decision.REJECTED)

    return approver
