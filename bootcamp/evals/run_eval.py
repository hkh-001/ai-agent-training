"""M12：评测 —— 把"感觉能用"变成数字。

为什么需要评测
--------------
M10 演示了三类问题"走对了工具"，但那是**我手写的脚本**决定的。
真实模型会不会选对、会不会引用、会不会编造——脚本测不出来。

评测要回答的是：
    1. 任务成功率：有多少问题真的被解决了？
    2. 工具选择准确率：选对工具了吗？
    3. 引用覆盖率：政策类回答有没有给出可核对的引用？
    4. 成本与效率：平均几步、多少 token、多少钱？
    5. 非法参数率：模型有没有频繁传错参数？

重要说明：这里的模型是一个"规则模型"（RuleModel），不是真实 LLM。
所以本评测测的是**链路正确性**（工具、检索、引用、预算、终止），
不是"某个 LLM 有多强"。换成真实模型只需要换掉 model 参数。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bootcamp.common.llm import ModelResponse, Usage
from bootcamp.common.types import ChatMessage, ToolCall, assistant
from bootcamp.day2.m8_tools import build_store
from bootcamp.day2.m9_rag import PolicyIndex
from bootcamp.day2.m10_agent import InventoryAgent

CASES_PATH = Path("bootcamp/evals/cases.jsonl")


# --------------------------------------------------------------------- 规则模型


class RuleModel:
    """确定性"假模型"：用关键词决定调什么工具、怎么措辞。

    它存在的意义：让评测可以离线、可复现地跑通整条链路。
    它**不能**代表真实 LLM 的能力上限——这一点必须记住。
    """

    name = "rule-model"

    _POLICY_HINTS = ("退", "运费", "包邮", "保修", "发票", "开票", "释放", "政策", "预留多久")
    _STOCK_HINTS = ("还有", "几件", "库存", "剩")

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def invoke(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict[str, Any]] = (),
        temperature: float | None = None,
    ) -> ModelResponse:
        last = messages[-1]
        prompt_tokens = sum(len(m.content or "") for m in messages) // 2 + 40
        self.calls.append({"last_role": last.role, "n_messages": len(messages)})

        if last.role == "tool":
            answer = self._final_answer(messages)
            return ModelResponse(
                message=assistant(answer),
                usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=len(answer) // 2),
                model=self.name,
            )

        question = last.content or ""
        plan = self._plan_tool_calls(question)
        if not plan:
            answer = "你好，我是客服助手，可以帮你查库存或解答退换货、运费、保修等政策问题。"
            return ModelResponse(
                message=assistant(answer),
                usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=len(answer) // 2),
                model=self.name,
            )
        return ModelResponse(
            message=assistant(tool_calls=plan),
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=20),
            model=self.name,
        )

    # ---------------------------------------------------------------- 决策

    def _plan_tool_calls(self, question: str) -> list[ToolCall]:
        sku_match = re.search(r"[A-Z][A-Z0-9]{1,5}", question)
        sku = sku_match.group(0) if sku_match else None
        wants_reserve = ("留" in question or "预留" in question) and "释放" not in question
        wants_policy = any(hint in question for hint in self._POLICY_HINTS)
        wants_stock = bool(sku) and any(hint in question for hint in self._STOCK_HINTS)

        calls: list[ToolCall] = []
        index = 0
        if wants_stock and sku:
            index += 1
            calls.append(ToolCall(f"call_{index}", "get_stock", {"sku": sku}))
        if wants_reserve and sku:
            qty = self._extract_qty(question)
            index += 1
            calls.append(
                ToolCall(f"call_{index}", "reserve_stock", {"sku": sku, "qty": qty})
            )
        if wants_policy:
            index += 1
            calls.append(ToolCall(f"call_{index}", "find_policy", {"keyword": question}))
        return calls

    @staticmethod
    def _extract_qty(question: str) -> int:
        match = re.search(r"(\d+)\s*件", question)
        return int(match.group(1)) if match else 1

    # ---------------------------------------------------------------- 收尾

    def _final_answer(self, messages: Sequence[ChatMessage]) -> str:
        results = [m for m in messages if m.role == "tool"]
        if not results:
            return "抱歉，我没有拿到结果。"

        parts: list[str] = []
        for message in results:
            payload = json.loads(message.content or "{}")
            if not payload.get("ok"):
                parts.append(f"查询失败：{payload.get('error', '未知原因')}")
                continue
            data = payload["data"]
            if "citations" in data:
                if not data["found"]:
                    parts.append("抱歉，没有查到相关条款，建议联系人工客服。")
                    continue
                top = data["citations"][0]
                snippet = " ".join(top["text"].split())[:80]
                parts.append(f"根据 [{top['ref']}]：{snippet}")
            elif "reserved" in data and "action" in data:
                parts.append(
                    f"已为你预留 {data['qty']} 件，{data['sku']} 当前可售 {data['available']} 件。"
                )
            elif "available" in data:
                parts.append(f"{data['sku']} 当前可售 {data['available']} 件。")
        return " ".join(parts) if parts else "抱歉，我没有拿到可用结果。"


# --------------------------------------------------------------------- 评测结构


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    passed: bool
    reasons: tuple[str, ...]
    tools: tuple[str, ...]
    tool_selection_ok: bool
    refs_ok: bool
    steps: int
    tokens: int
    cost_usd: float
    stop_reason: str
    invalid_args: bool


@dataclass(frozen=True)
class EvalReport:
    results: tuple[CaseResult, ...]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def success_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def tool_selection_accuracy(self) -> float:
        return sum(1 for r in self.results if r.tool_selection_ok) / self.total

    @property
    def citation_rate(self) -> float:
        policy = [r for r in self.results if r.refs_ok is not None]
        return sum(1 for r in policy if r.refs_ok) / len(policy) if policy else 0.0

    @property
    def invalid_arg_rate(self) -> float:
        return sum(1 for r in self.results if r.invalid_args) / self.total

    @property
    def avg_steps(self) -> float:
        return sum(r.steps for r in self.results) / self.total

    @property
    def avg_tokens(self) -> float:
        return sum(r.tokens for r in self.results) / self.total

    @property
    def avg_cost(self) -> float:
        return sum(r.cost_usd for r in self.results) / self.total

    @property
    def stop_reasons(self) -> dict[str, int]:
        return dict(Counter(r.stop_reason for r in self.results))

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "success_rate": round(self.success_rate, 3),
            "tool_selection_accuracy": round(self.tool_selection_accuracy, 3),
            "citation_rate": round(self.citation_rate, 3),
            "invalid_arg_rate": round(self.invalid_arg_rate, 3),
            "avg_steps": round(self.avg_steps, 2),
            "avg_tokens": round(self.avg_tokens, 1),
            "avg_cost_usd": round(self.avg_cost, 6),
            "stop_reasons": self.stop_reasons,
        }


# --------------------------------------------------------------------- 执行


def load_cases(path: str | Path = CASES_PATH) -> list[dict[str, Any]]:
    target = Path(path)
    return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]


def _invalid_args(run) -> bool:
    return any(
        (not result.ok) and ("参数" in (result.error or "") or "缺少" in (result.error or ""))
        for result in run.tool_results
    )


def run_case(case: dict[str, Any]) -> CaseResult:
    agent = InventoryAgent(
        model=RuleModel(),
        store=build_store(),
        index=PolicyIndex.from_directory(),
    )
    result = agent.ask(case["question"])
    run = result.run
    answer = run.final_answer or ""

    tool_names = tuple(
        name
        for message in run.messages
        if message.role == "assistant"
        for name in (c.name for c in message.tool_calls)
    )

    reasons: list[str] = []
    if sorted(tool_names) != sorted(case["expect_tools"]):
        reasons.append(f"工具不符: 期望 {case['expect_tools']} 实际 {list(tool_names)}")

    refs_ok: bool | None = None
    if case["expect_refs"]:
        refs_ok = any(ref in answer for ref in case["expect_refs"])
        if not refs_ok:
            reasons.append(f"缺少引用: {case['expect_refs']}")

    for keyword in case["expect_answer_contains"]:
        if keyword not in answer:
            reasons.append(f"答案缺少关键内容: {keyword}")

    return CaseResult(
        case_id=case["id"],
        passed=not reasons,
        reasons=tuple(reasons),
        tools=tool_names,
        tool_selection_ok=sorted(tool_names) == sorted(case["expect_tools"]),
        refs_ok=refs_ok if refs_ok is not None else True,
        steps=run.steps,
        tokens=result.usage.total_tokens,
        cost_usd=result.cost_usd,
        stop_reason=run.stop_reason.value,
        invalid_args=_invalid_args(run),
    )


def evaluate(cases: Sequence[dict[str, Any]] | None = None) -> EvalReport:
    items = list(cases) if cases is not None else load_cases()
    return EvalReport(results=tuple(run_case(case) for case in items))


def format_report(report: EvalReport) -> str:
    lines = [f"{'case':<18}{'结果':<6}{'工具':<28}{'步数':>4}{'tokens':>7}  说明"]
    lines.append("-" * 96)
    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        tools = ",".join(result.tools) or "-"
        note = "；".join(result.reasons)[:46]
        lines.append(
            f"{result.case_id:<18}{status:<6}{tools:<28}{result.steps:>4}"
            f"{result.tokens:>7}  {note}"
        )
    lines.append("-" * 96)
    summary = report.as_dict()
    lines.append(f"通过 {summary['passed']}/{summary['total']}  "
                 f"成功率={summary['success_rate']}  "
                 f"工具选择准确率={summary['tool_selection_accuracy']}  "
                 f"引用覆盖率={summary['citation_rate']}")
    lines.append(f"平均步数={summary['avg_steps']}  "
                 f"平均 tokens={summary['avg_tokens']}  "
                 f"平均成本=${summary['avg_cost_usd']}  "
                 f"非法参数率={summary['invalid_arg_rate']}")
    lines.append(f"终止原因分布={summary['stop_reasons']}")
    return "\n".join(lines)


def main() -> None:
    report = evaluate()
    print(format_report(report))
    if report.passed != report.total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
