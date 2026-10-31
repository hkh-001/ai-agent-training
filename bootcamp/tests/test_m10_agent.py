"""M10 验收测试：多工具选择、RAG 接入、多轮对话。

运行：uv run pytest bootcamp/tests/test_m10_agent.py -q

核心验收点：
    1. 三类问题走对工具（库存 / 政策 / 两者都要）
    2. 政策答案必须带引用编号
    3. 工具描述变了要能被测试发现（描述漂移 = 选择质量漂移）
"""

import json

from bootcamp.common.llm import MockChatModel
from bootcamp.common.types import ToolCall, assistant
from bootcamp.day2.m8_tools import build_store
from bootcamp.day2.m9_rag import PolicyIndex
from bootcamp.day2.m10_agent import SYSTEM_PROMPT, InventoryAgent, build_registry


def make_agent(script: list) -> InventoryAgent:
    return InventoryAgent(
        model=MockChatModel(script),
        store=build_store(),
        index=PolicyIndex.from_directory(),
    )


def tool_data(content: str) -> dict:
    return json.loads(content)["data"]


# ------------------------------------------------------------------ 工具集


def test_registry_exposes_three_tools() -> None:
    registry = build_registry(build_store(), PolicyIndex.from_directory())

    assert registry.names() == ["find_policy", "get_stock", "reserve_stock"]


def test_tool_descriptions_are_distinguishable() -> None:
    """描述是模型选工具的唯一依据——必须能一眼分清读/写/检索。"""
    registry = build_registry(build_store(), PolicyIndex.from_directory())

    assert "只读" in registry.get("get_stock").description
    assert "写操作" in registry.get("reserve_stock").description
    assert "检索" in registry.get("find_policy").description


def test_find_policy_returns_citations() -> None:
    registry = build_registry(build_store(), PolicyIndex.from_directory())

    payload = registry.get("find_policy").func(keyword="退货运费谁承担")

    assert payload["found"] is True
    refs = [c["ref"] for c in payload["citations"]]
    assert any(ref.startswith("refund.md#") for ref in refs)
    assert all("text" in c and "score" in c for c in payload["citations"])


def test_find_policy_reports_miss() -> None:
    registry = build_registry(build_store(), PolicyIndex.from_directory())

    payload = registry.get("find_policy").func(keyword="怎么开发票")

    assert payload["found"] is False
    assert payload["citations"] == []
    assert "阈值" in payload["reason"]


def test_system_prompt_requires_citation() -> None:
    assert "[refund.md#2]" in SYSTEM_PROMPT
    assert "found=false" in SYSTEM_PROMPT


# ------------------------------------------------------------------ 三类问题


def test_stock_only_question() -> None:
    agent = make_agent(
        [
            assistant(tool_calls=[ToolCall("c1", "get_stock", {"sku": "A100"})]),
            assistant("A100 现在可售 22 件。"),
        ]
    )

    result = agent.ask("A100 还有货吗")

    assert result.run.final_answer == "A100 现在可售 22 件。"
    assert len(result.run.tool_results) == 1
    assert tool_data(result.run.tool_results[0].content)["available"] == 22


def test_policy_only_question() -> None:
    agent = make_agent(
        [
            assistant(
                tool_calls=[ToolCall("c1", "find_policy", {"keyword": "退货运费"})]
            ),
            assistant("质量问题退货运费由商家承担 [refund.md#2]。"),
        ]
    )

    result = agent.ask("质量问题退货运费谁出")

    assert "refund.md#2" in (result.run.final_answer or "")
    assert tool_data(result.run.tool_results[0].content)["found"] is True


def test_combined_question_uses_both_tools() -> None:
    agent = make_agent(
        [
            assistant(
                tool_calls=[
                    ToolCall("c1", "get_stock", {"sku": "A100"}),
                    ToolCall("c2", "find_policy", {"keyword": "退货"}),
                ]
            ),
            assistant("A100 可售 22 件；退货规则见 [refund.md#1]。"),
        ]
    )

    result = agent.ask("A100 还有货吗？顺便说下退货规则")

    assert [r.tool_call_id for r in result.run.tool_results] == ["c1", "c2"]
    assert all(r.ok for r in result.run.tool_results)
    assert result.run.steps == 2


def test_write_intent_reserves_stock() -> None:
    agent = make_agent(
        [
            assistant(tool_calls=[ToolCall("c1", "reserve_stock", {"sku": "A100", "qty": 2})]),
            assistant("已为你预留 2 件。"),
        ]
    )

    result = agent.ask("帮我留 2 件 A100")

    assert result.run.tool_results[0].ok is True
    assert agent.store.inventory.get("A100").available == 20


def test_missing_policy_is_reported_not_invented() -> None:
    agent = make_agent(
        [
            assistant(tool_calls=[ToolCall("c1", "find_policy", {"keyword": "开发票"})]),
            assistant("抱歉，没有查到开票相关条款，建议联系人工客服。"),
        ]
    )

    result = agent.ask("怎么开发票")

    assert tool_data(result.run.tool_results[0].content)["found"] is False
    assert "没有查到" in (result.run.final_answer or "")


# ------------------------------------------------------------------ 多轮与观测


def test_multi_turn_keeps_history() -> None:
    model = MockChatModel(
        [
            assistant(tool_calls=[ToolCall("c1", "get_stock", {"sku": "A100"})]),
            assistant("A100 可售 22 件。"),
            assistant("刚才说过，A100 可售 22 件。"),
        ]
    )
    agent = InventoryAgent(model=model, store=build_store(), index=PolicyIndex.from_directory())

    agent.ask("A100 还有货吗")
    second = agent.ask("再说一遍")

    # 第二轮请求里应包含第一轮的全部消息（system + 2 轮对话）
    assert len(model.calls[2].messages) >= 5
    assert second.run.final_answer == "刚才说过，A100 可售 22 件。"


def test_each_turn_has_its_own_trace() -> None:
    model = MockChatModel([assistant("你好"), assistant("还在")])
    agent = InventoryAgent(model=model, store=build_store(), index=PolicyIndex.from_directory())

    first = agent.ask("你好")
    second = agent.ask("还在吗")

    assert next(e.kind for e in first.trace) == "run_start"
    assert next(e.kind for e in second.trace) == "run_start"
    assert len(second.trace) < 10  # 不是累加，是每轮独立
