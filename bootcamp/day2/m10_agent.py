"""M10：组装 —— 把库存、预留、政策检索装进同一个 Agent。

前面的模块都是零件：
    M1 消息、M2 上下文、M3 菜单、M4 执行、M5 循环、M6 观测、M7 领域、M8 工具、M9 检索

M10 要解决一个新问题：**工具变多了，模型怎么选对？**

这里有一个关键的架构选择：**RAG 该做成"工具"，还是"预注入"？**

    预注入（每次都检索 → 塞进 system prompt）
        优点：模型一定会看到证据
        缺点：① 每轮都烧 token；② 问题不需要查资料时也在浪费；
              ③ 检索错了模型也只能将错就错

    作为工具（由模型决定何时调用）  ← 本项目采用
        优点：按需检索，省 token；模型可以自己换关键词再查
        缺点：模型可能"忘了查"就开答 → 靠系统提示 + 评测来约束

工具的选择质量由三件事决定（这就是 M10 的验收重点）：
    1. description 写得够不够区分（get_stock vs reserve_stock vs find_policy）
    2. 参数 schema 是否严格（防止模型塞错字段）
    3. 返回值是否好读（结构化、带引用）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bootcamp.common.config import RunConfig
from bootcamp.common.llm import ChatModel
from bootcamp.common.loop import AgentRun
from bootcamp.common.tools import ToolRegistry
from bootcamp.common.types import ChatMessage, system, user
from bootcamp.day2.m8_tools import Store
from bootcamp.day2.m9_rag import PolicyIndex
from bootcamp.harness.runner import Runner, RunResult

SYSTEM_PROMPT = """你是某电商店铺的客服助手。必须遵守以下规则：

1. 只能依据工具返回的事实回答，绝不编造库存、价格或政策条款。
2. 回答政策问题时，必须标注引用编号，格式如 [refund.md#2]。
3. 工具返回 found=false 或报错时，如实告诉用户"没有查到相关条款"，
   并建议联系人工客服，不要猜测。
4. 库存数量一律以 get_stock 的 available 为准，不要用 reserved 或其它字段代替。
5. 预留库存是写操作，只有用户明确要求时才调用，一次调用只处理一个 SKU。
"""


@dataclass
class InventoryAgent:
    """一个可多轮对话的库存客服 Agent。

    设计上把 model 注入进来（而不是在内部 new 一个）：
    这样 Mock 模型和真实模型可以互换，测试完全离线。
    """

    model: ChatModel
    store: Store
    index: PolicyIndex
    config: RunConfig = field(default_factory=lambda: RunConfig(run_name="inventory-agent"))
    _messages: list[ChatMessage] = field(default_factory=lambda: [system(SYSTEM_PROMPT)])

    def __post_init__(self) -> None:
        self.registry = build_registry(self.store, self.index)

    @property
    def messages(self) -> tuple[ChatMessage, ...]:
        return tuple(self._messages)

    def ask(self, text: str, *, config: RunConfig | None = None) -> RunResult:
        """问一个问题，跑一整轮 Agent（可能包含多次工具调用）。"""
        self._messages = [*self._messages, user(text)]
        runner = Runner(self.model, self.registry, config or self.config)
        result = runner.run(self._messages)
        # 把这一轮产生的全部消息（含 tool 结果）留在历史里，支持多轮追问
        self._messages = list(result.run.messages)
        return result

    def reset(self) -> None:
        self._messages = [system(SYSTEM_PROMPT)]


def build_registry(store: Store, index: PolicyIndex) -> ToolRegistry:
    """三个工具：查库存（读）、预留（写）、查政策（RAG）。"""
    registry = ToolRegistry()

    @registry.register
    def get_stock(sku: str) -> dict:
        """查询某商品当前的可售库存数量（只读，可反复调用）。

        available 表示现在真正能卖的数量，已扣除其他订单的预留。
        用户问"还有货吗""还有几件""能不能买"时使用本工具。

        Args:
            sku: 商品编码，例如 A100
        """
        return store.check(sku)

    @registry.register
    def reserve_stock(sku: str, qty: int, _idempotency_key: str = "") -> dict:
        """为顾客预留库存（写操作，会减少可售数量）。

        只在用户明确表示要下单/预留时调用；一次只处理一个商品。

        Args:
            sku: 商品编码，例如 A100
            qty: 预留件数，必须是正整数
        """
        return store.reserve(sku, qty, _idempotency_key)

    @registry.register
    def find_policy(keyword: str, top_k: int = 3) -> dict:
        """检索售后政策条款（退换货、运费、保修、库存规则等）。

        返回命中的条款原文与引用编号；未命中时 found 为 false，
        此时应告诉用户没有查到，不要自己编造政策。

        Args:
            keyword: 检索关键词，例如"退货运费""保修多久"
            top_k: 返回条款数量，默认 3
        """
        result = index.search(keyword, top_k=top_k)
        if not result.found:
            return {
                "found": False,
                "query": result.query,
                "citations": [],
                "reason": result.reason,
            }
        return {
            "found": True,
            "query": result.query,
            "citations": [
                {
                    "ref": citation.ref,
                    "score": citation.score,
                    "text": citation.text,
                }
                for citation in result.citations
            ],
        }

    return registry


def summarize(run_result: RunResult) -> dict[str, Any]:
    """把一次运行压成给日志/评测用的摘要（不含消息正文）。"""
    run: AgentRun = run_result.run
    return {
        "stop_reason": run.stop_reason.value,
        "steps": run.steps,
        "attempts": run_result.attempts,
        "tool_calls": len(run.tool_results),
        "tool_ok": sum(1 for r in run.tool_results if r.ok),
        "cancelled": sum(1 for r in run.tool_results if r.cancelled),
        "tokens": run_result.usage.total_tokens,
        "cost_usd": round(run_result.cost_usd, 6),
        "final_answer": run.final_answer,
    }
