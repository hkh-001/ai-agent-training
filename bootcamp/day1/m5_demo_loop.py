"""M5 手工验证脚本：用脚本化的 Mock 模型跑一次完整 Loop，并打印轨迹。

跑法：
    uv run python -m bootcamp.day1.m5_demo_loop

它不联网、不需要 API Key，结果 100% 确定——这是"能反复看轨迹"的前提。
"""

from __future__ import annotations

import json
from pathlib import Path

from bootcamp.common.dispatch import ToolBusinessError
from bootcamp.common.llm import MockChatModel
from bootcamp.common.loop import LoopConfig, run_agent
from bootcamp.common.tools import ToolRegistry
from bootcamp.common.types import ToolCall, assistant, system, user
from bootcamp.harness.recorder import append_run, format_trace

TRACE_PATH = Path("bootcamp/data/traces/demo.jsonl")


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()

    @registry.register
    def get_stock(sku: str) -> dict:
        """查询某商品的可售库存。

        Args:
            sku: 商品编码
        """
        return {"sku": sku, "available": 5}

    @registry.register
    def reserve_stock(sku: str, qty: int) -> dict:
        """预留库存。

        Args:
            sku: 商品编码
            qty: 预留件数
        """
        if qty > 5:
            raise ToolBusinessError("库存不足：当前可用 5 件")
        return {"sku": sku, "reserved": qty}

    return registry


def scripted_model() -> MockChatModel:
    """脚本化模型：模拟"先查库存 → 再预留 → 报告结果"。

    注意第 2 次调用故意给了 qty=99，用来演示"M4 把业务错误变成数据，
    模型据此改口"的完整闭环。
    """
    return MockChatModel(
        [
            assistant(tool_calls=[ToolCall("c1", "get_stock", {"sku": "A1"})]),
            assistant(tool_calls=[ToolCall("c2", "reserve_stock", {"sku": "A1", "qty": 99})]),
            assistant("A1 目前最多只能预留 5 件，要我帮你留 5 件吗？"),
        ]
    )


def main() -> None:
    messages = [system("你是电商客服，只能依据工具结果回答。"), user("A1 还有货吗？帮我留 99 件")]
    run = run_agent(
        scripted_model(),
        build_registry(),
        messages,
        config=LoopConfig(max_steps=5, max_consecutive_tool_errors=3),
    )

    print(format_trace(run))
    print()
    append_run(run, TRACE_PATH)
    print(f"轨迹已追加到 {TRACE_PATH}")
    record = json.loads(TRACE_PATH.read_text(encoding="utf-8").splitlines()[-1])
    print("落盘后的字段:", sorted(record))


if __name__ == "__main__":
    main()
