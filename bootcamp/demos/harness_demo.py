"""M1–M6 学习用 Demo：4 个场景，全部离线可跑。

跑法：
    uv run python -m bootcamp.demos.harness_demo

每个场景都会真实执行，并根据真实结果输出 PASS / FAIL
（不是预先写好的成功文案）。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from bootcamp.common.config import RunConfig
from bootcamp.common.context import check_invariants
from bootcamp.common.dispatch import ToolBusinessError, dispatch
from bootcamp.common.llm import MockChatModel, ModelResponse, Usage
from bootcamp.common.loop import LoopConfig
from bootcamp.common.tools import ToolRegistry
from bootcamp.common.types import ToolCall, assistant, system, user
from bootcamp.harness.runner import Runner

START = [system("你是电商客服，只能依据工具结果回答。"), user("A100 还有货吗？")]


# ----------------------------------------------------------------- 工具与模型


def build_registry(counter: dict[str, int]) -> ToolRegistry:
    registry = ToolRegistry()

    @registry.register
    def get_stock(sku: str) -> dict:
        """查询某商品的可售库存。

        Args:
            sku: 商品编码，例如 A100
        """
        counter["get_stock"] = counter.get("get_stock", 0) + 1
        return {"sku": sku, "available": 25}

    @registry.register
    def reserve_stock(sku: str, qty: int) -> dict:
        """预留库存。

        Args:
            sku: 商品编码
            qty: 预留件数
        """
        if qty > 25:
            raise ToolBusinessError("库存不足：当前可用 25 件")
        return {"sku": sku, "reserved": qty}

    return registry


class FlakyModel:
    """第一次调用抛异常（模拟网络抖动），之后正常。"""

    def __init__(self) -> None:
        self.attempts = 0

    def invoke(self, messages, *, tools=(), temperature=None):
        self.attempts += 1
        if self.attempts == 1:
            raise ConnectionError("模拟网络抖动")
        return ModelResponse(message=assistant("A100 还有 25 件。"), model="flaky")


# ----------------------------------------------------------------- 展示辅助


def print_result(title: str, runner: Runner, result, checks: list[tuple[str, bool]]) -> None:
    run = result.run
    ok_tools = sum(1 for r in run.tool_results if r.ok)
    cancelled = sum(1 for r in run.tool_results if r.cancelled)
    failed = len(run.tool_results) - ok_tools - cancelled
    total = sum(1 for _ in result.trace)

    print("\n" + "=" * 62)
    print(f"场景：{title}")
    print("=" * 62)
    print(f"Run Name:        {runner.config.run_name}")
    print(f"Stop Reason:     {run.stop_reason.value}")
    print(f"Final Answer:    {run.final_answer or '—'}")
    print(
        f"Steps/Attempts:  {run.steps} / {result.attempts}"
        f"   Retries: {result.retries}"
    )
    print(f"Tool Calls:      {len(run.tool_results)} (ok={ok_tools}, failed={failed}, cancelled={cancelled})")
    print(
        f"Tokens:          prompt {result.usage.prompt_tokens} + completion "
        f"{result.usage.completion_tokens} = {result.usage.total_tokens}"
    )
    print(f"Cost:            ${result.cost_usd:.6f}")
    print(f"Elapsed:         {result.elapsed_ms:.2f} ms    Events: {total}")

    print("\n-- Trace --")
    for event in result.trace:
        detail = ", ".join(f"{k}={v}" for k, v in event.data.items())
        print(f"  {event.seq:03d}  {event.kind:<15} {detail}")

    print("\n-- Messages --")
    for msg in run.messages:
        if msg.role == "assistant" and msg.tool_calls:
            calls = ", ".join(f"{c.name}({c.arguments})" for c in msg.tool_calls)
            print(f"  assistant  ToolCall: {calls}")
        elif msg.role == "tool":
            print(f"  tool       [{msg.tool_call_id}] {msg.content}")
        else:
            print(f"  {msg.role:<10} {msg.content}")

    print("\n-- Checks --")
    for label, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'} - {label}")


# ----------------------------------------------------------------- 四个场景


def scenario_normal() -> None:
    counter: dict[str, int] = {}
    model = MockChatModel(
        [
            ModelResponse(
                message=assistant(tool_calls=[ToolCall("c1", "get_stock", {"sku": "A100"})]),
                usage=Usage(120, 40),
            ),
            ModelResponse(message=assistant("A100 还有 25 件。"), usage=Usage(160, 25)),
        ]
    )
    runner = Runner(model, build_registry(counter), RunConfig(run_name="normal", log_level="WARNING"))
    result = runner.run(START)

    kinds = [e.kind for e in result.trace]
    checks = [
        ("ToolCall 与 ToolResult 配对", kinds.count("tool_start") == kinds.count("tool_end") == 1),
        ("stop_reason 为 final", result.run.stop_reason.value == "final"),
        ("最终答案来自工具结果", "25" in (result.run.final_answer or "")),
        ("usage 正确累加", result.usage.total_tokens == 345),
        ("业务函数真的被调用 1 次", counter.get("get_stock") == 1),
    ]
    print_result("1. 正常 Tool Calling", runner, result, checks)


def scenario_retry() -> None:
    counter: dict[str, int] = {}
    runner = Runner(
        FlakyModel(),
        build_registry(counter),
        RunConfig(run_name="retry", model_retries=2, log_level="WARNING"),
    )
    result = runner.run(START)

    kinds = [e.kind for e in result.trace]
    checks = [
        ("发生了 1 次重试", result.retries == 1),
        ("物理尝试 2 次、逻辑步数 1 步", result.attempts == 2 and result.run.steps == 1),
        ("trace 里有 model_retry 事件", "model_retry" in kinds),
        ("重试后拿到答案", result.run.final_answer == "A100 还有 25 件。"),
    ]
    print_result("2. 模型异常与重试", runner, result, checks)


def scenario_bad_argument() -> None:
    counter: dict[str, int] = {}
    model = MockChatModel(
        [
            assistant(tool_calls=[ToolCall("c1", "get_stock", {"sku": 123})]),  # 类型错
            assistant("抱歉，我需要商品编码才能查询。"),
        ]
    )
    runner = Runner(model, build_registry(counter), RunConfig(run_name="bad-arg", log_level="WARNING"))
    result = runner.run(START)

    failed = [r for r in result.run.tool_results if not r.ok]
    checks = [
        ("工具调用被拒绝", len(failed) == 1),
        ("业务函数没有被执行（计数为 0）", counter.get("get_stock", 0) == 0),
        ("错误信息可被模型读取", "类型" in (failed[0].error or "")),
        ("模型据错误改了说法", "编码" in (result.run.final_answer or "")),
    ]
    print_result("3. 参数校验失败", runner, result, checks)


def scenario_batch_budget() -> None:
    counter: dict[str, int] = {}
    model = MockChatModel(
        [
            assistant(
                tool_calls=[
                    ToolCall("c1", "get_stock", {}),                  # 失败：缺 sku
                    ToolCall("c2", "get_stock", {}),                  # 失败：触发预算
                    ToolCall("c3", "get_stock", {"sku": "A100"}),     # 被取消
                ]
            )
        ]
    )
    runner = Runner(
        model,
        build_registry(counter),
        RunConfig(run_name="batch-budget", loop=LoopConfig(max_consecutive_tool_errors=2), log_level="WARNING"),
    )
    result = runner.run(START)

    cancelled = [r for r in result.run.tool_results if r.cancelled]
    invariant_ok = True
    try:
        check_invariants(result.run.messages)
    except Exception:  # noqa: BLE001
        invariant_ok = False

    checks = [
        ("3 个调用都有对应结果", len(result.run.tool_results) == 3),
        ("最后一个是取消（不是成功也不是失败）", len(cancelled) == 1 and cancelled[0].tool_call_id == "c3"),
        ("轨迹满足协议不变量", invariant_ok),
        ("stop_reason 为 tool_error_budget", result.run.stop_reason.value == "tool_error_budget"),
    ]
    print_result("4. 批量调用 + 连续失败预算", runner, result, checks)


def scenario_unserializable() -> None:
    registry = ToolRegistry()

    @registry.register
    def order_info(order_id: str) -> dict:
        """查订单信息（故意返回不可序列化对象）。

        Args:
            order_id: 订单号
        """
        return {"order_id": order_id, "created_at": datetime(2026, 10, 28, tzinfo=UTC)}

    result = dispatch(ToolCall("c1", "order_info", {"order_id": "A1"}), registry)
    print("\n" + "=" * 62)
    print("场景：5. 工具返回值不可序列化（P0-2 修复验证）")
    print("=" * 62)
    print(f"  ok={result.ok}  retryable={result.retryable}")
    print(f"  content={result.content}")
    print("\n-- Checks --")
    print(f"  {'PASS' if not result.ok else 'FAIL'} - dispatch 没有抛异常")
    print(f"  {'PASS' if '序列化' in (result.error or '') else 'FAIL'} - 错误信息说明了原因")
    print(f"  {'PASS' if json.loads(result.content)['ok'] is False else 'FAIL'} - content 仍是合法 JSON")


def main() -> None:
    print("=" * 62)
    print("       AGENT HARNESS DEMO（M1–M6，离线可跑）")
    print("=" * 62)
    scenario_normal()
    scenario_retry()
    scenario_bad_argument()
    scenario_batch_budget()
    scenario_unserializable()
    print("\n" + "=" * 62)
    print("全部场景执行完毕。所有 PASS/FAIL 均由真实断言产生。")
    print("=" * 62)


if __name__ == "__main__":
    main()
