"""真实模型联调脚本：验证"Mock 里跑通的东西，接到真 API 还能不能跑"。

跑法（需要 .env 里的 DEEPSEEK_API_KEY）：
    uv run python -m bootcamp.demos.real_model_check

四步验证，每步都会打印真实返回，失败会给出原因：
    1. 最小调用（不带工具）：鉴权 + 网络 + 文本返回
    2. 带工具调用：模型能否输出合法的 tool_calls（wire 格式兼容性）
    3. 跑完一整轮 Agent（真实模型 + 我们的工具 + RAG + 引用）
    4. 打印 usage / 成本 / trace（真实 token 数）
"""

from __future__ import annotations

import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from bootcamp.common.config import RunConfig
from bootcamp.common.llm_openai import ModelCallError, OpenAICompatChatModel
from bootcamp.common.types import system, user
from bootcamp.day2.m8_tools import build_store
from bootcamp.day2.m9_rag import PolicyIndex
from bootcamp.day2.m10_agent import InventoryAgent, build_registry
from bootcamp.harness.runner import Runner


def with_retry(func, attempts: int = 3, pause: float = 2.0):
    last = None
    for index in range(attempts):
        try:
            return func()
        except ModelCallError as exc:
            last = exc
            print(f"  第 {index + 1} 次失败: {exc}")
            time.sleep(pause)
    raise SystemExit(f"连续 {attempts} 次失败，放弃：{last}")


def step1_minimal(model: OpenAICompatChatModel) -> None:
    print("\n[1/4] 最小调用（无工具）")
    result = with_retry(lambda: model.invoke([system("你是简洁的助手。"), user("用一句话说明你是谁。")]))
    print(f"  content : {result.message.content}")
    print(f"  model   : {result.model}")
    print(f"  usage   : prompt={result.usage.prompt_tokens} completion={result.usage.completion_tokens}")
    print(f"  latency : {result.latency_ms:.0f} ms")


def step2_tool_call(model: OpenAICompatChatModel) -> None:
    print("\n[2/4] 带工具调用（验证 wire 格式兼容性）")
    registry = build_registry(build_store(), PolicyIndex.from_directory())
    tools = registry.to_openai_tools()
    print(f"  菜单里有 {len(tools)} 个工具: {[t['function']['name'] for t in tools]}")

    result = with_retry(
        lambda: model.invoke(
            [system("你是电商客服，只能依据工具结果回答。"), user("A100 还有货吗？")],
            tools=tools,
        )
    )
    calls = result.message.tool_calls
    print(f"  content    : {result.message.content!r}")
    print(f"  tool_calls : {[(c.name, c.arguments) for c in calls]}")
    if not calls:
        print("  ⚠️ 模型没有调用工具（可能直接回答了），这不一定是错，但值得记录")
    else:
        print("  ✅ arguments 已被正确解析成 dict（说明 from_wire 与真实 API 兼容）")


def step3_full_agent(model: OpenAICompatChatModel) -> None:
    print("\n[3/4] 完整一轮 Agent（真实模型 + 工具 + RAG）")
    agent = InventoryAgent(
        model=model,
        store=build_store(),
        index=PolicyIndex.from_directory(),
        config=RunConfig(run_name="real-model", log_level="WARNING"),
    )
    result = agent.ask("A100 还有货吗？顺便说下质量问题退货运费谁承担")
    run = result.run

    print(f"  stop_reason : {run.stop_reason.value}")
    for message in run.messages[1:]:
        if message.role == "assistant" and message.tool_calls:
            for call in message.tool_calls:
                print(f"  模型→工具   : {call.name}({call.arguments})")
        elif message.role == "tool":
            print(f"  工具→模型   : {(message.content or '')[:90]}")
        elif message.role == "assistant":
            print(f"  助手        : {message.content}")
    print(f"  tokens      : {result.usage.total_tokens}  cost=${result.cost_usd:.6f}")


def step4_observability(model: OpenAICompatChatModel) -> None:
    print("\n[4/4] 观测：真实 usage + trace")
    runner = Runner(
        model,
        build_registry(build_store(), PolicyIndex.from_directory()),
        RunConfig(run_name="trace-check", log_level="WARNING"),
    )
    result = with_retry(lambda: runner.run([system("你是客服。"), user("A900 还有几件？")]))
    for event in result.trace:
        detail = ", ".join(f"{k}={v}" for k, v in event.data.items())
        print(f"  {event.seq:03d} {event.kind:<14} {detail}")


def main() -> None:
    print("=" * 68)
    print("真实模型联调（DeepSeek / OpenAI 兼容协议）")
    print("=" * 68)

    model = OpenAICompatChatModel()
    print(f"base_url = {model._client.base_url}")
    print(f"model    = {model.model}")

    step1_minimal(model)
    step2_tool_call(model)
    step3_full_agent(model)
    step4_observability(model)

    print("\n" + "=" * 68)
    print("联调完成：M1 的 to_wire/from_wire 与真实 API 兼容。")
    print("=" * 68)


if __name__ == "__main__":
    main()
