"""M6 验收测试：配置、trace、成本、重试、安全日志。

运行：uv run pytest bootcamp/tests/test_m6_harness.py -q

重点不是"能不能跑"，而是"出问题时能不能查到"：
事件序号连续、start/end 成对、成本可算、日志不含正文。
"""

import logging
from dataclasses import FrozenInstanceError

import pytest

from bootcamp.common.config import RunConfig
from bootcamp.common.llm import ModelResponse, Usage
from bootcamp.common.loop import LoopConfig, StopReason
from bootcamp.common.telemetry import estimate_cost
from bootcamp.common.types import ToolCall, assistant, system, user
from bootcamp.harness.runner import HarnessError, Runner
from bootcamp.tests.test_m5_loop import make_registry


def start() -> list:
    return [system("你是电商客服"), user("A1 还有货吗")]


def run_once(model, config: RunConfig | None = None) -> "object":
    return Runner(model, make_registry(), config).run(start())


class FlakyModel:
    """前 fail_times 次抛异常，之后正常返回。"""

    def __init__(self, fail_times: int, response: str = "好了") -> None:
        self.fail_times = fail_times
        self.response = response
        self.calls = 0

    def invoke(self, messages, *, tools=(), temperature=None):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise TimeoutError("网络抖动")
        return ModelResponse(message=assistant(self.response), model="flaky")


# ------------------------------------------------------------------ 基本产出


def test_runner_returns_result() -> None:
    from bootcamp.common.llm import MockChatModel

    result = run_once(MockChatModel([assistant("A1 还有 5 件。")]))

    assert result.run.stop_reason is StopReason.FINAL
    assert result.run.final_answer == "A1 还有 5 件。"
    assert result.elapsed_ms >= 0
    assert result.retries == 0
    assert len(result.trace) > 0


def test_trace_event_sequence() -> None:
    from bootcamp.common.llm import MockChatModel

    model = MockChatModel(
        [
            assistant(tool_calls=[ToolCall("c1", "get_stock", {"sku": "A1"})]),
            assistant("A1 还有 5 件。"),
        ]
    )
    result = run_once(model)

    assert [e.kind for e in result.trace] == [
        "run_start",
        "model_start",
        "model_end",
        "tool_start",
        "tool_end",
        "model_start",
        "model_end",
        "run_end",
    ]


def test_trace_seq_is_continuous() -> None:
    from bootcamp.common.llm import MockChatModel

    result = run_once(MockChatModel([assistant("你好")]))

    seqs = [e.seq for e in result.trace]
    assert seqs == list(range(1, len(seqs) + 1))  # 1,2,3... 无跳跃


def test_tool_events_are_paired() -> None:
    from bootcamp.common.llm import MockChatModel

    model = MockChatModel(
        [
            assistant(
                tool_calls=[
                    ToolCall("c1", "get_stock", {"sku": "A1"}),
                    ToolCall("c2", "reserve_stock", {"sku": "A1", "qty": 99}),
                ]
            ),
            assistant("最多留 5 件。"),
        ]
    )
    result = run_once(model)

    kinds = [e.kind for e in result.trace]
    # 改成"每个工具一条 start"后，断言也更强：start/end 数量相等且 id 一一对应
    assert kinds.count("tool_start") == 2
    assert kinds.count("tool_end") == 2
    started = {e.data["tool_call_id"] for e in result.trace if e.kind == "tool_start"}
    ended = {e.data["tool_call_id"] for e in result.trace if e.kind == "tool_end"}
    assert started == ended == {"c1", "c2"}


# ------------------------------------------------------------------ 成本


def test_usage_aggregated_from_trace() -> None:
    from bootcamp.common.llm import MockChatModel

    model = MockChatModel(
        [
            # 第一轮必须带 tool_calls，否则循环第 1 步就结束了（只看得到 1 次调用）
            ModelResponse(
                message=assistant(tool_calls=[ToolCall("c1", "get_stock", {"sku": "A1"})]),
                usage=Usage(100, 20),
            ),
            ModelResponse(message=assistant("第二次"), usage=Usage(150, 30)),
        ]
    )
    result = run_once(model)

    assert result.usage.prompt_tokens == 250
    assert result.usage.completion_tokens == 50
    assert result.usage.total_tokens == 300


def test_cost_is_computed_from_price_table() -> None:
    usage = Usage(prompt_tokens=1000, completion_tokens=1000)
    cost = estimate_cost(usage, price_prompt_per_1k=0.5, price_completion_per_1k=1.5)

    assert cost == pytest.approx(2.0)


def test_runner_cost_uses_config_prices() -> None:
    from bootcamp.common.llm import MockChatModel

    model = MockChatModel([ModelResponse(message=assistant("好"), usage=Usage(1000, 1000))])
    config = RunConfig(price_prompt_per_1k=0.5, price_completion_per_1k=1.5)

    result = run_once(model, config)

    assert result.cost_usd == pytest.approx(2.0)


# ------------------------------------------------------------------ 重试


def test_retry_then_success() -> None:
    model = FlakyModel(fail_times=1)

    result = run_once(model, RunConfig(model_retries=2))

    assert result.run.final_answer == "好了"
    assert result.retries == 1
    assert "model_retry" in [e.kind for e in result.trace]


def test_retry_exhausted_raises_harness_error() -> None:
    model = FlakyModel(fail_times=5)

    with pytest.raises(HarnessError, match="模型调用失败"):
        run_once(model, RunConfig(model_retries=2))


# ------------------------------------------------------------------ 配置与日志


def test_config_is_frozen() -> None:
    config = RunConfig(run_name="eval-1")

    with pytest.raises(FrozenInstanceError):
        config.run_name = "改不了"  # type: ignore[misc]


def test_log_contains_meta_but_not_message_content(caplog) -> None:
    from bootcamp.common.llm import MockChatModel

    with caplog.at_level(logging.INFO, logger="bootcamp.agent"):
        run_once(MockChatModel([assistant("A1 还有 5 件，敏感内容不应该进日志")]))

    text = caplog.text
    assert "stop=final" in text          # 元信息在
    assert "cost=$" in text
    assert "A1 还有 5 件" not in text     # 正文不在（隐私/密钥安全）


def test_loop_config_comes_from_run_config() -> None:
    from bootcamp.common.llm import MockChatModel

    call = ToolCall("c1", "get_stock", {"sku": "A1"})
    model = MockChatModel([assistant(tool_calls=[call]) for _ in range(10)])
    config = RunConfig(loop=LoopConfig(max_steps=2))

    result = run_once(model, config)

    assert result.run.stop_reason is StopReason.MAX_STEPS
    assert result.run.steps == 2


def test_failure_still_leaves_trace() -> None:
    """回归测试：run() 抛异常时，失败现场仍然可读（run_error 兜底）。

    这个缺陷是写 m6_demo_harness.py 时才发现的：
    之前异常一抛，RunResult 不存在，trace 也就拿不到——现场全丢。
    """
    runner = Runner(FlakyModel(5), make_registry(), RunConfig(model_retries=0))

    with pytest.raises(HarnessError):
        runner.run(start())

    kinds = [e.kind for e in runner.trace]
    assert "model_start" in kinds       # 进去了
    assert "model_end" not in kinds     # 没出来 → 死在模型调用
    assert kinds[-1] == "run_error"     # 现场有结论
