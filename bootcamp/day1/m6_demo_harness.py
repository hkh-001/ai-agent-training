"""M6 观察脚本：亲眼看 harness 的五个作用。

跑法：
    uv run python -m bootcamp.day1.m6_demo_harness

五个实验：
    1. trace 长什么样（六类事件、连续序号、成对）
    2. 重试：同一个抖动模型，有重试 vs 没重试
    3. 成本：步数越多越贵（同一任务不同配置）
    4. 崩溃定位：事件缺一半 = 死在哪一步
    5. 日志安全：元信息进日志，正文不进
"""

from __future__ import annotations

from bootcamp.common.config import RunConfig
from bootcamp.common.llm import MockChatModel, ModelResponse, Usage
from bootcamp.common.types import ToolCall, assistant, system, user
from bootcamp.harness.runner import HarnessError, Runner
from bootcamp.tests.test_m5_loop import make_registry

START = [system("你是电商客服"), user("A1 还有货吗")]


def show_trace(result) -> None:
    for event in result.trace:
        data = ", ".join(f"{k}={v}" for k, v in event.data.items())
        print(f"   #{event.seq:<2} {event.kind:<12} {data}")


class FlakyModel:
    """前 N 次抛异常（模拟网络抖动），之后正常返回。"""

    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.calls = 0

    def invoke(self, messages, *, tools=(), temperature=None):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise TimeoutError("模拟网络抖动")
        return ModelResponse(message=assistant("好了，A1 有 5 件。"), model="flaky")


def experiment_1() -> None:
    print("\n【实验 1】trace 长什么样")
    model = MockChatModel(
        [
            ModelResponse(
                message=assistant(tool_calls=[ToolCall("c1", "get_stock", {"sku": "A1"})]),
                usage=Usage(120, 40),
            ),
            ModelResponse(message=assistant("A1 还有 5 件。"), usage=Usage(180, 25)),
        ]
    )
    result = Runner(model, make_registry(), RunConfig(run_name="exp1", log_level="WARNING")).run(START)
    show_trace(result)
    print(f"   → usage={result.usage.total_tokens} tokens, cost=${result.cost_usd:.6f}")


def experiment_2() -> None:
    print("\n【实验 2】重试：同一个抖动模型，两种配置")
    print("  (a) model_retries=0")
    try:
        Runner(
            FlakyModel(1), make_registry(), RunConfig(run_name="no-retry", model_retries=0)
        ).run(START)
        print("     意外成功")
    except HarnessError as exc:
        print(f"     💥 {exc}")

    print("  (b) model_retries=2")
    model = FlakyModel(1)
    result = Runner(model, make_registry(), RunConfig(run_name="retry", model_retries=2, log_level="WARNING")).run(START)
    show_trace(result)
    print(f"     → retries={result.retries}, answer={result.run.final_answer}")


def experiment_3() -> None:
    print("\n【实验 3】成本：步数越多越贵")
    call = ToolCall("c1", "get_stock", {"sku": "A1"})
    for target_steps in (1, 3, 5):
        # 前 target_steps-1 轮调工具，最后一轮给答案
        script = [
            ModelResponse(
                message=assistant(tool_calls=[call]), usage=Usage(500 * (i + 1), 50)
            )
            for i in range(target_steps - 1)
        ]
        script.append(ModelResponse(message=assistant("完毕"), usage=Usage(1000, 30)))
        result = Runner(
            MockChatModel(script),
            make_registry(),
            RunConfig(run_name=f"steps{target_steps}", log_level="WARNING"),
        ).run(START)
        print(
            f"   steps={result.run.steps}  tokens={result.usage.total_tokens:<6} "
            f"cost=${result.cost_usd:.6f}"
        )


def experiment_4() -> None:
    print("\n【实验 4】崩溃定位：事件缺一半 = 死在哪一步")
    runner = Runner(
        FlakyModel(1), make_registry(), RunConfig(run_name="crash", model_retries=0, log_level="WARNING")
    )
    try:
        runner.run(START)
    except HarnessError as exc:
        print(f"   崩溃了：{exc}")
        print("   失败现场（trace 依然可读）：")
        show_trace(type("R", (), {"trace": runner.trace})())
        print("   → model_start 有、model_end 没有 ⇒ 死在模型调用内部，不是工具")


def experiment_5() -> None:
    print("\n【实验 5】日志安全（看上面这行 [INFO]，再对比下面的正文）")
    model = MockChatModel([assistant("用户的手机号是 138****8888，不要写进日志")])
    Runner(model, make_registry(), RunConfig(run_name="safe", log_level="INFO")).run(START)
    print("   → 日志里有 stop/steps/cost，但没有消息正文（手机号不会泄露）")


def main() -> None:
    experiment_1()
    experiment_2()
    experiment_3()
    experiment_4()
    experiment_5()


if __name__ == "__main__":
    main()
