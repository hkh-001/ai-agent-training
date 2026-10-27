"""M2 验收测试：上下文不变量 + 裁剪。

运行：uv run pytest bootcamp/tests/test_m2_context.py -q

先读测试，它就是你需要满足的合同。注意每个断言里的**精确数字**，
它们来自 estimate_tokens 的约定（每条消息 4 + 内容字符数）。
"""

import pytest

from bootcamp.common.context import (
    ContextError,
    check_invariants,
    estimate_tokens,
    trim_history,
)
from bootcamp.common.types import ToolCall, assistant, system, tool_result, user


def _plain_turn(n: int) -> list:
    """一轮普通对话：user + assistant，各 2 个字符，合计 4+2 + 4+2 = 12。"""
    return [user(f"q{n}"), assistant(f"a{n}")]


# ---------------------------------------------------------------- estimate_tokens


def test_estimate_empty_is_zero() -> None:
    assert estimate_tokens([]) == 0


def test_estimate_counts_overhead_and_content() -> None:
    # system("s") = 4 + 1 = 5
    assert estimate_tokens([system("s")]) == 5


def test_estimate_is_monotonic() -> None:
    short = [system("s"), user("q1")]
    long = [system("s"), user("q1"), assistant("a1")]
    assert estimate_tokens(long) > estimate_tokens(short)


# ----------------------------------------------------------------- 不变量检查


def test_invariants_accept_plain_dialog() -> None:
    check_invariants([system("s"), user("q1"), assistant("a1")])


def test_invariants_accept_tool_round() -> None:
    check_invariants(
        [
            system("s"),
            user("A1 还有货吗"),
            assistant(tool_calls=[ToolCall("c1", "get_stock", {"sku": "A1"})]),
            tool_result("c1", '{"available":5}'),
            assistant("还有 5 件"),
        ]
    )


def test_invariants_accept_parallel_calls_in_any_order() -> None:
    check_invariants(
        [
            user("查 A1 和 A2"),
            assistant(
                tool_calls=[
                    ToolCall("c1", "get_stock", {"sku": "A1"}),
                    ToolCall("c2", "get_stock", {"sku": "A2"}),
                ]
            ),
            tool_result("c2", '{"available":3}'),
            tool_result("c1", '{"available":5}'),
        ]
    )


def test_invariants_reject_orphan_tool_message() -> None:
    with pytest.raises(ContextError, match="tool"):
        check_invariants([system("s"), user("q"), tool_result("c9", "{}")])


def test_invariants_reject_missing_tool_reply() -> None:
    with pytest.raises(ContextError):
        check_invariants(
            [
                user("A1 还有货吗"),
                assistant(tool_calls=[ToolCall("c1", "get_stock", {})]),
            ]
        )


def test_invariants_reject_partial_tool_replies() -> None:
    with pytest.raises(ContextError):
        check_invariants(
            [
                user("查两个"),
                assistant(
                    tool_calls=[
                        ToolCall("c1", "get_stock", {}),
                        ToolCall("c2", "get_stock", {}),
                    ]
                ),
                tool_result("c1", "{}"),  # 只回了一个
            ]
        )


def test_invariants_reject_wrong_tool_call_id() -> None:
    with pytest.raises(ContextError):
        check_invariants(
            [
                user("A1 还有货吗"),
                assistant(tool_calls=[ToolCall("c1", "get_stock", {})]),
                tool_result("c2", "{}"),  # id 对不上
            ]
        )


def test_invariants_reject_user_between_call_and_result() -> None:
    with pytest.raises(ContextError):
        check_invariants(
            [
                user("A1 还有货吗"),
                assistant(tool_calls=[ToolCall("c1", "get_stock", {})]),
                user("算了别查了"),  # 插在中间
                tool_result("c1", "{}"),
            ]
        )


# ------------------------------------------------------------------ trim_history


def _three_rounds() -> list:
    return [system("s"), *_plain_turn(1), *_plain_turn(2), *_plain_turn(3)]


def test_trim_drops_oldest_rounds_keeps_system() -> None:
    # 总计 5 + 12*3 = 41；预算 25 → 丢 2 轮 → 5 + 12 = 17
    result = trim_history(_three_rounds(), max_tokens=25)

    assert result.dropped_rounds == 2
    assert result.estimated_tokens == 17
    assert [m.role for m in result.messages] == ["system", "user", "assistant"]
    assert result.messages[1].content == "q3"


def test_trim_keeps_last_round_even_if_over_budget() -> None:
    result = trim_history(_three_rounds(), max_tokens=1)

    assert result.dropped_rounds == 2          # 只能丢前两轮
    assert result.messages[-1].content == "a3"  # 最后一轮仍在
    assert result.estimated_tokens > 1          # 明知超预算也保留


def test_trim_never_splits_tool_group() -> None:
    messages = [
        system("s"),
        user("查库存"),
        assistant(tool_calls=[ToolCall("c1", "get_stock", {})]),
        tool_result("c1", '{"available":5}'),
        user("好的"),
        assistant("嗯"),
    ]
    # 轮 1 估算 30（7+4+19），轮 2 估算 11，system 5；预算 20 → 只能留轮 2
    result = trim_history(messages, max_tokens=20)

    assert result.dropped_rounds == 1
    assert [m.role for m in result.messages] == ["system", "user", "assistant"]
    check_invariants(result.messages)          # 不能切出半个工具回合


def test_trim_is_idempotent() -> None:
    once = trim_history(_three_rounds(), max_tokens=25)
    twice = trim_history(once.messages, max_tokens=25)

    assert twice.messages == once.messages
    assert twice.dropped_rounds == 0
