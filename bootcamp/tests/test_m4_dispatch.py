"""M4 验收测试：调度器的六条失败路径 + 成功路径。

运行：uv run pytest bootcamp/tests/test_m4_dispatch.py -q

核心断言思路：**不管出什么错，dispatch 都不抛异常**，
而是返回一个 content 可被 json.loads 的 ToolResult。
"""

import json
import time

from bootcamp.common.dispatch import (
    ToolBusinessError,
    dispatch,
    dispatch_many,
    validate_arguments,
)
from bootcamp.common.tools import ToolRegistry, ToolSpec
from bootcamp.common.types import ToolCall

# ------------------------------------------------------------------ 测试用工具


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()

    @registry.register
    def get_stock(sku: str, qty: int = 1) -> dict:
        """查询某商品的可售库存。

        Args:
            sku: 商品编码
            qty: 需要核对的件数
        """
        return {"sku": sku, "available": 5, "qty": qty}

    @registry.register
    def reserve_stock(sku: str, qty: int) -> dict:
        """预留库存；库存不足时抛出业务错误。

        Args:
            sku: 商品编码
            qty: 预留件数
        """
        if qty > 5:
            raise ToolBusinessError("库存不足：当前可用 5 件，请减少数量")
        return {"sku": sku, "reserved": qty}

    @registry.register
    def boom() -> dict:
        """总是抛内部异常的工具。"""
        raise RuntimeError("数据库连接断了")

    @registry.register
    def slow(seconds: float) -> str:
        """睡眠指定秒数，用于测试超时。

        Args:
            seconds: 睡眠时长
        """
        time.sleep(seconds)
        return "done"

    return registry


def _payload(result) -> dict:
    """content 必须是可解析的 JSON —— 这是模型的读取方式。"""
    return json.loads(result.content)


# ------------------------------------------------------------------ 成功路径


def test_success_wraps_data_and_keeps_id() -> None:
    result = dispatch(ToolCall("c1", "get_stock", {"sku": "A1"}), make_registry())

    assert result.ok is True
    assert result.tool_call_id == "c1"
    assert _payload(result) == {"ok": True, "data": {"sku": "A1", "available": 5, "qty": 1}}
    assert isinstance(result.content, str)
    assert result.latency_ms >= 0


def test_success_applies_default_when_arg_omitted() -> None:
    result = dispatch(ToolCall("c1", "get_stock", {"sku": "A1"}), make_registry())

    assert _payload(result)["data"]["qty"] == 1


def test_result_converts_to_tool_message() -> None:
    result = dispatch(ToolCall("c1", "get_stock", {"sku": "A1"}), make_registry())
    message = result.to_message()

    assert message.role == "tool"
    assert message.tool_call_id == "c1"
    assert message.content == result.content


# ------------------------------------------------------------------ 失败路径


def test_unknown_tool_is_retryable() -> None:
    result = dispatch(ToolCall("c1", "delete_all", {}), make_registry())

    assert result.ok is False
    assert result.retryable is True
    assert "delete_all" in result.error
    assert "get_stock" in result.error  # 错误里带上可用清单，模型才有机会改


def test_missing_required_arg() -> None:
    result = dispatch(ToolCall("c1", "get_stock", {}), make_registry())

    assert result.ok is False
    assert result.retryable is True
    assert "缺少必填参数" in result.error
    assert "sku" in result.error


def test_wrong_type_arg() -> None:
    result = dispatch(
        ToolCall("c1", "get_stock", {"sku": "A1", "qty": "2"}), make_registry()
    )

    assert result.ok is False
    assert result.retryable is True
    assert "qty" in result.error and "integer" in result.error


def test_bool_is_not_accepted_as_integer() -> None:
    """True 在 Python 里是 int 的子类，但作为 qty 必须被拒绝。"""
    result = dispatch(
        ToolCall("c1", "get_stock", {"sku": "A1", "qty": True}), make_registry()
    )

    assert result.ok is False
    assert "qty" in result.error


def test_extra_arg_rejected() -> None:
    result = dispatch(
        ToolCall("c1", "get_stock", {"sku": "A1", "force": True}), make_registry()
    )

    assert result.ok is False
    assert "多余参数" in result.error
    assert "force" in result.error


def test_business_error_is_not_retryable() -> None:
    result = dispatch(
        ToolCall("c1", "reserve_stock", {"sku": "A1", "qty": 99}), make_registry()
    )

    assert result.ok is False
    assert result.retryable is False  # 重试无意义
    assert "库存不足" in result.error


def test_internal_error_does_not_leak_traceback() -> None:
    result = dispatch(ToolCall("c1", "boom", {}), make_registry())

    assert result.ok is False
    assert result.retryable is True
    assert "RuntimeError" in result.error
    assert "Traceback" not in result.content
    assert "数据库连接断了" not in result.content  # 内部细节只进日志


def test_timeout_is_retryable() -> None:
    result = dispatch(
        ToolCall("c1", "slow", {"seconds": 0.5}), make_registry(), timeout_s=0.05
    )

    assert result.ok is False
    assert result.retryable is True
    assert "超时" in result.error


# ------------------------------------------------------------------ 批量与校验


def test_dispatch_many_keeps_order_and_ids() -> None:
    results = dispatch_many(
        [ToolCall("c1", "get_stock", {"sku": "A1"}), ToolCall("c2", "boom", {})],
        make_registry(),
    )

    assert [r.tool_call_id for r in results] == ["c1", "c2"]
    assert results[0].ok is True
    assert results[1].ok is False


def test_validate_arguments_accepts_clean_input() -> None:
    registry = make_registry()
    spec = registry.get("get_stock")

    assert validate_arguments(spec, {"sku": "A1"}) is None


def test_validate_arguments_checks_enum() -> None:
    spec = ToolSpec.from_function(make_enum_tool())

    assert validate_arguments(spec, {"direction": "hold"}) is None
    problem = validate_arguments(spec, {"direction": "HOLD"})
    assert problem is not None and "HOLD" in problem


def make_enum_tool():
    from typing import Literal

    def move(direction: Literal["hold", "release"]) -> str:
        """移动库存。

        Args:
            direction: 方向
        """
        return direction

    return move
