"""M8 验收测试：工具化、幂等、审计、读写分离。

运行：uv run pytest bootcamp/tests/test_m8_tools.py -q

最关键的一条：**同一次调用被重复执行（重试），库存只能扣一次。**
"""

import json

import pytest

from bootcamp.common.dispatch import dispatch
from bootcamp.common.types import ToolCall
from bootcamp.day2.m8_tools import build_store, register_tools


@pytest.fixture
def store():
    return build_store()


@pytest.fixture
def registry(store):
    return register_tools(store)


def call(tool: str, args: dict, call_id: str = "c1") -> ToolCall:
    return ToolCall(call_id, tool, args)


def payload(result) -> dict:
    return json.loads(result.content)


# ------------------------------------------------------------------ 工具契约


def test_idempotency_key_is_not_in_schema(registry) -> None:
    """下划线参数是宿主注入的，绝不该出现在模型菜单里。"""
    schema = registry.get("reserve_stock").parameters

    assert set(schema["properties"]) == {"sku", "qty"}
    assert schema["required"] == ["sku", "qty"]


def test_read_tool_returns_available(registry) -> None:
    result = dispatch(call("get_stock", {"sku": "A100"}), registry)

    assert result.ok is True
    assert payload(result)["data"]["available"] == 22


def test_write_tool_reserves_stock(registry) -> None:
    result = dispatch(call("reserve_stock", {"sku": "A100", "qty": 2}), registry)

    assert result.ok is True
    assert payload(result)["data"]["available"] == 20


# ------------------------------------------------------------------ 幂等


def test_same_call_id_is_idempotent(registry, store) -> None:
    first = dispatch(call("reserve_stock", {"sku": "A100", "qty": 2}, "call_x"), registry)
    second = dispatch(call("reserve_stock", {"sku": "A100", "qty": 2}, "call_x"), registry)

    assert payload(first) == payload(second)          # 结果完全相同
    assert store.inventory.get("A100").available == 20  # 只扣了一次


def test_different_call_id_does_execute_twice(registry, store) -> None:
    dispatch(call("reserve_stock", {"sku": "A100", "qty": 2}, "call_1"), registry)
    dispatch(call("reserve_stock", {"sku": "A100", "qty": 2}, "call_2"), registry)

    assert store.inventory.get("A100").available == 18


def test_same_key_with_different_args_is_rejected(registry) -> None:
    dispatch(call("reserve_stock", {"sku": "A100", "qty": 2}, "call_x"), registry)
    result = dispatch(call("reserve_stock", {"sku": "A100", "qty": 3}, "call_x"), registry)

    assert result.ok is False
    assert "幂等键冲突" in result.error


def test_release_is_idempotent(registry, store) -> None:
    dispatch(call("release_stock", {"sku": "A100", "qty": 2}, "r1"), registry)
    dispatch(call("release_stock", {"sku": "A100", "qty": 2}, "r1"), registry)

    assert store.inventory.get("A100").reserved == 1  # 3 - 2，只释放一次


# ------------------------------------------------------------------ 业务错误


def test_insufficient_stock_is_not_retryable(registry) -> None:
    result = dispatch(call("reserve_stock", {"sku": "A900", "qty": 5}), registry)

    assert result.ok is False
    assert result.retryable is False          # 改参数也没用，得改策略
    assert "库存不足" in result.error


def test_unknown_sku_is_not_retryable(registry) -> None:
    result = dispatch(call("get_stock", {"sku": "NOPE"}), registry)

    assert result.ok is False
    assert result.retryable is False
    assert "未知商品" in result.error


def test_invalid_qty_is_rejected_before_touching_state(registry, store) -> None:
    result = dispatch(call("reserve_stock", {"sku": "A100", "qty": 0}), registry)

    assert result.ok is False
    assert store.inventory.get("A100").available == 22  # 状态没被动过


# ------------------------------------------------------------------ 审计


def test_audit_records_writes_only(registry, store) -> None:
    dispatch(call("get_stock", {"sku": "A100"}, "c1"), registry)
    dispatch(call("reserve_stock", {"sku": "A100", "qty": 2}, "c2"), registry)

    assert [entry.action for entry in store.audit] == ["reserve"]  # 读不记审计


def test_audit_marks_reused_calls(registry, store) -> None:
    dispatch(call("reserve_stock", {"sku": "A100", "qty": 2}, "same"), registry)
    dispatch(call("reserve_stock", {"sku": "A100", "qty": 2}, "same"), registry)

    entries = store.audit
    assert len(entries) == 2
    assert entries[0].reused is False
    assert entries[1].reused is True


def test_audit_records_failure_without_caching(registry, store) -> None:
    dispatch(call("reserve_stock", {"sku": "A900", "qty": 5}, "fail"), registry)

    entry = store.audit[-1]
    assert entry.ok is False
    assert "库存不足" in (entry.error or "")
    # 失败不写缓存：修正数量后用同一个键重试应该能成功
    ok = dispatch(call("reserve_stock", {"sku": "A900", "qty": 2}, "fail"), registry)
    assert ok.ok is True


def test_audit_captures_before_and_after(registry, store) -> None:
    dispatch(call("reserve_stock", {"sku": "A100", "qty": 2}, "k1"), registry)

    entry = store.audit[-1]
    assert entry.before["available"] == 22
    assert entry.after["available"] == 20
    assert entry.idempotency_key == "k1"


def test_host_injects_call_id_as_key(registry, store) -> None:
    dispatch(call("reserve_stock", {"sku": "A100", "qty": 1}, "call_abc"), registry)

    assert store.audit[-1].idempotency_key == "call_abc"


def test_state_stays_valid_after_operations(registry, store) -> None:
    dispatch(call("reserve_stock", {"sku": "A100", "qty": 2}, "a"), registry)
    dispatch(call("release_stock", {"sku": "A100", "qty": 1}, "b"), registry)

    item = store.inventory.get("A100")
    assert item.available == 21
    assert item.reserved == 4
