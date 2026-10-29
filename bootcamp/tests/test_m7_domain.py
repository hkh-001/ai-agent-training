"""M7 验收测试：领域模型的纯函数行为与不变量。

运行：uv run pytest bootcamp/tests/test_m7_domain.py -q

重点看三件事：
    1. 纯函数：操作返回新对象，原对象不变（可回放的前提）
    2. 不变量：非法状态必须在构造时就炸
    3. 口径：available = on_hand - reserved，金额是整数分
"""

import json

import pytest

from bootcamp.day2.m7_domain import (
    DomainError,
    Inventory,
    Sku,
    assert_invariants,
    check_stock,
    format_price,
    load_inventory,
    release,
    reserve,
    ship,
    to_dict,
)

DATA = "bootcamp/data/inventory.json"


@pytest.fixture
def inv() -> Inventory:
    return load_inventory(DATA)


# ------------------------------------------------------------------ 加载与口径


def test_load_reads_all_skus(inv: Inventory) -> None:
    assert sorted(inv.skus) == ["A100", "A200", "A900"]


def test_available_is_on_hand_minus_reserved(inv: Inventory) -> None:
    a100 = inv.get("A100")

    assert (a100.on_hand, a100.reserved, a100.available) == (25, 3, 22)


def test_price_is_integer_cents(inv: Inventory) -> None:
    price = inv.get("A100").price_cents

    assert isinstance(price, int) and not isinstance(price, bool)
    assert format_price(price) == "¥399.00"


def test_json_roundtrip(inv: Inventory, tmp_path) -> None:
    path = tmp_path / "inv.json"
    path.write_text(json.dumps(to_dict(inv)), encoding="utf-8")

    reloaded = load_inventory(path)

    assert to_dict(reloaded) == to_dict(inv)


def test_unknown_field_in_json_rejected(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps({"skus": [{"sku": "X1", "name": "x", "price_cents": 1.5,
                              "on_hand": 1, "reserved": 0}]}),
        encoding="utf-8",
    )

    with pytest.raises(DomainError, match="price_cents"):
        load_inventory(bad)


# ------------------------------------------------------------------ 纯函数性质


def test_reserve_returns_new_state_and_keeps_original(inv: Inventory) -> None:
    updated = reserve(inv, "A100", 2)

    assert updated.get("A100").reserved == 5      # 新状态变了
    assert inv.get("A100").reserved == 3          # 原对象没变（纯净性）
    assert updated is not inv


def test_inventory_is_read_only(inv: Inventory) -> None:
    with pytest.raises(TypeError):
        inv.skus["A100"] = inv.get("A200")  # type: ignore[index]


# ------------------------------------------------------------------ 领域规则


def test_reserve_more_than_available_raises(inv: Inventory) -> None:
    with pytest.raises(DomainError, match="库存不足"):
        reserve(inv, "A100", 999)


def test_reserve_unknown_sku_raises(inv: Inventory) -> None:
    with pytest.raises(DomainError, match="未知商品"):
        reserve(inv, "NOPE", 1)


@pytest.mark.parametrize("qty", [0, -1, True, 1.5, "2"])
def test_reserve_rejects_invalid_qty(inv: Inventory, qty) -> None:
    with pytest.raises(DomainError, match="qty"):
        reserve(inv, "A100", qty)  # type: ignore[arg-type]


def test_release_reduces_reserved(inv: Inventory) -> None:
    updated = release(inv, "A100", 3)

    assert updated.get("A100").reserved == 0
    assert updated.get("A100").available == 25


def test_release_more_than_reserved_raises(inv: Inventory) -> None:
    with pytest.raises(DomainError, match="释放失败"):
        release(inv, "A100", 4)


def test_ship_reduces_both_fields(inv: Inventory) -> None:
    updated = ship(inv, "A100", 3)

    assert (updated.get("A100").on_hand, updated.get("A100").reserved) == (22, 0)


def test_ship_requires_enough_reserved(inv: Inventory) -> None:
    with pytest.raises(DomainError, match="预留量不足"):
        ship(inv, "A100", 4)


# ------------------------------------------------------------------ 不变量


def test_invariant_rejects_negative_on_hand() -> None:
    bad = Sku(sku="X", name="x", price_cents=100, on_hand=-1, reserved=0)

    with pytest.raises(DomainError, match="on_hand"):
        Inventory(skus={"X": bad})


def test_invariant_rejects_reserved_over_on_hand() -> None:
    bad = Sku(sku="X", name="x", price_cents=100, on_hand=1, reserved=2)

    with pytest.raises(DomainError, match="预留量"):
        Inventory(skus={"X": bad})


def test_assert_invariants_passes_on_valid_state(inv: Inventory) -> None:
    assert_invariants(inv)  # 不抛异常即通过


# ------------------------------------------------------------------ 工具视角


def test_check_stock_shape_is_model_friendly(inv: Inventory) -> None:
    payload = check_stock(inv, "A100")

    assert payload["available"] == 22
    assert payload["price_display"] == "¥399.00"
    assert set(payload) == {
        "sku",
        "name",
        "available",
        "reserved",
        "price_cents",
        "price_display",
    }
