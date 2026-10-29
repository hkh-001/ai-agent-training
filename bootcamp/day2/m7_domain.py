"""M7：电商库存领域模型（纯数据 + 不变量 + 纯函数）。

为什么 Agent 项目要先做领域建模，而不是直接写工具函数：

1. **工具是门面，领域才是事实**。模型看到的是一句 `{"available": 22}`，
   但那个数字必须由一套**可验证的规则**算出来，否则 Agent 就是在编数据；
2. **不变量是 reward 的雏形**。"库存不能为负""预留不能超过在库"——
   这些判断以后会直接变成评测（M12）和 RL 的 verifier；
3. **纯函数让轨迹可回放**。同样的输入必须得到同样的输出，
   否则你的 trajectory 复现不出来（RL 的大忌）。

本模块的三条设计原则：

    A. 状态不可变：所有操作都返回新的 Inventory，绝不原地修改
    B. 金额用整数分：float 会累积误差（0.1 + 0.2 != 0.3）
    C. 失败用异常表达：DomainError 带可读文案，由上层（M8）翻译成工具错误
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any


class DomainError(ValueError):
    """业务规则被违反（库存不足、未知商品、数量非法）。

    注意：它是 ValueError 的子类，表示"输入有问题"，
    与程序 bug（TypeError/KeyError）区分开。M8 会把它翻译成 `retryable=false`
    的工具结果——因为改参数也没用，得改策略。
    """


@dataclass(frozen=True)
class Sku:
    """一个商品的库存快照。

    字段含义（这套口径要固定下来，否则数字永远对不上）：
        on_hand  实际在库数量（仓库里躺着多少）
        reserved 已被预留但未发出的数量
        available = on_hand - reserved   ← 可以卖的数量
    """

    sku: str
    name: str
    price_cents: int
    on_hand: int
    reserved: int

    @property
    def available(self) -> int:
        return self.on_hand - self.reserved


@dataclass(frozen=True)
class Inventory:
    """全部商品的快照。只读视图，任何"修改"都会返回新对象。"""

    skus: Mapping[str, Sku]

    def __post_init__(self) -> None:
        # frozen 只保证"字段不能重新赋值"，不保证 dict 内部不可改。
        # 用 MappingProxyType 包一层，把"可改的 dict"变成"只读视图"。
        # （这正是 M1 教案里说的"补法 2：不可变结构"）
        object.__setattr__(self, "skus", MappingProxyType(dict(self.skus)))
        assert_invariants(self)

    def get(self, sku: str) -> Sku:
        item = self.skus.get(sku)
        if item is None:
            raise DomainError(
                f"未知商品: {sku!r}（现有: {', '.join(sorted(self.skus))}）"
            )
        return item

    def with_sku(self, item: Sku) -> Inventory:
        """返回一个替换了某个 SKU 的新 Inventory（不修改自己）。"""
        return Inventory(skus={**self.skus, item.sku: item})


def _require_positive_int(value: Any, field_name: str) -> int:
    """整数字段校验：bool 也要拒绝（True 是 int 的子类，M4 学过的坑）。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise DomainError(f"{field_name} 必须是整数，收到 {type(value).__name__}")
    if value <= 0:
        raise DomainError(f"{field_name} 必须大于 0，收到 {value}")
    return value


def assert_invariants(inventory: Inventory) -> None:
    """领域不变量检查。任何一次状态变化之后都应该能过这一关。"""
    for item in inventory.skus.values():
        if item.on_hand < 0:
            raise DomainError(f"{item.sku}: on_hand 不能为负（{item.on_hand}）")
        if item.reserved < 0:
            raise DomainError(f"{item.sku}: reserved 不能为负（{item.reserved}）")
        if item.reserved > item.on_hand:
            raise DomainError(
                f"{item.sku}: 预留量 {item.reserved} 不能超过在库量 {item.on_hand}"
            )
        if item.price_cents <= 0:
            raise DomainError(f"{item.sku}: 价格必须为正（{item.price_cents} 分）")


# --------------------------------------------------------------------- 读写


def load_inventory(path: str | Path) -> Inventory:
    """从 JSON 文件加载库存，并逐字段校验。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    skus: dict[str, Sku] = {}
    for row in data.get("skus", []):
        try:
            sku = str(row["sku"])
            price = _require_positive_int(row["price_cents"], f"{sku}.price_cents")
            on_hand = row["on_hand"]
            reserved = row["reserved"]
            if isinstance(on_hand, bool) or not isinstance(on_hand, int) or on_hand < 0:
                raise DomainError(f"{sku}: on_hand 必须是非负整数")
            if isinstance(reserved, bool) or not isinstance(reserved, int) or reserved < 0:
                raise DomainError(f"{sku}: reserved 必须是非负整数")
            skus[sku] = Sku(
                sku=sku,
                name=str(row["name"]),
                price_cents=price,
                on_hand=on_hand,
                reserved=reserved,
            )
        except KeyError as exc:
            raise DomainError(f"库存数据缺字段: {exc}") from exc

    if not skus:
        raise DomainError("库存数据为空")
    return Inventory(skus=skus)


def to_dict(inventory: Inventory) -> dict[str, Any]:
    """转成可 JSON 序列化的 dict（工具返回值 / 落盘都用它）。"""
    return {
        "skus": [
            {
                "sku": item.sku,
                "name": item.name,
                "price_cents": item.price_cents,
                "on_hand": item.on_hand,
                "reserved": item.reserved,
                "available": item.available,
            }
            for item in sorted(inventory.skus.values(), key=lambda s: s.sku)
        ]
    }


# --------------------------------------------------------------------- 领域操作


def check_stock(inventory: Inventory, sku: str) -> dict[str, Any]:
    """只读查询：返回给模型看的库存事实。"""
    item = inventory.get(sku)
    return {
        "sku": item.sku,
        "name": item.name,
        "available": item.available,
        "reserved": item.reserved,
        "price_cents": item.price_cents,
        "price_display": format_price(item.price_cents),
    }


def reserve(inventory: Inventory, sku: str, qty: int) -> Inventory:
    """预留 qty 件（reserved += qty），返回新状态。

    这是**写操作**：它改变了业务事实。所以 M8 会给它加幂等键。
    """
    qty = _require_positive_int(qty, "qty")
    item = inventory.get(sku)
    if qty > item.available:
        raise DomainError(
            f"库存不足：{sku} 当前可用 {item.available} 件，你请求 {qty} 件"
        )
    return inventory.with_sku(replace(item, reserved=item.reserved + qty))


def release(inventory: Inventory, sku: str, qty: int) -> Inventory:
    """释放预留（reserved -= qty）。幂等的另一半：取消时用。"""
    qty = _require_positive_int(qty, "qty")
    item = inventory.get(sku)
    if qty > item.reserved:
        raise DomainError(
            f"释放失败：{sku} 当前只预留了 {item.reserved} 件，无法释放 {qty} 件"
        )
    return inventory.with_sku(replace(item, reserved=item.reserved - qty))


def ship(inventory: Inventory, sku: str, qty: int) -> Inventory:
    """发货（on_hand 和 reserved 同时减少）。"""
    qty = _require_positive_int(qty, "qty")
    item = inventory.get(sku)
    if qty > item.reserved:
        raise DomainError(f"发货失败：{sku} 预留量不足（{item.reserved} < {qty}）")
    return inventory.with_sku(
        replace(item, on_hand=item.on_hand - qty, reserved=item.reserved - qty)
    )


def format_price(price_cents: int) -> str:
    """整数分 → 展示用字符串。永远不要用 float 做金额计算。"""
    return f"¥{price_cents / 100:.2f}"
