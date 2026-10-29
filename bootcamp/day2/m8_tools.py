"""M8：把领域函数包装成 Agent 工具（幂等 + 审计 + 读写分离）。

M7 是"纯函数的世界"，M8 是"可被模型操作的世界"。这一步引入三个新问题：

1. **写操作必须幂等**
   超时重试、断线重连、模型重复调用——同一个请求可能被执行多次。
   如果每次扣库存，用户订 2 件会被扣成 4 件。
   解法：幂等键。同一个键 + 同样的参数 → 直接返回上次结果，不重复执行。

2. **幂等键不能交给模型编**
   模型编的键可能重复、可能每次都变。真正的幂等键应该由**宿主**生成——
   这里用 `tool_call_id`：同一次调用被重试时它不变，天然幂等。

3. **写操作必须留审计**
   出事时要能回答"谁在什么时候、用什么键、把库存从多少改到了多少"。
   审计是 SRE 视角的 trace，不是给模型看的。

另外，工具分两类，待遇完全不同：
    只读（get_stock）  → 可以随便重试
    写（reserve/release）→ 必须幂等 + 审计
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import wraps
from typing import Any

from bootcamp.common.dispatch import ToolBusinessError
from bootcamp.common.tools import ToolRegistry
from bootcamp.day2.m7_domain import (
    DomainError,
    Inventory,
    check_stock,
    load_inventory,
    release,
    reserve,
)


@dataclass(frozen=True)
class AuditEntry:
    """一条写操作审计记录。"""

    seq: int
    ts: float
    action: str                       # "reserve" / "release"
    sku: str
    qty: int
    idempotency_key: str
    ok: bool
    reused: bool                      # True 表示命中了幂等缓存（没有真正执行）
    before: dict[str, Any]
    after: dict[str, Any]
    error: str | None = None


class Store:
    """允许有状态的业务世界。

    整个项目里，**只有这里会变**。其余模块都是纯函数或无状态对象，
    这是刻意的：状态越集中，越容易回放、测试和推理。
    """

    def __init__(self, inventory: Inventory) -> None:
        self._inventory = inventory
        self._audit: list[AuditEntry] = []
        self._seq = 0
        # 幂等缓存：key -> (参数指纹, 结果)
        self._idempotency: dict[str, tuple[str, dict[str, Any]]] = {}

    @property
    def inventory(self) -> Inventory:
        return self._inventory

    @property
    def audit(self) -> tuple[AuditEntry, ...]:
        return tuple(self._audit)

    # ---------------------------------------------------------------- 只读

    def check(self, sku: str) -> dict[str, Any]:
        """只读查询，不写审计（读操作太多，记了也是噪音）。"""
        return check_stock(self._inventory, sku)

    # ---------------------------------------------------------------- 写

    def reserve(self, sku: str, qty: int, key: str) -> dict[str, Any]:
        return self._write("reserve", sku, qty, key)

    def release(self, sku: str, qty: int, key: str) -> dict[str, Any]:
        return self._write("release", sku, qty, key)

    def _write(self, action: str, sku: str, qty: int, key: str) -> dict[str, Any]:
        if not key:
            raise DomainError(
                "写操作缺少幂等键（应由宿主注入 tool_call_id，不应由模型提供）"
            )

        fingerprint = f"{action}|{sku}|{qty}"
        cached = self._idempotency.get(key)
        if cached is not None:
            cached_fingerprint, cached_result = cached
            if cached_fingerprint != fingerprint:
                # 同一个键配不同参数 = 调用方用错了键，必须报错而不是"随便挑一个"
                raise DomainError(
                    f"幂等键冲突: {key!r} 上次用于 {cached_fingerprint}，"
                    f"本次是 {fingerprint}"
                )
            # 命中幂等缓存：状态没变，before == after == 当前快照
            self._audit_entry(
                action, sku, qty, key, True, cached_result, None,
                before=self._snapshot(sku),
            )
            return cached_result

        before = self._snapshot(sku)
        try:
            if action == "reserve":
                self._inventory = reserve(self._inventory, sku, qty)
            else:
                self._inventory = release(self._inventory, sku, qty)
        except DomainError as exc:
            # 失败也记审计（允许失败后修正参数重试），但不写入幂等缓存
            self._audit_entry(action, sku, qty, key, False, {}, str(exc))
            raise

        result = {"ok": True, "action": action, "sku": sku, "qty": qty, **self._snapshot(sku)}
        self._idempotency[key] = (fingerprint, result)
        self._audit_entry(action, sku, qty, key, False, result, None, before=before)
        return result

    # ---------------------------------------------------------------- 内部

    def _snapshot(self, sku: str) -> dict[str, Any]:
        item = self._inventory.get(sku)
        return {
            "sku": item.sku,
            "on_hand": item.on_hand,
            "reserved": item.reserved,
            "available": item.available,
        }

    def _audit_entry(
        self,
        action: str,
        sku: str,
        qty: int,
        key: str,
        reused: bool,
        result: dict[str, Any],
        error: str | None,
        *,
        before: dict[str, Any] | None = None,
    ) -> None:
        self._seq += 1
        self._audit.append(
            AuditEntry(
                seq=self._seq,
                ts=time.time(),
                action=action,
                sku=sku,
                qty=qty,
                idempotency_key=key,
                ok=error is None,
                reused=reused,
                before=before or {},
                after=dict(result) if result else {},
                error=error,
            )
        )


def _as_business_error(func):
    """把领域异常翻译成"业务拒绝"。

    为什么翻译放在工具层，而不是让 dispatch 认识 DomainError：
    领域层不该知道"Agent 工具"这种东西的存在（依赖方向必须是 上层 → 下层）。

    为什么必须是 ToolBusinessError：dispatch 会把它映射成 retryable=False。
    如果让 DomainError（ValueError 子类）直接冒出去，dispatch 会当成"内部错误"
    标成 retryable=True —— 模型就会对着"库存不足"无限重试。
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except DomainError as exc:
            raise ToolBusinessError(str(exc)) from exc

    return wrapper


def build_store(path: str = "bootcamp/data/inventory.json") -> Store:
    return Store(load_inventory(path))


def register_tools(store: Store) -> ToolRegistry:
    """把领域能力注册成模型可用的工具。

    注意 `_idempotency_key` 是下划线参数：不进菜单、不需要模型填，
    由 dispatch 在执行前注入 `tool_call_id`。
    """
    registry = ToolRegistry()

    @registry.register
    @_as_business_error
    def get_stock(sku: str) -> dict:
        """查询某商品当前的可售库存（只读，可安全重复调用）。

        Args:
            sku: 商品编码，例如 A100
        """
        return store.check(sku)

    @registry.register
    @_as_business_error
    def reserve_stock(sku: str, qty: int, _idempotency_key: str = "") -> dict:
        """为顾客预留库存（写操作）。预留成功后该数量不再对外可售。

        Args:
            sku: 商品编码，例如 A100
            qty: 预留件数，必须是正整数
        """
        return store.reserve(sku, qty, _idempotency_key)

    @registry.register
    @_as_business_error
    def release_stock(sku: str, qty: int, _idempotency_key: str = "") -> dict:
        """释放已经预留的库存（写操作，用于取消订单）。

        Args:
            sku: 商品编码
            qty: 释放件数，不能超过当前已预留数量
        """
        return store.release(sku, qty, _idempotency_key)

    return registry
