"""M4：工具调度器（Dispatcher）——把 ToolCall 变成 ToolResult。

为什么需要一个专门的调度层，而不是直接 `spec.func(**args)`：

1. **信任边界**：`args` 来自模型，是不可信输入。必须校验后再执行；
2. **错误也是数据**：任何失败都要变成模型能读懂的 tool 消息，而不是抛穿程序；
3. **可观测**：每次调用都要有耗时、成败、错误类型，M6 的 trace 直接消费它。

content 的稳定契约（模型据此判断下一步）::

    成功：{"ok": true,  "data": ...}
    失败：{"ok": false, "error": "...", "retryable": true|false}

`retryable` 的含义：
    True  → 改参数 / 换个时机重试，可能成功（未知工具、参数错、超时、瞬时异常）
    False → 重试没有意义，必须换策略（业务规则拒绝，如库存不足）
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from typing import Any

from bootcamp.common.tools import ToolRegistry, ToolSpec, UnknownToolError
from bootcamp.common.types import ChatMessage, ToolCall, tool_result

# 线程池：用于给同步工具加超时。注意——线程无法被强杀，只能"放弃等待"。
_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool")


class ToolBusinessError(RuntimeError):
    """业务规则拒绝（库存不足、超卖、权限不够……）。

    与普通异常的区别：
    - 它是**预期内**的结果，不是 bug；
    - 重试没有意义，所以映射为 retryable=False；
    - 它的 message 会**原样**给模型看，所以必须是给模型读的文案。
    """


@dataclass(frozen=True)
class ToolResult:
    """一次工具执行的结果。"""

    tool_call_id: str
    ok: bool
    content: str
    error: str | None = None
    retryable: bool = False
    latency_ms: float = 0.0
    # True 表示"宿主主动没有执行"（如达到错误预算）。
    # 它既不是成功也不是失败，必须能与两者区分，否则轨迹会骗人。
    cancelled: bool = False

    def to_message(self) -> ChatMessage:
        """转成可以追加进历史的 tool 消息（id 原样带回）。"""
        return tool_result(self.tool_call_id, self.content)


def _fail(
    call: ToolCall,
    error: str,
    *,
    retryable: bool,
    started: float,
) -> ToolResult:
    """构造一条失败结果（content 统一为 JSON，便于模型解析）。"""
    return ToolResult(
        tool_call_id=call.id,
        ok=False,
        content=json.dumps(
            {"ok": False, "error": error, "retryable": retryable},
            ensure_ascii=False,
        ),
        error=error,
        retryable=retryable,
        latency_ms=(time.perf_counter() - started) * 1000,
    )


_JSON_TYPE_CHECKS: dict[str, Any] = {
    "string": lambda v: isinstance(v, str),
    # bool 是 int 的子类，必须显式排除，否则 True 会被当成合法 integer
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


def validate_arguments(spec: ToolSpec, args: dict[str, Any]) -> str | None:
    """按 schema 校验参数，返回错误说明；合法则返回 None。

    为什么用 schema 而不是 `inspect.signature`：
    schema 是**模型看到的同一份契约**，用它校验能保证"模型被要求什么"和
    "宿主检查什么"完全一致，不会漂移。
    """
    properties: dict[str, Any] = spec.parameters.get("properties", {})
    required: list[str] = spec.parameters.get("required", [])

    missing = [name for name in required if name not in args]
    if missing:
        return f"缺少必填参数: {', '.join(missing)}"

    extra = [name for name in args if name not in properties]
    if extra:
        return f"存在多余参数: {', '.join(extra)}（该工具不接受这些字段）"

    for name, value in args.items():
        schema = properties.get(name, {})
        expected = schema.get("type")
        check = _JSON_TYPE_CHECKS.get(expected or "")
        if check is not None and not check(value):
            return (
                f"参数 {name} 类型错误：需要 {expected}，"
                f"收到 {type(value).__name__}（值 {value!r}）"
            )
        allowed = schema.get("enum")
        if allowed is not None and value not in allowed:
            return (
                f"参数 {name} 取值必须是 {allowed} 之一，"
                f"收到 {value!r}"
            )

    return None


def _invoke_with_timeout(
    func: Any, kwargs: dict[str, Any], timeout_s: float
) -> Any:
    """在线程池里执行工具，超时则放弃等待。

    注意：超时后线程仍可能继续运行（Python 无法强杀线程）。所以写操作
    必须自己做幂等（M8 会用 idempotency_key 保证）。
    """
    future = _EXECUTOR.submit(func, **kwargs)
    try:
        return future.result(timeout=timeout_s)
    except FuturesTimeout:
        future.cancel()
        raise TimeoutError(f"工具执行超过 {timeout_s}s") from None


def dispatch(
    call: ToolCall,
    registry: ToolRegistry,
    *,
    timeout_s: float = 5.0,
) -> ToolResult:
    """执行一次工具调用。**任何情况下都不抛异常**，失败也返回 ToolResult。"""
    started = time.perf_counter()

    # ① 查白名单：模型只能调用注册过的工具
    try:
        spec = registry.get(call.name)
    except UnknownToolError as exc:
        return _fail(call, str(exc), retryable=True, started=started)

    # ② 校验参数：模型给的参数是不可信输入
    problem = validate_arguments(spec, call.arguments)
    if problem is not None:
        return _fail(call, problem, retryable=True, started=started)

    # ③ 执行（分三类异常，语义完全不同）
    try:
        result = _invoke_with_timeout(spec.func, call.arguments, timeout_s)
    except ToolBusinessError as exc:
        # 业务拒绝：预期内的结果，重试无意义，message 直接给模型看
        return _fail(call, str(exc), retryable=False, started=started)
    except TimeoutError as exc:
        return _fail(call, f"工具执行超时（{exc}），请稍后重试或换个思路", retryable=True, started=started)
    except Exception as exc:  # noqa: BLE001 - 工具代码不可控，必须兜住
        # 内部异常：只暴露异常类型，堆栈留在日志里（M6），不喂给模型
        return _fail(
            call,
            f"工具内部错误: {type(exc).__name__}（这不是你的参数问题，可稍后重试）",
            retryable=True,
            started=started,
        )

    # ④ 成功：序列化成字符串
    # 注意：工具可能返回 datetime/Decimal/set 等不可序列化对象。
    # 如果这里让 TypeError 抛出去，就违反了"dispatch 永不抛异常"的契约，
    # 整个 Agent 会崩，而模型什么也收不到。（P0-2）
    try:
        content = json.dumps({"ok": True, "data": result}, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        return _fail(
            call,
            f"工具返回值无法序列化为 JSON（{type(exc).__name__}），"
            f"这是工具实现的问题，不是你的参数问题",
            retryable=False,
            started=started,
        )

    return ToolResult(
        tool_call_id=call.id,
        ok=True,
        content=content,
        latency_ms=(time.perf_counter() - started) * 1000,
    )


def dispatch_many(
    calls: list[ToolCall] | tuple[ToolCall, ...],
    registry: ToolRegistry,
    *,
    timeout_s: float = 5.0,
) -> list[ToolResult]:
    """批量执行（模型一轮可能返回多个调用）。保持原有顺序与 id。"""
    return [dispatch(call, registry, timeout_s=timeout_s) for call in calls]
