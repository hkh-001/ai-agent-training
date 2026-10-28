"""M3：从 Python 函数签名生成 JSON Schema（也就是"模型看到的菜单"）。

核心认知：
    模型看不到你的函数，它只看到一段 JSON Schema。
    schema 里的 name / description / 参数说明，**就是你写给模型看的提示词**。

    函数签名是**给程序看的约束**（Python 会强制），
    JSON Schema 是**给模型看的描述**（模型只是"尽量遵守"）。
    两者必须从同一处生成，否则会漂移：改了函数忘了改 schema，模型就会传错误的参数。

数据流::

    带类型标注 + docstring 的函数
            │  tool_schema()
            ▼
    {"type": "function",
     "function": {"name": "get_stock",
                  "description": "查询某商品的可售库存",
                  "parameters": {"type": "object",
                                 "properties": {"sku": {"type": "string",
                                                        "description": "商品编码"}},
                                 "required": ["sku"],
                                 "additionalProperties": False}}}
            │  放进 tools=[...]
            ▼
    模型据此决定"调哪个工具、传什么参数"
"""

from __future__ import annotations

import inspect
import types
from collections.abc import Callable, Sequence
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

# Python 类型 → JSON Schema 类型
_PRIMITIVES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    dict: "object",
}

# docstring 里 Args 段结束的标志
_SECTION_END = {
    "Returns:",
    "Raises:",
    "Yields:",
    "Note:",
    "Example:",
    "Examples:",
}


class SchemaError(ValueError):
    """函数无法被转换成合法的 tool schema。"""


def _type_schema(annotation: Any) -> dict[str, Any]:
    """把单个类型标注转成 JSON Schema 片段。"""
    # 注意：模块里写了 `from __future__ import annotations` 时，
    # 运行时注解会变成字符串（如 'str'）。调用方应先用 get_type_hints 解析；
    # 走到这里的字符串说明解析失败（比如引用了未导入的名字）。
    if isinstance(annotation, str):
        raise SchemaError(
            f"类型标注 {annotation!r} 是字符串且无法解析（检查名字是否已导入）"
        )

    if annotation in _PRIMITIVES:
        return {"type": _PRIMITIVES[annotation]}

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Literal["a", "b"] → 枚举
    if origin is Literal or annotation is Literal:
        values = list(args)
        if not values:
            raise SchemaError("Literal 至少要有一个候选取值")
        return {"type": _infer_literal_type(values), "enum": values}

    # Optional[str] / str | None → 去掉 None 再映射
    if origin is Union or origin is types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _type_schema(non_none[0])
        raise SchemaError(f"不支持的多类型联合: {annotation!r}")

    # list[str] / list[int] → array
    if origin is list or origin is tuple:
        if origin is tuple:
            raise SchemaError("不支持 tuple 参数，请用 list[T]")
        if not args:
            return {"type": "array", "items": {"type": "string"}}
        return {"type": "array", "items": _type_schema(args[0])}

    raise SchemaError(f"不支持的类型标注: {annotation!r}（请用 str/int/float/bool/list/Literal）")


def _infer_literal_type(values: Sequence[Any]) -> str:
    """根据 Literal 的候选值推断 JSON Schema 类型。"""
    first = values[0]
    if isinstance(first, bool):
        return "boolean"
    if isinstance(first, int):
        return "integer"
    if isinstance(first, float):
        return "number"
    return "string"


def _parse_param_docs(doc: str | None) -> dict[str, str]:
    """从 Google 风格 docstring 的 Args 段解析参数说明。

    支持::

        Args:
            sku: 商品编码
            qty (int): 数量，默认 1

    Returns:
        {"sku": "商品编码", "qty": "数量，默认 1"}
    """
    if not doc:
        return {}

    docs: dict[str, str] = {}
    in_args = False

    for raw_line in doc.splitlines():
        line = raw_line.strip()
        if line in {"Args:", "Arguments:", "Parameters:"}:
            in_args = True
            continue
        if not in_args:
            continue
        if line in _SECTION_END:
            break
        if not line:
            continue

        head, sep, desc = line.partition(":") if ":" in line else line.partition("：")
        if not sep:
            continue
        name = head.split("(")[0].strip()
        if name:
            docs[name] = desc.strip()

    return docs


def tool_schema(
    func: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """把一个函数转成 OpenAI 兼容的 tool schema。

    契约：
    - name 默认取 `func.__name__`
    - description 默认取 docstring 第一行；两者都没有 → SchemaError
    - 每个参数都必须有类型标注，且必须在 docstring 的 Args 段里有说明，
      否则 → SchemaError（描述就是提示词，缺了模型只能瞎猜）
    - 有默认值的参数不进入 required，并把默认值写进 property
    - 统一设置 additionalProperties: False（防止模型自造参数）
    - 不支持 *args / **kwargs / tuple
    """
    if name is None:
        name = func.__name__

    if description is None:
        first_line = (func.__doc__ or "").strip().splitlines()
        description = first_line[0].strip() if first_line else ""
    if not description:
        raise SchemaError(f"工具 {name} 缺少 description（请写 docstring 第一行）")

    param_docs = _parse_param_docs(func.__doc__)

    # 关键：用 get_type_hints 把字符串注解（future annotations）解析成真实类型。
    # 解析失败时（比如引用了未导入的名字）退回原始注解，由 _type_schema 报错。
    try:
        resolved_hints = get_type_hints(func)
    except Exception:  # noqa: BLE001 - 解析失败要让用户看到"哪个参数有问题"
        resolved_hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    signature = inspect.signature(func)
    for param_name, param in signature.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise SchemaError(f"工具 {name} 不支持 *args / **kwargs 参数: {param_name}")

        annotation = resolved_hints.get(param_name, param.annotation)
        if annotation is inspect.Parameter.empty:
            raise SchemaError(f"工具 {name} 的参数 {param_name} 缺少类型标注")

        schema = _type_schema(annotation)

        doc = param_docs.get(param_name)
        if not doc:
            raise SchemaError(
                f"工具 {name} 的参数 {param_name} 缺少说明（请在 docstring 的 Args: 里写）"
            )
        schema["description"] = doc

        if param.default is not inspect.Parameter.empty:
            schema["default"] = param.default
        else:
            required.append(param_name)

        properties[param_name] = schema

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }
