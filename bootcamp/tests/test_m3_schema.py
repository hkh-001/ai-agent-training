"""M3 验收测试：函数签名 → JSON Schema，以及工具注册表。

运行：uv run pytest bootcamp/tests/test_m3_schema.py -q
"""

from typing import Literal

import pytest

from bootcamp.common.schema import SchemaError, tool_schema
from bootcamp.common.tools import ToolRegistry, ToolSpec, UnknownToolError


def get_stock(sku: str, qty: int = 1) -> dict:
    """查询某商品的可售库存。

    Args:
        sku: 商品编码，例如 A1
        qty: 需要核对的件数，默认 1

    Returns:
        库存信息
    """
    return {"sku": sku, "available": 5, "qty": qty}


def search_policy(keyword: str, top_k: int = 3) -> list:
    """检索售后政策条款。

    Args:
        keyword: 检索词
        top_k: 返回条数
    """
    return []


# ---------------------------------------------------------------- 基础结构


def test_schema_uses_function_name_and_docstring() -> None:
    schema = tool_schema(get_stock)

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "get_stock"
    assert schema["function"]["description"] == "查询某商品的可售库存。"


def test_schema_parameters_shape() -> None:
    params = tool_schema(get_stock)["function"]["parameters"]

    assert params["type"] == "object"
    assert set(params["properties"]) == {"sku", "qty"}
    assert params["additionalProperties"] is False


def test_required_only_contains_params_without_default() -> None:
    params = tool_schema(get_stock)["function"]["parameters"]

    assert params["required"] == ["sku"]  # qty 有默认值，不进 required


def test_default_value_is_recorded() -> None:
    params = tool_schema(get_stock)["function"]["parameters"]

    assert params["properties"]["qty"]["default"] == 1
    assert "default" not in params["properties"]["sku"]


def test_type_mapping() -> None:
    params = tool_schema(get_stock)["function"]["parameters"]

    assert params["properties"]["sku"]["type"] == "string"
    assert params["properties"]["qty"]["type"] == "integer"


def test_param_description_comes_from_docstring() -> None:
    params = tool_schema(get_stock)["function"]["parameters"]

    assert params["properties"]["sku"]["description"] == "商品编码，例如 A1"


def test_explicit_name_and_description_override() -> None:
    schema = tool_schema(get_stock, name="stock_query", description="查库存")

    assert schema["function"]["name"] == "stock_query"
    assert schema["function"]["description"] == "查库存"


def test_more_types() -> None:
    def demo(
        a: float,
        b: bool,
        c: list[str],
        d: Literal["asc", "desc"],
        e: str | None = None,
    ) -> None:
        """演示多种类型映射。

        Args:
            a: 浮点
            b: 布尔
            c: 字符串列表
            d: 排序方向
            e: 可选值
        """

    props = tool_schema(demo)["function"]["parameters"]["properties"]

    assert props["a"]["type"] == "number"
    assert props["b"]["type"] == "boolean"
    assert props["c"] == {"type": "array", "items": {"type": "string"}, "description": "字符串列表"}
    assert props["d"]["enum"] == ["asc", "desc"]
    assert props["e"]["type"] == "string"


# ---------------------------------------------------------------- 错误路径


def test_missing_annotation_raises() -> None:
    def bad(sku) -> None:  # type: ignore[no-untyped-def]
        """缺类型标注。

        Args:
            sku: 商品编码
        """

    with pytest.raises(SchemaError, match="类型标注"):
        tool_schema(bad)


def test_missing_param_description_raises() -> None:
    def bad(sku: str) -> None:
        """缺参数说明。"""

    with pytest.raises(SchemaError, match="缺少说明"):
        tool_schema(bad)


def test_missing_description_raises() -> None:
    def bad(sku: str) -> None:
        pass

    with pytest.raises(SchemaError, match="description"):
        tool_schema(bad)


def test_var_args_rejected() -> None:
    def bad(*args: str) -> None:
        """不支持可变参数。

        Args:
            args: 若干参数
        """

    with pytest.raises(SchemaError, match="args"):
        tool_schema(bad)


# ---------------------------------------------------------------- 注册表


def test_registry_register_and_list() -> None:
    registry = ToolRegistry()
    registry.register(get_stock)
    registry.register(search_policy)

    assert registry.names() == ["get_stock", "search_policy"]
    assert len(registry) == 2

    tools = registry.to_openai_tools()
    assert [t["function"]["name"] for t in tools] == ["get_stock", "search_policy"]
    assert tools[0]["type"] == "function"


def test_registry_decorator_returns_original_function() -> None:
    registry = ToolRegistry()

    @registry.register
    def ping() -> str:
        """健康检查。"""
        return "pong"

    assert ping() == "pong"
    assert "ping" in registry


def test_registry_unknown_tool() -> None:
    registry = ToolRegistry()
    registry.register(get_stock)

    with pytest.raises(UnknownToolError, match="get_stock"):
        registry.get("delete_everything")


def test_registry_rejects_duplicates() -> None:
    registry = ToolRegistry()
    registry.register(get_stock)

    with pytest.raises(ValueError, match="重名"):
        registry.register(get_stock)


def test_registry_subset_is_a_whitelist() -> None:
    registry = ToolRegistry()
    registry.register(get_stock)
    registry.register(search_policy)

    sub = registry.subset(["search_policy"])

    assert sub.names() == ["search_policy"]
    assert len(registry) == 2  # 原注册表不受影响


def test_spec_from_function_roundtrip() -> None:
    spec = ToolSpec.from_function(get_stock)

    assert spec.name == "get_stock"
    assert spec.parameters["required"] == ["sku"]
    assert spec.to_openai()["function"]["name"] == "get_stock"


def test_string_annotations_are_resolved() -> None:
    """回归测试：模块写了 from __future__ import annotations 时，
    运行时注解决是字符串（'str'），必须先用 get_type_hints 解析。

    这个 bug 在 84 个测试全绿的情况下，被一个真实脚本抓出来了
    —— 因为测试里的工具都没用 future annotations。
    """

    def tool(x: "str") -> dict:
        """演示字符串注解。

        Args:
            x: 参数
        """
        return {}

    props = tool_schema(tool)["function"]["parameters"]["properties"]

    assert props["x"]["type"] == "string"
