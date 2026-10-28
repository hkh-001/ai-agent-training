"""M3（续）：工具注册表 —— 把"函数"变成"模型可选的动作"。

为什么需要注册表：
模型只会返回一个**字符串名字**（`get_stock`）。宿主要能：
1. 用这个名字找到对应的函数；
2. 用这个名字的白名单生成 schema 列表（菜单）；
3. 拒绝菜单之外的任何名字（**权限边界**）。

注意：schema 里的 description 是"给模型看的提示词"，
而注册表是"给宿主看的白名单"，两者职责不同，不能混。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from bootcamp.common.schema import tool_schema


class UnknownToolError(LookupError):
    """模型点名了一个不在菜单里的工具。"""


@dataclass(frozen=True)
class ToolSpec:
    """一个可被模型调用的工具。"""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]

    @classmethod
    def from_function(
        cls,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> ToolSpec:
        schema = tool_schema(func, name=name, description=description)
        fn_part = schema["function"]
        return cls(
            name=fn_part["name"],
            description=fn_part["description"],
            parameters=fn_part["parameters"],
            func=func,
        )

    def to_openai(self) -> dict[str, Any]:
        """转成 tools=[...] 里的一个元素。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """工具白名单。模型只能调用注册过的工具。"""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Any:
        """支持两种用法::

            @registry.register
            def get_stock(...): ...

            registry.register(get_stock)
        """

        def _wrap(target: Callable[..., Any]) -> Callable[..., Any]:
            spec = ToolSpec.from_function(target, name=name, description=description)
            if spec.name in self._specs:
                raise ValueError(f"工具重名: {spec.name}")
            self._specs[spec.name] = spec
            return target

        if func is None:
            return _wrap
        return _wrap(func)

    def get(self, name: str) -> ToolSpec:
        """按名字取工具；不存在时抛出带"可用列表"的错误。"""
        spec = self._specs.get(name)
        if spec is None:
            raise UnknownToolError(
                f"未知工具: {name!r}，可用工具: {sorted(self._specs)}"
            )
        return spec

    def names(self) -> list[str]:
        return sorted(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """生成这次请求要传给模型的菜单（顺序稳定，便于测试与缓存）。"""
        return [self._specs[n].to_openai() for n in self.names()]

    def subset(self, names: Sequence[str]) -> ToolRegistry:
        """按白名单取子集：不同场景给模型不同的菜单。"""
        sub = ToolRegistry()
        for name in names:
            spec = self.get(name)
            sub._specs[name] = spec
        return sub
