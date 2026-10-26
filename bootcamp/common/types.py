"""M1：Agent 的通用语言——消息（Message）与工具调用（ToolCall）。

内部表示 vs 传输表示（wire）::

    内部：ChatMessage(role, content, tool_calls=[ToolCall(id, name, arguments: dict)],
                      tool_call_id)
    边界：{"role": ..., "content": ...,
           "tool_calls": [{"id": ..., "type": "function",
                           "function": {"name": ..., "arguments": "<JSON 字符串>"}}]}

三条铁律：
1. 内部一律用「已解析的 dict」表示参数，JSON 字符串只存在于边界；
2. 消息不可变（frozen），避免记录下来的轨迹被后续逻辑改坏；
3. 校验集中在 validate()，to_wire() 与 from_wire() 都必须先过校验。
"""


from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

Role = str  # 只允许 "system" / "user" / "assistant" / "tool"
VALID_ROLES: frozenset[str] = frozenset({"system", "user", "assistant", "tool"})


@dataclass(frozen=True)
class ToolCall:
    """模型「请求」的一次工具调用。

    注意：这是请求，不是结果。id 由模型生成，宿主程序必须原样回填到
    tool 消息里，否则下一次请求会被服务端拒绝（tool_call_id 对不上）。
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        """转成 OpenAI 兼容的 wire 结构。

        契约：
        - 形如 {"id": ..., "type": "function",
                "function": {"name": ..., "arguments": <JSON 字符串>}}
        - arguments 必须序列化为 JSON 字符串（ensure_ascii=False, 紧凑分隔符）
        - id / name 为空时抛 ValueError
        """
        if not self.id:
            raise ValueError("ToolCall.id 不能为空")
        if not self.name:
            raise ValueError("ToolCall.name 不能为空")
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(
                    self.arguments, ensure_ascii=False, separators=(",", ":")
                ),
            },
        }


@dataclass(frozen=True)
class ChatMessage:
    """一条消息。四种 role 的字段约束见 validate() 的契约。"""

    role: Role
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    def validate(self) -> None:
        """校验字段组合是否合法，不合法抛 ValueError，错误信息里带上字段名。

        契约：
        - role 必须属于 VALID_ROLES，否则 ValueError 且信息含 "role"
        - system / user：content 必须是非空白 str；tool_calls 必须为空；
          tool_call_id 必须为 None
        - assistant：content 允许 None 或 str；tool_calls 允许 0..n 个，
          但每个 id 必须非空且在消息内唯一；tool_call_id 必须为 None
        - tool：content 必须是 str（允许 ""）；tool_call_id 必须非空；
          tool_calls 必须为空
        """
        # 0) role 本身是否合法
        if self.role not in VALID_ROLES:
            raise ValueError(f"非法的 role: {self.role!r}")

        # 1) tool 消息：必须有 tool_call_id，content 必须是字符串，不能带 tool_calls
        if self.role == "tool":
            if not self.tool_call_id:
                raise ValueError("tool 消息必须带 tool_call_id")
            if not isinstance(self.content, str):
                raise ValueError("tool 消息的 content 必须是字符串")
            if self.tool_calls:
                raise ValueError("tool 消息不能带 tool_calls")
            return

        # 2) assistant 消息：content 可为 None，但 tool_call id 必须唯一非空
        if self.role == "assistant":
            if self.content is not None and not isinstance(self.content, str):
                raise ValueError("assistant 的 content 必须是 None 或字符串")
            if self.tool_call_id is not None:
                raise ValueError("assistant 消息不能带 tool_call_id")
            seen: set[str] = set()
            for call in self.tool_calls:
                if not call.id:
                    raise ValueError("tool_calls 的 id 不能为空")
                if call.id in seen:
                    raise ValueError(f"tool_calls 的 id 必须唯一: {call.id}")
                seen.add(call.id)
            return

        # 3) system / user：content 必须非空白，不能带工具相关字段
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError(f"{self.role} 消息的 content 不能为空")
        if self.tool_calls:
            raise ValueError(f"{self.role} 消息不能带 tool_calls")
        if self.tool_call_id is not None:
            raise ValueError(f"{self.role} 消息不能带 tool_call_id")

    def to_wire(self) -> dict[str, Any]:
        """转成 wire dict，并保证先通过 validate()。"""
        self.validate()
        wire: dict[str, Any] = {"role": self.role}
        # 1) content：None 就不输出这个键（wire 上不出现 content 字段）
        if self.content is not None:
            wire["content"] = self.content

        # 2) tool_calls：只有 assistant 消息才可能
        if self.tool_calls:
            wire["tool_calls"] = [call.to_wire() for call in self.tool_calls]
        #    提示：self.tool_calls 是个 tuple，用列表推导式，元素是 ToolCall

        # 3) tool_call_id：只有 tool 消息才有
        if self.tool_call_id is not None:
            wire["tool_call_id"] = self.tool_call_id

        # 4) name：可选字段，这里照契约输出
        if self.name is not None:
            wire["name"] = self.name

        return wire

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> ChatMessage:
        """从 wire dict 反序列化，并保证结果通过 validate()。

        契约：
        - role 缺失或不在 VALID_ROLES 中 → ValueError，信息含 "role"
        - tool_calls[].function.arguments 是 JSON 字符串：
          * 解析失败 → ValueError，信息含 "arguments" 并带上原始片段
          * 解析结果不是 dict → ValueError，信息含 "arguments"
        - 其余字段按需读取，缺失即 None / ()
        """
        # 1) role 先查，未知 role 直接拒（别硬塞）
        role = data.get("role")
        if role not in VALID_ROLES:
            raise ValueError(f"非法的 role: {role!r}")

        # 2) tool_calls：先确认它是数组，再把 JSON 字符串解析成 dict
        raw_calls = data.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raise ValueError(  # noqa: TRY004
                f"tool_calls 必须是数组，收到: {type(raw_calls).__name__}"
            )

        tool_calls: list[ToolCall] = []
        for index, raw in enumerate(raw_calls):
            if not isinstance(raw, dict):
                raise ValueError(f"tool_calls[{index}] 必须是对象")  # noqa: TRY004
            fn = raw.get("function") or {}
            raw_args = fn.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError as exc:
                raise ValueError(f"arguments 解析失败: {raw_args!r}") from exc
            if not isinstance(args, dict):
                # 对外部数据统一用 ValueError，上层统一捕获；故忽略 TRY004
                raise ValueError(  # noqa: TRY004
                    f"arguments 必须是 JSON 对象: {raw_args!r}"
                )
            tool_calls.append(
                ToolCall(
                    id=raw.get("id", ""),
                    name=fn.get("name", ""),
                    arguments=args,
                )
            )

        # 3) 组装 + 出门前质检
        msg = cls(
            role=role,
            content=data.get("content"),
            tool_calls=tuple(tool_calls),
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
        )
        msg.validate()
        return msg


def system(content: str) -> ChatMessage:
    """构造 system 消息。"""
    return ChatMessage(role="system", content=content)


def user(content: str) -> ChatMessage:
    """构造 user 消息。"""
    return ChatMessage(role="user", content=content)


def assistant(
    content: str | None = None,
    tool_calls: Sequence[ToolCall] = (),
) -> ChatMessage:
    """构造 assistant 消息（可以是纯文本，也可以是纯工具调用）。"""
    return ChatMessage(role="assistant", content=content, tool_calls=tuple(tool_calls))


def tool_result(tool_call_id: str, content: str) -> ChatMessage:
    """构造一条 tool 结果消息。content 必须是字符串（工具结果需要序列化）。"""
    return ChatMessage(role="tool", content=content, tool_call_id=tool_call_id)


def messages_to_wire(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    """把消息列表整体转成 wire 列表。"""
    return [m.to_wire() for m in messages]
