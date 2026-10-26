"""使用原生 Python 管理消息历史和轮次裁剪。"""

from copy import deepcopy
from typing import Literal, TypedDict

MessageRole = Literal["system", "user", "assistant"]


class Message(TypedDict):
    """一条最小聊天消息。"""

    role: MessageRole
    content: str


def build_message(role: MessageRole, content: str) -> Message:
    """创建经过基础校验的消息。"""
    normalized_content = content.strip()
    if not normalized_content:
        raise ValueError("消息内容不能为空")
    return {"role": role, "content": normalized_content}


def trim_to_recent_rounds(messages: list[Message], max_rounds: int) -> list[Message]:
    """保留 System Message 和最近若干个用户轮次。

    一个用户轮次从 ``user`` 消息开始，并可包含紧随其后的 ``assistant``
    消息。函数按最后几个 user 起点裁剪，避免从孤立的 assistant 消息开始。
    """
    if max_rounds < 1:
        raise ValueError("max_rounds 必须大于等于 1")
    if not messages:
        return []

    system_messages = [message for message in messages if message["role"] == "system"]
    dialogue = [message for message in messages if message["role"] != "system"]
    user_indexes = [
        index for index, message in enumerate(dialogue) if message["role"] == "user"
    ]

    if len(user_indexes) <= max_rounds:
        return deepcopy(messages)

    first_kept_index = user_indexes[-max_rounds]
    return deepcopy(system_messages[:1] + dialogue[first_kept_index:])


class Conversation:
    """维护单个会话的消息列表。"""

    def __init__(self, system_prompt: str, max_rounds: int = 3) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds 必须大于等于 1")
        self.max_rounds = max_rounds
        self._messages: list[Message] = [build_message("system", system_prompt)]

    @property
    def messages(self) -> list[Message]:
        """返回副本，避免调用方绕过校验修改内部状态。"""
        return deepcopy(self._messages)

    def add_user(self, content: str) -> None:
        """追加用户消息，并在发送请求前裁剪旧轮次。"""
        self._messages.append(build_message("user", content))
        self._messages = trim_to_recent_rounds(self._messages, self.max_rounds)

    def add_assistant(self, content: str) -> None:
        """追加模型回复。"""
        self._messages.append(build_message("assistant", content))

