"""能读取消息历史的确定性 Mock Chat Model。"""

import re

from day02.examples.conversation import Message

NICKNAME_PATTERN = re.compile(
    r"我叫(?:是|：|:|\s)*(?!什么)([^，。！？,.!?\s]+)"
)


class MockChatModel:
    """根据消息历史演示“会话内记忆”，不模拟真实语言能力。"""

    def generate(self, messages: list[Message]) -> str:
        """读取完整历史并返回可预测回答。"""
        user_messages = [
            message["content"] for message in messages if message["role"] == "user"
        ]
        if not user_messages:
            raise ValueError("至少需要一条 user 消息")

        latest_input = user_messages[-1]
        nickname = self._find_latest_nickname(user_messages)

        if "我叫什么" in latest_input or "我的名字" in latest_input:
            if nickname:
                return f"你之前说你叫{nickname}。"
            return "当前保留的消息历史中没有你的名字。"
        if NICKNAME_PATTERN.search(latest_input):
            return f"好的，我会在当前消息历史中记住你叫{nickname}。"
        return f"这是当前保留历史中的第 {len(user_messages)} 条用户消息。"

    @staticmethod
    def _find_latest_nickname(user_messages: list[str]) -> str | None:
        for content in reversed(user_messages):
            match = NICKNAME_PATTERN.search(content)
            if match:
                return match.group(1)
        return None
