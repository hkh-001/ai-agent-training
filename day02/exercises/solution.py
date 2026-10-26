"""Day 2 参考实现：有限历史的昵称助手。"""

import json

from day02.examples.conversation import Conversation, Message
from day02.examples.mock_chat_model import MockChatModel
from day02.examples.prompt_builder import build_tutor_prompt


class NicknameAssistant:
    """组合会话状态与 Mock Model 的小型助手。"""

    def __init__(self, max_rounds: int = 3) -> None:
        self.max_rounds = max_rounds
        self.model = MockChatModel()
        self.conversation = self._new_conversation()

    def _new_conversation(self) -> Conversation:
        return Conversation(build_tutor_prompt(), max_rounds=self.max_rounds)

    @property
    def messages(self) -> list[Message]:
        """返回当前消息历史副本。"""
        return self.conversation.messages

    def reply(self, user_input: str) -> str:
        """追加用户消息，生成回答，再保存 assistant 消息。"""
        self.conversation.add_user(user_input)
        answer = self.model.generate(self.conversation.messages)
        self.conversation.add_assistant(answer)
        return answer

    def reset(self) -> None:
        """创建全新会话，仅保留 System Message。"""
        self.conversation = self._new_conversation()


def main() -> None:
    """运行带历史查看与重置命令的交互程序。"""
    assistant = NicknameAssistant(max_rounds=3)
    print("昵称助手已启动：/history 查看历史，/reset 重置，/quit 退出。")

    while True:
        user_input = input("\n你：").strip()
        if user_input == "/quit":
            print("对话结束。")
            return
        if user_input == "/history":
            print(json.dumps(assistant.messages, ensure_ascii=False, indent=2))
            continue
        if user_input == "/reset":
            assistant.reset()
            print("会话已重置。")
            continue
        if not user_input:
            print("输入不能为空。")
            continue

        print(f"助手：{assistant.reply(user_input)}")


if __name__ == "__main__":
    main()

