"""无需 API Key 的多轮消息与裁剪演示。"""

import json

from day02.examples.conversation import Conversation
from day02.examples.mock_chat_model import MockChatModel
from day02.examples.prompt_builder import build_tutor_prompt


def print_messages(conversation: Conversation) -> None:
    """打印下一次模型调用会收到的完整消息。"""
    print("\n--- 当前发送给模型的 messages ---")
    print(json.dumps(conversation.messages, ensure_ascii=False, indent=2))
    print("----------------------------------")


def main() -> None:
    """运行最多保留 3 个用户轮次的 Mock 多轮对话。"""
    conversation = Conversation(build_tutor_prompt(), max_rounds=3)
    model = MockChatModel()

    print("Mock 多轮助手已启动；输入 /quit 退出。")
    print("建议依次输入：我叫小林、第一条、第二条、我叫什么？")
    while True:
        user_input = input("\n你：").strip()
        if user_input == "/quit":
            print("对话结束。")
            return
        if not user_input:
            print("输入不能为空。")
            continue

        conversation.add_user(user_input)
        print_messages(conversation)
        answer = model.generate(conversation.messages)
        conversation.add_assistant(answer)
        print(f"助手：{answer}")


if __name__ == "__main__":
    main()

