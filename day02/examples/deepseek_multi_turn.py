"""使用 DeepSeek Chat Completions API 运行多轮对话。"""

import json
import os
import sys

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from day02.examples.conversation import Conversation
from day02.examples.prompt_builder import build_tutor_prompt


def read_required_env(name: str) -> str:
    """读取必需环境变量。"""
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"缺少环境变量 {name}")
    return value


def create_client() -> OpenAI:
    """创建带有限超时和重试的 DeepSeek 兼容客户端。"""
    return OpenAI(
        api_key=read_required_env("DEEPSEEK_API_KEY"),
        base_url=read_required_env("DEEPSEEK_BASE_URL"),
        timeout=30.0,
        max_retries=2,
    )


def ask_deepseek(client: OpenAI, conversation: Conversation) -> tuple[str, int]:
    """发送完整消息历史，返回回答与本次总 Token 数。"""
    response = client.chat.completions.create(
        model=read_required_env("DEEPSEEK_MODEL"),
        messages=conversation.messages,  # type: ignore[arg-type]
        stream=False,
        extra_body={"thinking": {"type": "disabled"}},
    )
    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("模型响应中没有可显示的文本")
    total_tokens = response.usage.total_tokens if response.usage else 0
    return answer, total_tokens


def print_messages(conversation: Conversation) -> None:
    """显示本轮实际发送的消息历史。"""
    print("\n--- 本轮发送的 messages ---")
    print(json.dumps(conversation.messages, ensure_ascii=False, indent=2))
    print("-----------------------------")


def main() -> None:
    """运行真实多轮对话。"""
    load_dotenv()
    conversation = Conversation(build_tutor_prompt(), max_rounds=3)

    try:
        client = create_client()
    except ValueError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return

    print("DeepSeek 多轮助手已启动；输入 /quit 退出。")
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
        try:
            answer, total_tokens = ask_deepseek(client, conversation)
        except AuthenticationError:
            print("鉴权失败：请检查 DEEPSEEK_API_KEY。", file=sys.stderr)
            return
        except APITimeoutError:
            print("请求超时，请检查网络后重试。", file=sys.stderr)
            continue
        except APIConnectionError:
            print("连接失败，请检查网络或代理。", file=sys.stderr)
            continue
        except RateLimitError:
            print("达到速率或额度限制。", file=sys.stderr)
            continue
        except APIStatusError as exc:
            print(f"API 返回错误状态 {exc.status_code}。", file=sys.stderr)
            continue
        except RuntimeError as exc:
            print(f"响应错误：{exc}", file=sys.stderr)
            continue

        conversation.add_assistant(answer)
        print(f"助手：{answer}")
        print(f"本次请求总 Token：{total_tokens}")


if __name__ == "__main__":
    main()
