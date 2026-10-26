"""使用 OpenAI Python SDK 调用 DeepSeek 的兼容 API。"""

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


def read_required_env(name: str) -> str:
    """读取必需环境变量，拒绝不存在或只有空白的配置。"""
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"缺少环境变量 {name}")
    return value


def ask_deepseek(user_input: str) -> str:
    """通过 DeepSeek Chat Completions API 返回一次文本回答。"""
    normalized_input = user_input.strip()
    if not normalized_input:
        raise ValueError("用户输入不能为空")

    client = OpenAI(
        api_key=read_required_env("DEEPSEEK_API_KEY"),
        base_url=read_required_env("DEEPSEEK_BASE_URL"),
        timeout=30.0,
        max_retries=2,
    )
    response = client.chat.completions.create(
        model=read_required_env("DEEPSEEK_MODEL"),
        messages=[
            {
                "role": "system",
                "content": "你是一名耐心的 AI 入门导师。请用简洁中文回答。",
            },
            {"role": "user", "content": normalized_input},
        ],
        stream=False,
    )

    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("模型响应中没有可显示的文本")
    return answer


def main() -> None:
    """加载本机配置，读取问题，并处理常见 API 错误。"""
    load_dotenv()
    question = input("请输入问题：")

    try:
        answer = ask_deepseek(question)
    except ValueError as exc:
        print(f"配置或输入错误：{exc}", file=sys.stderr)
    except AuthenticationError:
        print("鉴权失败：请检查新建的 DEEPSEEK_API_KEY。", file=sys.stderr)
    except APITimeoutError:
        print("请求超时：请检查网络后再试。", file=sys.stderr)
    except APIConnectionError:
        print("连接失败：请检查网络或代理设置。", file=sys.stderr)
    except RateLimitError:
        print("达到速率或额度限制：请稍后重试并检查账户余额。", file=sys.stderr)
    except APIStatusError as exc:
        print(f"API 返回错误状态 {exc.status_code}。", file=sys.stderr)
    except RuntimeError as exc:
        print(f"响应错误：{exc}", file=sys.stderr)
    else:
        print(f"模型回答：{answer}")


if __name__ == "__main__":
    main()
