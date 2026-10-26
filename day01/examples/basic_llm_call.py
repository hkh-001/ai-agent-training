"""使用 OpenAI Responses API 完成一次真实模型调用。"""

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
    """读取必需环境变量，缺失时给出明确错误。"""
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"缺少环境变量 {name}")
    return value


def ask_model(user_input: str) -> str:
    """向模型发送一次请求并返回纯文本结果。"""
    normalized_input = user_input.strip()
    if not normalized_input:
        raise ValueError("用户输入不能为空")

    api_key = read_required_env("OPENAI_API_KEY")
    model_name = read_required_env("OPENAI_MODEL")

    # 超时和有限重试用于避免网络异常时无限等待。
    client = OpenAI(api_key=api_key, timeout=30.0, max_retries=2)
    response = client.responses.create(
        model=model_name,
        instructions="你是一名耐心的 AI 入门导师。请用简洁中文回答。",
        input=normalized_input,
    )

    if not response.output_text:
        raise RuntimeError("模型响应中没有可显示的文本")
    return response.output_text


def main() -> None:
    """读取配置与用户问题，并统一处理常见 API 错误。"""
    load_dotenv()
    question = input("请输入问题：")

    try:
        answer = ask_model(question)
    except ValueError as exc:
        print(f"配置或输入错误：{exc}", file=sys.stderr)
    except AuthenticationError:
        print("鉴权失败：请检查 OPENAI_API_KEY。", file=sys.stderr)
    except APITimeoutError:
        print("请求超时：请检查网络后再试。", file=sys.stderr)
    except APIConnectionError:
        print("连接失败：请检查网络或代理设置。", file=sys.stderr)
    except RateLimitError:
        print("达到速率或额度限制：请稍后重试并检查账户额度。", file=sys.stderr)
    except APIStatusError as exc:
        print(f"API 返回错误状态 {exc.status_code}。", file=sys.stderr)
    except RuntimeError as exc:
        print(f"响应错误：{exc}", file=sys.stderr)
    else:
        print(f"模型回答：{answer}")


if __name__ == "__main__":
    main()

