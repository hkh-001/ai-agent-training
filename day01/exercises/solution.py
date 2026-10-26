"""Day 1 小型实践参考实现：可切换的问答器。"""

import os
import sys
from collections.abc import Callable

from dotenv import load_dotenv
from openai import OpenAIError

from day01.examples.deepseek_llm_call import ask_deepseek
from day01.examples.mock_model import MockModel


def use_real_model() -> bool:
    """判断是否显式开启真实模型。"""
    return os.getenv("USE_REAL_MODEL", "false").strip().lower() == "true"


def build_answer_function(real_model: bool) -> Callable[[str], str]:
    """根据模式返回统一的问答函数。"""
    if real_model:
        return ask_deepseek

    mock_model = MockModel()
    return lambda question: mock_model.generate(question).output_text


def main() -> None:
    """运行命令行问答器。"""
    load_dotenv()
    real_model = use_real_model()
    mode_name = "真实模型" if real_model else "Mock Model"
    print(f"当前模式：{mode_name}")

    question = input("请输入问题：").strip()
    if not question:
        print("输入错误：问题不能为空。", file=sys.stderr)
        return

    answer_function = build_answer_function(real_model)
    try:
        answer = answer_function(question)
    except (OpenAIError, RuntimeError, ValueError) as exc:
        print(f"调用失败：{exc}", file=sys.stderr)
        return

    print(f"回答：{answer}")


if __name__ == "__main__":
    main()
