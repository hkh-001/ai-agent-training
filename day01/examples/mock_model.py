"""不需要 API Key 的最小 Mock Model。

Mock 的目标不是模拟大模型的智能，而是提供稳定、可重复的接口，帮助我们先
理解“输入 -> 模型接口 -> 输出”的数据流。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MockResponse:
    """模拟模型的一次响应。"""

    output_text: str


class MockModel:
    """根据少量规则返回确定性结果的假模型。"""

    def generate(self, user_input: str) -> MockResponse:
        """根据用户输入生成固定规则的回复。

        Args:
            user_input: 用户输入的文本。

        Returns:
            带有 ``output_text`` 字段的模拟响应。

        Raises:
            ValueError: 输入为空或只包含空白字符。
        """
        normalized_input = user_input.strip()
        if not normalized_input:
            raise ValueError("用户输入不能为空")

        if "你好" in normalized_input:
            answer = "你好！我是 Day 1 的 Mock Model。"
        elif "agent" in normalized_input.lower():
            answer = "Agent 不只是调用一次模型；它通常还需要决策、行动和反馈循环。"
        else:
            answer = f"Mock 收到：{normalized_input}"

        return MockResponse(output_text=answer)


def main() -> None:
    """运行一次交互式 Mock 调用。"""
    model = MockModel()
    question = input("请输入问题：")

    try:
        response = model.generate(question)
    except ValueError as exc:
        print(f"输入错误：{exc}")
        return

    print(f"模型回答：{response.output_text}")


if __name__ == "__main__":
    main()

