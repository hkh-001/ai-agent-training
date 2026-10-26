"""Day 1 Mock Model 单元测试。"""

import pytest

from day01.examples.mock_model import MockModel


def test_mock_model_returns_greeting() -> None:
    model = MockModel()

    response = model.generate("你好")

    assert "Day 1" in response.output_text


def test_mock_model_explains_agent_boundary() -> None:
    model = MockModel()

    response = model.generate("什么是 Agent？")

    assert "不只是调用一次模型" in response.output_text


@pytest.mark.parametrize("invalid_input", ["", "   ", "\n"])
def test_mock_model_rejects_empty_input(invalid_input: str) -> None:
    model = MockModel()

    with pytest.raises(ValueError, match="不能为空"):
        model.generate(invalid_input)

