"""System Prompt 构造测试。"""

import pytest

from day02.examples.prompt_builder import build_system_prompt


def test_prompt_contains_all_sections() -> None:
    prompt = build_system_prompt("导师", ["规则甲", "规则乙"], "简洁")

    assert "# 角色\n导师" in prompt
    assert "1. 规则甲" in prompt
    assert "2. 规则乙" in prompt
    assert "# 输出风格\n简洁" in prompt


def test_prompt_requires_at_least_one_rule() -> None:
    with pytest.raises(ValueError, match="至少需要一条规则"):
        build_system_prompt("导师", ["", "  "], "简洁")

