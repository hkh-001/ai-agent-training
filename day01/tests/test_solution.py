"""Day 1 小型实践中模式选择逻辑的测试。"""

from day01.exercises.solution import build_answer_function, use_real_model


def test_real_model_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("USE_REAL_MODEL", raising=False)

    assert use_real_model() is False


def test_real_model_flag_is_case_insensitive(monkeypatch) -> None:
    monkeypatch.setenv("USE_REAL_MODEL", "TrUe")

    assert use_real_model() is True


def test_mock_answer_function_needs_no_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    answer_function = build_answer_function(real_model=False)

    answer = answer_function("测试")

    assert answer == "Mock 收到：测试"
