"""M12 验收测试：评测本身是否可信。

运行：uv run pytest bootcamp/tests/test_m12_eval.py -q

关键：**评测器必须能识别"错误答案"**。
如果只有全 PASS 的用例，你永远不知道评测是不是形同虚设。
"""

import json

from bootcamp.evals.run_eval import evaluate, format_report, load_cases, run_case


def test_cases_file_is_well_formed() -> None:
    cases = load_cases()

    assert len(cases) >= 10
    for case in cases:
        assert {"id", "question", "expect_tools", "expect_refs",
                "expect_answer_contains"} <= set(case)


def test_all_cases_pass_with_rule_model() -> None:
    report = evaluate()

    assert report.passed == report.total, format_report(report)


def test_report_has_metrics() -> None:
    summary = evaluate().as_dict()

    assert summary["success_rate"] == 1.0
    assert summary["tool_selection_accuracy"] == 1.0
    assert summary["citation_rate"] == 1.0
    assert summary["invalid_arg_rate"] == 0.0
    assert summary["avg_steps"] > 0
    assert summary["avg_tokens"] > 0
    assert summary["avg_cost_usd"] >= 0
    assert summary["stop_reasons"] == {"final": summary["total"]}


def test_evaluator_detects_wrong_tool_expectation() -> None:
    """故意把期望写错：评测必须报 FAIL，而不是放水。"""
    bad_case = {
        "id": "deliberately-wrong",
        "question": "A100 还有货吗？",
        "expect_tools": ["find_policy"],          # 实际会调 get_stock
        "expect_refs": [],
        "expect_answer_contains": ["22"],
    }

    result = run_case(bad_case)

    assert result.passed is False
    assert any("工具不符" in reason for reason in result.reasons)


def test_evaluator_detects_missing_answer_keyword() -> None:
    bad_case = {
        "id": "wrong-keyword",
        "question": "A100 还有货吗？",
        "expect_tools": ["get_stock"],
        "expect_refs": [],
        "expect_answer_contains": ["这句话不会出现"],
    }

    result = run_case(bad_case)

    assert result.passed is False
    assert any("答案缺少" in reason for reason in result.reasons)


def test_evaluator_detects_missing_citation() -> None:
    bad_case = {
        "id": "wrong-ref",
        "question": "质量问题退货运费谁出？",
        "expect_tools": ["find_policy"],
        "expect_refs": ["does_not_exist.md#9"],
        "expect_answer_contains": ["商家"],
    }

    result = run_case(bad_case)

    assert result.passed is False
    assert any("缺少引用" in reason for reason in result.reasons)


def test_format_report_is_readable() -> None:
    text = format_report(evaluate())

    assert "PASS" in text
    assert "成功率" in text
    assert "工具选择准确率" in text


def test_case_result_is_json_serializable() -> None:
    result = run_case(load_cases()[0])

    json.dumps(
        {
            "case_id": result.case_id,
            "passed": result.passed,
            "tools": list(result.tools),
            "steps": result.steps,
        }
    )
