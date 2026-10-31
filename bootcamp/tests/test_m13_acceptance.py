"""M13 验收测试：8 个场景必须全部通过，且可重复。

运行：uv run pytest bootcamp/tests/test_m13_acceptance.py -q
"""

from bootcamp.app.inventory_agent import SCENARIOS, run_all


def test_all_eight_scenarios_pass() -> None:
    results = run_all()

    failures = {r.sid: r.reasons for r in results if not r.passed}
    assert not failures, failures
    assert len(results) == 8


def test_scenarios_are_deterministic() -> None:
    first = [(r.sid, r.passed) for r in run_all()]
    second = [(r.sid, r.passed) for r in run_all()]

    assert first == second


def test_scenario_ids_are_unique() -> None:
    ids = [scenario().sid for scenario in SCENARIOS]

    assert len(ids) == len(set(ids))
