"""M5 验收测试（续）：轨迹记录器的往返正确性。

运行：uv run pytest bootcamp/tests/test_m5_recorder.py -q

为什么轨迹要测"往返"：
记录下来的轨迹，未来要能被读回来做训练/评测。如果 to_wire / from_wire 不对称，
落盘的轨迹读回来就变形了——而且这种 bug 只有在训练时才会暴露（数据已经脏了）。
"""

import json

from bootcamp.common.llm import MockChatModel
from bootcamp.common.loop import run_agent
from bootcamp.common.types import ToolCall, assistant, system, user
from bootcamp.harness.recorder import (
    append_run,
    format_trace,
    load_runs,
    messages_from_record,
    run_to_record,
)
from bootcamp.tests.test_m5_loop import make_registry


def _run():
    model = MockChatModel(
        [
            assistant(tool_calls=[ToolCall("c1", "get_stock", {"sku": "A1"})]),
            assistant("A1 还有 5 件。"),
        ]
    )
    return run_agent(model, make_registry(), [system("客服"), user("A1 有货吗")])


def test_record_is_json_serializable() -> None:
    record = run_to_record(_run())

    text = json.dumps(record, ensure_ascii=False)  # 不抛异常即可
    assert '"stop_reason": "final"' in text


def test_record_contains_trace_fields() -> None:
    record = run_to_record(_run())

    assert record["stop_reason"] == "final"
    assert record["steps"] == 2
    assert record["final_answer"] == "A1 还有 5 件。"
    assert len(record["tool_results"]) == 1
    assert record["tool_results"][0]["ok"] is True


def test_messages_roundtrip() -> None:
    run = _run()
    restored = messages_from_record(run_to_record(run))

    assert restored == run.messages  # 往返必须无损


def test_append_and_load_jsonl(tmp_path) -> None:
    path = tmp_path / "traces" / "run.jsonl"
    append_run(_run(), path)
    append_run(_run(), path)

    runs = load_runs(path)

    assert len(runs) == 2  # JSONL：一行一条，追加写
    assert runs[0]["stop_reason"] == "final"


def test_load_missing_file_returns_empty(tmp_path) -> None:
    assert load_runs(tmp_path / "nope.jsonl") == []


def test_format_trace_is_readable() -> None:
    text = format_trace(_run())

    assert "[step 1] assistant" in text
    assert "get_stock" in text
    assert "tool<-c1" in text
    assert "stop=final" in text
