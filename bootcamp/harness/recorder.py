"""最小轨迹记录器：把一次 AgentRun 落盘成 JSONL，并能读回来回放。

为什么必须落盘：
1. **排障**：线上出问题时，你要能回答"模型当时看到了什么"；
2. **复现**：同一个输入 + 同一份轨迹，应该能重跑出同样的结论；
3. **数据**：这些 JSONL 就是未来 SFT / RL 的原始素材。

为什么用 JSONL 而不是 JSON 数组：
- 一行一条，追加写（append）不需要重写整个文件；
- 中途崩溃时，已写入的行仍然可读；
- 便于用 grep / jq / pandas 直接处理。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bootcamp.common.loop import AgentRun
from bootcamp.common.types import ChatMessage


def run_to_record(run: AgentRun) -> dict[str, Any]:
    """把一次运行转成可序列化的 dict（消息用 wire 格式，保证可逆）。"""
    return {
        "stop_reason": run.stop_reason.value,
        "steps": run.steps,
        "context_tokens": run.context_tokens,
        "final_answer": run.final_answer,
        # 用 to_wire()：这是唯一可逆的序列化方式（M1）
        "messages": [m.to_wire() for m in run.messages],
        "tool_results": [
            {
                "tool_call_id": r.tool_call_id,
                "ok": r.ok,
                "retryable": r.retryable,
                "latency_ms": round(r.latency_ms, 2),
                "content": r.content,
            }
            for r in run.tool_results
        ],
    }


def messages_from_record(record: dict[str, Any]) -> tuple[ChatMessage, ...]:
    """从记录里还原消息（回放 / 二次分析用）。"""
    return tuple(ChatMessage.from_wire(item) for item in record["messages"])


def append_run(run: AgentRun, path: str | Path) -> Path:
    """追加一条轨迹到 JSONL 文件（文件不存在则创建）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(run_to_record(run), ensure_ascii=False) + "\n")
    return target


def load_runs(path: str | Path) -> list[dict[str, Any]]:
    """读取文件里的全部轨迹。"""
    target = Path(path)
    if not target.exists():
        return []
    with target.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def format_trace(run: AgentRun) -> str:
    """把轨迹渲染成人类可读的文本（单元测试失败时也用它打日志）。"""
    lines: list[str] = []
    step = 0
    for msg in run.messages:
        if msg.role == "assistant":
            step += 1
            calls = ", ".join(f"{c.name}({c.arguments})" for c in msg.tool_calls)
            head = f"[step {step}] assistant"
            lines.append(f"{head}  tool_calls: {calls}" if calls else f"{head}  {msg.content}")
        elif msg.role == "tool":
            lines.append(f"           tool<-{msg.tool_call_id}  {msg.content}")
        else:
            lines.append(f"[{'system' if msg.role == 'system' else 'user'}]  {msg.content}")

    lines.append(
        f"--- stop={run.stop_reason.value} steps={run.steps} "
        f"tools={len(run.tool_results)} tokens={run.context_tokens}"
    )
    if run.final_answer:
        lines.append(f"--- answer: {run.final_answer}")
    return "\n".join(lines)
