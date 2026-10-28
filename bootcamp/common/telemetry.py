"""M6：可观测性 —— trace 事件、成本估算、安全日志。

三个原则：

1. **事件要有连续序号**：没有序号，你就无法判断"是漏了一条还是顺序乱了"；
2. **成对事件**：model_start/model_end、tool_start/tool_end 必须成对，
   缺一个就说明程序在中间崩了（这是排查崩溃点最有效的手段）；
3. **日志不记正文**：只记条数、耗时、token、工具名。
   消息正文里可能有用户隐私、API Key、内部提示词——**日志是最容易泄露的地方**。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from bootcamp.common.llm import Usage

_LOGGER = logging.getLogger("bootcamp.agent")


@dataclass(frozen=True)
class TraceEvent:
    """一条 trace 事件。seq 全局递增，便于排序与查漏。"""

    seq: int
    kind: str
    ts: float
    data: dict[str, Any]


@dataclass
class Tracer:
    """收集 trace 事件。刻意做成"内存里攒着"，方便测试断言。"""

    events: list[TraceEvent] = field(default_factory=list)
    _seq: int = 0

    def emit(self, kind: str, **data: Any) -> TraceEvent:
        self._seq += 1
        event = TraceEvent(seq=self._seq, kind=kind, ts=time.perf_counter(), data=data)
        self.events.append(event)
        return event

    def kinds(self) -> list[str]:
        return [event.kind for event in self.events]

    def to_dicts(self) -> list[dict[str, Any]]:
        return [
            {"seq": e.seq, "kind": e.kind, "ts": round(e.ts, 6), "data": e.data}
            for e in self.events
        ]


def estimate_cost(
    usage: Usage,
    *,
    price_prompt_per_1k: float,
    price_completion_per_1k: float,
) -> float:
    """按 token 用量估算成本（美元）。"""
    return (
        usage.prompt_tokens / 1000 * price_prompt_per_1k
        + usage.completion_tokens / 1000 * price_completion_per_1k
    )


def configure_logging(level: str = "INFO") -> None:
    """配置一个简洁的控制台 logger（重复调用不会叠加 handler）。"""
    _LOGGER.setLevel(level.upper())
    if not _LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        )
        _LOGGER.addHandler(handler)
    _LOGGER.propagate = False


def log_run_summary(
    *,
    run_name: str,
    stop_reason: str,
    steps: int,
    tool_calls: int,
    usage: Usage,
    cost_usd: float,
    elapsed_ms: float,
    retries: int,
) -> None:
    """只记录元信息，**绝不记录消息正文**。

    这条规则不是洁癖：日志会进 ELK/云服务、会被同事随手粘贴、会被长期保留。
    一条误记的 API Key 或用户隐私，代价远大于"日志不够详细"。
    """
    _LOGGER.info(
        "run=%s stop=%s steps=%d tools=%d prompt=%d completion=%d cost=$%.6f "
        "elapsed=%.1fms retries=%d",
        run_name,
        stop_reason,
        steps,
        tool_calls,
        usage.prompt_tokens,
        usage.completion_tokens,
        cost_usd,
        elapsed_ms,
        retries,
    )
