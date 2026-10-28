"""M6：运行配置 —— 一次 Agent 运行的全部可调参数。

为什么要单独一个 config：
1. **可复现**：一次运行的成败必须能被"一组参数"完整描述；
2. **可对比**：评测时只改一个字段、其他不变，才知道是谁的影响；
3. **可审计**：线上事故复盘时，"当时用的是哪套参数"是第一个要回答的问题。

注意：循环相关的参数复用 M5 的 LoopConfig（不重复定义），
RunConfig 只增加"工程层"的字段（重试、价格、日志级别）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bootcamp.common.loop import LoopConfig


@dataclass(frozen=True)
class RunConfig:
    """一次运行的完整配置。frozen 保证运行期间不会被偷偷改掉。"""

    run_name: str = "default"

    # 循环/预算参数（来自 M5，避免两处定义导致漂移）
    loop: LoopConfig = field(default_factory=LoopConfig)

    # 模型调用最多重试几次（只针对"异常"，不针对"模型回答得不好"）
    model_retries: int = 1

    # 成本单价（美元 / 千 token）。默认值仅用于演示，真实项目按服务商价格填。
    price_prompt_per_1k: float = 0.0005
    price_completion_per_1k: float = 0.0015

    # 日志级别：DEBUG/INFO/WARNING
    log_level: str = "INFO"
