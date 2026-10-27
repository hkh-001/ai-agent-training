"""M2：上下文（Context）构造、不变量检查与历史裁剪。

M1 解决"消息长什么样"，M2 解决"这一轮到底发什么"。

三个必须记住的事实：
1. 模型没有记忆，每一轮都要把**整份历史**重发一次；
2. 历史必须满足协议不变量：`assistant(tool_calls)` 与 `tool` 消息严格配对，
   否则服务端直接 400；
3. 历史会随轮次线性增长，所以必须能"按轮"裁剪——而且**不能把工具回合切一半**。

一轮（round）的定义：
    从一条 user 消息开始，到**下一条 user 消息之前**的全部消息（含工具回合）。
    system 消息不属于任何一轮，永不裁剪。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from bootcamp.common.types import ChatMessage


class ContextError(ValueError):
    """历史不合法（孤儿 tool 消息、缺回应、顺序错乱等）。"""


@dataclass(frozen=True)
class TrimResult:
    """裁剪结果。带上统计信息，便于日志与评测使用。"""

    messages: tuple[ChatMessage, ...]
    dropped_rounds: int
    estimated_tokens: int


def estimate_tokens(messages: Sequence[ChatMessage]) -> int:
    """粗略估算这段历史的 token 数（不调用任何 tokenizer）。

    契约：
    - 空列表返回 0
    - 每条消息固定开销 4，加上内容的字符数；`content` 为 None 时按 "" 算
    - 结果必须确定性（同样输入 → 同样输出），且随内容单调递增

    说明：这是刻意做的"廉价近似"。真实项目会用 tiktoken 或服务端返回的
    usage.prompt_tokens 校准；但在**预算决策**（要不要裁剪）上，近似值足够，
    且没有依赖、跑得快。
    """
    raise NotImplementedError("TODO(M2): 实现 estimate_tokens")


def check_invariants(messages: Sequence[ChatMessage]) -> None:
    """检查历史是否满足协议不变量，不合法抛 ContextError。

    规则（也是服务端的规则）：
    - 每条 `tool` 消息都必须能对应到**前面某条** assistant 消息里的 tool_call id
    - 有 tool_calls 的 assistant 消息，其后必须紧跟**恰好**那些 id 的 tool 消息
    - 不允许在"调用"和"结果"之间插入 user/system 消息
    - 全部消息处理完后，不允许有"未被回应"的 tool_call

    为什么必须查：这不是"洁癖"，是 400 的直接来源。而且一旦发生，
    错误会出现在**下一次请求**，离真正的 bug 现场很远。
    """
    raise NotImplementedError("TODO(M2): 实现 check_invariants")


def trim_history(messages: Sequence[ChatMessage], max_tokens: int) -> TrimResult:
    """按「轮」裁剪历史，使估算 token 不超过 max_tokens。

    契约：
    - 开头的 system 消息永不裁剪
    - 轮 = 从一条 user 消息开始，到下一个 user 消息之前（含工具回合），
      整轮保留或整轮丢弃，**绝不允许切一半**
    - 从**最老**的轮开始丢，直到 estimated_tokens <= max_tokens
    - 最后一轮永远保留（哪怕它自己就超预算）——否则模型连当前问题都看不到
    - 输出必须能通过 check_invariants

    返回 TrimResult(messages=保留的消息, dropped_rounds=丢了几轮,
                   estimated_tokens=裁剪后估算值)。
    """
    raise NotImplementedError("TODO(M2): 实现 trim_history")
