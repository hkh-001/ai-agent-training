# bootcamp：两天 Agent 核心强化训练

原生 Python 手写 Agent 核心链路（**不使用 LangChain / LangGraph**），两天后交付
一个可运行的电商库存 Agent。

## 先读这三个文件

1. [PLAN_2DAYS.md](PLAN_2DAYS.md) — 可行性结论、15 天→2 天映射、时间表、模块卡片、
   SFT/RL 对应关系
2. [ACCEPTANCE.md](ACCEPTANCE.md) — 每关的验收命令与反模式清单
3. [PROGRESS.md](PROGRESS.md) — 进度台账（每过一关更新）

## 教案（每完成一个模块追加一份）

- [M1-教案-消息与工具调用.md](M1-教案-消息与工具调用.md) — 概念讲解 + 12 个疑问 + 12 个易错点 + 面试答题卡 + 自测清单

## 运行方式

```bash
cd /Users/hkh/Desktop/project/agent学习

uv run pytest bootcamp/tests -q          # 跑测试
uv run pytest bootcamp/tests -k m1 -v    # 只跑 M1
uv run ruff check bootcamp               # 静态检查
uv run python -m bootcamp.day1.m5_demo_loop   # 某个演示脚本（M5 之后才有）
```

当前环境：Python 3.12.13 / pytest 9.1.1 / ruff 0.16.1（`uv.lock` 锁定）。

## 目录与模块对应

| 路径 | 模块 | 内容 |
|---|---|---|
| `common/types.py` | M1 | 消息、ToolCall、wire 序列化 |
| `common/llm.py` | M1 | ChatModel 协议、Mock、Usage |
| `common/context.py` | M2 | 上下文构造、裁剪、不变量 |
| `common/schema.py` `common/tools.py` | M3 | 签名 → JSON Schema、工具注册表 |
| `common/dispatch.py` | M4 | 工具调度与错误语义 |
| `common/loop.py` | M5 | Agent Loop、终止条件 |
| `common/config.py` `common/telemetry.py` `harness/` | M6 | 配置、日志、最小 Harness |
| `day2/m7_domain.py` | M7 | 电商领域模型 |
| `day2/m8_business_tools.py` | M8 | 业务工具 |
| `day2/m9_rag.py` | M9 | RAG |
| `day2/m10_agent.py` | M10 | 组装 |
| `day2/m11_hitl.py` | M11 | 人工确认门禁 |
| `evals/` | M12 | 评测集与指标 |
| `app/inventory_agent.py` | M13 | 最终 CLI |

## 当前状态

- M1 骨架已生成，测试处于**红色**（`20 failed, 1 passed`）——这是设计好的起点。
- 你的任务：读 `common/types.py` 里的契约注释 + `tests/test_m1_*.py`，把 4 个
  `NotImplementedError` 全部实现掉。

## 约定

- 卡住时按需索取：「给我提示」→「给我代码片段」→「我要参考实现」。
- 不用真实 API 跑测试；真实调用只做冒烟，且失败不算阻塞。
- 每个模块结束有一次代码审查 + 3–5 道口头面试题。
