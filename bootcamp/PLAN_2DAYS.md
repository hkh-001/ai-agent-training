# 两天 Agent 核心强化训练计划（原生 Python · 9h × 2）

> 目标不是「学完 15 天」，而是「两天后能独立写出可运行、可测试、可评测的
> Agent 核心链路，并能在面试里讲清数据流」。
>
> 本训练**全程不使用 LangChain / LangGraph**，只用 Python 标准库 + `openai`
> SDK（走 OpenAI 兼容协议，可直连 DeepSeek）+ `pytest`。

---

## 1. 可行性结论

**可行，但必须裁剪。** 这不是"把 15 天塞进 2 天"，而是"换一条更窄但更深的路线"。

### 1.1 为什么可行

| 事实 | 说明 |
|---|---|
| 15 天里有 6 天是框架教学（Day 9–14） | 你的要求是第一阶段不碰框架 → 这 6 天直接不占预算 |
| 真正的"核心链路"只有 4 个概念 | Messages、Tool Schema、Tool Call ID 闭环、Agent Loop |
| 15 天是「每天一个小实验」的碎片化设计 | 换成"两天一个连续项目"，反而更省上下文切换成本 |
| 你已有 Python 基础和概念认知 | 不需要从变量、函数、HTTP 讲起 |

### 1.2 两天后你「能」做到

- 不查文档，从零写出一条 `user → model → tool_call → execute → tool_result →
  model → final` 的完整闭环；
- 用 Pydantic/JSON Schema 描述工具，并解释模型看到的 schema 与宿主执行的函数
  是**两套东西**；
- 手写 `tool_call_id` 配对、上下文裁剪、最大步数、重试、超步预算终止；
- 实现一个不依赖向量库的 RAG（chunk → 确定性 embedding → top-k → 引用），并
  把它作为**一个工具**接进 Agent；
- 写出带结构化日志、Trace 事件、成本统计、HITL 门禁、离线评测的最小 Harness；
- 能对着代码讲：轨迹（trajectory）怎么变成 SFT 样本，reward 从哪来。

### 1.3 两天后你「不能」做到（诚实的边界）

- 独立阅读/修改基于 LangGraph 的生产项目（Checkpoint、Thread、interrupt/resume
  的框架语义没练）；
- 多智能体协作、异步并发、部署与可观测性平台接入；
- 长期记忆的向量库工程、复杂 Planning/Reflection 策略调优。

这三块我放在"第三阶段路线"里，Day 2 末尾给你 1 页衔接清单（不占用训练时间）。

### 1.4 两个前提条件（不满足就砍模块，不要硬撑）

1. **Mock 优先**：所有测试离线可跑。真实 API 只做冒烟（M2），网络抖动不阻塞进度。
2. **每模块 Gate 制**：测试全绿 + 能口述数据流，才进下一模块。卡住超过 45 分钟，
   我按「提示 → 局部代码 → 完整实现」三级降级。

---

## 2. 15 天 → 2 天 映射表

| 原 15 天模块 | 2 天归属 | 处理方式 |
|---|---|---|
| Day 1 LLM 与一次调用 | **M1** | 压缩：只保留 messages/usage 数据结构 + Mock |
| Day 2 消息与 Prompt | **M1 + M2** | 合并进「上下文构造与裁剪」 |
| Day 3 结构化输出/Pydantic | **M3** | 升级为"函数签名 → JSON Schema" |
| Day 4 Tool Calling | **M3 + M4** | 拆成「声明」与「执行」两半 |
| Day 5 Agent Loop | **M5** | 原样保留，加轨迹记录 |
| Day 6 Memory/State | **M5 + M6** | 只保留「会话状态 + 步数状态」；长期记忆**剪掉** |
| Day 7 RAG | **M9** | 保留，向量改用确定性哈希 embedding（无需下载模型） |
| Day 8 Planning/Reflection/可靠性 | **M6 + M11** | 保留重试/预算/HITL；**剪掉** Reflection 自我反思循环 |
| Day 9–11 LangChain | — | **剪掉**（用户明确要求） |
| Day 12–14 LangGraph/Checkpoint | — | **剪掉**，换成 M5 手写状态机 + M11 手写中断确认 |
| Day 15 工程化/Tracing/Eval | **M6 + M11 + M12** | 保留并强化（日志、trace、评测集、指标） |
| （原计划外） | **M7 + M8 + M10 + M13** | 新增：电商领域建模、业务工具、组装、综合验收 |

净效果：**11 个代码模块 + 2 次代码审查**，覆盖 15 天里 70% 的"能动手"部分和
100% 的"Agent 核心链路"部分。

---

## 3. Day 1 时间表（9h，含 60min 缓冲）

主题：**从一次模型调用，到能自己跑圈子的 Agent Loop**

| 时段 | 模块 | min | 产出 | Gate |
|---|---|---|---|---|
| 09:00–09:20 | 开场：数据流全景 + 环境校验 | 20 | 环境自检输出 | 能画出 messages 流转图 |
| 09:20–10:30 | **M1** 消息与工具调用数据结构 | 70 | `common/types.py` `common/llm.py` | `test_m1_*.py` 全绿 |
| 10:30–11:30 | **M2** 上下文构造、裁剪、不变量 | 60 | `common/context.py` | tool_call_id 配对断言通过 |
| 11:30–12:45 | **M3** Tool Schema：签名 → JSON Schema | 75 | `common/schema.py` `common/tools.py` | 3 个工具 schema 快照通过 |
| 12:45–13:45 | 午休 | 60 | — | — |
| 13:45–14:55 | **M4** Tool Dispatcher 与错误语义 | 70 | `common/dispatch.py` | 6 条错误路径可复现 |
| 14:55–16:15 | **M5** Agent Loop + 轨迹记录 | 80 | `common/loop.py` `harness/recorder.py` | 4 条终止路径通过 |
| 16:15–17:10 | **M6** 最小 Harness：配置/日志/预算/重试 | 55 | `common/config.py` `common/telemetry.py` `harness/runner.py` | 一次运行产出完整 trace |
| 17:10–17:50 | **代码审查 + 5 道面试题** | 40 | `PROGRESS.md` 更新 | 口述通过 |
| 17:50–18:00 | 缓冲 | 10 | — | — |

> 缓冲 60min 已内含在午休外的间隙，实际编码时间约 7h，符合"9h 含休息"。

## 4. Day 2 时间表（9h，含 60min 缓冲）

主题：**RAG + 电商库存 Agent + Harness + 评测**

| 时段 | 模块 | min | 产出 | Gate |
|---|---|---|---|---|
| 09:00–09:45 | **M7** 领域建模：SKU/库存/订单/政策 | 45 | `data/inventory.json` `day2/m7_domain.py` | 领域不变量断言通过 |
| 09:45–11:00 | **M8** 业务工具：只读 vs 写操作、幂等 | 75 | `day2/m8_business_tools.py` | 4 个工具 + 幂等测试通过 |
| 11:00–12:25 | **M9** RAG：chunk → embed → top-k → 引用 | 85 | `day2/m9_rag.py` `data/docs/*.md` | 命中/未命中/排序测试通过 |
| 12:25–13:25 | 午休 | 60 | — | — |
| 13:25–14:35 | **M10** 组装：多工具 + RAG 接入 Loop | 70 | `day2/m10_agent.py` | 3 条组合场景通过 |
| 14:35–15:35 | **M11** Harness 强化：trace/成本/HITL 门禁 | 60 | `day2/m11_hitl.py` `harness/runner.py` | 批准/拒绝/无确认三条路径 |
| 15:35–16:30 | **M12** 评测：数据集 + 指标 + 回归 | 55 | `evals/cases.jsonl` `evals/run_eval.py` | 8 条用例跑出指标表 |
| 16:30–17:30 | **M13** 综合验收 + CLI 8 场景 | 60 | `app/inventory_agent.py` | 见 `ACCEPTANCE.md` |
| 17:30–18:00 | **代码审查 + 5 道面试题 + SFT/RL 路线** | 30 | `PROGRESS.md` 更新 | 口述通过 |

---

## 5. 模块卡片（目标 / 关键数据流 / Gate）

### M1 消息与工具调用数据结构
- **目标**：把"聊天"变成可校验、可序列化的类型；理解 internal dict vs wire JSON。
- **数据流**：`ChatMessage(role, content, tool_calls, tool_call_id)` ⇄ wire dict。
- **Gate**：`test_m1_types.py` + `test_m1_llm.py` 全绿；能解释 `content=None`
  为什么不能写成 `""`。

### M2 上下文构造、裁剪与不变量
- **目标**：控制"发给模型的历史"；保证 assistant.tool_calls 与 tool 消息一一配对。
- **数据流**：`system + 窗口内历史 + 本轮工具结果 → messages`；裁剪只能切在"轮"边界。
- **Gate**：孤儿 tool 消息被拒绝；超长历史按轮裁剪后 token 估算下降。

### M3 Tool Schema：签名 → JSON Schema
- **目标**：理解"模型看到的只是描述"，schema 是约束与提示的词法。
- **数据流**：`def get_stock(sku: str, qty: int = 1)` → `{"name","description","parameters"}`。
- **Gate**：必填/可选/枚举/描述齐全；`additionalProperties: false`。

### M4 Tool Dispatcher 与错误语义
- **目标**：宿主才是执行者；所有失败都要变成**可被模型读懂的** tool 消息。
- **数据流**：`ToolCall(id,name,args)` → 查注册表 → 校验 → 执行 → `tool_result(id, json)`。
- **Gate**：未知工具/参数缺失/类型错/业务异常/超时 五类错误都是结构化输出，不抛穿。

### M5 Agent Loop 与轨迹
- **目标**：写下 while 循环的四个出口；产出可回放的 trajectory。
- **数据流**：`state = {messages, steps, trace}`；每步追加 assistant / tool 消息。
- **Gate**：final / max_steps / tool_error_exhausted / budget_exceeded 四条路径。

### M6 最小 Harness
- **目标**：一次运行 = 配置 + 预算 + 日志 + 结果；可复现、可统计。
- **数据流**：`RunConfig → Runner.run(task) → RunResult(trace, usage, stop_reason)`。
- **Gate**：trace 事件序号连续、包含 tool 前后成对事件、成本可累加。

### M7 电商领域建模
- **目标**：区分"事实数据"与"政策文本"；写操作要有不变量（不超卖）。
- **Gate**：并发预留不超卖、金额/数量类型正确。

### M8 业务工具
- **目标**：只读工具随便调，写工具必须幂等 + 可审计。
- **Gate**：重复 idempotency_key 返回同一结果且库存只扣一次。

### M9 RAG
- **目标**：chunk/embed/top-k/引用/拒答五件事，不引入向量库。
- **数据流**：`docs → chunks → vectors → cosine → top_k → [Citation]`。
- **Gate**：相关问题命中正确文档、无关问题返回"未找到证据"、引用可定位到原文行。

### M10 组装
- **目标**：工具数量变多后，靠描述与 schema 让模型选对工具。
- **Gate**：库存 + 政策 + 下单三类问题走对工具组合。

### M11 Harness 强化（HITL + 成本）
- **目标**：高风险写操作前真正停下，等待人工裁决。
- **Gate**：批准执行、拒绝转安全回复、超时默认拒绝，三条路径都留审计记录。

### M12 评测
- **目标**：把"感觉能用"变成数字。
- **Gate**：8 条用例输出任务成功率、工具选择准确率、平均步数、非法参数率、
  平均 token 成本。

### M13 综合验收
- **Gate**：见 `ACCEPTANCE.md` 的 8 条场景 + 反模式清单。

---

## 6. 每个模块如何变成 SFT / Trajectory / RL 资产

| 模块 | 对 SFT 的意义 | 对 Trajectory 数据构建 | 对 Agentic RL |
|---|---|---|---|
| M1 | `wire dict` 就是 SFT 样本的 `messages` 字段；tool_calls 的 JSON 字符串格式必须与训练数据一致 | 决定样本的序列化 schema（训练/推理必须同构） | action space 的语法层 |
| M2 | 系统提示 + 历史裁剪 = 样本长度控制；tool 配对错位是最常见的脏数据来源 | 同一轨迹可生成"短窗口/长窗口"两个版本 | 上下文策略（context engineering）本身就是可学习策略 |
| M3 | tool schema 是 function-calling 样本的"动作定义"，描述质量直接影响泛化 | 采集成 (schema, 用户请求, 选中工具) 三元组 | action space 的语义层：工具集合 = 动作集合 |
| M4 | 错误回传文本是"负样本修复"的关键；训练模型学会读错误并改参数 | 每次失败都是高价值负例（error → repair 对） | reward 的一部分：工具错误率、非法参数率 |
| M5 | 一条完整 trajectory = 多轮 agentic SFT 样本（含中间 tool 轮） | 循环的每个 step 是训练样本的一个 turn | rollout 本体：RL 的 episode 就是一次 Loop 运行 |
| M6 | 记录 usage/stop_reason，才能做样本过滤（丢弃超预算轨迹） | trace 事件顺序 = 数据血缘 | 环境与执行器（可复现 = RL 可复现） |
| M7 | 领域不变量让样本"可判定对错" | 状态快照可做 counterfactual 数据增强 | reward 的 verifier（是否超卖/是否越权） |
| M8 | 写操作样本稀缺，必须靠幂等/审计保证采集安全 | 幂等键 = 轨迹去重的天然 ID | 稀疏奖励来源（下单成功/失败） |
| M9 | 检索片段 + 引用 = grounded SFT 样本，可训练"引证"行为 | (query, chunk, cited?) 可直接做检索监督 | reward 的 grounding 分项（是否编造） |
| M10 | 多工具选择的困难样本集中在此 | 工具组合序列 = 计划（plan）数据 | 长程 credit assignment 的主要战场 |
| M11 | HITL 轨迹本身是"安全对齐"数据 | 审批事件写入轨迹，标注 human label | 风险敏感 RL（action 成本不对称） |
| M12 | 评测集 = SFT 的 dev set 与 RL 的 test set | 用例 → 轨迹 → 指标 三者对齐 | reward model 与自动化 verifier 的基础 |

一句话总结：**M1–M6 决定你的数据能不能用于训练，M7–M12 决定你的 reward 有没有意义。**

---

## 7. 目录结构（本项目实际结构）

```text
bootcamp/
├── README.md                 # 训练入口、运行方式、导航
├── PLAN_2DAYS.md             # 本文件
├── ACCEPTANCE.md             # 验收标准与 Gate 清单
├── PROGRESS.md               # 功能完成情况台账（我每过一关就更新）
├── common/                   # 可复用领域代码（不是"前一天的业务代码"）
│   ├── types.py              # M1 消息 / ToolCall / wire 序列化
│   ├── llm.py                # M1 ChatModel 协议 + Mock
│   ├── context.py            # M2 上下文构造与不变量
│   ├── schema.py             # M3 签名 → JSON Schema
│   ├── tools.py              # M3 ToolSpec + 注册表
│   ├── dispatch.py           # M4 调度器与错误语义
│   ├── loop.py               # M5 Agent Loop
│   ├── telemetry.py          # M6 结构化日志 / Trace
│   └── config.py             # M6 RunConfig（预算/开关）
├── harness/
│   ├── runner.py             # M6/M11 最小 Harness
│   └── recorder.py           # M5/M11 轨迹落盘 jsonl
├── day1/
│   └── m5_demo_loop.py       # 计算器 + 天气 Mock 的 Loop 演示
├── day2/
│   ├── m7_domain.py          # 库存/订单领域模型
│   ├── m8_business_tools.py  # 业务工具（只读 + 写）
│   ├── m9_rag.py             # chunk/embed/top-k/引用
│   ├── m10_agent.py          # 组装 Agent
│   └── m11_hitl.py           # 人工确认门禁
├── app/
│   └── inventory_agent.py    # 最终交付 CLI（8 场景）
├── data/
│   ├── inventory.json        # 库存事实数据
│   └── docs/                 # RAG 语料（退换货政策、FAQ）
├── evals/
│   ├── cases.jsonl           # 评测集
│   └── run_eval.py           # 指标计算
└── tests/
    ├── test_m1_types.py      # ✅ 已生成（当前为红）
    ├── test_m1_llm.py        # ✅ 已生成（当前为红）
    └── test_m2..m12_*.py     # 按模块逐个生成
```

---

## 8. 训练规则（你和我之间的契约）

1. **任务驱动**：每个模块先讲原理与数据结构 → 给骨架/TODO/测试 → 你实现 →
   我 review → 过 Gate 才下一步。
2. **三级降级**：你卡住时，我先给**提示**（不写代码）→ 再给**局部代码片段** →
   最后才给**完整参考实现**。请明确说"给我提示 / 给我片段 / 我要答案"。
3. **红绿灯**：`uv run pytest` 全绿 + `ruff check` 干净 + 能口述数据流 = 通过。
4. **不跳步**：`PROGRESS.md` 里状态不是 ✅ 的模块，不进入下一模块。
5. **每模块结束**：代码审查（我指出 3 个具体问题）+ 3–5 道口头面试题。
6. **记录**：每个模块的完成情况、耗时、卡点写进 `PROGRESS.md`。

---

## 9. 风险与应对

| 风险 | 概率 | 应对 |
|---|---|---|
| 真实 API 不稳定/限流 | 中 | 全程 Mock 优先，M2 只做一次冒烟；失败不阻塞 |
| M3 的 JSON Schema 手工拼装太耗时 | 高 | 允许先用 `pydantic` 生成，再手写一份对照理解 |
| M5 循环调试卡住 | 中 | 用 Mock 脚本固定每步返回，禁止用真实模型调循环 |
| M9 embedding 想接真模型 | 中 | 先用确定性哈希向量跑通全链路，真模型作为可选 |
| 时间超支 | 高 | 优先保 M1–M6 + M8/M9/M12；M11 的 HITL 可降级为"函数式确认" |
