# Day 1 半程总结：M1–M5（消息 → 上下文 → 菜单 → 执行 → 循环）

> 里程碑：Agent 核心链路已经**闭环**。剩下的 M6（Harness）是"工程化包装"，
> M7–M13 是"业务与评测"。**换句话说：技术骨架已经搭完了。**

---

## 1. 五关一览

| 模块 | 解决什么 | 文件 | LOC | 测试 |
|---|---|---|---|---|
| **M1** | 消息长什么样、怎么进出 | `common/types.py` `common/llm.py` | 357 | 21 |
| **M2** | 这一轮到底该发什么 | `common/context.py` | 147 | 15 |
| **M3** | 模型看到的菜单从哪来 | `common/schema.py` `common/tools.py` | 352 | 19 |
| **M4** | 模型点菜后谁来做、做砸了怎么说 | `common/dispatch.py` | 199 | 14 |
| **M5** | 什么时候继续、什么时候停 | `common/loop.py` `harness/recorder.py` | 223 | 16 |
| 合计 | | 11 个源文件 | **1365** | **85** |

验收命令（每次都是这两条）：

```bash
uv run pytest bootcamp/tests -q     # 85 passed
uv run ruff check bootcamp          # All checks passed!
```

---

## 2. 全景图：一次真实请求的完整路径

```
用户输入 "A1 还有货吗？帮我留 99 件"
   │
   ├─① system() / user()                         M1  消息构造
   │
   ├─② check_invariants(history)                 M2  进场自检（抓孤儿 tool）
   ├─③ estimate_tokens / trim_history            M2  预算与裁剪（按轮丢）
   │
   ├─④ registry.to_openai_tools()                M3  生成菜单
   │
   ├─⑤ model.invoke(history, tools=菜单)         M1  模型边界（Mock/真实）
   │     └─ 返回 assistant(tool_calls=[...])      ← 模型只"说"，不"做"
   │
   ├─⑥ dispatch_many(calls)                      M4  执行
   │     ├─ registry.get(name)                   M3  白名单
   │     ├─ validate_arguments(spec, args)       M4  参数校验（按 schema）
   │     ├─ spec.func(**args)                    ← 唯一真正"做事"的一行
   │     └─ 失败 → {"ok":false,"error":...,"retryable":...}
   │
   ├─⑦ result.to_message() → history.append      M1/M4  id 原样回填
   │
   └─⑧ 回到 ⑤（while 循环）                      M5  四个出口
        final / max_steps / tool_error_budget / budget_exceeded
             │
             └─⑨ append_run(trajectory.jsonl)    M5  落盘（可回放/可训练）
```

**排查任何问题时，先定位在这条链的哪一环。**（见 §4 故障对照表）

---

## 3. 五个核心认知（每关一句）

| 模块 | 一句话 | 面试最爱问的角度 |
|---|---|---|
| M1 | **四种 role 是"署名"不是"发送者"；模型是失忆的，历史由宿主保管** | tool_call_id 的作用、None vs ""、wire 格式 |
| M2 | **历史既是资产也是负债**：不变量保证合法，按轮裁剪保证不超窗 | 为什么整轮丢、为什么 system 不裁 |
| M3 | **模型看不到你的代码，只看到 schema；描述就是提示词** | required 语义、enum vs 描述、为什么用 schema 校验 |
| M4 | **错误必须是数据，不是异常**，否则 Agent 无法自我修复 | retryable、超时真相、参数校验位置 |
| M5 | **终止是宿主的责任，不是模型的承诺** | 四个出口、连续失败计数、预算前置 |

---

## 4. 故障对照表（现象 → 哪一环 → 怎么改）

| 现象 | 环节 | 根因 | 修法 |
|---|---|---|---|
| `400 tool_calls must be followed by tool messages` | M1/M2 | tool 消息没回、id 不匹配、裁剪切了一半 | `check_invariants` 前置 + 按轮裁剪 |
| 模型重复调同一个工具 | M3/M4 | 描述含糊 或 错误文案读不懂 | 改 description；错误信息补"哪个字段/期望什么" |
| 模型把字符串当数字 | M3/M4 | 缺类型标注 / 宿主没校验 | 类型标注 + `validate_arguments` |
| 模型自己编造库存数字 | M2/M4 | 工具结果没进历史 / 结果被丢弃 | 检查历史 role 序列；确认 tool 消息已 append |
| 循环停不下来 | M5 | 只有 final 一个出口 | 加 max_steps / 连续失败 / 预算 |
| 对话越来越慢、越来越贵 | M2 | 历史 O(n²) 增长且没裁剪 | `trim_history` + 工具输出裁剪 |
| 工具卡死，Agent 无响应 | M4 | 没有超时 | 线程池 + timeout（注意：超时不等于没执行） |
| 扣库存扣了两次 | M4/M8 | 超时后重试 + 非幂等 | 幂等键（M8） |
| 脚本崩但测试全绿 | 工程 | 测试没覆盖真实路径 | 冒烟脚本 + 回归测试 |
| `不支持的类型标注: 'str'` | M3 | `from __future__ import annotations` 把注解变字符串 | `get_type_hints` 解析 |

---

## 5. 你踩过的 8 个坑（都是真实发生的）

| # | 坑 | 教训 |
|---|---|---|
| 1 | `import json` 写在 `from __future__` 上面 | import 顺序有硬规则，`from __future__` 必须最先 |
| 2 | 改了代码但没保存；盯着旧截图排查 | 先 `py_compile` / `grep` 确认落盘，再看测试 |
| 3 | `if self.tool_calls is not None:` 恒为真 | 默认值是 `()` 不是 `None`；空容器用真值判断 |
| 4 | 分支末尾漏 `return`，掉进下一个分支 | early return 的代价：每条分支自己负责结束 |
| 5 | `get(k, [])` 在"值是 null"时不生效 | `get` 的默认值只防"键不存在"，`or` 才防"值是空" |
| 6 | `qty=True` 被当成 1 件库存 | Python 的 `bool` 是 `int` 子类；JSON 里它们不同型 |
| 7 | 84 个测试全绿，脚本一跑就崩 | future annotations → 注解变字符串；需要冒烟脚本 |
| 8 | 轨迹文件被提交进 git | 运行时产物要 `.gitignore` |

> 这 8 条比任何"最佳实践清单"都有用，因为每一条你都亲眼见过它炸。

---

## 6. 现在你已经能做什么

- [x] 从零定义消息类型、写 `to_wire/from_wire/validate`，并解释三层表示
- [x] 讲清 `tool_call_id` 在**协议层**和**语义层**的作用
- [x] 手写"按轮裁剪"的上下文管理，并说明为什么不能切半个工具回合
- [x] 从函数签名 + docstring 生成 JSON Schema（含 enum、required、additionalProperties）
- [x] 写一个"六类失败都不崩"的工具调度器，并给出正确的 `retryable` 语义
- [x] 手写一个带四个出口的 Agent Loop，产出可回放的 trajectory
- [x] 用 Mock 驱动整套链路做离线测试（85 个，不需要 API Key）
- [x] 看轨迹定位问题：role 序列、id 配对、token 增长、stop_reason

## 7. 你**还**不能做什么（诚实清单）

| 缺什么 | 归属 |
|---|---|
| 真实模型接入（`OpenAICompatChatModel`） | M6 之后补，或现在补 |
| 日志/trace/成本统计/重试策略 | **M6** |
| 电商领域建模、写操作幂等 | M7 / M8 |
| RAG（chunk/embed/top-k/引用） | M9 |
| 多工具组装与描述调优 | M10 |
| HITL 人工确认门禁 | M11 |
| 评测数据集与指标 | M12 |
| 完整 CLI 交付 | M13 |

---

## 8. 进度 vs 计划

| 模块 | 计划 | 实际 | 偏差 |
|---|---|---|---|
| M1 | 70 | 115 | +45（补 Python 基础） |
| M2 | 60 | 150 | +90（`raise`/`set`/切分概念） |
| M3 | 75 | 35 | −40（改为我实现、你理解） |
| M4 | 70 | 90 | +20 |
| M5 | 80 | 75 | −5 |
| **合计** | **355** | **465** | **+110（约 7.75h）** |

**结论**：Day 1 的 9 小时里已用约 7.75h，还剩 M6（55min）+ 代码审查（40min）。
时间紧但可控。**真正的经验：Python 基础概念的补课成本必须提前预留。**

---

## 9. 三句话记住整条链路

```
M1 决定"消息长什么样"
M2 决定"这一轮发什么"
M3 决定"模型看到什么菜单"
M4 决定"点菜后怎么做、做砸了怎么说"
M5 决定"什么时候停"
```

**缺任何一环，Agent 都会以某种方式失效**：

- 缺 M1 → 协议层 400
- 缺 M2 → 上下文超窗或被静默截断
- 缺 M3 → 模型选错工具
- 缺 M4 → 一崩到底，无法自我修复
- 缺 M5 → 死循环烧钱

---

## 10. 下一步选项

| 选项 | 内容 | 适合 |
|---|---|---|
| **A** | M6 最小 Harness（日志/trace/成本/重试） | 按计划推进，Day 1 收官 |
| **B** | 补真实模型接入，跑一次真链路 | 想验证"真的能调通" |
| **C** | Day 1 代码审查 + 面试巡礼（把 5 关的题串一遍） | 面试临近 |

推荐顺序：**A → B → C**。A 完成后 Day 1 的工程骨架就完整了。
