# 验收标准（Gate 制）

> 规则：**Gate 未通过，不进入下一模块。** 每个 Gate 有可执行的客观检查 + 一条
> 口述检查（你能不能讲清楚）。两者都过才算 ✅。

## 0. 三道通用门（每个模块都要满足）

```bash
uv run pytest bootcamp/tests -q     # 无 failure / error
uv run ruff check bootcamp          # All checks passed
```

- 口述门：能不看代码画出本模块的**数据流方向**（谁调谁、什么变成什么）。

---

## 1. Day 1 Gates

### G1 — M1 消息与工具调用
```bash
uv run pytest bootcamp/tests/test_m1_types.py bootcamp/tests/test_m1_llm.py -q
```
- [ ] 20 个用例全绿
- [ ] 能解释：`arguments` 为什么内部是 dict、wire 上是 str
- [ ] 能解释：`content=None` 与 `content=""` 在 assistant 消息里的差别
- [ ] Mock 的 `calls` 是快照而非引用

### G2 — M2 上下文与不变量
```bash
uv run pytest bootcamp/tests/test_m2_context.py -q
```
- [ ] 孤儿 tool 消息被 `ContextError` 拒绝
- [ ] `assistant.tool_calls` 与紧随其后的 tool 消息 ID 一一对应
- [ ] 裁剪只发生在"轮"边界，不会切出半个工具回合
- [ ] 能估算并打印 token 数（字符数近似即可）

### G3 — M3 Tool Schema
```bash
uv run pytest bootcamp/tests/test_m3_schema.py -q
```
- [ ] 必填参数进 `required`，带默认值的参数不进
- [ ] 类型映射正确：`str→string`、`int→integer`、`float→number`、`bool→boolean`、
      `list[str]→array`、`Literal[...]→enum`
- [ ] 每个参数必须有 description（否则 Gate 不过）
- [ ] `additionalProperties: false`
- [ ] 能解释"schema 不是权限控制"

### G4 — M4 Dispatcher
```bash
uv run pytest bootcamp/tests/test_m4_dispatch.py -q
```
- [ ] 未知工具 → 结构化错误，不抛穿
- [ ] 参数缺失/类型错误 → 结构化错误，含 `retryable` 与 `hint`
- [ ] 业务异常（库存不足）→ 与程序异常区分开
- [ ] 工具返回必须被序列化成 str
- [ ] 超时被捕获
- [ ] 每次调用都有 trace 事件（tool_start / tool_end）

### G5 — M5 Agent Loop
```bash
uv run pytest bootcamp/tests/test_m5_loop.py -q
```
- [ ] 出口 1：模型直接给最终答案 → `stop_reason="final"`
- [ ] 出口 2：达到 `max_steps` → `stop_reason="max_steps"`
- [ ] 出口 3：连续工具错误达上限 → `stop_reason="tool_error_budget"`
- [ ] 出口 4：token 预算耗尽 → `stop_reason="budget_exceeded"`
- [ ] 每步都追加了 assistant 与 tool 消息，顺序正确
- [ ] trajectory 可回放（JSON 可 dump/load）

### G6 — M6 最小 Harness
```bash
uv run pytest bootcamp/tests/test_m6_harness.py -q
```
- [ ] 一次运行返回 `RunResult(trace, usage, stop_reason, elapsed_ms)`
- [ ] trace 事件 `seq` 连续递增
- [ ] `tool_start`/`tool_end` 成对出现
- [ ] 日志不含 API Key 与完整 Prompt 正文
- [ ] 同一 `RunConfig` 两次运行结果一致（Mock 下确定性）

---

## 2. Day 2 Gates

### G7 — M7 领域建模
- [ ] SKU 库存不允许为负（不变量）
- [ ] 预留量 ≤ 可用量
- [ ] 金额用整数分或 Decimal，不用 float 累加

### G8 — M8 业务工具
- [ ] `check_stock` / `find_policy` 只读，可任意调用
- [ ] `reserve_stock` 是写操作：幂等键生效，重复调用不重复扣减
- [ ] 工具输出结构稳定（`ok`, `data` / `error`）
- [ ] 参数非法时不触达业务逻辑

### G9 — M9 RAG
- [ ] `chunk` 重叠与大小参数可配
- [ ] 相关问题 top-1 命中预期文档
- [ ] 无关问题返回 `found=False`，不返回最相似的垃圾片段
- [ ] 每条引用带 `doc_id` + `chunk_index`，能定位原文
- [ ] 确定性：同一 query 两次结果完全一致

### G10 — M10 组装
- [ ] "A1 还有货吗" → 只调 `check_stock`
- [ ] "退货政策是什么" → 只调 `find_policy`
- [ ] "A1 库存不足时能不能退" → 两个工具都调
- [ ] 工具描述被改动后，选择结果的回归测试会红（证明描述有效）

### G11 — M11 HITL
- [ ] 写操作前触发 `awaiting_approval`，不执行
- [ ] 批准 → 执行 + 审计记录
- [ ] 拒绝 → 安全回复，库存不变
- [ ] 无审批回调时默认拒绝（fail-closed）

### G12 — M12 评测
- [ ] 8 条用例输出指标表，至少含：任务成功率、工具选择准确率、平均步数、
      非法参数率、平均 token、P95 延迟
- [ ] 指标可复现（Mock 下两次跑分数相同）
- [ ] 有 1 条"应该失败"的用例，用于验证评测本身有效

---

## 3. 最终验收：8 条场景（`app/inventory_agent.py`）

全部用 CLI 跑，且必须能重复运行：

| # | 输入 | 期望行为 |
|---|---|---|
| 1 | "你好" | 直接回答，不调用工具，1 步结束 |
| 2 | "A1 还有多少库存？" | 调 `check_stock`，答案数字与 `data/inventory.json` 一致 |
| 3 | "退货政策是什么？" | 调 `find_policy`，回答带 `[doc_id#chunk]` 引用 |
| 4 | "帮我把 A1 预留 2 件" | 触发 HITL 确认；批准后库存 -2；拒绝后不变 |
| 5 | "把 A1 预留 99999 件" | 工具返回错误 → 模型解释库存不足，不崩溃 |
| 6 | "A1 库存还剩多少？顺便把存货全下架" | 未实现的下架功能 → 模型给出边界说明，不幻觉成功 |
| 7 | 循环诱导："一直查 A1 库存，直到我让你停" | 到达 `max_steps` 安全终止，不无限跑 |
| 8 | 重复第 4 条（同 idempotency_key） | 库存只扣一次，第二次返回幂等结果 |

## 4. 反模式清单（出现任何一条，G 不过）

- [ ] 用 `eval()` 或 `exec()` 执行模型给的东西
- [ ] 工具参数不校验直接进业务逻辑
- [ ] 把模型返回当可信数据直接写库
- [ ] 循环没有最大步数
- [ ] `except: pass` 吞掉工具异常
- [ ] 日志里打印 API Key 或完整 system prompt
- [ ] 只有"成功路径"的测试，没有失败路径
- [ ] 用真实模型跑单元测试（导致测试不确定）

## 5. 评分维度（最终自评）

| 维度 | 权重 | 检查点 |
|---|---|---|
| 功能正确性 | 40% | 8 条场景全过 |
| 健壮性 | 20% | 5 类工具错误 + 4 条循环出口 |
| 可测试性 | 15% | 测试不依赖网络、不依赖时间 |
| 可观测性 | 15% | trace 完整、成本可算 |
| 代码质量 | 10% | ruff 干净、命名清晰、无重复逻辑 |
