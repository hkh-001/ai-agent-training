# Day 2 学习总结

## 今天真正建立的能力

你现在可以观察和控制模型每一轮真正看到的内容：

```text
System Prompt
    + 最近若干轮 user/assistant 历史
    + 当前 user 消息
    -> 模型回答
    -> 保存 assistant 消息
```

Day 2 的关键不是“做出一个聊天框”，而是理解连续对话背后的消息重发机制。

## 四句话复盘

- Message 的角色和顺序共同决定上下文含义。
- Prompt 是完整输入组织，不只是用户最后一句话。
- 无状态 API 通过客户端重发历史形成多轮对话。
- 历史增长会消耗 Token；裁剪能控制成本，但也可能删除重要信息。

## 运行记录

完成学习时填写：

- 学习日期：
- 实际投入时长：
- Prompt Builder 是否运行成功：
- Mock 多轮示例是否观察到裁剪：
- DeepSeek 多轮示例是否验证（可填“暂未运行”）：
- 第 1 轮总 Token：
- 第 2 轮总 Token：
- 你观察到 Token 如何变化：
- 今天最不清楚的概念：

## 自测清单

- [ ] 我能画出两轮对话实际发送的消息列表。
- [ ] 我知道 assistant 回复也必须保存到历史。
- [ ] 我能解释无状态不等于不能多轮。
- [ ] 我能运行 `day02.examples.messages_demo`。
- [ ] 我能运行并读懂 `prompt_builder.py`。
- [ ] 我能解释裁剪为什么从 user 消息开始。
- [ ] 我能证明昵称被裁剪后 Mock 会忘记。
- [ ] 我知道 Prompt 不能替代安全权限。
- [ ] 我完成了昵称助手练习后才阅读参考实现。
- [ ] `uv run python -m pytest day02/tests` 显示 11 passed。
- [ ] `uv run ruff check day02` 全部通过。

## 与 Day 3 的连接

Day 2 的模型最终仍返回自由文本。自由文本适合聊天，但程序很难稳定判断字段、类型和范围。Day 3 将把输出变成 JSON 和 Pydantic 数据模型，区分语法解析、Schema 校验与事实正确性，并为后续 Tool Calling 的参数校验打基础。
