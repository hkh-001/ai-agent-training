# Day 2 概念笔记与代码导读

## 一页心智模型

```text
多轮对话不是模型自动记忆

第 1 轮：system + user1                         -> assistant1
第 2 轮：system + user1 + assistant1 + user2   -> assistant2
第 3 轮：system + ... + assistant2 + user3     -> assistant3
```

每一行都是独立 API 请求。历史之所以存在，是客户端维护并再次发送。

## 术语表

| 中文 | 英文 | 本日解释 |
|---|---|---|
| 消息 | Message | 带角色和内容的结构化上下文项 |
| 角色 | Role | 标明消息来源与职责的字段 |
| 系统消息 | System Message | 应用提供的角色、规则和行为要求 |
| 用户消息 | User Message | 用户当前需求或输入 |
| 助手消息 | Assistant Message | 模型先前的回答 |
| 提示 | Prompt | 本次推理的完整输入及组织方式 |
| 多轮对话 | Multi-turn Conversation | 多次请求通过重发历史形成的连续交互 |
| 无状态 | Stateless | 服务端不替客户端自动保存前次请求上下文 |
| 历史裁剪 | History Trimming | 请求前选择保留哪些历史消息 |
| 提示注入 | Prompt Injection | 非可信内容被模型当作指令，导致行为偏离的风险 |

## `TypedDict` 为什么适合 Message

Day 2 定义：

```python
class Message(TypedDict):
    role: MessageRole
    content: str
```

运行时它仍是普通字典：

```python
{"role": "user", "content": "你好"}
```

但类型标注能告诉编辑器：必须存在 `role` 和 `content`，角色只能是规定字符串。它不会替代运行时校验，所以 `build_message()`仍会拒绝空内容。

## 为什么 `Conversation.messages` 返回副本

如果返回内部列表：

```python
conversation.messages.append({"role": "hacker", "content": "..."})
```

调用者就能绕过 `add_user()` 和 `add_assistant()`。返回深拷贝能保护教学状态的不变量：

```text
内部状态只能通过规定方法改变
```

生产系统还需要更完整的 Schema 校验；Day 3 会引入 Pydantic。

## 轮次与消息条数不同

一个完整对话轮次通常包含：

```text
user + assistant
```

但请求发出前，最新轮次可能只有：

```text
user
```

因此不能假设每轮永远恰好两条消息。工具调用后还可能出现更多消息类型，这也是按固定消息条数裁剪容易出错的原因。

## “记忆”的四个层次不要混淆

1. **模型参数中的知识**：训练得到，不是当前用户的会话记忆。
2. **当前上下文**：本次请求实际发送的消息。
3. **会话历史**：客户端暂存并在后续请求重发的消息。
4. **长期记忆**：跨会话保存、选择和重新注入的信息，Day 6 学习。

Day 2 的昵称实验只属于第 2 和第 3 层。

## Prompt 的职责划分

推荐最小结构：

```text
角色：模型在当前应用扮演什么职责
规则：必须遵循哪些可验证行为
输出：希望结果如何表达
```

避免：

- 重复同一规则多次；
- 用抽象形容词代替可观察要求；
- 把 API Key 或秘密写入 Prompt；
- 认为“绝对不要失败”能替代错误处理。

## 裁剪策略的取舍

| 策略 | 优点 | 缺点 |
|---|---|---|
| 最近 N 条消息 | 实现最简单 | 可能破坏完整轮次 |
| 最近 N 轮 | 容易理解 | 不等于精确 Token 预算 |
| Token 预算 | 更贴近模型限制 | 需要模型对应 tokenizer |
| 摘要旧历史 | 压缩明显 | 摘要可能遗漏或失真 |
| 语义检索 | 按当前问题找相关内容 | 需要 Embedding/Retriever |

Day 2 使用最近 N 轮，是为了先掌握控制流，不宣称它适合所有生产场景。

## Day 2 边界

今天已经有：消息、Prompt、多轮历史、裁剪和真实/Mock Chat Model。

今天仍然没有：结构化输出、工具、Agent Loop、持久化 Memory、RAG、LangChain 或 LangGraph。

