# Day 1 当日小型实践：可切换的问答器

## 任务目标

用 30～60 分钟实现一个命令行问答器：默认使用 Mock Model；当环境变量
`USE_REAL_MODEL=true` 时，才调用真实模型。

## 输入与输出

- 输入：用户在终端输入的一段非空文本。
- 输出：以 `回答：` 开头的模型文本，或者清晰、可操作的错误提示。

## 实现要求

1. 读取 `USE_REAL_MODEL`，不区分大小写地识别 `true`。
2. Mock 模式不能要求 API Key。
3. 真实模式复用 `day01/examples/deepseek_llm_call.py` 中的 `ask_deepseek()`。
4. 空输入必须被拒绝。
5. 不得在代码中写入真实密钥或写死模型名。

## 建议步骤

1. 先只完成 Mock 路径并运行。
2. 将“选择模型”的判断放进独立函数。
3. 增加空输入处理。
4. 有 Key 时再验证真实路径；没有 Key 不影响完成练习。
5. 运行 `uv run python -m pytest day01/tests` 验收。

## 验收方法

```bash
uv run python -m day01.exercises.solution
USE_REAL_MODEL=true uv run python -m day01.exercises.solution
uv run python -m pytest day01/tests
```

第一条必须在没有 API Key 时正常运行。第二条只有配置了 `.env` 才需要成功。

## 可选扩展

- 输出当前使用的是 Mock 还是真实模式。
- 统计本次输入的字符数，并说明字符数不等于 Token 数。
- 为真实请求记录耗时，但不要记录 API Key。
