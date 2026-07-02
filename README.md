# FileGraph Agents v0

FileGraph Agents (FGA) 是一个极简 Python 原型：把代码仓库里的每个文件当成一个隐藏的 actor-style file-agent，以文件路径为地址，用 `talk(path, prompt)` 做消息传递，让代码修改任务沿文件/依赖图局部传播、局部协商、局部整合。

这个版本刻意保持简单，目标不是做工业级 IDE agent，而是验证一个假设：

> 弱模型失败的主要原因之一，是 repo 级任务所需的有效上下文信息超过了单模型注意力能力；如果把同一目标下的上下文收集函数按文件边界分解，弱模型也可能完成更复杂的软件工程任务。

## 核心规则

- MainAgent 只能 `ls/search/talk/create_file/delete_file/shell`。
- MainAgent 不能直接 read/write 源码。
- FileAgent 只能 `read/write` 自己绑定的文件。
- 其他文件的信息只能通过 `talk(other_path, prompt)` 询问对应 file-agent。
- `search(content)` 只返回命中的文件路径和次数，不返回代码内容。
- `shell(command)` 只给 MainAgent，用于 test/build/typecheck/lint/verifier，不能用于阅读或编辑代码。
- `talk` 是 one-shot，但 file-agent 是 actor：同一个文件只有一个持续上下文。等待某个 talk 回复时，这个 agent 仍可处理新的 talk 请求，且使用同一份局部 memory。

## 安装

```bash
pip install -e .
```

或直接在源码目录运行：

```bash
python -m filegraph_agents.cli /path/to/repo "fix the failing test"
```

## 配置模型

FGA 通过 [litellm](https://github.com/BerriAI/litellm) 调用任意 OpenAI-compatible `/chat/completions` 接口，并使用原生 function calling（tool calls）。

```bash
export DEEPSEEK_API_KEY="..."
export FGA_BASE_URL="https://api.deepseek.com"
export FGA_MODEL="deepseek-v4-flash"
```

也可以显式传：

```bash
fga /path/to/repo --model deepseek-v4-flash "implement ..."
```

如果你的模型服务是标准 OpenAI-compatible `/v1/chat/completions`，设置：

```bash
export FGA_BASE_URL="https://your-provider.example/v1"
export FGA_API_KEY="..."
export FGA_MODEL="your-model-name"
```

## Python 用法

```python
from filegraph_agents import FGAConfig, FGARuntime, LiteLLMModel

config = FGAConfig.from_env()
runtime = FGARuntime(
    root="/path/to/repo",
    config=config,
    model=LiteLLMModel(config),
)

result = runtime.run("Fix the bug described in the issue. Run tests when done.")
print(result)
```

## 工具协议

当前版本使用模型原生的 function calling（tool calls）。每个 actor 根据自己的权限暴露一组工具，模型通过调用工具来行动，通过直接输出纯文本来回复调用者。

所有 actor 通用的工具：

- `ls(path?)`：列目录（不返回代码内容）。
- `search(content, max_results?)`：全仓库字面量搜索，只返回命中路径和次数。
- `talk(path, prompt)`：向另一个 file-agent 提问或委托修改。
- `create_file(path)` / `delete_file(path)`：创建/删除文件并管理其 file-agent。
- `memory_update(items)`：写入持久化的局部 memory。

MainAgent 额外拥有：

- `shell(command)`：运行 test/build/typecheck/lint/verifier（禁止 cat/sed/grep 等读代码命令）。

FileAgent 额外拥有（只作用于自己绑定的文件）：

- `read(start_line, offset)`：读取自己文件的若干行。
- `write(start_line, end_line, content)`：替换自己文件的 `[start_line, end_line]`（含端点）；插入用 `end_line = start_line - 1`。

回复调用者不需要工具，直接输出纯文本即可。MainAgent 完成任务时同样直接输出最终总结文本。

## Actor / talk stack 语义

FGA v0 的 `talk` 是同步 one-shot，但 actor 可以重入：

```text
A talk B
B talk C
C talk A
```

此时 A 虽然正在等待 B 的回复，但 A 这个 actor 没有死。C 对 A 的新 talk 会被注入同一个 A 上下文。A 回答 C 后，再继续等待 B。实现上是同步递归调用，但 actor memory 是同一个对象，因此不会产生 A0/A1 这种上下文分裂。

## v0 刻意不做

- 不做 hash write。
- 不做 AST patch。
- 不做并发执行。
- 不做 LSP。
- 不做复杂 CLI 展示。
- 不自动构建依赖图。

第一版只验证：文件边界 + actor talk + 局部整合，是否能让弱模型超过单体基线。

## 运行测试

```bash
python -m unittest discover -s tests
```
