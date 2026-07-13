# 02 上下文、记忆与 Compaction

## 1. 为什么上下文是 Agent 的瓶颈

Coding agent 的任务往往跨越多个文件、命令输出、测试日志、用户偏好和历史决策。模型的上下文窗口再大，也会遇到三个问题：

- **容量问题**：代码库、日志、diff、工具输出很快超过窗口；
- **信噪比问题**：无关信息会稀释关键约束，导致模型注意力漂移；
- **时间问题**：长任务跨越多轮对话、压缩边界甚至多次会话恢复。

因此生产级 agent 需要 context engineering，而不是简单把所有东西塞进 prompt。

可以把 live context 抽象成：

```math
C_{live}=C_{system}+C_{memory}+C_{conversation}+C_{tools}+C_{results}+C_{runtime}
```

其中真正难管理的是 `C_conversation` 和 `C_results`：它们增长最快，也最容易把模型带偏。

## 2. Claude Code 的两类记忆

Claude Code 官方文档把跨会话知识分成两种机制：

1. **CLAUDE.md**：用户或团队写的持久指令，适合记录项目结构、编码规范、测试命令、工作流。
2. **Auto Memory**：Claude 根据用户纠正、调试模式和偏好自动沉淀的学习内容。

二者的差别是：

| 维度 | CLAUDE.md | Auto Memory |
|---|---|---|
| 写入者 | 用户、团队、组织 | Agent 自己 |
| 内容 | 明确规则、约定、项目事实 | 从交互中学到的偏好和经验 |
| 作用 | 启动时加载为上下文 | 启动时加载为上下文 |
| 风险 | 太长会占上下文，规则可能过期 | 自动写入可能沉淀错误经验 |
| 约束力 | 软约束，不是强制策略 | 软约束，不是强制策略 |

面试要强调：**记忆不是权限系统。记忆会影响模型行为，但不能替代 hooks、deny rules、sandbox 这些硬边界。**

## 3. CLAUDE.md 层级与按需加载

官方文档说明，Claude Code 会从当前目录向上查找 `CLAUDE.md` / `CLAUDE.local.md`，并在进入子目录时按需加载子目录下的规则。这个设计体现了两个原则：

- **越靠近工作目录，规则越具体**：项目根目录给全局约定，子目录给局部约定；
- **越大的团队越需要模块化规则**：把所有规则放进一个巨大文件会浪费上下文并降低遵循率。

一个好用的记忆结构通常是：

```text
CLAUDE.md                 # 项目级总规则
.claude/rules/testing.md  # 测试规则
.claude/rules/security.md # 安全规则
submodule/CLAUDE.md       # 子模块专属约定
```

这和 RAG 的思想类似：不是预先加载所有知识，而是在需要时把最相关的规则送入上下文。

## 4. 五层 Compaction 管线

论文最重要的发现之一是：Claude Code 不是用单一截断或单一总结处理长上下文，而是用了五层逐级加重的 compaction pipeline：

1. **Budget reduction**：对单个工具结果做大小限制，先控制局部输出。
2. **Snip**：轻量剪掉较旧历史，释放一部分 token。
3. **Microcompact**：更细粒度、缓存感知的压缩，减少缓存和上下文开销。
4. **Context collapse**：在读取时构造一个压缩后的历史视图，而不直接改写完整历史。
5. **Auto-compact**：当仍然超压时，用模型生成完整摘要。

这个顺序体现了 lazy degradation：先用最便宜、信息损失最小的方式；只有压力仍然存在时，才升级到更强、更贵、更不可逆的压缩。

## 5. 为什么不是简单截断

很多简单 agent 框架会采用“丢掉最旧消息”或“达到阈值就总结”的策略。Claude Code 的五层策略更复杂，但解决了几个生产问题：

- 工具输出往往是最大噪声源，应该先局部裁剪，而不是立刻总结整段对话；
- 有些历史可以压缩成摘要，有些必须保留原始证据，例如用户要求、关键错误、文件路径；
- cache、工具 schema、附件、计划状态都会影响真实成本；
- long-running task 需要 transcript 完整可恢复，而 live context 只是当前投影。

因此应区分：

- **durable transcript**：完整、append-only、可审计；
- **live context**：当前模型可见的压缩视图；
- **memory files**：跨会话加载的项目知识和偏好。

## 6. Context engineering 与 RAG 的关系

Anthropic 的上下文工程文章提出一个关键思想：just-in-time context。Agent 不应该预先塞入所有信息，而应保留轻量引用，在需要时通过工具读取。

这与 RAG 很像，但 coding agent 的上下文工程更复杂：

- RAG 多数处理文档片段检索；coding agent 还要处理 shell 输出、diff、测试日志、文件树、权限状态；
- RAG 关注召回相关知识；coding agent 还要决定哪些历史要压缩、哪些状态要重放；
- RAG 的知识通常是静态文档；coding agent 的上下文不断被工具调用改变。

面试中可以说：**Claude Code 的 context management 是 RAG、日志压缩、状态投影和长期记忆的组合。**

## 7. 常见追问

### 7.1 Auto Memory 会不会污染？

会。因为 auto memory 是 agent 根据交互自动写入的经验，可能把一次性偏好、错误归因或临时 workaround 固化。因此需要可审计、可编辑、可禁用，并且重要规则应由人工写入 CLAUDE.md 或硬策略。

### 7.2 Compaction 会不会丢关键信息？

会。任何压缩都会损失信息。Claude Code 的思路不是消除损失，而是分层降低损失：先裁剪工具输出和旧历史，最后才做模型摘要；同时保留 transcript，使得需要时可以恢复或回看。

### 7.3 为什么 memory 不是 database？

论文指出 Claude Code 更偏好文件和 append-only JSONL 这类透明、低依赖的方案。数据库查询能力更强，但增加部署复杂度，也降低用户对状态的可见性。

## 8. 面试速记

- 上下文管理目标：在 token 预算内最大化相关信息密度。
- CLAUDE.md：人写的持久规则；Auto Memory：agent 自动沉淀经验。
- 五层 compaction：budget reduction → snip → microcompact → context collapse → auto-compact。
- durable transcript 与 live context 要分离：前者完整可审计，后者压缩给模型看。
- Context engineering 不只是 RAG，还包括工具输出治理、状态重放、压缩边界和长期记忆。

## 参考资料

- [Dive into Claude Code: arXiv HTML, Context Construction and Memory](https://arxiv.org/html/2604.14228v1)
- [How Claude remembers your project](https://code.claude.com/docs/en/memory)
- [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Explore the context window - Claude Code Docs](https://code.claude.com/docs/en/context-window)
