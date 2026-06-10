# Claude Code Agent 架构研读：总览

> 主题来源：2026 年 Claude Code npm 包 source map 泄漏事件之后，VILA Lab/UCL 等作者基于公开 TypeScript 源码做了系统性论文分析：[1Dive into Claude Code: The Design Space of Today's and Future AI Agent Systems2](https://arxiv.org/abs/2604.14228)。本文档不引用、不传播泄漏源码，只整理论文、官方文档和公开报道中可学习的 Agent 架构设计点。

## 1. 为什么这篇论文重要

Claude Code 不是“一个模型 + 一个聊天框”，而是一个完整的 **agent harness**：它把大模型、工具执行、权限控制、上下文压缩、记忆、扩展机制、子 agent 和会话持久化组合成一个可运行的软件系统。

论文给出的关键判断是：核心 agent loop 很简单，近似为：

```text
while task_not_done:
    assemble_context()
    call_model()
    dispatch_tool_calls()
    observe_results()
    update_state()
```

真正的复杂度在 loop 周围：

- 如何给模型提供足够但不过载的上下文；
- 如何让模型调用 shell、文件、外部服务等工具；
- 如何在自动化和安全之间做权限边界；
- 如何让长期任务跨越上下文窗口和会话边界；
- 如何把任务拆给子 agent，又不让上下文和权限失控；
- 如何通过 MCP、插件、技能、hooks 接入外部能力。

## 2. 论文提炼出的五个设计价值

论文把 Claude Code 的设计动机归纳为五类价值：

1. **Human Decision Authority**：人类保留最终决策权，agent 可以执行但不能无限制越权。
2. **Safety, Security, and Privacy**：默认保护代码、数据、凭据和基础设施，尤其要防 prompt injection 与误操作。
3. **Reliable Execution**：长任务中持续对齐目标，能验证结果，能从错误和上下文边界中恢复。
4. **Capability Amplification**：不是简单补全，而是让开发者能完成原本不想或不能投入的复杂工作。
5. **Contextual Adaptability**：能适配项目、团队、用户习惯，并通过记忆与配置随时间改善。

## 3. 建议的 Agent 知识点拆分

本目录按面试和系统设计常问点拆成 4 个专题文件：

- [01-agent-loop-tools-permissions.md](01-agent-loop-tools-permissions.md)：Agent 循环、工具调用、权限边界、安全沙箱。
- [02-context-memory-compaction.md](02-context-memory-compaction.md)：上下文工程、CLAUDE.md/Auto Memory、五层压缩管线。
- [03-extension-mcp-skills-hooks.md](03-extension-mcp-skills-hooks.md)：MCP、Plugins、Skills、Hooks 四类扩展机制。
- [04-subagents-persistence-evaluation.md](04-subagents-persistence-evaluation.md)：子 agent 编排、隔离、sidechain、会话持久化与评估。

## 4. 面试中的总论回答模板

如果被问“Claude Code 这类 coding agent 的架构是什么样的？”，可以这样回答：

> Coding agent 的核心不是把 LLM 接到 IDE，而是构建一个围绕 LLM 的执行运行时。模型负责根据目标和上下文产生下一步意图，harness 负责把意图转成受控工具调用，并在权限、安全、上下文预算、记忆、持久化和多 agent 协作上提供确定性边界。Claude Code 的核心 loop 很朴素，但周边系统非常厚：它有工具池和权限管线，有面向长上下文的五层 compaction，有 CLAUDE.md 与 auto memory，有 MCP/插件/技能/hooks 扩展层，也有隔离式 subagent 和 append-only transcript。这样的设计说明，Agent 系统的壁垒往往不只在模型，而在模型外部的工程编排与安全控制。

## 5. 参考资料

- 论文：[1Dive into Claude Code: The Design Space of Today's and Future AI Agent Systems2](https://arxiv.org/abs/2604.14228)
- 论文 HTML 版：[1arXiv HTML 2604.14228v12](https://arxiv.org/html/2604.14228v1)
- 论文项目仓库：[1VILA-Lab/Dive-into-Claude-Code2](https://github.com/VILA-Lab/Dive-into-Claude-Code)
- Claude Code 记忆官方文档：[1How Claude remembers your project2](https://code.claude.com/docs/en/memory)
- Claude Code 权限模式官方文档：[1Permission modes2](https://code.claude.com/docs/en/permission-modes)
- Anthropic 工具调用文档：[1Tool use with Claude2](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)
- Anthropic Agent 架构文章：[1Building Effective Agents2](https://www.anthropic.com/engineering/building-effective-agents)
- Anthropic 上下文工程文章：[1Effective context engineering for AI agents2](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- 泄漏事件公开报道：[1TechRadar: Anthropic confirms source leak2](https://www.techradar.com/pro/security/anthropic-confirms-it-leaked-512-000-lines-of-claude-code-source-code-spilling-some-of-its-biggest-secrets)
