# 01 Agent 循环、工具调用与权限边界

## 1. Agent loop：Claude Code 的最小内核

论文认为 Claude Code 的核心不是复杂的显式规划器，而是一个很朴素的迭代循环：模型读取上下文，产生下一步动作，harness 执行工具，结果回流给模型，然后继续。这个设计更接近 ReAct 思路：

```text
Reason / decide next action
→ Act through tools
→ Observe environment feedback
→ Repeat until task is solved or stopped
```

和经典 ReAct 的区别是，Claude Code 把大量工程能力下沉到 harness：工具 schema、权限检查、工具结果裁剪、恢复机制、会话记录、hooks、子 agent 等都由运行时负责，而不是完全靠 prompt 约束。

面试表达可以抓住一句话：**Agent 的智能来自模型与环境反馈闭环，可靠性来自模型外部的确定性运行时。**

## 2. 工具调用：从“文本建议”到“环境操作”

Claude API 官方文档把工具调用分成两步：模型根据用户请求和工具描述选择工具，并返回结构化 tool call；应用侧或服务侧再执行该调用，把结果返回给模型。Claude Code 把这一机制产品化到开发环境中，典型工具包括：

- 读写文件：读取代码、修改代码、生成补丁；
- shell 命令：运行测试、构建、搜索、git 操作；
- 外部服务：通过 MCP 或插件访问 GitHub、数据库、文档系统等；
- 任务管理：计划、检查点、子 agent 调度等运行时状态。

工具调用的关键不是“能不能调函数”，而是四个工程问题：

1. **工具发现**：当前任务可用哪些工具？工具描述是否足够让模型正确选择？
2. **参数约束**：模型输出的参数是否可解析、可验证、可审计？
3. **执行回流**：工具输出如何进入上下文，避免日志、diff、测试输出淹没模型？
4. **权限边界**：哪些动作允许自动执行，哪些必须询问人类，哪些必须拒绝？

## 3. 权限管线：让 autonomy 有边界

Claude Code 的权限系统是论文重点之一。公开论文总结其包含多种 permission modes，并且在 auto mode 中引入分类器辅助判断。官方权限文档也强调：不同模式对应不同程度的自动化和人工确认。

可以把权限管线理解为多层过滤：

```text
model proposes tool call
→ pre-filter / static rule
→ hook policy
→ permission rule evaluation
→ classifier or human confirmation
→ execute or deny
```

这种设计解决的是 agent 系统里的核心矛盾：

- 如果每一步都问用户，长任务会被打断，用户也会出现 approval fatigue；
- 如果完全自动执行，agent 可能误删文件、泄露密钥、运行危险命令；
- 所以需要按风险分层，把低风险操作自动化，把高风险操作交给更强边界。

## 4. Auto Mode 与安全分类器

Anthropic 关于 auto mode 的文章指出，默认逐次审批虽然安全，但会导致大量点击确认；auto mode 的目标是在减少打扰的同时维持安全边界。论文进一步把它放到 Claude Code 的权限架构里看：分类器不是替代安全设计，而是权限管线中的一个判断组件。

面试中要避免说“有了 classifier 就安全”。更好的回答是：

> Auto mode 是一种风险分层策略。它把明显低风险的动作自动放行，把高风险或不确定动作交给人类或拒绝。但 classifier 本身可能误判，也可能被上下文污染，所以仍要配合静态 deny 规则、sandbox、hooks、审计日志和最小权限原则。

## 5. Shell sandbox：把可执行能力关进笼子

Claude Code 这类 coding agent 最危险也最有用的能力是 shell。Anthropic sandboxing 文章强调，Claude Code 在用户机器上运行，访问文件系统、shell 和网络是能力来源，但也必须受限。

Sandbox 的面试要点：

- **文件系统边界**：限制能读写哪些目录，避免越权访问私密文件；
- **网络边界**：限制对外请求，减少数据外泄；
- **命令边界**：对删除、权限修改、凭据读取、远程执行等命令提高确认级别；
- **可观测性**：让用户看到 agent 将要做什么、做了什么、失败在哪里；
- **可恢复性**：配合 git/checkpoint/append-only transcript，降低错误动作损失。

## 6. 与经典 Agent 框架的关系

Claude Code 的 loop 可以映射到常见 Agent 模式：

- **ReAct**：推理、行动、观察循环；
- **Plan-and-Execute**：对于复杂任务先形成计划，再逐步执行；
- **Evaluator-Optimizer**：生成与检查分离，测试、lint、review 充当环境反馈；
- **Orchestrator-Workers**：父 agent 把子任务交给 subagents；
- **Toolformer / Function Calling**：模型以结构化形式选择工具，运行时负责执行。

但 Claude Code 的启发在于：生产级 Agent 不是把这些模式机械拼起来，而是围绕风险、上下文、工具结果和用户控制做系统工程。

## 7. 面试速记

- Agent loop 本质：`LLM decision + tool execution + observation feedback`。
- 工具调用本质：把模型的文本意图变成可审计、可验证、可回滚的环境操作。
- 权限系统本质：在自动化效率和安全控制之间建立分层边界。
- Auto Mode 本质：低风险自动放行，高风险询问或拒绝，但不能替代 sandbox 和规则。
- 生产级 coding agent 的壁垒：不是只有模型能力，而是工具、权限、上下文和恢复机制的 harness。

## 参考资料

- [Dive into Claude Code: The Design Space of Today's and Future AI Agent Systems](https://arxiv.org/abs/2604.14228)
- [Tool use with Claude](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)
- [Claude Code permission modes](https://code.claude.com/docs/en/permission-modes)
- [How we built Claude Code auto mode](https://www.anthropic.com/engineering/claude-code-auto-mode)
- [Making Claude Code more secure and autonomous with sandboxing](https://www.anthropic.com/engineering/claude-code-sandboxing)
- [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
