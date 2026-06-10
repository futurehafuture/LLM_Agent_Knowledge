# 04 子 Agent、会话持久化与评估

## 1. 为什么需要子 Agent

当任务变长、变复杂时，单个 agent 会遇到两个瓶颈：

- **上下文瓶颈**：所有探索、日志、错误路径都塞进一个上下文窗口，会快速失焦；
- **工作流瓶颈**：有些子任务可以独立探索，例如定位 bug、读文档、写测试、搜索实现方案。

Claude Code 论文把 subagent delegation 看作核心设计点之一。它不是让所有 agent 共享一个巨大聊天记录，而是让子 agent 在隔离上下文中完成任务，然后把摘要返回给父 agent。

这背后的原则是：**多 agent 协作的关键不是“越多越好”，而是隔离、汇总和可控。**

## 2. 父子 Agent 架构

可以把 Claude Code 的子 agent 机制理解为：

```text
Parent agent
  ├─ decides whether to delegate
  ├─ specifies subtask and constraints
  ├─ launches isolated subagent context
  ├─ receives summary-only result
  └─ integrates result into main task
```

子 agent 适合：

- 独立探索代码库某个模块；
- 并行搜索不同假设；
- 生成候选方案但不直接改主线；
- 读取大量资料后产出摘要；
- 执行低耦合、边界清晰的子任务。

不适合：

- 父 agent 下一步立即依赖的阻塞任务；
- 需要强一致状态的共享编辑；
- 权限风险高且边界不清的操作；
- 多个子 agent 同时修改同一文件。

## 3. Isolation：隔离比共享更重要

论文强调 Claude Code 采用隔离式 subagent 边界。隔离通常包括：

- **上下文隔离**：子 agent 不继承完整父会话，减少污染；
- **权限隔离**：子 agent 的工具权限可以更受限；
- **工作区隔离**：通过 worktree 或类似机制降低文件冲突；
- **结果隔离**：父 agent 通常只接收摘要，而不是完整子会话 transcript。

这解决了多 agent 最常见的问题：上下文爆炸。如果所有子 agent 的完整对话都回流给父 agent，token 成本会随 agent 数量快速膨胀，父 agent 反而更难判断。

## 4. Sidechain transcript：子任务也要可审计

虽然父 agent 只需要摘要，但系统仍需要记录子 agent 的执行轨迹。论文提到 Claude Code 使用 subagent sidechains：子 agent 有独立 transcript 和 metadata。

这体现了 durable state 与 live context 的分离：

- 父 agent 的 live context 只保留子任务摘要；
- 子 agent 的完整轨迹写入 sidechain，便于调试、审计和恢复；
- 如果摘要有问题，可以回看子 agent 的证据链。

面试中可以说：**summary-only return 是上下文优化，sidechain transcript 是可审计性保障。**

## 5. 会话持久化：append-only transcript

论文把 Claude Code 的会话持久化描述为 append-oriented session storage。它会把消息、工具结果、compact 边界、子 agent 摘要等事件写入磁盘。

这种设计的优点：

- **透明**：JSONL/文本事件可读，不依赖复杂数据库；
- **可审计**：每一步发生了什么可以回放；
- **可恢复**：上下文窗口可以压缩，但完整会话仍在 transcript 中；
- **可分叉**：用户可以从历史会话恢复或 fork 新路径。

缺点也明显：

- 查询能力弱，不如数据库；
- 需要额外逻辑处理清理、压缩边界和历史重放；
- 长期知识如果只存在 transcript 里，不一定能被模型有效召回。

## 6. 为什么恢复会话不恢复权限

论文指出，Claude Code 的 resume/fork 不恢复 session-scoped permissions。这个细节非常重要：它说明权限不是普通对话状态。

原因是：

- 旧会话中的授权可能已经不适合新环境；
- resume/fork 可能发生在不同时间、目录、分支或安全上下文；
- 恢复消息可以提高连续性，但恢复权限可能扩大风险。

所以更安全的设计是：恢复历史对话，但重新建立权限上下文；不认识的请求回到 deny-first 或 ask-first。

## 7. 评估与自我修正

Agent 可靠性不能只看模型回答，还要看环境验证。Claude Code 这类 coding agent 常见评估信号包括：

- 测试是否通过；
- lint/typecheck 是否通过；
- diff 是否符合用户目标；
- 是否误改无关文件；
- 是否遵守权限和安全策略；
- 长任务中是否保持目标一致。

这对应 Anthropic “Building Effective Agents” 中的 evaluator-optimizer 思路：生成和评价最好分离。对 coding agent 来说，测试、编译器、静态分析、代码 review、人类确认都可以成为 evaluator。

## 8. 多 Agent 与经典架构模式

Claude Code 的 subagent 机制可对应到几类常见模式：

- **Orchestrator-Workers**：父 agent 作为 orchestrator，子 agent 作为 worker；
- **Blackboard / shared workspace**：通过文件、任务列表或锁进行间接协作；
- **Map-Reduce**：多个子 agent 分别探索，父 agent 汇总摘要；
- **Debate / Critique**：不同 agent 评审同一方案，但需要控制成本和幻觉；
- **Plan-and-Solve**：父 agent 规划，子 agent 执行局部步骤。

生产系统通常不会无脑采用 fully-connected 多 agent 聊天，因为那会带来上下文爆炸、冲突写入、责任不清和成本失控。

## 9. 面试速记

- 子 agent 价值：隔离探索、并行子任务、降低主上下文污染。
- 子 agent 风险：上下文爆炸、权限扩散、文件冲突、摘要失真。
- Claude Code 倾向：隔离上下文 + summary-only return + sidechain transcript。
- 会话持久化：append-only transcript，live context 可压缩，durable history 可审计。
- Resume/fork 不恢复权限：权限是安全上下文，不是普通历史消息。
- 评估闭环：测试、lint、typecheck、review 和 human approval 都是 agent 的外部反馈。

## 参考资料

- [Dive into Claude Code: arXiv HTML, Subagent Delegation and Orchestration](https://arxiv.org/html/2604.14228v1)
- [Dive into Claude Code: arXiv HTML, Session Persistence and Recovery](https://arxiv.org/html/2604.14228v1)
- [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
- [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Claude Code memory docs](https://code.claude.com/docs/en/memory)
