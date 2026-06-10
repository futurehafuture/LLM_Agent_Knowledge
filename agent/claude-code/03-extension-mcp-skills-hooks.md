# 03 MCP、Plugins、Skills 与 Hooks：Agent 扩展机制

## 1. 为什么 Agent 需要扩展层

生产级 coding agent 不可能把所有能力硬编码在核心 loop 里。它需要同时满足：

- 接入外部系统：GitHub、数据库、浏览器、工单、云平台；
- 复用团队工作流：测试流程、发布流程、安全检查；
- 注入领域知识：项目约定、框架知识、代码生成模板；
- 加入治理策略：操作前检查、操作后审计、敏感命令阻断。

论文把 Claude Code 的扩展面拆成四类：**MCP servers、Plugins、Skills、Hooks**。这四类不是重复设计，而是分别解决不同层级的问题。

## 2. MCP：外部工具与资源协议

MCP（Model Context Protocol）可以理解为 agent 访问外部能力的一种标准协议。它让外部系统把工具、资源、提示等暴露给模型运行时。

适合 MCP 的场景：

- 需要访问外部服务，例如 GitHub、数据库、Notion、Slack；
- 工具能力可以被结构化描述；
- 希望同一工具在不同 agent/客户端之间复用；
- 需要把外部资源按需加载进上下文。

面试可以这样回答：

> MCP 解决的是 agent 与外部世界的能力连接问题。它把外部系统包装成可发现、可调用、可权限管理的工具接口，让模型不是只会生成文本，而是能在运行时访问真实环境。

## 3. Plugins：产品级能力包

Plugins 更像可安装的能力包。它可以组合多种资源：命令、工具、skills、MCP 配置、工作流说明等。MCP 更偏协议和工具边界，plugin 更偏产品分发和能力组织。

适合 plugin 的场景：

- 团队希望发布一整套 agent 能力；
- 一个能力不只是单个 API，而是一组工具、说明、脚本和约定；
- 需要版本化、安装、启用/禁用；
- 希望用户以较低成本接入复杂工作流。

可以把 plugin 类比为 IDE 插件：不是每个功能都要写进核心产品，而是通过插件生态扩展。

## 4. Skills：可复用的任务知识模块

Skill 更像“按需加载的操作手册 + 脚本/模板/资产”。它适合描述某类任务怎么做，例如写 PR、生成 PPT、处理 PDF、调试 CI、整理论文等。

Skill 的关键价值是减少 prompt 重复和上下文污染：

- 不需要每次都把长流程写进用户 prompt；
- 只有任务相关时才加载，节省 context；
- 可以包含脚本、模板、检查清单，提高执行一致性；
- 适合沉淀团队经验，而不一定需要暴露成外部 API。

和 CLAUDE.md 的区别：

| 机制 | 适合内容 | 加载方式 | 风险 |
|---|---|---|---|
| CLAUDE.md | 每次都需要的项目规则 | 启动/目录相关时加载 | 太长会占用上下文 |
| Skill | 某类任务才需要的流程知识 | 触发时加载 | 触发条件和内容需要维护 |

## 5. Hooks：确定性策略与治理点

Hooks 是扩展机制中最像“控制面”的部分。它们在特定生命周期点触发，例如工具调用前、工具调用后、压缩前等。

Hooks 适合做模型不应该自由决定的事情：

- 阻止危险命令，例如删除根目录、读取密钥；
- 在工具调用前注入额外检查；
- 在工具调用后记录审计日志；
- 在 compact 前加入必须保留的摘要要求；
- 根据团队策略强制运行测试或格式化。

面试中要强调：**hooks 是硬工程边界，memory/skills 是软行为引导。** 如果某条规则必须执行，应该放进 hook、permission 或 sandbox，而不是只写进提示词。

## 6. 四类扩展机制的分工

| 机制 | 解决的问题 | 典型例子 | 约束强度 |
|---|---|---|---|
| MCP | 连接外部工具和资源 | GitHub、数据库、文件搜索 | 中，取决于权限管线 |
| Plugins | 分发一组能力 | 团队工具包、领域插件 | 中 |
| Skills | 复用任务流程知识 | 写报告、修 CI、生成测试 | 软约束 |
| Hooks | 生命周期治理 | PreToolUse、PostToolUse、安全检查 | 强约束 |

Claude Code 同时提供这四类机制，说明 production agent 的扩展不是一个维度，而是“能力连接、能力打包、知识复用、治理控制”四个维度。

## 7. 与大模型知识点的联系

扩展机制本质上是在补足 LLM 的几个短板：

- **知识过期**：通过 MCP/RAG 访问最新外部资源；
- **上下文有限**：通过 skill 按需加载，而不是常驻 prompt；
- **不可执行**：通过工具协议把文本意图转成外部动作；
- **不可靠遵循**：通过 hooks 和权限系统提供确定性边界；
- **领域迁移成本高**：通过 plugins 分发领域能力包。

这也解释了为什么仅微调模型不能解决所有 agent 问题。LoRA、SFT、RLHF 可以改变模型倾向，但工具、权限、审计、上下文和外部状态仍需要系统层设计。

## 8. 面试速记

- MCP：agent 连接外部工具和资源的协议层。
- Plugin：把多种能力打包、安装、分发的产品层。
- Skill：按需加载的任务流程知识，减少重复提示和上下文污染。
- Hook：生命周期中的确定性控制点，适合安全、审计、策略执行。
- 关键判断：软知识放 memory/skill，硬约束放 permission/hook/sandbox。

## 参考资料

- [Dive into Claude Code: arXiv HTML, Extensibility](https://arxiv.org/html/2604.14228v1)
- [Extend Claude Code - official docs](https://code.claude.com/docs/en/overview#extend-claude-code)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Tool use with Claude](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)
- [How Claude remembers your project](https://code.claude.com/docs/en/memory)
