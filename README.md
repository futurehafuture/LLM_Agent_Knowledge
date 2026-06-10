<p align="center">
  <img src="https://img.shields.io/badge/LLM-Agent-blue" alt="LLM Agent">
  <img src="https://img.shields.io/badge/Architecture-Transformer-orange" alt="Transformer">
  <img src="https://img.shields.io/badge/Training-RLHF%20%7C%20SFT%20%7C%20LoRA-green" alt="Training">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen" alt="PRs welcome">
</p>

<h1 align="center">🧠 LLM & Agent 知识库</h1>

<p align="center">
  <strong>从面经问题出发，用 LLM 辅助深度学习</strong><br>
  <sub>🧹 目前还很乱，欢迎一起整理</sub>
</p>

---

## 💡 这个仓库是怎么来的

日常刷小红书、刷面经，看到好的面试题就会丢给大模型：

> *"帮我详细讲讲 MHA 和 GQA 的区别"*  
> *"PPO 和 DPO 的本质差异是什么"*  
> *"Claude Code 的 Agent Loop 是怎么设计的"*

然后把回答整理、查证、补充，沉淀成结构化的笔记。

这样做的好处是：**每个主题都不是干背八股，而是带着问题出发，有上下文、有对比、有来源**。

---

## 🧹 但是目前还比较乱

- 有些笔记已经比较完善，有些还只是初稿
- 目录结构还在迭代中
- 部分内容偏面试导向，部分偏工程理解，风格不完全统一

**如果你也在准备类似方向，或者对某个主题有自己的理解，非常欢迎一起完善。**

---

## 📂 当前覆盖

| 模块 | 内容 |
|------|------|
| **大模型架构** | MHA · GQA · MLA · RoPE · RMSNorm · SwiGLU · MoE · Qwen3 系列 |
| **训练 & 对齐** | SFT · RLHF · PPO · DPO · GRPO · LoRA · AdamW · 混合精度 |
| **Agent 系统** | Claude Code 源码级分析：Agent Loop · 上下文压缩 · MCP · 子代理 |
| **CS 基础** | TCP · HTTP · 数据库索引 · B+树 · 操作系统 |
| **算法** | LeetCode Hot 100（链表 · DP · 二分 · 二叉树） |
| **论文 & 课程** | WebGPT · WebGLM · Stanford Scaling Laws |

---

## ✍️ 内容原则：让初学者也能看懂

补充任何一个知识点时，希望尽量做到：

- **从「是什么」开始** — 不要假设读者已经知道，用一句话讲清楚这个概念的直觉含义
- **解释为什么** — 不是罗列公式，而是说明这套设计解决了什么问题、有哪些 trade-off
- **配上人话版** — 正式定义之前，先用通俗类比讲一遍。比如「RMSNorm 就是只除标准差、不减去均值，比 LayerNorm 更快」
- **有对比才有理解** — 比如讲 GQA 时，把它和 MHA、MQA 放在一起比较：它们各自省了什么、代价是什么
- **参考文献要可追溯** — 引用原论文、官方文档或可信来源，让读者能自行深入

> 目标不是堆砌面经答案，而是让一个刚入门的人看完后能真正理解这个主题。

---

## 🤝 一起构建

如果你：

- 发现了错误或过时的内容
- 对某个主题有更好的理解或组织方式
- 想补充新的主题（推理优化、RAG、多模态、Agent 评估…）

欢迎提 Issue 或 PR，一起把这份知识库变得更扎实。

---

## 🚀 使用

笔记为 Obsidian Markdown 格式，支持 `[[双向链接]]`：

```bash
git clone https://github.com/futurehafuture/LLM_Agent_Knowledge.git
```

---

<p align="center">
  <sub>🤖 AI-assisted · 👤 Human-curated · 🤝 Community-welcomed</sub>
</p>
