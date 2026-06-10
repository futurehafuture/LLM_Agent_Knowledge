---
tags: [面试, 大模型, Qwen3, Qwen3-Next, MoE, 长上下文]
created: 2026-06-03
source: 面经
复习状态: 未复习
难度: 中等
---

# Qwen3 / Qwen3-Next 模型架构

## 资料来源地图

1. 来源类型：官方模型卡 / 官方模型仓库  
   为什么可信：Qwen3-Next 的参数、架构布局、上下文长度、使用方式属于模型发布信息，应以 Qwen 官方 Hugging Face 模型卡为准。  
   本文主要参考：Qwen3-Next-80B-A3B-Instruct 的 highlights、模型结构、参数量、长上下文、部署要求。  
   链接：[Qwen3-Next-80B-A3B-Instruct Model Card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)

2. 来源类型：技术报告  
   为什么可信：Qwen3 主线架构与训练设计来自 Qwen 团队技术报告。  
   本文主要参考：Qwen3 Dense/MoE 双线、thinking / non-thinking、模型家族定位。  
   链接：[Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)

3. 来源类型：已有 Obsidian 笔记  
   为什么可信：本地已有 Qwen3 详细架构笔记，本文只补 Qwen3-Next 并给面试表达。  
   本文主要参考：Qwen3 的 RMSNorm、GQA、RoPE、QK-Norm、MoE 对比。  
   链接：[[07-Qwen3-模型架构]]

## 这篇解决什么问题

- 原始面经问题：
  - 讲讲 Qwen3 / Qwen3-Next。

- 你需要掌握的核心能力：
  - 能讲 Qwen3 的主线架构特点。
  - 能讲 Qwen3-Next 为什么叫 Next：重点是效率架构，而不是只换一个模型名。
  - 能对比 dense、MoE、高稀疏 MoE、GQA、混合注意力、长上下文。

## 先讲人话版

Qwen3 是“主线大模型家族”：它覆盖 dense 和 MoE，强调统一 thinking / non-thinking、多语言、代码、数学、agent 能力。

Qwen3-Next 是“下一代高效架构探索”：它不只是把参数变大，而是试图让模型在总参数很大的同时，每个 token 激活参数很少，并且更擅长超长上下文。它的关键词是：

- 80B total / 约 3B activated。
- 高稀疏 MoE。
- Gated DeltaNet + Gated Attention 的 hybrid attention。
- Multi-Token Prediction。
- 原生 256K 上下文，可用 YaRN 扩到约 1M。

面试中可以这样总括：

> Qwen3 是 Qwen 主线 LLM 家族，重点是 dense/MoE 双线、RMSNorm、GQA、RoPE、SwiGLU/MoE、QK-Norm，以及 thinking 和 non-thinking 的统一。Qwen3-Next 则更关注 scaling efficiency，用高稀疏 MoE、混合注意力和 MTP，在只激活少量参数的情况下提升长上下文和推理吞吐。

## 必备前置知识

| 概念 | 短定义 | 为什么重要 |
|---|---|---|
| Dense model | 每个 token 使用全部参数 | Qwen3 有 dense 小中模型 |
| MoE | 每个 token 只选择部分专家 | Qwen3 和 Qwen3-Next 的容量/效率关键 |
| Active parameters | 每 token 实际参与计算的参数量 | 判断 MoE 推理成本不能只看总参数 |
| GQA | 多个 Q head 共享较少 K/V head | 节省 KV Cache |
| Linear attention / DeltaNet | 用递推或线性复杂度方式建模长上下文 | Qwen3-Next 混合注意力的一部分 |
| MTP | 一次预测多个未来 token 的辅助目标或推理加速机制 | 提升训练信号和 speculative decoding 效率 |
| YaRN | RoPE 长上下文扩展方法 | Qwen3-Next 扩到约 1M 上下文时使用 |

## 核心原理

### 1. Qwen3 主线架构

Qwen3 是 decoder-only causal language model。一个典型 block 可概括为：

```text
Embedding
  -> N 层 Transformer Decoder Block
       -> RMSNorm
       -> Causal Attention / GQA + RoPE + QK-Norm
       -> Residual
       -> RMSNorm
       -> SwiGLU FFN 或 MoE FFN
       -> Residual
  -> Final RMSNorm
  -> LM Head
```

Qwen3 的面试关键词：

| 关键词 | 怎么讲 |
|---|---|
| Dense + MoE | 既有全参数激活的 dense 模型，也有总参数大、激活参数小的 MoE 模型 |
| RMSNorm | 替代 LayerNorm，计算更轻，适合大模型 |
| GQA | 降低推理 KV Cache，占用和带宽更友好 |
| RoPE | 把位置信息注入 Q/K，天然表达相对位置 |
| QK-Norm | 对 Q/K 做归一化，控制 attention logits 尺度 |
| SwiGLU | 门控 FFN，提高表达能力 |
| Thinking / Non-thinking | 复杂推理与快速回答统一在同一模型框架 |

### 2. Qwen3-Next 的模型定位

官方模型卡把 Qwen3-Next 的动机放在两个趋势上：

1. 总参数继续扩大。
2. 上下文长度继续变长。

但直接堆参数和上下文会让训练、推理成本爆炸，所以 Qwen3-Next 的核心问题是：

> 如何在总容量更大、上下文更长的同时，让每 token 的计算成本更低？

### 3. Qwen3-Next-80B-A3B 的关键参数

以官方模型卡中的 `Qwen3-Next-80B-A3B-Instruct` 为例：

| 项目 | 数值 / 设计 |
|---|---|
| 模型类型 | Causal Language Model |
| 总参数 | 80B |
| 激活参数 | 约 3B activated |
| 训练 | Pretraining 15T tokens + Post-training |
| 层数 | 48 |
| hidden dimension | 2048 |
| MoE experts | 512 experts |
| 激活专家 | 10 activated experts |
| shared expert | 1 |
| 原生上下文 | 262,144 tokens |
| YaRN 扩展 | 官方模型卡称验证到约 1,010,000 tokens |
| Instruct 模式 | 仅 non-thinking，不输出 `<think></think>` |

注意：面试中不要把 `A3B` 机械理解成精确 3.000B。模型卡写的是 80B total 和 3B activated，具体实现统计可能在不同平台显示为约 3B 或 3.9B active。

### 4. Hybrid Attention：Gated DeltaNet + Gated Attention

Qwen3-Next 不完全使用标准 attention，而是混合：

- **Gated DeltaNet**：偏线性注意力 / state-space 风格，适合高效长上下文建模。
- **Gated Attention**：保留标准注意力类机制，补足精确 token 交互能力。

官方模型卡给出的 layout：

```text
12 * (3 * (Gated DeltaNet -> MoE) -> 1 * (Gated Attention -> MoE))
```

直觉：

> 每 4 个子结构里，3 个用更高效的 DeltaNet 路径处理长上下文，1 个用 attention 路径保留强表达能力。这是“效率”和“质量”的折中。

面试表达：

> 标准 self-attention 对序列长度是二次复杂度，长上下文成本高。Qwen3-Next 用 Gated DeltaNet 和 Gated Attention 混合，让大部分层用更适合长上下文的高效模块，同时周期性保留 attention 来维持精确建模能力。

### 5. High-Sparsity MoE

Qwen3-Next 的 MoE 很稀疏：

- 512 个 experts。
- 每 token 激活 10 个 experts。
- 另有 1 个 shared expert。

这意味着模型总容量很大，但每个 token 只走很小一部分专家。

和普通 MoE 的区别：

| 维度 | 普通 MoE | 高稀疏 MoE |
|---|---|---|
| experts 数 | 较少或中等 | 很多 |
| 激活比例 | top-k，占比不一定极低 | 激活比例很低 |
| 目标 | 增加容量、控制计算 | 更激进地提升参数效率 |
| 难点 | 路由、负载均衡、通信 | 更依赖稳定路由和训练技巧 |

### 6. Stability Optimizations

官方模型卡提到 Qwen3-Next 包含稳定性优化，例如：

- zero-centered layernorm。
- weight-decayed layernorm。
- 其他预训练和后训练稳定技巧。

面试里不用展开到实现细节，除非对方追问。可以说：

> 这种高稀疏 MoE + 混合注意力架构比标准 dense Transformer 更难训，所以必须额外处理归一化、权重衰减和训练稳定性，否则容易出现路由不均衡、梯度不稳或长上下文退化。

### 7. Multi-Token Prediction

MTP 的直觉：

> 普通 next-token 训练每个位置只预测下一个 token；MTP 让模型同时预测多个未来 token，能提供更密集的训练信号，也能配合推理框架做 speculative decoding，提高吞吐。

注意边界：

- 官方模型卡提示 Hugging Face Transformers 中 MTP 不一定通用可用。
- 实际吞吐收益依赖 vLLM / SGLang 等推理框架实现。

## Qwen3 vs Qwen3-Next 对比

| 维度 | Qwen3 | Qwen3-Next |
|---|---|---|
| 定位 | 主线模型家族 | 下一代高效架构探索 |
| 架构主体 | Decoder-only Transformer | Hybrid Transformer / linear-attention-style 架构 |
| 模型形态 | Dense + MoE | 高稀疏 MoE |
| 注意力 | GQA + RoPE + QK-Norm | Gated DeltaNet + Gated Attention |
| 长上下文 | Qwen3 系列支持长上下文扩展 | 原生 256K，YaRN 可扩展到约 1M |
| 训练/推理效率 | 依模型尺寸不同 | 重点强调 scaling efficiency |
| 面试关键词 | RMSNorm、GQA、RoPE、QK-Norm、SwiGLU/MoE、thinking mode | 80B/A3B、512 experts、hybrid attention、MTP、ultra-long context |

## 面试怎么答

### 30 秒版

> Qwen3 是 Qwen 的主线大模型家族，decoder-only 架构，既有 dense 也有 MoE，常见组件包括 RMSNorm、GQA、RoPE、SwiGLU、QK-Norm，并统一 thinking 和 non-thinking。Qwen3-Next 更强调高效 scaling，以 80B-A3B 为代表，总参数 80B 但每 token 只激活约 3B，用高稀疏 MoE、Gated DeltaNet + Gated Attention 的混合注意力，以及 MTP 来提升长上下文和推理效率。

### 2 分钟版

> 我会先把 Qwen3 和 Qwen3-Next 分开讲。Qwen3 是主线模型家族，核心还是 decoder-only Transformer。它在 block 内用 RMSNorm 做 pre-norm，用 GQA 降低 KV Cache，用 RoPE 做位置编码，用 QK-Norm 稳定 attention logits，用 SwiGLU 或 MoE 做 FFN，并支持 thinking/non-thinking 的统一。
>
> Qwen3-Next 则更像下一代高效架构实验。它面对的是两个扩展压力：总参数越来越大，上下文越来越长。直接用标准 attention 堆 dense 参数会很贵，所以它用高稀疏 MoE 保留大容量，但每个 token 只激活少量专家；同时用 Gated DeltaNet 和 Gated Attention 混合，让大部分层更适合长上下文，又保留部分 attention 的精确交互能力。MTP 还能提供更密集训练信号，并在专门推理框架下帮助加速。

## 常见追问

### Q1：Qwen3-Next 为什么不直接全用标准 attention？

标准 attention 的计算和显存随序列长度增长很快，长上下文成本高。混合 Gated DeltaNet 可以让长上下文建模更高效，但完全放弃 attention 可能损失精确 token 交互，所以采用混合布局。

### Q2：80B-A3B 中 A3B 是什么意思？

大意是 80B 总参数、约 3B 激活参数。MoE 模型不能只看总参数，推理计算更接近每 token 激活参数，但显存和部署仍要考虑总参数存储。

### Q3：Qwen3-Next 和 Qwen3-235B-A22B 谁更强？

不能简单按总参数排序。Qwen3-235B-A22B 总容量更大；Qwen3-Next-80B-A3B 更强调长上下文和效率。官方模型卡称 Qwen3-Next-Instruct 在部分 benchmark 上接近 Qwen3-235B-A22B-Instruct-2507，并在超长上下文任务上有优势。面试中最好说“取决于任务和部署约束”。

### Q4：Qwen3-Next Instruct 会输出 thinking 吗？

官方模型卡明确说明 `Qwen3-Next-80B-A3B-Instruct` 只支持 instruct non-thinking 模式，不生成 `<think></think>` blocks。不要把它和 Thinking 版本混淆。

### Q5：MTP 和 speculative decoding 是一回事吗？

不是一回事。MTP 是模型训练/结构上的多 token 预测能力；speculative decoding 是推理算法。MTP 可以为 speculative decoding 提供 draft tokens，但实际加速取决于推理框架实现。

## 容易踩坑

- 把 Qwen3-Next 当成普通 Qwen3 的一个尺寸变体，而忽略架构变化。
- 只说 80B，不说 active parameters。
- 把 Instruct 版本说成 thinking 模型。
- 把 YaRN 扩展的 1M 上下文说成原生上下文；原生是 262,144 tokens。
- 把高稀疏 MoE 等同于训练一定更容易；实际更需要稳定性优化。
- 说 MTP 在所有 Transformers 推理里都有加速；官方模型卡提示实现依赖框架。

## 复习清单

- Qwen3：decoder-only、dense + MoE、RMSNorm、GQA、RoPE、QK-Norm、SwiGLU、thinking/non-thinking。
- Qwen3-Next：80B total、约 3B active、高稀疏 MoE。
- Qwen3-Next MoE：512 experts、10 activated experts、1 shared expert。
- 混合注意力：Gated DeltaNet + Gated Attention。
- 长上下文：原生 256K，YaRN 可扩展到约 1M。
- MTP：多 token 预测，训练信号更密集，也可配合专门推理框架加速。
- 面试主线：Qwen3 讲“主线架构”，Qwen3-Next 讲“效率架构”。

## 参考资料

1. [Qwen/Qwen3-Next-80B-A3B-Instruct Model Card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) - 官方模型卡；Qwen3-Next 架构与参数来源。
2. [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) - Qwen3 技术报告；Qwen3 主线家族来源。
3. [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) - Qwen3-Next 长上下文扩展相关方法。
4. [[07-Qwen3-模型架构]] - 本地 Qwen3 详细复习笔记。

