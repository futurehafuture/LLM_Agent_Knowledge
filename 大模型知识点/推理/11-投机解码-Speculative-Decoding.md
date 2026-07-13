---
tags: [大模型, 推理, Speculative-Decoding, Medusa, EAGLE]
created: 2026-07-13
updated: 2026-07-13
---

# 投机解码：Speculative Decoding

> 核心目标：用便宜方法一次提出多个候选 token，再让大模型用一次并行 forward 验证，减少昂贵的大模型串行调用次数，同时保持目标分布不变。

## 1. 为什么能加速

普通自回归每生成一个 token 都调用一次目标模型。若 draft 一次提出 $`k`$ 个 token：

```text
Draft:  a1 a2 a3 a4
Target: 一次并行验证这 4 个位置
```

若平均接受 $`A`$ 个 token，目标模型每次调用推进约 $`A+1`$ 个位置。是否更快取决于 draft 成本、验证成本和接受长度。

## 2. Draft 与 Target

- 目标分布 $`p(x_t\mid x_{<t})`$：最终必须保持；
- draft 分布 $`q(x_t\mid x_{<t})`$：更便宜，用于提出候选。

Draft 可以是小模型、n-gram、额外预测头、特征预测模型或目标模型自身的 MTP 头。

## 3. 接受—拒绝规则

对 draft 采样到的 token $`x`$，接受概率为：

```math
\alpha(x)=\min\left(1,\frac{p(x)}{q(x)}\right).
```

若拒绝，不能简单改成 target 的普通采样，否则会破坏整体分布。应从修正分布采样：

```math
p'(x)=\frac{\max(0,p(x)-q(x))}{\sum_y\max(0,p(y)-q(y))}.
```

这保证最终样本仍严格来自 $`p`$。

## 4. 为什么一次能验证多个位置

Draft 已给出完整候选前缀，因此目标模型可以像 prefill 一样并行计算这些位置的 logits。验证仍使用 causal mask，每个位置只能看到候选中它之前的 token。

如果第 $`j`$ 个候选被拒绝，后面的候选全部失效，因为它们建立在错误前缀上。

## 5. 性能模型

设目标模型一次验证成本 $`C_T(k)`$、draft 成本 $`C_D(k)`$、平均推进 $`L`$ 个 token：

```math
T_{per\ token}\approx\frac{C_D(k)+C_T(k)}{L}.
```

候选越多，可能推进更多，但验证张量更大且后部候选接受率下降。最佳 $`k`$ 随模型、batch、任务和硬件变化。

## 6. 方法演进

### Draft Model

用同 tokenizer 的小模型逐 token 草拟。实现直观，但小模型也有权重读取和调度成本。

### n-gram / Prompt Lookup

从 prompt 或已生成文本中匹配重复片段。没有额外模型，代码、模板和重复性文本中效果好，开放式生成命中有限。

### Medusa

在目标模型顶部训练多个解码头，分别预测未来不同位置；用树结构组织候选并通过 tree attention 验证。

### EAGLE 系列

EAGLE 利用目标模型中间特征训练轻量 draft。EAGLE-3 改为直接 token prediction，并融合多层特征，通过 training-time test 提高 draft 的可扩展性。

### MTP

模型训练时自带 Multi-Token Prediction 模块，推理时可作为 draft。优点是与目标模型联合训练，缺点是需要模型架构支持。

## 7. Tree Speculation

单链只有一个候选路径。Tree 方法为不确定位置保留多个分支，提高至少一条路径命中的概率，但会增加验证 token 数、mask 复杂度和 KV 管理开销。

## 8. 什么时候收益最大

- batch 较小、目标模型 decode 未充分利用 GPU；
- draft 与 target 分布接近；
- 代码、固定格式或低温度生成可预测性高；
- 验证 kernel 与 KV 提交实现高效。

大 batch 下目标模型已经接近高利用率，额外 draft 工作可能使吞吐下降。

## 9. KV Cache 提交与回滚

验证候选时会临时产生多位置 KV。只允许把已接受 token 的 KV 提交到主序列；被拒绝位置及其后缀必须丢弃。Paged KV 管理可以用临时 block、引用计数和 block 回收完成。

## 10. 评测指标

- acceptance rate；
- accepted length；
- target calls/token；
- 单请求 TPOT；
- 聚合 throughput；
- draft 与验证占用的额外显存；
- 与非投机采样的分布一致性。

## 11. 常见误区

1. 投机解码不是让小模型决定最终答案。
2. 严格算法不会降低模型质量；近似变体需要单独评估。
3. 接受率高不等于端到端加速高。
4. 论文中的 batch=1 加速不能直接外推到高并发服务。

## 12. 面试回答

> Draft 模型先提出多个 token，Target 用一次并行 forward 验证。候选按 $`\min(1,p/q)`$ 接受，拒绝时从 $`(p-q)_+`$ 归一化后的修正分布采样，因此最终分布仍是 Target 分布。加速取决于平均接受长度是否覆盖 draft、树验证和 KV 管理成本，小 batch 通常更受益。

## 参考资料

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Accelerating LLM Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- [Medusa](https://arxiv.org/abs/2401.10774)
- [EAGLE-3](https://arxiv.org/abs/2503.01840)
- [vLLM Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/)
