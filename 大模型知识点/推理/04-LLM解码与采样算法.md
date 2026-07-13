---
tags: [大模型, 推理, Sampling, Top-k, Top-p, Beam-Search]
created: 2026-07-13
updated: 2026-07-13
---

# LLM 解码与采样算法

> 推理不只是一遍 forward。模型输出 logits 后，如何选择 token 决定了确定性、多样性、质量和一部分系统开销。

## 1. 从 logits 到概率

模型在词表 $`V`$ 上输出 logits $`z_i`$。温度为 $`\tau`$ 时：

```math
p_i=\frac{\exp(z_i/\tau)}{\sum_{j\in V}\exp(z_j/\tau)}.
```

- $`\tau<1`$：分布更尖锐；
- $`\tau>1`$：分布更平坦；
- $`\tau\to0`$：趋近 greedy。

工程实现会先减去最大 logit，避免指数溢出。

## 2. Greedy Decoding

```math
x_{t+1}=\arg\max_i z_i.
```

优点是确定、便宜；缺点是局部最优，容易产生重复或缺乏多样性。代码补全、格式化输出经常适用，创作类任务不一定适用。

## 3. Top-k Sampling

只保留概率最大的 $`k`$ 个 token，再归一化采样：

```math
S_k=\operatorname{TopK}(p,k),\qquad
\tilde p_i=\frac{p_i\mathbf1[i\in S_k]}{\sum_{j\in S_k}p_j}.
```

固定 $`k`$ 不适应分布形状：模型很确定时可能保留太多噪声，很不确定时又可能删除合理候选。

## 4. Top-p / Nucleus Sampling

将 token 按概率降序，选择累计概率首次达到 $`p`$ 的最小集合：

```math
S_p=\min\left\{S:\sum_{i\in S}p_i\ge p\right\}.
```

候选集合大小随上下文变化，比固定 top-k 更自适应。Top-k 和 top-p 可以同时应用。

## 5. Min-p 与 Typical Sampling

- **Min-p**：保留 $`p_i\ge p_{min}\cdot p_{max}`$ 的 token，阈值随最可能 token 调整。
- **Typical sampling**：保留信息量接近当前分布熵的 token，避免极端意外或过度平凡的候选。

这些方法本质上都在解决“如何截断长尾但不误删合理候选”。

## 6. 重复与长度控制

### 6.1 Repetition penalty

对已出现 token 的 logits 做缩放。它是启发式规则，不等价于概率模型中的严格约束。

### 6.2 Frequency / presence penalty

```math
z_i'=z_i-\lambda_f c_i-\lambda_p\mathbf1[c_i>0],
```

其中 $`c_i`$ 是 token 已出现次数。frequency penalty 随次数增加，presence penalty 只关心是否出现过。

### 6.3 Stop condition

EOS、stop strings、最大 token 数和结构化语法都可以结束生成。字符串停止条件可能跨 token 边界，不能只检查最新 token。

## 7. Beam Search

Beam Search 每一步保留累计分数最好的 $`B`$ 条序列：

```math
s(x_{1:t})=\sum_{j=1}^{t}\log p(x_j\mid x_{<j}).
```

长序列累计 log probability 更负，因此常加入长度归一化。Beam Search 适合翻译等较确定任务，但会放大 KV Cache 和候选管理成本；开放式对话中未必优于采样。

## 8. 采样的系统开销

大词表下，softmax、top-k、排序和随机采样会成为 decode 尾部的小 kernel。Batch 较大或模型较小时，这部分比例会上升。优化手段包括：

- fused logits processing；
- GPU 上完成 top-k/top-p；
- 避免把完整 logits 拷回 CPU；
- 只返回 top-logprobs，而非整个词表。

## 9. 随机性与可复现

固定随机种子仍不一定获得跨硬件、跨 batch 的完全相同输出，因为并行归约、浮点舍入和调度顺序可能不同。生产接口应把“尽力确定”与“位级确定”区分开。

## 10. 参数选择思路

| 场景 | 常见倾向 |
|---|---|
| 数学/代码验证 | 低温度、greedy 或小 top-p |
| 通用对话 | 中等温度与 top-p |
| 创意生成 | 较高温度，保留更宽候选集 |
| 多样本搜索 | 多次采样，再由 verifier/reranker 选择 |
| 严格 JSON | 约束解码，而不是只降低温度 |

## 11. 面试回答

> Temperature 改变整个分布的尖锐程度；top-k 固定保留 k 个候选；top-p 保留累计概率达到阈值的最小集合。Beam Search 维护多条高分序列，成本和 KV Cache 近似随 beam 数增长。严格结构输出应使用 FSM/CFG 约束，而不是期待采样参数碰巧生成合法格式。

## 参考资料

- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
- [Hugging Face Generation Strategies](https://huggingface.co/docs/transformers/main/generation_strategies)
- [vLLM Sampling Parameters](https://docs.vllm.ai/en/latest/api/vllm/sampling_params.html)
