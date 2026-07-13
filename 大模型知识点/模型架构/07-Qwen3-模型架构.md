---
tags: [面试, 大模型, Qwen3, MoE, 架构]
created: 2026-05-10
updated: 2026-05-10
sources:
  - https://arxiv.org/abs/2505.09388
  - https://zhuanlan.zhihu.com/p/1902019286836449827
  - https://zhuanlan.zhihu.com/p/1905945819108079268
  - https://zhuanlan.zhihu.com/p/2010381324368769292
  - https://zhuanlan.zhihu.com/p/2023781244639429693
  - https://zhuanlan.zhihu.com/p/1903789599664350106
  - https://zhuanlan.zhihu.com/p/1905664754569185060
---

# Qwen3 模型架构

> 面经问题：**"Qwen3 的模型架构有什么特点？和之前版本相比改了什么？"**

---

## 1. 模型系列概览

Qwen3 提供 **Dense（稠密）** 和 **MoE（混合专家）** 两条产品线：

| 类型 | 型号 | 总参数 | 激活参数 |
|------|------|--------|----------|
| Dense | Qwen3-0.6B | 0.6B | 0.6B |
| Dense | Qwen3-1.7B | 1.7B | 1.7B |
| Dense | Qwen3-4B | 4B | 4B |
| Dense | Qwen3-8B | 8B | 8B |
| Dense | Qwen3-14B | 14B | 14B |
| Dense | Qwen3-32B | 32B | 32B |
| MoE | Qwen3-30B-A3B | ~30B | ~3B |
| MoE | **Qwen3-235B-A22B** | ~235B | ~22B |

![[qwen3-dense-moe-compare.jpg]]
> Dense 就是每个 token 走全部参数；MoE 像有 128 个诊室，每个 token 只进其中 8 个，激活率仅 6.25%。

---

## 2. 整体数据流（Decoder-only）

Qwen3 是 **Decoder-only 因果语言模型**，无 Encoder，无交叉注意力。

![[qwen3-total-structure1.jpg]]
![[qwen3-total-structure2.jpg]]
> 模型由四部分组成：embed_tokens（嵌入层）→ 多个 Decoder Layer 堆叠 → 最终 RMSNorm → LM Head（线性输出层）

```
输入文本
  └─ BBPE Tokenizer (vocab=151,669)
       └─ token ID 序列 [L]
            └─ Embedding 查表 → X ∈ ℝ^(L × d_model)
                 └─ 堆叠 N 个 Transformer Block
                      ├─ Pre-Norm（RMSNorm）
                      ├─ 因果 GQA（含 RoPE + QK-Norm）
                      ├─ 残差
                      ├─ Pre-Norm（RMSNorm）
                      ├─ SwiGLU FFN 或 MoE
                      └─ 残差
                 └─ 最终 RMSNorm
                      └─ LM Head → logits → 下一个 token 概率
```

---

## 3. 单层 Transformer Block 详解

### 3.1 Qwen2.5 结构（对比基准）

![[qwen25-structure.jpg]]
> Qwen2.5-32B 由 64 层组成，Q/K/V Linear **带 Bias**，Attention 后接 Add/RMSNorm，MLP 由 Gate/Up/Down Linear 构成，支持 Sliding Window。

### 3.2 Qwen3 Dense 结构（以 32B 为例）

![[qwen3-dense-structure.jpg]]
> 与 Qwen2.5 相比，最大变化：**Q/K Linear 之后新增 RMS-LayerNorm（即 QK-Norm）**；去掉了 Sliding Window；hidden_size 与 Q-head-num × head-dim 的关系在某些型号略有不同。

单层 Block 数据流：

```
本层输入 h ∈ ℝ^(L × d)
  │
  ├─ [子层1: 注意力]
  │   ├─ RMSNorm(h)                      ← Pre-Norm
  │   ├─ Q/K/V 投影（无 bias）
  │   ├─ QK-Norm（每头独立 RMSNorm）      ← Qwen3 新增
  │   ├─ RoPE 旋转位置编码
  │   ├─ 因果 Mask + Scaled Dot-Product
  │   ├─ 拼头 + O 投影
  │   └─ 残差：h = h + Attn_out
  │
  └─ [子层2: FFN]
      ├─ RMSNorm(h)                      ← Pre-Norm
      ├─ SwiGLU MLP（或 MoE 层）
      └─ 残差：h = h + FFN_out
```

---

## 4. 各组件深入解析

### 4.1 RMSNorm（Root Mean Square Layer Norm）

```math
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \cdot \gamma
```

- 只做**尺度归一化**，去掉了 LayerNorm 的减均值步骤，计算量更小
- Qwen3 用 **Pre-Norm** 结构（先 Norm 再过子层），深层网络训练更稳

---

### 4.2 GQA（Grouped Query Attention）

![[qwen3-gqa-attention.jpg]]
> 以 Qwen3-30B-A3B 为例：hidden_size=2048，Q 头数=32，KV 头数=4，每组 8 个 Q 共享 1 套 KV。逻辑扩展：把 4 个 KV 头各复制 8 份，凑成 32 个与 Q 一一对应。

**MHA / GQA / MQA 三者关系**：

| 机制 | KV 头数 | KV Cache | 速度 | 质量 |
|------|---------|----------|------|------|
| MHA | = Q 头数 | 最大 | 慢 | 最高 |
| GQA | < Q 头数（分组共享） | 中 | 较快 | 接近 MHA |
| MQA | = 1 | 最小 | 最快 | 下降明显 |

以 **Qwen3-32B** 为例：Q=64 头，KV=8 头，group ratio=8:1，KV Cache 仅为 MHA 的 **1/8**。

---

### 4.3 RoPE（旋转位置编码）

RoPE 不把位置加到 Embedding，而是在每层注意力中对 Q、K 施加旋转矩阵：

```math
\text{Attn}(q_m, k_n) = (R_m q_m)^T (R_n k_n) = q_m^T R_{n-m} k_n
```

只有**相对位置 (n-m)** 参与注意力计算。对第 $`i`$ 维，旋转角度为：

```math
\theta_i = \text{base}^{-2i/d}
```

**Qwen3 的 RoPE base 大幅提升（ABF 技术）**：

| 模型 | rope_theta | max_position |
|------|-----------|--------------|
| Qwen2 | 10,000 | 32,768 |
| Qwen3 Base | **1,000,000** | 32,768 |
| Qwen3 Instruct | **5,000,000** | **262,144**（≈256K） |

**为什么 base 越大越好**：base 越大 → θ_i 越小 → Q/K 旋转越慢 → 远距离 token 的角度差异不会饱和 → 长文建模更稳定。

---

### 4.4 QK-Norm（Qwen3 核心新增）

在 Q/K 投影之后、RoPE 之前，对**每个注意力头独立做 RMSNorm**：

```python
Q = self.q_proj(x)          # [B, L, num_heads, head_dim]
K = self.k_proj(x)          # [B, L, num_kv_heads, head_dim]

Q = self.q_norm(Q)          # 每头独立 RMSNorm  ← Qwen3 新增
K = self.k_norm(K)          # 每头独立 RMSNorm  ← Qwen3 新增

# 再做 RoPE，再做注意力
```

**为什么需要 QK-Norm？**

注意力 logit = Q·K^T / √d，若 Q 或 K 模长很大，logit 爆炸，导致：
1. softmax 饱和 → 梯度消失
2. 深层网络 attention entropy collapse（全注意力集中在少数位置）

QK-Norm 直接归一化每个头的 Q/K 向量尺度，从根源防止 logit 爆炸。

**与去掉 QKV Bias 的关系**：Qwen2 用 bias 做了一定尺度补偿；Qwen3 去掉 bias 后，QK-Norm 接替稳定训练的职责，且效果更直接。

---

### 4.5 SwiGLU FFN

```math
\text{FFN}(x) = \text{down\_proj}\bigl(\text{SiLU}(\text{gate\_proj}(x)) \odot \text{up\_proj}(x)\bigr)
```

- gate 和 up 两条支路，逐元素相乘后过 down 投影
- SiLU 激活：$`\text{SiLU}(x) = x \cdot \sigma(x)`$，平滑无死区
- 相比标准 FFN 多一个 gate\_proj，表达能力更强

---

### 4.6 三代对比总表

| 项目 | Qwen 1 | Qwen 2 | Qwen 3 |
|------|--------|--------|--------|
| 注意力 | MHA（部分 GQA） | 全系 GQA | 全系 GQA |
| QKV Bias | 有 | 有 | **无** |
| QK-Norm | 无 | 无 | **有**（每头 RMSNorm） |
| RoPE base | ~10,000 | 1,000,000 | Base: 1M / Instruct: 5M |
| 词表大小 | ~152K | 151,646 | **151,669** |
| 长上下文 | NTK 等推理技巧 | YARN + DCA | ABF + YARN + DCA（三阶段预训练） |
| MoE 设计 | — | 共享专家 + Top-K | **纯 Top-K，无共享专家** |

---

## 5. MoE 架构详解

### 5.1 整体结构

![[qwen3-moe-decoder-layer.jpg]]
> Qwen3-MoE（以 30B-A3B 为例：48 层，hidden_size=2048，128 专家，top-8）。每个 Decoder Layer 由 Self-Attention（GQA）和 MoE Block 两个子模块组成，注意力层与 Dense 完全相同，稀疏性只发生在 FFN 上。

![[qwen3-235b-moe-structure.jpg]]
> Qwen3-235B-A22B 的 MoE 整体结构。Attention 模块与 Dense 几乎一致，唯一区别是 KV Head 数从 8 减为 4。

### 5.2 路由机制详解

#### 简单示例（4 专家选 2）

![[qwen3-moe-routing-simple.jpg]]
> 路由逻辑：token 矩阵 × 路由矩阵 → MoE Logits → Softmax → Top-K 选出专家编号和权重 → 归一化权重（使 k 个权重之和 = 1）

#### Qwen3 完整流程（128 专家选 8）

![[qwen3-moe-routing-full.jpg]]
> 实际 Qwen3 MoE 流程：输入 [seq_len, 2048] → Gate Linear → 128 维 Logits → Softmax → Top-8 → 得到专家编号矩阵和权重矩阵 → 各专家分别做 SwiGLU FFN → 按权重加权求和 → FFN Output

**路由伪代码**：

```python
# Gate（Router）
gate_out = linear(hidden, num_experts=128)    # [seq, 128]
probs    = softmax(gate_out, dim=-1)          # [seq, 128]
weights, experts = topk(probs, k=8, dim=-1)   # [seq, 8]
weights  = weights / weights.sum(-1, keepdim=True)  # 归一化

# 各 Expert 计算（SwiGLU MLP）
output = sum(weights[i] * expert_i(hidden) for i in top8_experts)
```

### 5.3 专家参数配置

| 型号 | 总专家 | 每 token 激活 | 共享专家 | 激活率 |
|------|--------|---------------|----------|--------|
| Qwen3-30B-A3B | 128 | 8 | **无** | 6.25% |
| Qwen3-235B-A22B | 128 | 8 | **无** | 6.25% |
| （对比）Qwen2.5-MoE | 64 | 8 | **有** | — |

### 5.4 为什么去掉共享专家？

Qwen2.5-MoE 设有 1~2 个**共享专家**（所有 token 都经过），用途是提供通用基础知识、充当稳定器。

Qwen3 去掉的理由：
1. **专家数量翻倍**（64 → 128），路由空间足够大，不需要兜底
2. **Global-batch 负载均衡**（见 5.5）替代了共享专家的稳定器职责
3. 共享专家与路由专家产生耦合，影响路由专家的专化程度
4. 纯 Top-K 结构更简洁，GPU 计算效率更高

### 5.5 Global-Batch 负载均衡损失

![[qwen3-lb-problem.jpg]]
> 不加约束时的问题：少数专家初期表现略好 → 得到更多 token → 训练更多 → 进一步变强 → 大多数专家形同虚设（Expert Collapse）。

**解法**：在总 loss 中加辅助均衡项（借鉴 Switch Transformer）：

![[qwen3-lb-formula.jpg]]

```math
L_{\text{aux}} = N_e \cdot \sum_{i=1}^{N_e} f_i \cdot P_i
```

- $`N_e = 128`$：专家总数
- $`f_i`$：专家 $`i`$ 在**整个 global batch** 中被选中的 token 比例
- $`P_i`$：路由器给专家 $`i`$ 的平均概率

**"Global-batch"的关键**：$`f_i`$ 在整个 global batch（多机多 GPU 所有 token）上统计，而非 per-sample 或 per-GPU：
- 统计量更稳定（样本数更多，方差更小）
- 避免单 GPU 样本分布偏斜导致虚假"均衡"
- 让专家在整体数据分布上真正特化

**代码实现**：所有层的 gate_logits 拼接后统一计算一次 loss，跨层跨 GPU 聚合：

```python
# 所有 MoE 层的 gate logits 拼接（dim=0 = token 维度）
concatenated = torch.cat([layer_gate for layer_gate in gate_logits], dim=0)
routing_weights = softmax(concatenated, dim=-1)
_, selected = topk(routing_weights, k=top_k, dim=-1)

expert_mask = one_hot(selected, num_experts)          # [tokens, k, experts]
f_i = expert_mask.float().mean(dim=0)                 # token 分布
P_i = routing_weights.mean(dim=0)                     # 概率分布

L_aux = num_experts * (f_i * P_i.unsqueeze(0)).sum()
```

### 5.6 层配置灵活性

```python
# 每一层灵活选择 MoE 或普通 MLP
if (layer_idx not in config.mlp_only_layers) and \
   (config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0):
    self.mlp = SparseMoeBlock(config)  # MoE 层
else:
    self.mlp = Qwen3MoeMLP(config)    # 普通 Dense MLP
```

`decoder_sparse_step=1` 表示每层都是 MoE（默认），可通过 `mlp_only_layers` 指定某些层用普通 MLP。

---

## 6. 长上下文方案

### 6.1 三阶段预训练

| 阶段 | 数据量 | 序列长度 | 目标 |
|------|--------|----------|------|
| S1 通用 | ~30T tokens，119 语言 | 4,096 | 通用知识 + 多语言基础 |
| S2 推理 | ~5T，增 STEM/代码/合成比例 | 4,096 | 增强推理能力 |
| S3 长上下文 | 数百亿，长文数据（75% 在 16K~32K） | **32,768** | 支持 32K 上下文 |

### 6.2 ABF（Attention Base Frequency）

直接把 RoPE 的 `base` 从 10,000 调大到 **1,000,000**。

本质：让各频率分量旋转变慢，token 之间的相对角度在更长距离内不会饱和，模型在 S3 训练时能真正感知 32K 范围内的位置关系。

### 6.3 YaRN（Yet Another RoPE extensioN）

**问题**：训练 32K，推理时想支持 128K+，直接外推 RoPE 模型从未见过这么大的旋转角度，性能急剧下降。

**YaRN 思路**：对不同频率的 RoPE 分量做**差异化插值**：
- 高频分量（旋转快）：做位置插值（拉伸）
- 低频分量（旋转慢）：保持不变，保留远程关系

推理时 `position_ids` 按比例缩放，让模型感受到的"位置密度"与训练时一致。

**结果**：推理时序列长度可扩展到约 **4× 训练窗口**（32K → ~128K）。

### 6.4 DCA（Dual Chunk Attention）

**问题**：YaRN 在中等长度（几倍训练长度）效果好；到更极端长度时，插值带来的位置漂移让注意力质量下降。

**DCA 思路**：将长序列切成 Chunk，**块内完整注意力 + 块间稀疏注意力**：

```
序列:  [Chunk1 | Chunk2 | Chunk3 | ... | ChunkN]
          ↓完整 attn   ↓稀疏 cross-chunk attn
         局部精确          全局稀疏感知
最后一个 token 额外做全局 global attention
```

**YaRN + DCA 分工**：
- YaRN 解决"训练短/推理长"的**位置外推**问题
- DCA 解决超长序列下**注意力质量和计算复杂度**问题

两者作用于不同维度，叠加使用，Instruct 版最终支持 **262K tokens**。

---

## 7. Thinking Mode（思考模式）

Thinking Mode **不是新增的神经网络层**，是**后训练（post-training）+ 聊天模板**共同塑造的行为。

### 7.1 四阶段后训练

```
阶段 1：Long-CoT 冷启动 SFT
  └─ 用带长推理链的数学/代码数据微调，建立 <think>...</think> 格式

阶段 2：专注推理的 RL（强化学习）
  └─ 以数学/编程为主，奖励基于可验证正确性（答案对/代码过测试）
     类似 DeepSeek-R1 的 GRPO 路线，不依赖奖励模型打分

阶段 3：思考 + 非思考混合 SFT
  └─ 有/无推理路径的数据合并，让模型学会按需思考

阶段 4：通用领域 RL
  └─ 所有下游任务 RL，加入人类偏好信号，提升泛化性
```

对小模型，额外使用**强到弱蒸馏**（大模型 on-policy/off-policy 知识迁移）替代 RL，效果更好。

### 7.2 生成格式

```
<think>
[推理过程，token 数可受 Thinking Budget 限制]
</think>
[最终答案]
```

关闭思考时，`<think>` 块为空或完全不生成。

### 7.3 软切换（Soft Switch）

在 `enable_thinking=True` 时，用户可在 prompt 中动态控制：

```
/think    → 当轮开启推理（<think> 块有内容）
/no_think → 当轮跳过推理（<think></think> 为空）
```

多轮对话可逐轮切换，无需换模型。

### 7.4 硬切换（Hard Switch）

```python
# 完全禁用思考，行为等同 Qwen2.5-Instruct
outputs = model.generate(**inputs, enable_thinking=False)
# 此时 /think /no_think 软切换标签失效
```

### 7.5 Thinking Budget

用户可指定 `<think>` 块的最大 token 数，模型自适应分配计算资源：
- 简单问题：budget 低，快速响应
- 复杂推理：budget 高，深度思考
- 实验发现：思考 token 数增加，各类任务性能**单调提升**

---

## 8. 参数配置表（来自 Technical Report）

### Dense 模型

![[qwen3-dense-params-table.jpg]]

| 型号 | Layers | Hidden | Q heads | KV heads | FFN dim | Context |
|------|--------|--------|---------|----------|---------|---------|
| 0.6B | 28 | 1024 | 16 | 8 | 2816 | 32K |
| 1.7B | 28 | 2048 | 16 | 8 | 11008 | 32K |
| 4B | 36 | 2560 | 32 | 8 | 6912 | 32K |
| 8B | 36 | 4096 | 32 | 8 | 22016 | 32K |
| 14B | 40 | 5120 | 40 | 8 | 17920 | 32K |
| 32B | 64 | 5120 | 64 | 8 | 25600 | 32K |

### MoE 模型

![[qwen3-moe-params-table.jpg]]

| 型号 | Layers | Hidden | Q heads | KV heads | 总专家 | 激活专家 |
|------|--------|--------|---------|----------|--------|----------|
| 30B-A3B | 48 | 2048 | 32 | 4 | 128 | 8 |
| 235B-A22B | 94 | 7168 | 64 | 4 | 128 | 8 |

---

## 9. 常见面试追问

**Q1：QK-Norm 加在哪里？和 Pre-Norm 的 RMSNorm 有什么区别？**

> Pre-Norm 的 RMSNorm 作用于**整个子层的输入**（对全维 hidden state 归一化），粒度是 layer 级；QK-Norm 作用于注意力内部的**每个头的 Q/K 向量**，粒度是 head 级。两者互不替代：Pre-Norm 防止残差链上的尺度漂移，QK-Norm 防止注意力 logit 爆炸。

**Q2：为什么 Qwen3-MoE 去掉共享专家，而 Qwen3.5-MoE 又加回来了？**

> Qwen3 去掉是因为 128 专家 + Global-batch 均衡足够稳定。Qwen3.5-MoE 是多模态模型，视觉和文本 token 的分布差异很大，路由器压力更高，共享专家重新充当"通用基础能力"的缓冲层。

**Q3：RoPE base 从 10k 提到 1M，推理时还需要 YaRN 吗？为什么？**

> 需要。ABF（提升 base）解决"训练时位置角度饱和"的问题，让 32K 内的相对位置编码更稳定；但推理时序列超过训练长度（32K → 128K+），模型依然遇到从未见过的绝对位置值，需要 YaRN 插值对齐分布。两个技术解决不同问题。

**Q4：Global-batch 负载均衡和 per-sample 均衡有什么实质差异？**

> per-sample 在单条样本内统计专家负载，统计量不稳定且一条样本未必覆盖所有专家；Global-batch 在几千甚至几万 token 上统计，能真实反映专家在整个数据分布上的利用率，从而给出更准确的惩罚信号，鼓励专家在语料层面特化。

**Q5：Qwen3 的 Thinking Mode RL 用的什么奖励信号？**

> 第二阶段 RL 聚焦数学和编程，奖励基于**可验证的正确性**（数学答案对/错、代码能否过测试用例），类似 DeepSeek-R1 的 GRPO 路线，不依赖奖励模型打分，降低奖励 hacking 风险。第四阶段通用 RL 再引入人类偏好信号。

**Q6：Dense 和 MoE 在推理时怎么选？**

> Dense：全部参数参与每次 forward，延迟低，适合显存充裕、追求低延迟的场景。MoE：激活参数仅 ~6%，decode 吞吐更高，但需要加载全部专家权重（显存要求不低），路由 overhead 在 batch 极小时不划算。高吞吐推理选 MoE，延迟敏感的在线服务选 Dense。

---

## 参考

- [Qwen3 Technical Report (arXiv:2505.09388)](https://arxiv.org/abs/2505.09388)
- [知乎：Qwen3 模型结构（骑虎南下）](https://zhuanlan.zhihu.com/p/1902019286836449827)
- [知乎：Qwen3 技术报告全文（吕阿华）](https://zhuanlan.zhihu.com/p/1905945819108079268)
- [知乎：Qwen3-MoE vs Qwen3.5-MoE 架构深度解析（还是菜）](https://zhuanlan.zhihu.com/p/2010381324368769292)
- [知乎：Transformer 20. Qwen3 架构介绍（小超同学）](https://zhuanlan.zhihu.com/p/2023781244639429693)
- [知乎：Qwen3 MoE 负载均衡解析（GenerAI）](https://zhuanlan.zhihu.com/p/1903789599664350106)
- [知乎：图解 Qwen3 MoE 推理过程（黄向平）](https://zhuanlan.zhihu.com/p/1905664754569185060)
