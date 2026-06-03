---
tags:
  - 面试
  - 大模型
  - 注意力机制
  - GQA
  - KV Cache
created: 2026-05-11
updated: 2026-05-11
---

# GQA：分组查询注意力

> 核心问题：为什么 LLM 不用 MHA 而用 GQA？GQA 和 MQA 的区别是什么？这和 KV Cache 有什么关系？

---

## 0. 问题起源：MHA 的推理瓶颈

### 0.1 推理时发生了什么

训练时可以并行看到整个序列。但推理（生成）是**逐 token 自回归**的：

```
Step 1: 输入 "今天" → 输出 "天气"
Step 2: 输入 "今天天气" → 输出 "很"
Step 3: 输入 "今天天气很" → 输出 "好"
...
```

### 0.2 KV Cache 是什么

每生成一个新 token，都需要和之前**所有**历史 token 做 Attention。

如果每步重算一次所有历史 token 的 K 和 V，计算量会爆炸（$O(T^2)$ 每一步都重算）。

**KV Cache 的做法**：把算过的 K、V 存起来，新 token 只算自己的 Q、K、V，然后 Q 和所有缓存的 K 算 Attention。

```
Step 1: 算 K₁, V₁ → 缓存 [K₁], [V₁]
Step 2: 算 Q₂, K₂, V₂；Q₂ 和 [K₁, K₂] 做 Attention → 缓存 [K₁, K₂], [V₁, V₂]
Step 3: 算 Q₃, K₃, V₃；Q₃ 和 [K₁, K₂, K₃] 做 Attention → ...
```

### 0.3 KV Cache 的内存占用

在 MHA 中，**每个头**都有独立的 $W^K$ 和 $W^V$，所以每个 token 要存 $h$ 套 K、V。

每层的 KV Cache 大小：

$$
2 \times \text{batch} \times h \times T \times d_k
$$

一个例子：LLaMA-7B，32 层，32 头，$d_k=128$，序列长度 2048，batch=1：

$$
2 \times 32 \times 32 \times 2048 \times 128 \times 2\text{ bytes (FP16)} \approx 1\text{ GB}
$$

**序列越长，KV Cache 越大。** 这就是 MQA/GQA 要解决的问题。

---

## 1. 三种注意力机制对比

一张图看懂区别：

```
MHA (Multi-Head Attention)    MQA (Multi-Query Attention)    GQA (Grouped-Query Attention)
─────────────────────────     ─────────────────────────      ──────────────────────────
Q: ■ ■ ■ ■  (每个头独立)     Q: ■ ■ ■ ■  (每个头独立)      Q: ■ ■ ■ ■  (每个头独立)
K: ■ ■ ■ ■  (每个头独立)     K: ■ . . .  (所有头共享)      K: ■ ■ . .  (分组共享)
V: ■ ■ ■ ■  (每个头独立)     V: ■ . . .  (所有头共享)      V: ■ ■ . .  (分组共享)
```

### MHA（多头注意力）

- $h$ 个 Q 头，$h$ 个 K 头，$h$ 个 V 头
- 每个头完全独立
- KV 参数：$h \times d_k$ 个
- KV Cache 最大，表示能力最强

### MQA（多查询注意力）

- $h$ 个 Q 头，**1** 个 K 头，**1** 个 V 头
- 所有头共享同一套 K、V
- KV 参数：$1 \times d_k$ 个
- KV Cache 最小（$1/h$），但表示能力下降明显

### GQA（分组查询注意力）

- $h$ 个 Q 头，**g** 个 K 头，**g** 个 V 头（$1 < g < h$）
- Q 头分成 g 组，每组共享一套 K、V
- KV 参数：$g \times d_k$ 个
- **MHA 和 MQA 之间的折中方案**

---

## 2. GQA 详解

### 2.1 分组逻辑

假设 $h=8$，$g=4$（即 4 个 KV 组）：

```
Q heads:  Q₀  Q₁  Q₂  Q₃  Q₄  Q₅  Q₆  Q₇
           │   │   │   │   │   │   │   │
Groups:  └─┬─┘  └─┬─┘  └─┬─┘  └─┬─┘
           │      │      │      │
KV:       KV₀    KV₁    KV₂    KV₃
```

- Q₀、Q₁ 共享 KV₀（组 0）
- Q₂、Q₃ 共享 KV₁（组 1）
- Q₄、Q₅ 共享 KV₂（组 2）
- Q₆、Q₇ 共享 KV₃（组 3）

**每个组内的 Q 头使用相同的 K 和 V，但各自投影的 Q 不同**，仍然可以关注不同的模式。

### 2.2 参数对比

| | MHA | GQA (g=4) | MQA |
|---|---|---|---|
| Q 参数 | $h \times d_k$ | $h \times d_k$ | $h \times d_k$ |
| K 参数 | $h \times d_k$ | $g \times d_k$ | $1 \times d_k$ |
| V 参数 | $h \times d_k$ | $g \times d_k$ | $1 \times d_k$ |
| KV Cache 大小 | 100%（基准） | $g/h = 50\%$（当 g=4, h=8） | $1/h = 12.5\%$ |
| 表示能力 | 最强 | 中等 | 较弱 |

### 2.3 GQA 在性能-效率上的权衡

GQA 论文中的实验结论：

- GQA 性能仅比 MHA 略低，但时间开销远小于 MHA
- GQA 时间开销只比 MQA 略高，但性能明显好于 MQA
- 相比直接减少头数（不改变机制），GQA 在相同参数缩减下效果更好

---

## 3. PyTorch 实现

```python
import torch
import torch.nn as nn
import math


class GroupedQueryAttention(nn.Module):
    """
    GQA: 将 Q 头分成 num_kv_groups 组，每组共享一套 K、V。

    当 num_kv_groups == num_heads  → 退化为 MHA
    当 num_kv_groups == 1          → 退化为 MQA
    """

    def __init__(self, d_model, num_heads, num_kv_groups, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_kv_groups == 0  # 必须能整除

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.d_k = d_model // num_heads
        self.heads_per_group = num_heads // num_kv_groups  # 每组几个 Q 头

        # Q 投影：仍然每个头独立
        self.W_q = nn.Linear(d_model, d_model)

        # K, V 投影：只有 num_kv_groups 套
        self.W_k = nn.Linear(d_model, num_kv_groups * self.d_k)
        self.W_v = nn.Linear(d_model, num_kv_groups * self.d_k)

        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, D = x.shape

        # Q: [B, T, D] → [B, h, T, d_k]
        Q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # K, V: [B, T, D] → [B, g, T, d_k]
        K = self.W_k(x).view(B, T, self.num_kv_groups, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_kv_groups, self.d_k).transpose(1, 2)

        # 关键步骤：将 K, V 从 [B, g, T, d_k] 扩展到 [B, h, T, d_k]
        # 每组内的 Q 头共享同一套 K, V
        K = K.unsqueeze(2).expand(B, self.num_kv_groups, self.heads_per_group, T, self.d_k)
        K = K.reshape(B, self.num_heads, T, self.d_k)

        V = V.unsqueeze(2).expand(B, self.num_kv_groups, self.heads_per_group, T, self.d_k)
        V = V.reshape(B, self.num_heads, T, self.d_k)

        # 缩放点积注意力（和 MHA 完全一样）
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.dropout(torch.softmax(scores, dim=-1))

        out = attn @ V  # [B, h, T, d_k]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)
```

---

## 4. 演进路线总结

```
MHA (Transformer, 2017)
 │  问题：推理时 KV Cache 太大
 │
 ├→ MQA (2019)
 │    方案：所有头共享一套 K、V
 │    代价：表示能力下降明显
 │
 └→ GQA (2023)
      方案：分组共享 K、V（在 MHA 和 MQA 之间折中）
      效果：接近 MHA 的性能，大幅降低 KV Cache
```

### 各模型采用的注意力机制

| 模型 | 注意力机制 |
|---|---|
| GPT-3, BERT | MHA |
| PaLM, Falcon | MQA |
| LLaMA 2/3, Mistral, Qwen | GQA |
| DeepSeek-V2/V3 | MLA（更激进的优化） |

---

## 5. 面试高频追问

### Q1: GQA 为什么能减少内存但对性能影响不大？

不同 Q 头关注不同模式，但它们共享的 K、V 来自同一个 token 表示，本身就有一定重叠。分组共享让相近的 Q 头使用相同的 K、V，相当于"相近模式共用一套标签"，对多数场景影响不大。

### Q2: MQA 为什么不够好？

一个 token 只有一套 K、V 来匹配所有 Q 头（可能 32 个甚至更多），K、V 被迫同时满足多套 Q 的查询需求，表达能力受限。GQA 提供了更平滑的折中——多套 K、V 可以分组匹配不同的 Q。

### Q3: GQA 的 num_kv_groups 怎么选？

- 越大越接近 MHA（性能好、内存大）
- 越小越接近 MQA（性能差、内存小）
- LLaMA 2 70B 用 $g=8$（$h=64$），LLaMA 3 8B 用 $g=4$（$h=32$）
- 一般取 $g = h/4$ 到 $h/8$

### Q4: GQA 训练和推理各有什么优势？

- 训练：K、V 参数少，显存占用小
- 推理：KV Cache 小 $g/h$ 倍，可支持更长上下文或更大 batch

---

## 参考资料

- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, EMNLP 2023
- Shazeer, *Fast Transformer Decoding: One Write-Head Is All You Need* (MQA), 2019
- [【LLM】一文详解 MHA、GQA、MQA 原理](https://zhuanlan.zhihu.com/p/1890128241)
- [GQA 分组查询注意力学习笔记与代码实现](https://zhuanlan.zhihu.com/p/1924817502799631379)
- [极简代码复现 MHA、MQA、GQA——面试终结者](https://zhuanlan.zhihu.com/p/1905369792745017821)
