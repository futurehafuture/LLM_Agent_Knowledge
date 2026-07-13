---
tags:
  - 面试
  - 大模型
  - 位置编码
  - RoPE
created: 2026-05-11
updated: 2026-07-13
---

# RoPE：旋转位置编码

> 核心问题：为什么需要位置编码？RoPE 为什么比 sin/cos 更好？旋转到底是什么意思？base 参数控制什么？

---

## 0. 问题的起点：Attention 看不到顺序

### 0.1 Attention 是位置无关的

先验证一件事：Attention 本身**不知道 token 的顺序**。

假设两个 token A、B 的 embedding 分别是 $[1, 0]$ 和 $[0, 1]$，$W_Q=W_K=W_V=I$：

- 输入 `[A, B]` → 注意力分数矩阵：

  ```math
  \begin{bmatrix}
  1 & 0 \\
  0 & 1
  \end{bmatrix}
  ```

- 输入 `[B, A]` → 注意力分数矩阵：

  ```math
  \begin{bmatrix}
  1 & 0 \\
  0 & 1
  \end{bmatrix}
  ```

**完全一样！** 顺序换了，每个 token 的输出向量毫无变化。这就是置换不变性。

**如果不加位置信息，模型无法区分"狗咬人"和"人咬狗"。**

---

## 1. 位置编码的演进

### 1.1 绝对位置编码（Sinusoidal PE）

Transformer 论文最早使用的方案——给每个位置生成一个固定的向量，直接加到词嵌入上：

```math
\begin{cases}
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right) \\
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
\end{cases}
```

- $pos$：token 在序列中的位置
- $i$：维度索引（0 到 $d/2-1$）
- $d$：embedding 维度

**为什么用不同频率？** 不同维度的波长不一样——低维维度变化慢（波长长，适合长距离），高维维度变化快（波长短，适合精细位置区分）。这样每个位置产生一个唯一的编码向量。

**问题**：
1. 和词嵌入**直接相加**，污染了语义信息
2. 相对位置仅通过内积隐式表达，没有显式约束
3. 超出训练长度后编码分布偏移（外推能力弱）

### 1.2 可学习位置编码

用一个可学习的 Embedding 表 $PE[pos]$，每个位置查表。BERT、GPT-2 用这种方法。

**问题**：训练时见过的最大位置是固定的（比如 512），推理时遇到更长的序列就完全不知道怎么办。

---

## 2. RoPE 的核心思想

### 2.1 目标：把位置信息注入 Attention，且只依赖相对位置

我们希望设计一种编码，使得 Attention 分数**天然地只依赖相对位置**：

```math
\text{Attention}(Q_m, K_n) = f(Q, K, m-n)
```

也就是说，经过编码后的 $Q_m$（位置 m 的查询）和 $K_n$（位置 n 的键）做内积时，结果只和它们的相对距离 $(m-n)$ 有关。

### 2.2 从 2D 开始理解：旋转

想象每个 token 的 Q、K 向量是 2D 平面上的一个点。RoPE 的做法是：

> **根据 token 的位置，把 Q 和 K 旋转相应的角度。**

- 位置 $m$ 的 Q 旋转角度 $m\theta$
- 位置 $n$ 的 K 旋转角度 $n\theta$

旋转矩阵：

```math
R(\theta) = \begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
```

编码后的 Q 和 K：

```math
Q'_m = R(m\theta) \cdot Q_m,\quad K'_n = R(n\theta) \cdot K_n
```

### 2.3 为什么旋转可以实现"只依赖相对位置"？

计算编码后的内积：

```math
\begin{aligned}
Q'_m \cdot K'_n &= (R(m\theta) Q_m)^T \cdot (R(n\theta) K_n) \\
               &= Q_m^T R(m\theta)^T R(n\theta) K_n \\
               &= Q_m^T R(-m\theta) R(n\theta) K_n \quad \text{（旋转矩阵的转置 = 反向旋转）} \\
               &= Q_m^T R((n-m)\theta) K_n \quad \text{（旋转可叠加）}
\end{aligned}
```

**只依赖 $(n-m)$！** 这就是 RoPE 的数学核心。

> 旋转矩阵的优雅性质：$R(a)^T R(b) = R(b-a)$，把绝对位置 $m$ 和 $n$ 消掉了，只留下差值。

---

## 3. 从 2D 推广到高维

### 3.1 分组旋转

一个 $d$ 维向量无法直接做 2D 旋转。但可以把 $d$ 维拆成 $d/2$ 对，每一对在各自的 2D 平面内独立旋转。

```math
R_{\Theta,m}^d = \begin{bmatrix}
\cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots \\
\sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \cdots \\
0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots \\
0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{bmatrix}
```

这是一个**分块对角矩阵**，沿对角线排列 $d/2$ 个 $2 \times 2$ 的旋转块。

### 3.2 不同频率

每一对使用不同的旋转频率 $\theta_i$：

```math
\theta_i = \frac{1}{\text{base}^{2i/d}},\quad i = 0, 1, \dots, d/2-1
```

| 维度索引 $i$ | 频率 $\theta_i$ | 效果 |
|---|---|---|
| 小（低维） | 大 | 旋转快，对短距离位置敏感（区分相邻 token） |
| 大（高维） | 小 | 旋转慢，对长距离位置敏感（保持远程依赖） |

> base 默认值 10000（来自最初的 Transformer Sinusoidal PE），LLaMA 等模型继续沿用。

---

## 4. 具体计算步骤

对于位置 $m$ 的向量 $\mathbf{x} \in \mathbb{R}^d$：

```
输入: x = [x₀, x₁, x₂, x₃, ...]

第 0 对: (x₀, x₁), 旋转角度 m·θ₀
  x₀' = cos(mθ₀)·x₀ - sin(mθ₀)·x₁
  x₁' = sin(mθ₀)·x₀ + cos(mθ₀)·x₁

第 1 对: (x₂, x₃), 旋转角度 m·θ₁
  x₂' = cos(mθ₁)·x₂ - sin(mθ₁)·x₃
  x₃' = sin(mθ₁)·x₂ + cos(mθ₁)·x₃
  ...

输出: x' = [x₀', x₁', x₂', x₃', ...]
```

### PyTorch 实现

```python
import torch


def precompute_freqs_cis(dim, max_seq_len, base=10000.0):
    """
    预计算旋转角度的 cos 和 sin 值。
    返回形状: [max_seq_len, dim/2, 2]，最后一维是 (cos, sin)
    """
    # 计算频率: θ_i = 1 / base^(2i/d)
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    # 计算每个位置的旋转角度: [max_seq_len, dim/2]
    t = torch.arange(max_seq_len).float()
    angles = torch.outer(t, freqs)  # m × θ_i
    # 返回 cos 和 sin
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)


def apply_rotary_emb(x, freqs_cis):
    """
    对输入 x 应用 RoPE。
    x: [batch, seq_len, dim]
    freqs_cis: [seq_len, dim/2, 2]
    """
    # 把 x 的相邻两个维度配对: [batch, seq_len, dim/2, 2]
    x_paired = x.float().reshape(*x.shape[:-1], -1, 2)

    cos, sin = freqs_cis[..., 0], freqs_cis[..., 1]

    # 旋转: x' = (cosθ, sinθ) ⊗ (x_even, x_odd)
    x_rotated = torch.stack([
        x_paired[..., 0] * cos - x_paired[..., 1] * sin,
        x_paired[..., 0] * sin + x_paired[..., 1] * cos,
    ], dim=-1)

    return x_rotated.flatten(-2)  # 恢复原始维度
```

---

## 5. RoPE 的优越性

| | Sinusoidal PE | 可学习 PE | RoPE |
|---|---|---|---|
| 位置注入方式 | 加到 embedding | 加到 embedding | 旋转 Q、K（乘法融入） |
| 相对位置 | 隐式（内积近似） | 无 | **显式（数学可证）** |
| 对语义的干扰 | 有（直接相加） | 有（直接相加） | 无（正交旋转，保范数） |
| 外推能力 | 弱 | 无 | 较强（旋转规则统一） |
| 额外参数 | 无 | $max_len \times d$ | 无 |
| 计算开销 | 极小 | 极小 | 极小 |

**RoPE 的三个核心优势**：

1. **语义与位置解耦**：旋转是正交变换，保向量模长和内积，不对语义信息产生扭曲
2. **显式相对位置**：Attention 分数严格只依赖 $m-n$，而非绝对位置
3. **自然外推**：训练长度为 $L$ 的旋转规则和 $L+1$ 的旋转规则一致（都按 $\theta_i$ 旋转），不存在分布突变

---

## 6. 面试高频追问

### Q1: RoPE 是绝对位置编码还是相对位置编码？

**RoPE 以绝对位置注入，但以相对位置生效。** 编码时按绝对位置 $m$ 旋转 Q、按绝对位置 $n$ 旋转 K——这是绝对编码。但内积计算后只依赖 $(m-n)$——这是相对编码的效果。

### Q2: RoPE 为什么只加到 Q 和 K，不加到 V？

Attention 分数（QK^T）决定了"谁关注谁"，位置影响的是这个匹配过程。V 只是被加权求和的内容，不需要位置信息——V 已经通过 $V \leftarrow \text{softmax}(QK^T) \cdot V$ 间接获得了位置依赖。

### Q3: base 参数的作用是什么？

$\theta_i = \text{base}^{-2i/d}$，base 控制所有频率的整体缩放。

- base 越大 → 频率越小 → 旋转越慢 → 长距离位置区分度越好 → **长上下文外推能力越强**
- 原始 base=10000，LLaMA 用 10000，Qwen3 提高到 1,000,000

增大 base 是免训练扩展上下文长度的常用技巧（如 NTK-aware scaling）。

### Q4: 实现中的 rotate_half 是什么？

代码中常见一种等价但更高效的实现：把向量前后各半的 cos/sin 旋转转化为符号变换。理论上是 $R(\theta)x$，实现上用向量化加速：

```python
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)     # 前后切两半
    return torch.cat([-x2, x1], dim=-1)  # 前半变后半的负，后半变前半

def apply_rotary_pos_emb(x, cos, sin):
    return x * cos + rotate_half(x) * sin
```

> 这和理论等价：$x \cdot \cos\theta + \text{rotate\_half}(x) \cdot \sin\theta$ 展开后就是旋转公式。

---

## 参考资料

- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021 ([arXiv:2104.09864](https://arxiv.org/abs/2104.09864))
- [说人话理解 RoPE：从 sin/cos 位置编码到旋转矩阵](https://zhuanlan.zhihu.com/p/2032881588019663675)
- [AI Infra 面试常考——主流大模型架构（含 RoPE 详解）](https://zhuanlan.zhihu.com/p/2010304685437916961)
- [AIGC 面试面经第六期：旋转位置编码 RoPE 相关问答](https://zhuanlan.zhihu.com/p/2001374727403481029)
