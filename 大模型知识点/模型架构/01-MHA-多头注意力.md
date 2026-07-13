---
tags:
  - 面试
  - 大模型
  - 注意力机制
  - MHA
created: 2026-05-11
updated: 2026-07-13
---

# MHA：多头注意力机制

> 核心问题：为什么 Transformer 用"多头"注意力而不是"单头"？MHA 中的 Q、K、V 到底是什么？多个头到底在做什么？

---

## 0. 前置理解：Self-Attention 在做什么

### 0.1 Attention 的本质：按相似度加权汇总

注意力机制的核心操作只有一句话：**用 Query 和 Key 计算相似度，按相似度对 Value 做加权求和。**

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
```

拆开看这个公式：

| 步骤                     | 操作                 | 含义                       |
| ---------------------- | ------------------ | ------------------------ |
| $QK^T$                 | 计算 Query 和 Key 的点积 | 得到"相似度矩阵"，值越大 = 越相关      |
| $\frac{1}{\sqrt{d_k}}$ | 除以 $\sqrt{d_k}$    | 防止点积过大导致 softmax 梯度消失    |
| softmax(...)           | 按行归一化              | 把相似度变成"权重"，每行加起来等于 1     |
| $\times V$             | 用权重对 Value 加权求和    | "根据相关程度，把 Value 的信息聚合过来" |

### 0.2 类比：数据库查询

```
Query  → 你想查什么（"找跟苹果有关的信息"）
Key    → 每个位置的标签（"我这里讲的是水果" / "我这里讲的是手机"）
Value  → 每个位置的实际内容（"苹果是一种蔷薇科植物..." / "iPhone 15 搭载..."）

相似度(Q, K) → 你的查询和每个标签的匹配程度
输出 = 按匹配程度，把所有内容加权混合
```

### 0.3 Self-Attention：自己注意自己

Self-Attention 的特殊之处：**Q、K、V 都来自同一个输入序列**。

```
输入序列 X (shape: [seq_len, d_model])
         │
    ┌────┼────┐
    ↓    ↓    ↓
   W^Q  W^K  W^V    三个可学习的权重矩阵
    ↓    ↓    ↓
    Q    K    V      三个投影，维度相同但含义不同
```

- $Q = X W^Q$：我作为"查询者"，想问别人什么
- $K = X W^K$：我作为"被查者"，我能提供什么标签
- $V = X W^V$：我作为"被查者"，我实际持有什内容

**为什么 Q、K、V 要分别投影？** 因为让模型在不同的表示空间里学习"查询"和"被查询"——查东西时用的特征和作为标签用的特征不需要一样。

---

## 1. 为什么需要"多头"？

### 1.1 单头的问题

单头 Attention 所有 token 之间只算一次相似度，只能捕捉**一种**关系模式。

但语言中的关系是多维度的：

```
"小明把苹果给了小红"
```

这个句子里同时存在：
- **语法关系**：主语（小明）→ 谓语（给）→ 宾语（苹果）→ 间接宾语（小红）
- **语义关系**：苹果 和 吃 的关联 vs 苹果 和 手机品牌的关联
- **位置关系**：近距离词之间的依赖

单头 Attention 无法同时捕捉这些不同维度的关系。

### 1.2 多头的直觉

**多头 = 让模型在多个不同的"子空间"里分别做 Attention。**

类比 CNN 的多通道：一个卷积核提取边缘，另一个提取纹理，另一个提取颜色。同理：

- Head 1 可能学会关注**语法依存**
- Head 2 可能学会关注**同义词/近义词**
- Head 3 可能学会关注**长距离依赖**
- Head 4 可能学会关注**位置相邻**

---

## 2. MHA 的计算流程

将 $d_{\text{model}}$ 维的 Q、K、V 切成 $h$ 个头，每个头维度 $d_k = d_{\text{model}} / h$。

```
输入 X: [batch, seq_len, d_model]
         │
    ┌────┼────┐
   W^Q  W^K  W^V          # 线性投影到 d_model 维
    ↓    ↓    ↓
    Q    K    V            # [batch, seq_len, d_model]
    │    │    │
    └────┼────┘
    split into h heads     # [batch, h, seq_len, d_k]
         │
    ┌────┼────┐
   head₁ head₂ ... head_h  # 每个头独立做 Attention
    │     │         │
    └────┼─────────┘
      concat              # [batch, seq_len, d_model]
         │
       W^O                # 输出投影
         ↓
    output: [batch, seq_len, d_model]
```

每个头的计算：

```math
\text{head}_i = \text{Attention}(Q W_i^Q,\; K W_i^K,\; V W_i^V)
```

最终输出：

```math
\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
```

---

## 3. 为什么除以 $\sqrt{d_k}$？（从零讲起）

> 这一节是 Transformer 面试**必考**。我假设你**完全不知道**为什么，从最基础的"softmax 怎么工作"一步一步讲到结论。

### 3.0 面试怎么答（30 秒电梯版）

> "如果不缩放，$QK^T$ 这个点积的方差会随着 $d_k$ 增大而变大（方差 ≈ $d_k$）。点积值越大，softmax 输出就越尖锐 —— 接近 one-hot 分布。
>
> 而 softmax 一旦变成 one-hot，**梯度几乎为 0**（梯度消失），训练时模型学不动。
>
> 除以 $\sqrt{d_k}$ 正好把点积的方差从 $d_k$ 拉回 1，softmax 输出温度合适、梯度健康，训练才稳。
>
> 用 $\sqrt{d_k}$ 而不是 $d_k$，是因为方差和标准差差一个开根号 —— 我们要的是缩放标准差。"

---
![[Pasted image 20260525105536.png]]
### 3.1 第一步：先搞懂 softmax 在干什么

Softmax 把一组实数 $(z_1, z_2, \dots, z_n)$ 转成概率分布（正数且和为 1）：

```math
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
```

**关键性质**：它对"输入的相对大小"非常敏感，对"绝对大小"不那么敏感（因为有归一化）。但它**对绝对幅度有反应**：

- 输入差距小 → 输出柔和、平均
- 输入差距大 → 输出尖锐、集中在最大值上

**用数字感受一下**。假设有 3 个 token，注意力 score 是：

| 情况 | 输入 z | softmax 输出 | 长什么样 |
|---|---|---|---|
| A（小尺度） | (1, 2, 3) | (0.09, 0.24, 0.67) | 比较柔和 |
| B（中尺度） | (5, 10, 15) | (0.0067, 0.99, 0.0..)| 已经很尖锐 |
| C（大尺度） | (10, 20, 30) | (≈0, ≈0, ≈1.0) | **几乎是 one-hot** |

> 看出来了：**输入幅度越大 → softmax 越接近 one-hot**。原因很简单：softmax 里有 $e^z$，z 大一点 $e^z$ 就指数级放大。

---

### 3.2 第二步：one-hot 的 softmax 为什么是灾难

机器学习里"训练"的核心是**反向传播算梯度**。softmax 的梯度长这样：

```math
\frac{\partial \text{softmax}(z_i)}{\partial z_j} = \begin{cases}
p_i (1 - p_i) & i = j \\
- p_i p_j & i \ne j
\end{cases}
```

其中 $p_i = \text{softmax}(z_i)$。

**当 softmax 接近 one-hot**（某一项 $p_i \approx 1$，其他 $p_j \approx 0$）：

```math
p_i(1-p_i) \approx 1 \times 0 = 0,\quad p_i p_j \approx 1 \times 0 = 0
```

→ **所有偏导都接近 0**！

后果：
- 反向传播经过 softmax 的梯度被乘以一个几乎全是 0 的矩阵
- **梯度消失** → Q、K、V 的权重更新接近 0
- 训练卡死，模型学不动

> 这就是为什么我们**不能让 softmax 的输入太大**。

---

### 3.3 第三步：为什么 $QK^T$ 会随 $d_k$ 变大

Attention 的分数是 Q 和 K 这两个**向量的点积**：

```math
\text{score} = q \cdot k = \sum_{i=1}^{d_k} q_i \cdot k_i
```

—— 这是 $d_k$ 个数相加。**加的项越多，结果的"波动范围"越大**，这是常识。

#### 直观体验：抛硬币

抛 1 次硬币（每次结果 ±1）：可能值在 [-1, 1]
抛 100 次求和：可能在 -100 到 100 之间，但**典型范围**会更大（标准差 = $\sqrt{100} = 10$）
抛 10000 次求和：可能在 -10000 到 10000，**典型范围**标准差 = 100

每次抛硬币方差是 1，**100 次求和方差是 100**，因为独立随机变量求和方差相加。

#### 用数学严谨写一遍

假设 $q_i$ 和 $k_i$ 是独立的随机变量，**均值 0、方差 1**（这是常见的初始化假设，比如 Xavier、He 初始化都让权重在这个范围）。

那么 $q_i \cdot k_i$ 的期望和方差是：

- $E[q_i k_i] = E[q_i] \cdot E[k_i] = 0 \cdot 0 = 0$
- $\text{Var}(q_i k_i) = E[q_i^2 k_i^2] - (E[q_i k_i])^2 = E[q_i^2] E[k_i^2] - 0 = 1 \cdot 1 = 1$

求和 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$：

- 期望：$E[q \cdot k] = \sum E[q_i k_i] = 0$
- 方差（独立项的方差相加）：$\text{Var}(q \cdot k) = \sum \text{Var}(q_i k_i) = d_k$

**结论**：点积的标准差 $\sigma = \sqrt{d_k}$。

> 也就是说，**$d_k$ 越大，点积的"典型大小"按 $\sqrt{d_k}$ 这个速度变大**。

#### 代入实际数字

| $d_k$ | 点积的典型大小（标准差 $\sqrt{d_k}$） |
|---|---|
| 8 | 2.83 |
| 64 | 8 |
| 128 | 11.3 |
| 256 | 16 |
| 512 | 22.6 |

对照 3.1 节的表：当输入达到 10、20、30 这种量级，softmax 已经接近 one-hot 了！

> 所以**不缩放的话，$d_k \ge 64$ 的 attention 训练会直接崩**。

---

### 3.4 第四步：为什么除以 $\sqrt{d_k}$ 而不是 $d_k$

我们的目标是让点积**回到方差 1**（也就是和 $d_k = 1$ 时一样的尺度），这样输入 softmax 才不会爆。

**方差的缩放规则**：

```math
\text{Var}(c \cdot X) = c^2 \cdot \text{Var}(X)
```

如果给点积乘以 $\frac{1}{\sqrt{d_k}}$：

```math
\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{1}{d_k} \cdot \text{Var}(q \cdot k) = \frac{1}{d_k} \cdot d_k = 1 \quad \checkmark
```

**完美**：方差回到 1。

如果除以 $d_k$（而不是 $\sqrt{d_k}$）：

```math
\text{Var}\left(\frac{q \cdot k}{d_k}\right) = \frac{1}{d_k^2} \cdot d_k = \frac{1}{d_k}
```

→ 方差变成 $1/d_k$，**过度缩小**，softmax 反而变得太平坦，所有 token 几乎被等权重对待，模型也学不到差异。

> **类比**：方差和标准差差一个开根号；我们想"控制典型大小"，就要按标准差缩放，所以除以 $\sqrt{d_k}$ 而不是 $d_k$。

---

### 3.5 验证：缩放前后的 softmax 对比

假设 $d_k = 64$，三个 token 的原始点积分数恰好是它们的"典型大小"附近，比如 (4, -2, 6)：

**不缩放**：

直接进 softmax：$\text{softmax}(4, -2, 6) \approx (0.12, 0.0002, 0.88)$ → 已经偏 one-hot

**缩放后**（除以 $\sqrt{64} = 8$）：

$(4/8, -2/8, 6/8) = (0.5, -0.25, 0.75)$ → $\text{softmax}(0.5, -0.25, 0.75) \approx (0.31, 0.15, 0.54)$ → 分布柔和、有梯度

> 看出来了：缩放之前 softmax 已经把第二个 token 的权重压到 0.0002（几乎没声音），缩放后是 0.15（明显存在）。**第二个 token 信息没被丢掉**。

---

### 3.6 一图总结

```
                  d_k 越大
                      │
                      ↓
        ┌──────────────────────────┐
        │  q · k 的方差 ≈ d_k       │   ←  独立项方差相加
        │  典型大小 ≈ √d_k          │
        └──────────────────────────┘
                      │
        不缩放        ↓        缩放（除以 √d_k）
            ┌─────────┴─────────┐
            ↓                   ↓
   softmax 输入太大        softmax 输入回到 ~1 范围
            │                   │
            ↓                   ↓
    输出接近 one-hot       输出是健康分布
            │                   │
            ↓                   ↓
       梯度 ≈ 0            梯度健康
            │                   │
            ↓                   ↓
       梯度消失 💀          训练正常 ✅
       模型学不动
```

---

### 3.7 面试追问

#### Q1：为什么是 $\sqrt{d_k}$，不是 $\sqrt{d_{model}}$、不是 $\sqrt{h}$？
缩的是**单个头的点积**，所以分母是单个头的维度 $d_k = d_{model} / h$。如果不分头（普通 attention）就用 $d_{model}$。

#### Q2：能不能不除以 $\sqrt{d_k}$，靠 LayerNorm 来稳定？
不能完全替代。LayerNorm 是在 token 维度归一化，作用在 attention 模块的**输入和输出**；而 softmax 爆炸发生在 attention **内部**（$QK^T$ 之后、softmax 之前），LayerNorm 触及不到。

#### Q3：训练时如果忘了除会怎样？
小模型小 $d_k$ 可能还能勉强训，大模型立刻崩 —— loss 不下降、梯度看上去是 0。这是 Transformer 实现的经典 bug。

#### Q4：为什么不用其他缩放因子，比如除以 $d_k$ 然后再乘个常数？
理论上任何能让方差稳定的因子都行，但 $\sqrt{d_k}$ 是**最自然的选择**（直接对应标准差），不需要额外调参。

#### Q5：如果初始化方差不是 1，结论还成立吗？
假设 $q_i, k_i$ 方差都是 $\sigma^2$，那么点积方差是 $d_k \cdot \sigma^4$，标准差是 $\sigma^2 \sqrt{d_k}$。所以**严格来说**应该除以 $\sigma^2 \sqrt{d_k}$。实践中现代神经网络的权重初始化和 LayerNorm 让 $\sigma \approx 1$，所以除以 $\sqrt{d_k}$ 已经够好。

#### Q6：训练后期 Q、K 的方差还是 1 吗？
不一定。但 attention 模块前面通常有 LayerNorm，会重新把激活拉回方差 1 附近，所以缩放因子用 $\sqrt{d_k}$ 仍然合理。这也是为什么近期一些工作（如 QK-Norm）在 Q、K 上再加一次 norm。

---

## 4. PyTorch 实现

```python
import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, D = x.shape

        # 1. 线性投影
        Q = self.W_q(x)  # [B, T, D]
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. 拆成多头: [B, T, D] → [B, h, T, d_k]
        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # 3. 缩放点积注意力
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # [B, h, T, T]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.dropout(torch.softmax(scores, dim=-1))

        # 4. 加权求和
        out = attn @ V  # [B, h, T, d_k]

        # 5. 合并多头: [B, h, T, d_k] → [B, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        # 6. 输出投影
        return self.W_o(out)
```

---

## 5. MHA 的参数量和计算量

| 组件 | 参数量 |
|---|---|
| $W^Q, W^K, W^V$ | $3 \times d_{\text{model}}^2$ |
| $W^O$ | $d_{\text{model}}^2$ |
| 总计 | $4d_{\text{model}}^2$ |

计算复杂度：$O(T^2 \cdot d_{\text{model}})$，其中 $T$ 是序列长度。

> $T^2$ 来自注意力矩阵 $QK^T$ 的形状 $[T, T]$。这也是长序列时 Attention 成为瓶颈的原因。

---

## 6. 面试高频追问

### Q1: 为什么多头比单头好？

不同头可以关注不同的表示子空间——有的头关注位置、有的关注语法、有的关注语义。多头 = 多个视角同时理解输入。

### Q2: 为什么 Q、K、V 要分开投影？

让模型在不同表示空间里分别学习"查询"和"被匹配"——查询时用的特征和作为标签用的特征不需要一样。

### Q3: 如果 d_k 很小会怎样？

每个头的表示能力下降。头的数量 × d_k = d_model 不变，需要在头数和每头维度之间权衡。通常每头 64 或 128 维。

### Q4: MHA 的推理瓶颈是什么？

KV Cache。推理时每个 token 都要存储所有历史 token 的 K、V，内存占用 = $2 \times \text{层数} \times \text{头数} \times \text{序列长度} \times d_k$。序列越长，KV Cache 越大。这也是 MQA/GQA 的改进动机。

---

## 参考资料

- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017
- [【LLM】一文详解 MHA、GQA、MQA 原理](https://zhuanlan.zhihu.com/p/1890128241)
- [手撕大模型 Attention：MLA、MHA、MQA 与 GQA](https://zhuanlan.zhihu.com/p/1909650875439387633)
- [极简代码复现 MHA、MQA、GQA——面试终结者](https://zhuanlan.zhihu.com/p/1905369792745017821)
