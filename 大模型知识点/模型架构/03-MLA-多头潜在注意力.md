---
tags:
  - 面试
  - 大模型
  - 注意力机制
  - MLA
  - DeepSeek
  - KV Cache
created: 2026-05-24
updated: 2026-05-24
---

# MLA：多头潜在注意力（Multi-head Latent Attention）

> 核心问题：MHA → MQA → GQA → MLA 这条线是怎么演进的？MLA 是怎么进一步压 KV Cache 的？为什么 MLA 在 KV Cache 节省的同时**性能不掉**？

---

## 0. 面试怎么答（30 秒电梯版）

> "MLA 是 DeepSeek-V2 提出的注意力变体。
>
> **演进逻辑**：MHA 每头独立 K/V，KV Cache 太大；MQA 所有头共享一对 K/V，Cache 极小但效果掉；GQA 折中分组共享。
>
> **MLA 的关键想法**：不直接缓存 K 和 V，而是**先把 K/V 通过低秩矩阵压缩到一个潜在向量 $c^{KV}_t$**（维度 $d_c \ll d_\text{model}$），**只缓存这个潜在向量**。推理时再升维投影回多头 K/V。
>
> **结果**：KV Cache 大小从"层数 × heads × head_dim × 2"降到"层数 × $d_c$"，**比 GQA 还小**，但通过保留多头表示能力**性能反而比 MHA 还好**。
>
> 配 RoPE 时有一个 trick：RoPE 不能直接套在压缩后的 K 上，所以 MLA 把 K 拆成"压缩部分（无 RoPE）+ 解耦部分（带 RoPE）"两份拼接。"

---

## 1. 演进背景回顾

| 方法 | KV 头数 | KV Cache 公式 | 性能 |
|---|---|---|---|
| **MHA** | $h$（与 Q 头数相同） | $2 \cdot L \cdot h \cdot d_h \cdot T$ | 最佳 |
| **MQA** | 1 | $2 \cdot L \cdot d_h \cdot T$（小 h 倍） | 显著下降 |
| **GQA** | $g$（分组数，$1 < g < h$） | $2 \cdot L \cdot g \cdot d_h \cdot T$ | 接近 MHA |
| **MLA** | "潜在" $d_c$ | $L \cdot d_c \cdot T$（不分 KV，只缓存一份压缩向量） | **优于 MHA** |

详见 [[01-MHA-多头注意力]]、[[02-GQA-分组查询注意力]]、[[01-KV-Cache-为什么不缓存Q]]。

---

## 2. MLA 的核心想法：低秩压缩 K/V

### 2.1 标准 MHA 的 K/V 投影

```
hidden h_t (d_model)
   │
   ├── W^K → K_t (d_model = h * d_head)  ← 缓存这个
   └── W^V → V_t (d_model)               ← 也缓存这个
```

KV Cache per token per layer = $2 \cdot h \cdot d_h$

### 2.2 MLA 的低秩压缩

```
hidden h_t (d_model)
   │
   └── W^{DKV} (下投影矩阵, d_model → d_c) → c^{KV}_t   ← 只缓存这一个 d_c 维向量！
                                              │
                ┌─────────────────────────────┤
                ↓                             ↓
            W^{UK} (升维到 K)            W^{UV} (升维到 V)
                ↓                             ↓
              K_t (重建出来用)            V_t (重建出来用)
```

数学上：

$$
c^{KV}_t = W^{DKV} h_t \quad \in \mathbb{R}^{d_c}
$$

$$
K_t = W^{UK} c^{KV}_t,\quad V_t = W^{UV} c^{KV}_t
$$

> 直觉：把"高维 K + 高维 V"压成"低维潜在向量"，需要时再展开。
>
> 类比：把一张 1080p 照片存成压缩的 latent code，需要看时再 decode。

### 2.3 KV Cache 大小对比

假设 $d_\text{model} = 5120$，$h = 128$，$d_h = 128$，$d_c = 512$，128 层：

| 方法 | 每 token 每层 cache | DeepSeek-V2 实际值 |
|---|---|---|
| MHA | $2 \cdot 128 \cdot 128 \times 2\text{B} = 65.5\text{KB}$ | 8.4MB / token |
| GQA (g=8) | $2 \cdot 8 \cdot 128 \times 2\text{B} = 4\text{KB}$ | 525KB / token |
| **MLA** | $d_c \times 2\text{B} = 1\text{KB}$ | **128KB / token** |

> **MLA 比 GQA 还小**，但性能接近甚至超过 MHA。

---

## 3. 关键技术细节：吸收矩阵（Absorption Trick）

### 3.1 朴素实现的问题

如果每次推理都先用 $c^{KV}_t$ 重建出 $K_t = W^{UK} c^{KV}_t$，再做 $Q_t^T K_t$，**升维操作每步都重做**，相当于多了一次大矩阵乘法。

### 3.2 吸收：把升维矩阵融到 Q 投影里

注意力分数：

$$
\text{score} = Q_t^T K_s = (W^Q h_t)^T (W^{UK} c^{KV}_s) = h_t^T \underbrace{(W^Q)^T W^{UK}}_{\text{融成一个矩阵}} c^{KV}_s
$$

**关键观察**：$(W^Q)^T W^{UK}$ 是固定矩阵，可以**离线融合**成一个新的等效 $\tilde{W}^Q$。

这样推理时**直接用 $\tilde{W}^Q h_t$ 和 $c^{KV}_s$ 做内积**，根本不需要重建 $K_s$！

V 侧同理：$W^{UV}$ 可以吸收进输出投影 $W^O$。

> 工程结果：**只缓存 $c^{KV}_s$（很小），推理时不需要显式重建 K/V**。这就是 MLA"小 cache + 不掉性能 + 不增加推理延迟"的核心。

---

## 4. MLA 怎么和 RoPE 配合？解耦 RoPE 设计

### 4.1 RoPE 与 MLA 的天然冲突

RoPE 通过对 Q、K 做位置相关的旋转来注入位置信息：

$$
Q'_t = R_t Q_t,\quad K'_t = R_t K_t
$$

详见 [[04-RoPE-旋转位置编码]]。

**问题**：MLA 的 K 来自 $K_t = W^{UK} c^{KV}_t$，那 RoPE 要怎么套？

如果套在 $c^{KV}_t$ 上：不同位置对应不同的 $R_t$，但 $c^{KV}_t$ 维度是 $d_c$ 而不是 $d_h$，RoPE 的二维旋转分组对不上。

如果先升维再 RoPE：$K_t = R_t W^{UK} c^{KV}_t$ —— 升维矩阵不能再被吸收（因为有位置依赖的 $R_t$ 在中间）→ 吸收 trick 失效。

### 4.2 解决方案：拆分成两部分

MLA 把 Q、K 各自**分成两段拼接**：

$$
Q_t = [Q^C_t;\ Q^R_t],\quad K_t = [K^C_t;\ K^R_t]
$$

| 部分 | 维度 | 作用 |
|---|---|---|
| $Q^C, K^C$ | "Content" | **无 RoPE**，通过低秩压缩 + 吸收 trick |
| $Q^R, K^R$ | "RoPE" | **带 RoPE**，常规走法 |

注意力分数：

$$
Q_t^T K_s = (Q^C_t)^T K^C_s + (Q^R_t)^T K^R_s
$$

两部分独立计算后相加。

### 4.3 缓存什么

| 张量 | 是否缓存 | 大小 |
|---|---|---|
| $c^{KV}_s$（content 部分的压缩 K/V） | ✅ | $d_c$ |
| $K^R_s$（RoPE 部分的 K） | ✅ | $d_h^R$（小，DeepSeek-V2 取 $d_h^R = d_h / 2 = 64$） |
| $K^C_s$（重建出来的） | ❌（吸收掉） | — |
| $V_s$（重建出来的） | ❌（吸收掉） | — |

总缓存：$d_c + d_h^R$，DeepSeek-V2 里是 $512 + 64 = 576$ 维 per token per layer，约 1.1 KB（FP16）。

---

## 5. Q 也做低秩（训练显存优化）

MLA 还对 Q 做了类似的低秩压缩：

$$
c^Q_t = W^{DQ} h_t \in \mathbb{R}^{d'_c},\quad Q^C_t = W^{UQ} c^Q_t
$$

**目的不是 cache（Q 不需要 cache），而是节省训练时的激活显存和参数量**。

---

## 6. PyTorch 伪代码

```python
import torch
import torch.nn as nn

class MLA(nn.Module):
    def __init__(self, d_model, n_heads, d_head, d_c=512, d_qc=1536, d_rope=64):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head  # content 部分每头维度
        self.d_rope = d_rope

        # === Q 投影（带低秩压缩）===
        self.W_DQ = nn.Linear(d_model, d_qc, bias=False)
        self.W_UQ_content = nn.Linear(d_qc, n_heads * d_head, bias=False)
        self.W_UQ_rope    = nn.Linear(d_qc, n_heads * d_rope, bias=False)

        # === K/V 投影（共享低秩压缩 + 独立 RoPE K）===
        self.W_DKV = nn.Linear(d_model, d_c, bias=False)               # 压缩到 c_KV
        self.W_UK  = nn.Linear(d_c, n_heads * d_head, bias=False)      # content K
        self.W_UV  = nn.Linear(d_c, n_heads * d_head, bias=False)      # V
        self.W_KR  = nn.Linear(d_model, d_rope, bias=False)            # rope K (共享 across heads)

        # 输出投影
        self.W_O = nn.Linear(n_heads * d_head, d_model, bias=False)

    def forward(self, h_t, kv_cache=None, position_ids=None):
        B, T, D = h_t.shape

        # 1. Q 路径
        c_q = self.W_DQ(h_t)                                            # [B, T, d_qc]
        Q_c = self.W_UQ_content(c_q).view(B, T, self.n_heads, self.d_head)
        Q_r = self.W_UQ_rope(c_q).view(B, T, self.n_heads, self.d_rope)
        Q_r = apply_rope(Q_r, position_ids)

        # 2. K/V 压缩路径
        c_kv = self.W_DKV(h_t)                                          # [B, T, d_c]  ← 这是要缓存的
        K_r  = self.W_KR(h_t)                                           # [B, T, d_rope] ← 这也要缓存
        K_r  = apply_rope(K_r.unsqueeze(2), position_ids).squeeze(2)

        # 3. KV Cache（推理时累积，注意只缓存 c_kv 和 K_r）
        if kv_cache is not None:
            c_kv = torch.cat([kv_cache['c_kv'], c_kv], dim=1)
            K_r  = torch.cat([kv_cache['K_r'], K_r], dim=1)
            new_cache = {'c_kv': c_kv, 'K_r': K_r}
        else:
            new_cache = {'c_kv': c_kv, 'K_r': K_r}

        # 4. 重建 K/V（推理优化版会把这步吸收掉）
        K_c = self.W_UK(c_kv).view(B, -1, self.n_heads, self.d_head)
        V   = self.W_UV(c_kv).view(B, -1, self.n_heads, self.d_head)

        # 5. 拼接 content + rope 部分，计算注意力
        Q = torch.cat([Q_c, Q_r], dim=-1)              # [B, T, h, d_head + d_rope]
        K = torch.cat([K_c, K_r.unsqueeze(2).expand(-1, -1, self.n_heads, -1)], dim=-1)
        scores = torch.einsum('bthd,bshd->bhts', Q, K) / (self.d_head + self.d_rope) ** 0.5
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum('bhts,bshd->bthd', attn, V)
        out = out.reshape(B, T, -1)

        return self.W_O(out), new_cache
```

> 注意：上面是教学版，真正的高性能实现会用 absorption trick 完全避免 K_c / V 的重建。

---

## 7. MLA vs GQA：到底哪里更好

| 维度 | GQA | MLA |
|---|---|---|
| KV Cache 大小 | 小（除以 g 倍） | **更小**（$d_c \ll h \cdot d_h$） |
| 表示能力 | 弱于 MHA | **接近甚至超过 MHA** |
| 实现复杂度 | 简单（直接共享） | 复杂（要做吸收、RoPE 解耦） |
| 与 RoPE 兼容 | 天然兼容 | 需要拆分设计 |
| 训练显存 | 没节省 | Q 也压缩，省训练显存 |
| 适用规模 | 通用 | 主要是超大模型（V2/V3） |

> **结论**：MLA 是为**超大模型 + 长上下文 + 在线推理**场景设计的。中小模型用 GQA 性价比更高。

---

## 8. MLA 在 DeepSeek 系列里的实战参数

DeepSeek-V2 / V3：

| 参数 | 值 |
|---|---|
| $d_\text{model}$ | 5120 (V2) / 7168 (V3) |
| 头数 $h$ | 128 |
| $d_h$ (content) | 128 |
| $d_h^R$ (RoPE) | 64 |
| $d_c$ (KV 潜在维度) | 512 |
| $d'_c$ (Q 潜在维度) | 1536 |

V2 KV Cache：每 token 每层约 **128KB**（vs MHA 的 8.4MB）→ **65× 压缩**。

---

## 9. 面试高频追问

### Q1：MLA 比 GQA 强在哪？
两点：① **Cache 更小**（一个 $d_c$ 向量 vs $g$ 组 KV）；② **表示能力更强**（保留多头 K/V 的等效表达，而 GQA 是头之间硬共享）。

### Q2：为什么 MLA 不影响性能？
压缩矩阵 $W^{DKV}, W^{UK}, W^{UV}$ 是**可学习**的低秩分解，能学到对当前任务最重要的方向。这等价于在 K/V 上加了一个"信息瓶颈"，反而可能起到正则化作用。

### Q3：MLA 推理为什么不慢？
吸收 trick 把升维矩阵预先融到 Q 投影/O 投影里，**推理时不需要显式重建 K、V**，避免了"先解压再算"的开销。

### Q4：MLA 怎么和 RoPE 配合？
直接给压缩后的 K 加 RoPE 会破坏吸收 trick。MLA 把 Q、K **拆成 content（走压缩 + 无 RoPE）和 RoPE（走原始投影 + 有 RoPE）两段，分别计算 score 再相加**。

### Q5：训练显存怎么省？
Q 也做低秩分解（$d'_c$ 维），减少 Q 投影参数和激活；K/V 的压缩也减少了反向梯度的中间张量。

### Q6：MLA 能不能用在中小模型？
可以但**性价比不如 GQA**。MLA 的实现复杂度（吸收、RoPE 解耦）只在超大模型 + 长上下文场景才划算。

### Q7：MLA 和 MoE 的关系？
正交。MLA 是注意力优化，MoE 是 FFN 优化。DeepSeek-V2/V3 同时用了 MLA + MoE。

### Q8：MLA 的 $d_c$ 怎么选？
经验：$d_c \approx 4 \cdot d_h$（DeepSeek-V2 是 $4 \cdot 128 = 512$）。太小信息损失大，太大失去压缩意义。

---

## 关联阅读

- [[01-MHA-多头注意力]] —— 演进起点
- [[02-GQA-分组查询注意力]] —— MLA 的前一步
- [[01-KV-Cache-为什么不缓存Q]] —— 为什么 KV Cache 是瓶颈
- [[04-RoPE-旋转位置编码]] —— 解耦 RoPE 的前置知识

## 参考资料

- DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*, 2024
- DeepSeek-AI, *DeepSeek-V3 Technical Report*, 2024
- [手撕大模型 Attention：MLA、MHA、MQA 与 GQA](https://zhuanlan.zhihu.com/p/1909650875439387633)
- [一文看懂 MLA（Multi-head Latent Attention）](https://zhuanlan.zhihu.com/p/697846879)
