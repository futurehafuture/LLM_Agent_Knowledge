---
tags: [面试, 大模型, 推理优化, KV-Cache, Attention]
created: 2026-05-09
updated: 2026-05-12
topic: 为什么 LLM 推理只 cache K 和 V，不 cache Q
---

# 为什么 Transformer / LLM 推理时只 cache K 和 V，而不 cache Q

> 适用场景：大模型 / LLM Infra / 推理优化方向的面试技术准备  
> 关键词：KV Cache、Self-Attention、Autoregressive Decoding、Prefill / Decode、MQA / GQA / MLA、PagedAttention

---

## 1. 背景：什么是 KV Cache，为什么 LLM 推理离不开它

### 1.1 自回归生成的基本机制

LLM 推理本质上是自回归（autoregressive）解码：模型每次只生成一个 token，并把它拼回输入，再去预测下一个 token。形式化地，给定上下文 $`x_{1:t}`$，模型计算

```math
p(x_{t+1} \mid x_{1:t}) = \text{LM}(x_{1:t})
```

然后采样得到 $`x_{t+1}`$，再拼接成 $`x_{1:t+1}`$，反复迭代直到 EOS 或达到最大长度。

每一步都需要对当前完整序列 $`x_{1:t}`$ 跑一次前向。如果什么缓存都没有，每生成一个 token 就要从头算一遍 attention，*意味着第 $`n`$ 步的复杂度是 $`O(n^2)`$，整个生成过程是 $`O(n^3)`$*，且大量计算是**重复**的。

### 1.2 KV Cache 的作用

KV Cache 把每一层 self-attention 中已经算过的 Key 张量 $`K_{1:t}`$ 与 Value 张量 $`V_{1:t}`$ 缓存到显存中。下一步生成时，只需要：

1. 用新 token 计算它自己的 $`q_{t+1}, k_{t+1}, v_{t+1}`$（一行）；
2. 把 $`k_{t+1}, v_{t+1}`$ append 到缓存中，得到 $`K_{1:t+1}, V_{1:t+1}`$；
3. 直接复用缓存做 $`q_{t+1} K_{1:t+1}^\top`$ 与对 $`V_{1:t+1}`$ 的加权求和。

引入 KV Cache 后，单步 attention 复杂度从 $`O(t^2)`$ 降到 $`O(t)`$，整体生成从 $`O(n^3)`$ 降到 $`O(n^2)`$。HuggingFace 在 T4 GPU 上的实测显示，KV Cache 可以让生成快约 **5×**（61s → 11.7s），不同模型/硬件下普遍在 **3–5×** 之间。

不开 KV Cache，长上下文几乎无法实用化：每生成一个 token 都要从头跑一次 prefill，几百步以后单步延迟会急剧爆炸。

---

## 2. 核心原因：从 self-attention 公式推导为什么只缓存 K 和 V

### 2.1 标准 self-attention 公式

```math
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V
```

其中 $`M`$ 是因果掩码（causal mask），$`M_{ij} = -\infty`$ if $`j > i`$，保证位置 $`i`$ 只能看到 $`j \le i`$ 的位置。

### 2.2 Prefill 阶段：一次性吃完 prompt

设 prompt 长度为 $`T_p`$，hidden size 为 $`d`$，单头 head_dim 为 $`d_h`$。Prefill 阶段一次性计算：

```math
Q = X W_Q, \quad K = X W_K, \quad V = X W_V \in \mathbb{R}^{T_p \times d_h}
```

得到完整的注意力分数矩阵 $`S = QK^\top \in \mathbb{R}^{T_p \times T_p}`$。这个阶段计算量大、并行度高，是典型的 **compute-bound**（受算力约束）。

完成 prefill 后，**$`K`$ 和 $`V`$ 被原样写入缓存**，$`Q`$ 用完即丢——因为它接下来不会再被使用。

### 2.3 Decode 阶段：每步一个新 token

生成第 $`t+1`$ 个 token 时，输入是新 token 的 embedding $`x_{t+1} \in \mathbb{R}^{1 \times d}`$，于是：

```math
q_{t+1} = x_{t+1} W_Q,\quad k_{t+1} = x_{t+1} W_K,\quad v_{t+1} = x_{t+1} W_V \in \mathbb{R}^{1 \times d_h}
```

注意：

- 新 token 的 query 只有**一行** $`q_{t+1}`$。
- 把 $`k_{t+1}, v_{t+1}`$ append 到缓存：$`K_{1:t+1} = [K_{1:t}; k_{t+1}]`$，$`V_{1:t+1} = [V_{1:t}; v_{t+1}]`$。

这一步真正做的注意力计算是：

```math
\text{attn}_{t+1} = \text{softmax}\!\left(\frac{q_{t+1} K_{1:t+1}^\top}{\sqrt{d_k}}\right) V_{1:t+1}
```

即一个 $`1 \times (t+1)`$ 的注意力分布 × $`(t+1) \times d_h`$ 的 value 矩阵 → $`1 \times d_h`$ 的输出。

**这里就出现了根本性的不对称**：

| 量 | 在第 $`t+1`$ 步的角色 | 在第 $`t+2`$ 步还会被用到吗？ |
|---|---|---|
| $`q_{t+1}`$ | 只用来算第 $`t+1`$ 个 token 的输出 | **不会** —— 下一步用的是全新的 $`q_{t+2}`$ |
| $`k_{t+1}`$ | 参与 $`q_{t+1} K^\top`$ 中的一列 | **会** —— $`q_{t+2}, q_{t+3}, \dots`$ 都要点乘它 |
| $`v_{t+1}`$ | 在加权和中贡献一行 | **会** —— 后续每一步都要加权它 |

### 2.4 因果掩码（causal mask）的关键作用

为什么"历史的 $`K, V`$ 不会变"？因为 decoder 是 **causal** 的：位置 $`i`$ 的输出只依赖位置 $`\le i`$ 的输入。这意味着加入新 token $`x_{t+1}`$ 不会改变 $`K_{1:t}, V_{1:t}`$ 的数值——它们仅仅依赖 $`x_{1:t}`$。

如果是 BERT 那种 bidirectional self-attention，新加 token 会改变所有位置的 $`K, V`$，KV Cache 就根本不成立。Causal mask 是 KV Cache 能存在的**结构性前提**。

### 2.5 为什么 Q 不需要被缓存：一句话总结

> **每一步生成只需要"当前 step 的那一个 query"。这个 query 用完就丢，下一步用的是另一个全新的 $`q`$；而 $`K, V`$ 是历史信息载体，每一步都要被"现在的 q"再次点乘和加权——所以前者一次性，后者累积复用。**

### 2.6 补充：为什么 prefill 不能只算最后一个 token 的 Query

这里有一个很容易卡住的点：

> [!question] 既然生成下一个 token 只需要最后一个 query，那 prefill 阶段为什么还要算前面 token 的 query？  
> 只算 prompt 最后一个 token 的 query，不也能得到下一个 token 的 logits 吗？

答案是：**在最后一层、最后一个位置上，确实只需要最后一行 query；但为了走到最后一层，你必须先构造出每一层、每个历史 token 的 hidden state，才能得到每一层的历史 K/V cache。**

对第 $`\ell`$ 层来说，每个位置 $`i`$ 的 $`Q/K/V`$ 都来自上一层 hidden state：

```math
q_i^{(\ell)} = h_i^{(\ell-1)}W_Q^{(\ell)},\quad
k_i^{(\ell)} = h_i^{(\ell-1)}W_K^{(\ell)},\quad
v_i^{(\ell)} = h_i^{(\ell-1)}W_V^{(\ell)}
```

而第 $`i`$ 个位置进入下一层前，需要先算出自己的 attention 输出：

```math
h_i^{(\ell)}
=
\text{Block}^{(\ell)}\!\left(
h_i^{(\ell-1)},
\text{softmax}\!\left(\frac{q_i^{(\ell)}K_{1:i}^{(\ell)\top}}{\sqrt{d_k}}\right)V_{1:i}^{(\ell)}
\right)
```

注意这个依赖关系：**下一层的历史 K/V 不是从原始 embedding 直接来的，而是从上一层处理后的 $`h_i^{(\ell)}`$ 来的。** 所以如果 prefill 阶段不计算前面 token 的 query，就得不到前面 token 在这一层的输出 $`h_i^{(\ell)}`$，也就无法继续计算它们在下一层的 $`k_i^{(\ell+1)}, v_i^{(\ell+1)}`$。

#### 只算最后一个 query 会在哪里断掉

假设 prompt 有 $`T`$ 个 token，模型有多层 Transformer。

在**第 1 层**，如果你只想算最后位置 $`h_T^{(1)}`$，看起来可以只算 $`q_T^{(1)}`$，再配上所有 token 的 $`K_{1:T}^{(1)}, V_{1:T}^{(1)}`$。

但问题马上出现在**第 2 层**：

- 第 2 层最后 token 需要 attend 到第 2 层的历史 $`K_{1:T}^{(2)}, V_{1:T}^{(2)}`$；
- 这些 $`K_{1:T}^{(2)}, V_{1:T}^{(2)}`$ 需要由第 1 层输出 $`h_{1:T}^{(1)}`$ 投影得到；
- 你如果第 1 层只算了 $`h_T^{(1)}`$，那 $`h_1^{(1)}, \dots, h_{T-1}^{(1)}`$ 都没有；
- 所以第 2 层的历史 $`K/V`$ 根本构造不出来。

因此，**只算最后一个 query 的做法在一层模型里看似可行，但在真实多层 LLM 里会从第二层开始断掉**。

#### 正确理解：prefill 的 Q 是为了"建层间状态"，decode 的 Q 是为了"查缓存"

可以把 prefill 和 decode 的 query 分工分开看：

| 阶段 | 为什么需要 Q | Q 用完后是否缓存 |
|---|---|---|
| Prefill | 计算 prompt 中每个位置的每层输出 $`h_i^{(\ell)}`$，从而构造下一层输入和本层 KV cache | 不缓存 |
| Decode | 只计算当前最后一个位置的输出，用它产生下一个 token 的 logits | 不缓存 |

所以"只需要最后一个 query"这句话只适用于 **decode 单步的 attention 输出**：

```math
\text{last\_row}_T
=
\text{softmax}\!\left(\frac{q_T^{(L)}K_{1:T}^{(L)\top}}{\sqrt{d_k}}\right)V_{1:T}^{(L)}
```

但 prefill 不是只求最后一层最后一行这么简单。它还要为**所有层**建立：

```math
K_{1:T}^{(1)}, V_{1:T}^{(1)},\quad
K_{1:T}^{(2)}, V_{1:T}^{(2)},\quad
\dots,\quad
K_{1:T}^{(L)}, V_{1:T}^{(L)}
```

这些 cache 以后 decode 每一步都要用。前面 token 的 query 虽然不会被缓存，但它们在 prefill 里承担了一个必要任务：**把前面 token 的表示逐层更新到正确深度，让它们能成为后续层的 K/V。**

> [!tip] 面试版一句话
> Decode 只算最后一个 query，是因为每一步只需要最后位置的输出；Prefill 要算所有 query，是因为多层 Transformer 需要先算出所有 prompt token 在每一层的 hidden state，才能构造每一层的 KV cache。前面 token 的 Q 不缓存，但不能不算。

---

## 3. 张量形状（shape）层面的解释

以多头注意力 (MHA) 为例，标准 5D 形状记号 `(B, H, T, D_h)`：

- `B`：batch size
- `H`：head 数
- `T`：序列长度（缓存中累积的 token 数）
- `D_h`：每个 head 的维度（通常 $`d / H`$）

### 3.1 Prefill 阶段（prompt 长度 $`T_p`$）

- $`Q, K, V`$ 形状均为 $`(B, H, T_p, D_h)`$。
- 注意力分数 $`QK^\top`$：$`(B, H, T_p, T_p)`$，是个**方阵**。

### 3.2 Decode 阶段（生成第 $`t+1`$ 个 token）

- 新 query：$`q_{t+1}`$ 形状 $`(B, H, 1, D_h)`$ ——**T 维只有 1**。
- 缓存的 K：$`(B, H, t, D_h)`$；append 后变 $`(B, H, t+1, D_h)`$。
- 缓存的 V：$`(B, H, t, D_h)`$；append 后变 $`(B, H, t+1, D_h)`$。
- 注意力分数 $`q_{t+1} K^\top`$：$`(B, H, 1, t+1)`$ —— 只有 **一行**。
- 输出：$`(B, H, 1, D_h)`$。

显然，**新的 q 永远只有 1 行**，下一步又会被一行新的 q 替换。它的"生命周期"只有当前 step。把它存进 cache 没有任何下游消费者，写入是纯粹的浪费（多一次 HBM 写）。

而 $`K, V`$ 每一行都对应历史中一个固定 token，会被未来**所有 step** 的 $`q`$ 反复点乘——所以缓存它们能省下"重新算一遍线性投影 $`XW_K, XW_V`$"的所有工作。

---

## 4. 如果硬要 cache Q 会怎么样？为什么没有收益

可以从三个角度看："Q Cache" 既无收益，又有成本：

1. **没有未来消费者**：第 $`t+1`$ 步的 $`q_{t+1}`$ 不会出现在任何 $`t+2, t+3, \dots`$ 步的 attention 计算里。$`q_{t+2}`$ 是用 $`x_{t+2}`$ 重新算的，跟历史 $`q`$ 没有任何关系。
2. **Causal mask 的方向性**：mask 让"未来的 q 看历史的 k/v"，而**不是**"历史的 q 看未来的 k/v"。所以历史 q 即使保留，也没有合法的去处去参与新一步的 softmax。
3. **白白多花显存与带宽**：每一步多 append 一行 $`(B, H, 1, D_h)`$ 的 q，长上下文累计起来与 K Cache 同量级。Decode 是 memory-bound 阶段，多一次写入和潜在的读出会直接拖慢延迟。

> 一些初学者会说"那为啥不把所有 q 存下来当 prefix 用"——这其实是把"q 当 k 用"，已经偏离了 attention 的定义：q 与 k 来自不同投影矩阵 $`W_Q \neq W_K`$，语义不同。

---

## 5. 显存占用公式与瓶颈分析

### 5.1 KV Cache 显存公式

每层每个 token，需要存储一份 K 和一份 V，每份大小 $`H \cdot D_h`$：

```math
\boxed{\text{KV-Cache bytes} = 2 \times B \times L \times H \times T \times D_h \times \text{dtype\_size}}
```

其中 $`L`$ 是 transformer 层数，$`2`$ 来自 K 和 V 各一份。

**示例（Llama-2-7B, FP16）**：$`L=32`$, $`H=32`$, $`D_h=128`$, dtype=2 bytes，$`B=1`$：

```math
2 \times 1 \times 32 \times 32 \times T \times 128 \times 2 = 524{,}288 \cdot T \text{ bytes} \approx 0.5\ \text{MB} / \text{token}
```

- 4K 上下文：约 2 GB；
- 32K 上下文：约 16 GB —— 已经接近模型权重本身大小；
- 大 batch 下显存压力呈线性放大。

这就是为什么 KV Cache 已经成为 LLM 推理的**头号显存瓶颈**，也催生了一整批优化方向。

### 5.2 Prefill 是 compute-bound，Decode 是 memory-bound

- **Prefill**：一次吃完几百到几千 token，矩阵乘法形状大、并行度高，GPU 可以打满 TFLOPS 利用率 → **算力瓶颈**。
- **Decode**：每步只有 1 个 token，每层的 GEMV（矩阵 × 向量）算术强度（arithmetic intensity）极低，绝大多数时间花在**把模型权重和 KV Cache 从 HBM 搬到 SM 寄存器**上 → **显存带宽瓶颈**。

这一区分直接决定了优化策略：prefill 想办法 batch 起来吃满算力；decode 想办法降低每 token 需要搬运的字节数（KV 量化、MQA/GQA/MLA、PagedAttention 等）。

---

## 6. 围绕 KV Cache 的进阶优化（面试高频追问点）

### 6.1 MHA → MQA → GQA → MLA：减小每 token 的 KV 字节数

| 变体 | K/V head 数 | KV Cache 大小 | 代表模型 |
|---|---|---|---|
| **MHA**（标准多头） | $`H`$ | $`2 \cdot H \cdot D_h`$ | GPT-3、原始 Llama-1 |
| **MQA**（Multi-Query） | $`1`$ | $`2 \cdot D_h`$ | PaLM、Falcon |
| **GQA**（Grouped-Query） | $`G`$（$`1 < G < H`$） | $`2 \cdot G \cdot D_h`$ | Llama-2/3、Mistral |
| **MLA**（Multi-head Latent Attention） | 等价压缩到低秩潜变量 $`d_c`$ | $`\approx d_c`$（远小于 MHA） | DeepSeek-V2/V3/R1 |

- **MQA**：所有 query head 共用一对 K/V → 缓存压到 $`1/H`$，但效果有损。
- **GQA**：每 $`H/G`$ 个 query head 共用一对 K/V → 在效果和缓存之间折中。
- **MLA**（DeepSeek 的核心创新）：把 K/V 通过低秩矩阵 $`W^{DKV}`$ 压缩到一个潜变量 $`c_t \in \mathbb{R}^{d_c}`$，**只缓存 $`c_t`$**，attention 计算时再"按需解压"成每头独立的 K/V。论文显示 MLA 能在显著缩小 KV Cache 的同时**效果不输甚至略好于 MHA**（因为低秩瓶颈起到了正则化作用）。Decoupled RoPE 解决了旋转位置编码与低秩压缩的兼容性问题。

### 6.2 PagedAttention（vLLM）：用操作系统分页管理 KV

**问题**：传统实现把每个 sequence 的 KV Cache 在显存里**连续分配**到最大长度，导致 60–80% 的内部碎片浪费。

**核心思想**（论文 *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP'23）：

- 借鉴虚拟内存分页：把 KV Cache 切成固定大小的 **blocks**（页），可以散落在物理显存的任意位置；
- 每个请求维护一张 **block table**（页表），记录"逻辑块号 → 物理块号"的映射；
- 不同请求间还能 **share blocks**（如 system prompt 共享、beam search 各分支共享前缀）。

**效果**：内存浪费降到 < 4%，相比 FasterTransformer 等系统吞吐**提升 2–4×**。这是 vLLM 成为事实标准推理框架的根本原因。

### 6.3 KV Cache 量化（FP16 → FP8 / INT4）

把缓存中的 K, V 从 FP16 压到 FP8、INT8 甚至 INT4，按公式直接线性缩小显存占用。

代价是 attention 数值精度下降；实践中 K 通道的 outlier 较多，常需要逐 channel quantize 或者保留部分高精度通道。NVFP4 等新格式专门为长上下文 KV 量化设计。

### 6.4 Sliding Window + StreamingLLM（attention sink）

- **Sliding window attention**（Mistral）：每个 token 只 attend 最近 $`W`$ 个 token，KV Cache 大小封顶在 $`W`$。
- **StreamingLLM**（Xiao et al., ICLR'24）：发现单纯 sliding window 一旦"窗口滑过最初几个 token"，模型就会崩溃；原因是初始几个 token 充当 **attention sink**——softmax 的归一化导致 attention 总要"倒"在某些位置上，初始 token 因为可被所有后续位置看到，自然成为这个 sink。
- **方案**：始终保留最初的若干（通常 4 个）token 的 KV + 一个 rolling window，可让 4M+ token 持续生成而模型不崩；相比"每次重算窗口"快 22×。

### 6.5 Prefix Caching / 系统提示复用

如果多个请求共享同一段 system prompt 或 few-shot 示例，可以在框架层面**复用**这段 prefix 的 KV Cache，无需为每个请求重新 prefill。vLLM 的 *Automatic Prefix Caching* 与许多商业 LLM API 提供的 "prompt caching" 都是此思路。在长系统提示场景下能省掉 50%+ 的 prefill 算力与首 token 延迟。

---

## 7. 常见误区辨析

1. **"Attention 是对称的，q 和 k 角色对偶，应该一视同仁"** ——错。Self-attention 在数学形式上 $`q_i \cdot k_j`$ 看似对称，但**生成过程是不对称的**：q 来自"当前 step"，k/v 来自"所有历史 step"。Causal mask 又进一步打破时间对称性。
2. **"既然要存 K，那 Q 投影矩阵 $`W_Q`$ 是不是也可以省"** ——不可以。每一步仍然需要 $`W_Q x_{t+1}`$ 这次乘法去拿到 $`q_{t+1}`$，否则没有 query 去"问"历史。
3. **"KV Cache 是为了节省显存"** ——恰恰相反，**KV Cache 是用显存换算力**。它会显著增加显存占用（长上下文下甚至超过模型权重本身）。所有围绕 KV Cache 的二次优化（量化、MLA、Paged）都是在收回这部分显存成本。
4. **"Prefill 也能用 KV Cache 加速"** ——KV Cache 在 prefill 阶段没有加速作用，prefill 是从零开始建立 cache 的过程。Prefix Caching 是**跨请求**复用，不是单请求 prefill 内部加速。
5. **"GQA / MLA 是对训练的优化"** ——它们主要优化的是**推理**（缩小每 token KV 字节数 → 降低 decode 阶段的显存带宽压力），训练阶段相对收益小得多。

---

## 8. 一页式回答模板（面试快速作答用）

> **Q：为什么 LLM 推理只 cache K 和 V，不 cache Q？**
>
> 1. Self-attention 公式是 $`\text{softmax}(QK^\top/\sqrt{d_k})V`$。
> 2. Decoder 是 causal 的，新 token 不会改变历史 token 的 K/V，所以 K/V 可以 append-only 复用。
> 3. 每生成一步，**只需要当前最后一个 token 的那一行 q**，它跟所有历史 K/V 算一次 attention 后就被丢弃；下一步会算一个全新的 q。q 没有未来消费者。
> 4. 形状上：q 是 $`(B, H, 1, D_h)`$，K/V 是 $`(B, H, T, D_h)`$ —— 缓存 q 既没下游用途，又徒增显存写入和带宽。
> 5. 所以 KV Cache 把 attention 单步从 $`O(T^2)`$ 降到 $`O(T)`$，整个生成从 $`O(N^3)`$ 降到 $`O(N^2)`$，实测 3–5× 加速。
> 6. 代价是显存：$`2 B L H T D_h \cdot \text{dtype}`$，长上下文下成为头号瓶颈，由此衍生 MQA / GQA / MLA / PagedAttention / KV 量化 / StreamingLLM / Prefix Caching 等一系列优化。

---

## 参考资料

- [HuggingFace — KV Caching Explained: Optimizing Transformer Inference Efficiency](https://huggingface.co/blog/not-lain/kv-caching)
- [HuggingFace — KV Cache from scratch in nanoVLM](https://huggingface.co/blog/kv-cache)
- [HuggingFace — MLA: Redefining KV-Cache Through Low-Rank Projections and On-Demand Decompression](https://huggingface.co/blog/NormalUhr/mla-explanation)
- [HuggingFace docs — Cache strategies / Caching](https://huggingface.co/docs/transformers/en/kv_cache)
- [HuggingFace docs — Caching / Attention matrices](https://huggingface.co/docs/transformers/main/cache_explanation)
- [HuggingFace blog — Attention Sinks in LLMs for endless fluency](https://huggingface.co/blog/tomaarsen/attention-sinks)
- [HuggingFace blog — Unlocking Longer Generation with KV Cache Quantization](https://huggingface.co/blog/kv-cache-quantization)
- [HuggingFace blog — Prefill and Decode for Concurrent Requests](https://huggingface.co/blog/tngtech/llm-performance-prefill-decode-concurrent-requests)
- [Pierre Lienhart — LLM Inference Series: 3. KV caching explained](https://medium.com/@plienhar/llm-inference-series-3-kv-caching-unveiled-048152e461c8)
- [João Lages — Transformers KV Caching Explained](https://medium.com/@joaolages/kv-caching-explained-276520203249)
- [Sebastian Raschka — Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)
- [Neptune.ai — Transformers Key-Value Caching Explained](https://neptune.ai/blog/transformers-key-value-caching)
- [知乎 — 为什么加速LLM推断有KV Cache而没有Q Cache？](https://www.zhihu.com/question/653658936)
- [知乎 — 大模型推理加速：看图学KV Cache](https://zhuanlan.zhihu.com/p/662498827)
- [知乎 — 大模型推理性能优化之KV Cache解读](https://zhuanlan.zhihu.com/p/630832593)
- [知乎 — 动图看懂什么是KV Cache](https://zhuanlan.zhihu.com/p/19489285169)
- [博客园 — 探秘Transformer系列之（20）— KV Cache](https://www.cnblogs.com/rossiXYZ/p/18799503)
- [arXiv 2309.06180 — Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180)
- [vLLM docs — Paged Attention](https://docs.vllm.ai/en/latest/design/paged_attention/)
- [arXiv 2309.17453 — Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)
- [MIT Han Lab — StreamingLLM project page](https://hanlab.mit.edu/projects/streamingllm)
- [DeepSeek-V3 Explained 1: Multi-head Latent Attention (Towards Data Science)](https://towardsdatascience.com/deepseek-v3-explained-1-multi-head-latent-attention-ed6bee2a67c4/)
- [arXiv 2502.07864 — TransMLA: Multi-Head Latent Attention Is All You Need](https://arxiv.org/html/2502.07864v2)
- [Towards Data Science — Prefill Is Compute-Bound. Decode Is Memory-Bound.](https://towardsdatascience.com/prefill-is-compute-bound-decode-is-memory-bound-why-your-gpu-shouldnt-do-both/)
- [NVIDIA Technical Blog — Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
- [NVIDIA Technical Blog — Optimizing Inference for Long Context with NVFP4 KV Cache](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/)
- [BentoML — Prefill-Decode Disaggregation (LLM Inference Handbook)](https://bentoml.com/llm/inference-optimization/prefill-decode-disaggregation)
