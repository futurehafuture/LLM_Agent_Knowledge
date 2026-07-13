# Lecture 2: Resource Accounting (计算与内存资源核算) 深度笔记

本笔记基于斯坦福 CS336 (Language Modeling from Scratch) 第二讲的课堂内容整理，涵盖大模型训练与推理中的计算与内存资源核算（Resource Accounting），包括数据类型、GPU 存储与计算、Einops 库的使用、FLOPs 计算、算术强度与 Roofline 模型分析，以及训练显存分析与各种显存优化技术（如梯度累加、激活值重计算/检查点技术）。笔记力求详尽，公式丰富，适合复习和传阅。

> **课程信息**：CS336 · Spring 2026 · 主题：Resource Accounting (计算与内存资源核算)

---

# Part 1: Tensors, Memory and Data Types (张量、内存与数据类型)

## Slide 1: Announcements & Recap (课程公告与回顾)

![Marin Forecasts](https://pbs.twimg.com/media/HE1P1HmaUAAjLXF?format=jpg&name=medium)

### 讲解

本节课首先发布了一些课程公告，包括加入 CS336 Slack 频道、在 Modal 平台使用 Stanford 邮箱注册并阅读 AI 政策指南以及集群使用指南。

另外，主讲人分享了令人兴奋的研究进展：**Marin 1e23 FLOPs** 规模的语言模型训练已经顺利完成，其曲线完全符合之前的预测（[matched forecasts](https://x.com/WilliamBarrHeld/status/2039373983632814318)）。这进一步印证了通过较小规模实验拟合缩放定律（Scaling Laws）并预测更大规模训练结果的有效性。

本节课的核心主题是**资源核算（Resource Accounting / Systems）**。在固定的计算和内存资源预算下，我们应当如何最大化计算效率？在进行任何大模型训练和系统设计之前，我们必须首先理解计算和显存资源是如何被消耗的。

---

## Slide 2: Motivating Questions (引导性问题：大模型算力与显存估算)

### 讲解

在深入底层实现之前，我们先来看两个在实际大模型研发中非常经典的“估算问题”（Back-of-the-envelope calculation / Napkin math），这些粗略的计算可以帮助我们迅速获得系统瓶颈和资源消耗的宏观直觉：

#### **问题 1：在 1024 张 H100 GPU 上，将一个拥有 70B（700 亿）参数的模型在 15T（15 万亿）Tokens 的数据集上训练，需要花费多少天？**

我们先进行 FLOPs（浮点运算次数）的核算：
- 按照经验公式，对于 Transformer 架构，每个 Token 的训练（包含前向与反向传播）大约需要 $`6`$ 次 FLOPs 乘以参数量。因此总 FLOPs 计算如下：
  ```math
  \text{Total FLOPs} = 6 \times 70 \times 10^9 \times 15 \times 10^{12} = 6.3 \times 10^{24} \text{ FLOPs}
  ```
- 单张 H100 的峰值性能（BF16/FP16，不考虑稀疏性）为：
  ```math
  \text{H100 Peak Performance} = \frac{1979 \times 10^{12}}{2} \approx 989.5 \text{ TFLOP/s}
  ```
- 在实际大规模集群训练中，由于通信开销、数据存取瓶颈等，模型算力利用率（MFU, Model FLOPs Utilization）通常取 $`50\%`$ （$`0.5`$）。
- 那么，1024 张 H100 每天所能提供的实际 FLOPs 数量为：
  ```math
  \text{FLOPs per day} = 989.5 \times 10^{12} \times 0.5 \times 1024 \times 86400 \approx 4.38 \times 10^{22} \text{ FLOPs/day}
  ```
- 因此，训练所需要的天数为：
  ```math
  \text{Days} = \frac{6.3 \times 10^{24}}{4.38 \times 10^{22}} \approx 143.8 \text{ 天}
  ```

#### **问题 2：利用 8 张 H100 GPU 结合 AdamW 优化器，能够训练的最大模型参数量是多少？**

我们先来计算每个参数在训练过程中需要消耗的字节数：
- 在混合精度训练下，**模型参数（Parameters）**通常以 16 位浮点数（2 字节）存储。
- **梯度（Gradients）**同样以 16 位浮点数（2 字节）存储。
- **AdamW 优化器状态（Optimizer States）**需要为每个参数保存一阶矩 $`m_t`$（FP32，4 字节）和二阶矩 $`v_t`$（FP32，4 字节），共计 8 字节。
- 因此，每个参数在不考虑激活值（Activations）的情况下，显存基础消耗为：
  ```math
  \text{Bytes per parameter} = 2 \text{ (参数)} + 2 \text{ (梯度)} + 8 \text{ (优化器状态)} = 12 \text{ 字节}
  ```
- 8 张 80GB 显存的 H100 GPU 总显存为：
  ```math
  \text{Total HBM} = 8 \times 80 \times 10^9 = 640 \text{ GB}
  ```
- 假设将显存全部用于存储这三部分，那么理论上支持的最大模型参数量为：
  ```math
  \text{Max Parameters} = \frac{640 \times 10^9}{12} \approx 53.3 \text{ B (533 亿参数)}
  ```

*注：这是一个极度简化的上界，因为在实际训练中，前向传播过程中产生的**激活值（Activations）**需要存放在显存中以备反向传播使用，这取决于 Batch Size 与 Sequence Length，往往会占用极大的显存空间。*

---

## Slide 3: Tensor Basics (张量基础)

### 讲解

在深度学习和 Transformer 中，**张量（Tensor）**是存储和流转一切数据的核心单位，包括：
- **数据（Data）**：例如输入的 Token ID、目标标签。
- **模型参数（Parameters）**：例如多头注意力的投影矩阵 $`W_q, W_k, W_v`$ 以及 MLP 层的参数。
- **梯度（Gradients）**：用于参数更新的偏导数值。
- **优化器状态（Optimizer State）**：例如 Adam 中的一阶和二阶矩。
- **前向传播中间状态（Activations）**：前向计算时各层产生的临时张量。

张量的**秩（Rank）**指其维度的数量。例如：
- 秩 1 张量（一维向量）：`x = torch.zeros(4)`
- 秩 2 张量（二维矩阵）：`x = torch.zeros(4, 8)`
- 秩 3 张量（三维张量）：`x = torch.zeros(4, 8, 2)`

在 Transformer 中，我们通常会遇到**秩 4 张量**。例如在注意力机制中表示隐藏状态或输入序列特征：
```math
\text{Shape: } [B, S, H, D]
```
- $`B`$ (Batch size)：批次大小
- $`S`$ (Sequence length)：序列长度
- $`H`$ (Number of heads)：注意力头数
- $`D`$ (Hidden dimension per head)：每个注意力头的隐藏层维度大小

真实生产环境中的大模型，其参数量极其庞大。例如 [DeepSeek v3.2 model on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3.2?show_file_info=model.safetensors.index.json) 拥有的庞大参数张量被分割存储在数十个 `.safetensors` 分片中。

---

## Slide 4: Data Types: FP32, FP16, BF16 (数据类型：FP32, FP16, BF16 详解)

| 数据类型 | 符号位 (Sign) | 指数位 (Exponent) | 尾数位 (Mantissa) | 总位数 (Bits) | 动态范围 | 相对精度 | 字节数 (Bytes) |
|---|---|---|---|---|---|---|---|
| **FP32** | 1 | 8 | 23 | 32 | $`1.4 \times 10^{-45} \sim 3.4 \times 10^{38}`$ | 高 (24 bits) | 4 |
| **FP16** | 1 | 5 | 10 | 16 | $`6.1 \times 10^{-5} \sim 65504`$ | 中 (11 bits) | 2 |
| **BF16** | 1 | 8 | 7 | 16 | $`1.4 \times 10^{-45} \sim 3.4 \times 10^{38}`$ | 低 (8 bits) | 2 |

#### **FP32 (单精度浮点数)**
![FP32 Format](images/fp32.png)
- 传统的科学计算与深度学习默认类型，每个元素占用 4 字节（32 位）。
- 尽管精度和范围极大，但在大模型训练中直接采用 FP32 会面临巨大的显存与带宽压力。例如 GPT-3 包含的前馈层矩阵单精度占用达 2.3 GB 显存。

#### **FP16 (半精度浮点数)**
![FP16 Format](images/fp16.png)
- 仅占用 2 字节（16 位），可直接减少一半内存占用和传输带宽。
- **缺点**：指数位仅有 5 位，导致动态范围（Dynamic Range）较小。当数值小于 $`6.1 \times 10^{-5}`$ 时就会发生**下溢（Underflow）**，变为 0。在训练大模型时极易发生数值不稳定（如 NaN、梯度爆炸或消失）。

#### **BF16 (Brain Floating Point)**
![BF16 Format](images/bf16.png)
- Google Brain 专为机器学习开发，与 FP16 一样是 16 位，但分配给指数位 8 位（与 FP32 相同），尾数位 7 位。
- **优势**：拥有和 FP32 完全一样的动态范围，基本消除了下溢风险（例如，对于 $`10^{-8}`$ 不会发生下溢）。虽然尾数变少导致相对精度略逊，但深度学习对“范围”比对“精度”更敏感，因此 BF16 成为了现代大模型预训练的黄金标准。

---

## Slide 5: Mixed Precision & FP8/FP4 (混合精度训练与 FP8/FP4 前沿数据类型)

### 讲解

#### **混合精度训练 (Mixed Precision Training)**
尽管 BF16 拥有大动态范围，但在更新参数（参数累加小数值）时，低尾数精度可能造成数值丢失（Rounding Error）。
- **解决方案**：前向传播、反向传播中使用 BF16 以减少显存和计算时间；但在优化器状态中，使用 **FP32 副本**来累计梯度和更新参数，以确保训练的数值稳定性。
- PyTorch 提供了自动混合精度（AMP）库（如 [PyTorch AMP docs](https://pytorch.org/docs/stable/amp.html)），当调用 `torch.amp.autocast("cuda", dtype=torch.bfloat16)` 时，系统会自动选择将计算安全地 cast 为 BF16（例如矩阵乘法），而对一些对精度敏感的激活操作（例如指数运算 `exp`）保留更高精度。

#### **FP8 (8位浮点数)**
于 2022 年标准化，专为机器学习设计。NVIDIA H100 硬件天然支持 FP8。它包含两种格式（参考 [FP8 Primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)）：
- **E4M3**（4位指数，3位尾数）：范围为 $`[-448, 448]`$，精度相对较高，多用于前向传播的激活值和权重。
- **E5M2**（5位指数，2位尾数）：范围为 $`[-57344, 57344]`$，动态范围更大，多用于反向传播中范围波动较大的梯度。
- 引入 FP8 能够使计算吞吐量相比 16 位浮点数再次翻倍。

#### **FP4 (4位浮点数)**
NVIDIA 在 2025 年推出了 `nvfp4` 数据类型，每个值仅占 4 位，其值只能是 16 个离散点（例如 $`-6, -4, -3, \dots, 6`$）。通过采用分块比例因子（scale factor per block）来在微观上动态缩放，从而在推理中达到极高的效率。例如，Nemotron 3 Super 模型就是在 NVFP4 精度下完成部署的。

---

## Slide 6: Tensors on GPUs (GPU 上的张量存储与计算)

### 讲解

在 PyTorch 中，新建的张量默认存储在 CPU 内存中：
```python
x = torch.zeros(32, 32)
assert x.device == torch.device("cpu")
```

为了利用 GPU 的并行计算能力，需要通过以下方式显式地将张量移动到 GPU 显存（HBM）中：

![CPU GPU Memory](images/cpu-gpu.png)

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
```
或者在创建张量时直接在 GPU 上初始化：
```python
with torch.device(device):
    x = torch.zeros(32, 32)
    assert x.device == device
```

---

# Part 2: Tensor Algebra & Einops (张量代数与 Einops 库)

## Slide 7: Einops Motivation (传统维度操作痛点)

### 讲解

在传统的 PyTorch 中，改变张量形状和维度的操作（例如 `view`, `transpose`, `permute`, `squeeze` 等）不仅可读性极差，而且极其容易出错。

例如，实现一个简单的自注意力机制维度的矩阵乘法：
```python
# x 形状为 (batch, seq, hidden)
# y 形状为 (batch, seq, hidden)
# 传统的做法：
z = x @ y.transpose(-2, -1)  # 得到 (batch, seq, seq)
```
代码中的 `-2, -1` 或者是 `transpose(1, 2)` 需要开发者在脑海中追踪复杂的维度映射。随着张量维度的增加，这类代码会成为 Bug 的温床。

---

## Slide 8: Einops Einsum (爱因斯坦求和约定与 einsum 算子)

### 讲解

**Einops** 是一个改变张量维度操作范式的第三方库，其引入了**显式命名维度**的表达方式，底层基于 Einstein 求和约定（Einstein Summation Notation）。（参考 [Einops Tutorial](https://einops.rocks/1-einops-basics/)）

`einsum` 可以优雅地表示各种广义矩阵乘法，并进行清晰的维度簿记。

#### **基础矩阵乘法**
```python
x = torch.ones(3, 4)  # seq1 hidden
y = torch.ones(4, 3)  # hidden seq2

# 传统做法
z = x @ y  # 得到 (seq1, seq2)

# Einops 做法
z = einsum(x, y, "seq1 hidden, hidden seq2 -> seq1 seq2")
```

#### **批次注意力矩阵乘法**
```python
x = torch.ones(2, 3, 4)  # batch seq1 hidden
y = torch.ones(2, 3, 4)  # batch seq2 hidden

# 传统做法
z = x @ y.transpose(-2, -1)

# Einops 做法
z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")
```
在 `einsum` 字符串中：
- 逗号分隔各个输入张量的维度映射（例如 `"batch seq1 hidden"` 与 `"batch seq2 hidden"`）。
- 箭头 `->` 右侧定义了输出张量的目标维度映射。
- **核心约定**：在箭头右侧**未出现**但在左侧出现的维度名称（例如 `hidden`），代表在该维度上进行求和（Summation）。
- 另外，可以使用省略号 `...` 代表广播维度（Broadcasting）：
```python
z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")
```

---

## Slide 9: Einops Reduce (张量规约 reduce)

### 讲解

`reduce` 允许在指定的维度上进行标量规约操作（如 `sum`, `mean`, `max`, `min` 等），且维度含义极其直观：
```python
x = torch.ones(2, 3, 4)  # batch seq hidden

# 传统做法
y = x.sum(dim=-1)

# Einops 做法
y = reduce(x, "... hidden -> ...", "sum")
```
上面的 `... hidden -> ...` 字符串明确表示：消除最后一个维度 `hidden`，并在该维度上使用 `"sum"`（求和）操作进行规约。

---

## Slide 10: Einops Rearrange (张量重组 rearrange 与多头注意力维度重组)

### 讲解

在 Transformer（尤其是多头自注意力机制 Multi-Head Attention）中，我们经常需要将最后一个隐藏层维度拆分为“头数 $`\times`$ 每个头的维度”，在传统的 PyTorch 中这需要组合调用 `view` 和 `transpose`，而 `rearrange` 可以一步且安全地完成：

```python
x = torch.ones(3, 8)  # seq total_hidden
# total_hidden = heads * hidden1 (假设 heads = 2, hidden1 = 4)
w = torch.ones(4, 4)  # hidden1 hidden2

# 1. 拆分维度
x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)

# 2. 变换
x = einsum(x, w, "... hidden1, hidden1 hidden2 -> ... hidden2")

# 3. 合并维度回来
x = rearrange(x, "... heads hidden2 -> ... (heads hidden2)")
```
通过括号括起来的 `(heads hidden1)` 显式表明这两个维度是由 `total_hidden` 拆分出来的，且直接传递 `heads=2` 参数。这样的写法杜绝了隐式重组引入的物理含义错乱。

---

# Part 3: Compute Accounting & FLOPs (计算开销核算与 FLOPs 计算)

## Slide 11: FLOPs vs FLOP/s (FLOPs 与 FLOP/s 概念区分)

### 讲解

在系统资源核算中，有两个极易混淆且读音相同的缩写：
- **FLOPs (Floating-point operations)**：浮点运算次数。它是一个**数量单位**，用来衡量完成某项计算任务（如训练一个模型、运行一次前向传播）所需的总计算量。
- **FLOP/s 或 FLOPS (Floating-point operations per second)**：每秒浮点运算次数。它是一个**速度单位**，用来衡量计算机硬件的运算速率或吞吐量。

#### **工业界基准直觉**
- **GPT-3 (175B 参数)**：训练总计算量约为 $`3.14 \times 10^{23}`$ FLOPs。（参考 [Demystifying GPT-3](https://lambdalabs.com/blog/demystifying-gpt-3)）
- **GPT-4**：业界推测其训练总计算量达到了 $`2 \times 10^{25}`$ FLOPs 级别。（参考 [GPT-4 Details Revealed](https://patmcguinness.substack.com/p/gpt-4-details-revealed)）
- NVIDIA H100 标称单卡峰值性能为 $`1979 \text{ TFLOP/s}`$（混合精度，开启硬件缩放）。在不使用稀疏计算时，标称峰值为一半，即 $`989.5 \text{ TFLOP/s}`$。
- 假设我们拥有 8 张 H100 GPU 持续满负荷运转两周，理论上能提供的最大计算量为：
  ```math
  \text{Total FLOPs} = 8 \times 2 \times 7 \times 86400 \times 989.5 \times 10^{12} \approx 9.57 \times 10^{21} \text{ FLOPs}
  ```

---

## Slide 12: Peak Performance & Tensor Operations (硬件峰值性能与线性模型的 FLOPs 计算)

### 讲解

在深度学习中最基础的运算是矩阵乘法（Matrix Multiplication, GEMM）。我们来核算一个经典的线性层乘法所消耗的 FLOPs。

#### **矩阵乘法的 FLOPs 计数**
设输入矩阵 $`X \in \mathbb{R}^{B \times D}`$，权重矩阵 $`W \in \mathbb{R}^{D \times K}`$，输出 $`Y = XW \in \mathbb{R}^{B \times K}`$：
- 对于输出矩阵 $`Y`$ 的每一个元素 $`Y_{ik} = \sum_{j=1}^D X_{ij} W_{jk}`$，它是由 $`D`$ 个元素的乘法与 $`D-1`$ 个加法组成的点积。
- 在硬件计算中，通常将一次乘法和一次加法统称为 2 次浮点运算（或者称为 1 次 FMA - Fused Multiply-Add）。
- 故计算每个元素共需 $`2D`$ 次 FLOPs（严格为 $`2D-1`$）。
- 输出矩阵一共有 $`B \times K`$ 个元素，因此该矩阵乘法的总计算量为：
  ```math
  \text{FLOPs} = 2 \times B \times D \times K
  ```

#### **吞吐量性能基准测试**
我们可以在 PyTorch 中测试该操作在 GPU 上的执行时间：
```python
x = torch.ones(16384, 32768, device="cuda", dtype=torch.bfloat16)
w = torch.randn(32768, 8192, device="cuda", dtype=torch.bfloat16)
# 计算量 FLOPs = 2 * 16384 * 32768 * 8192 = 8.79e12 FLOPs

# 运行 benchmark 测试平均耗时 (s)
actual_time = benchmark(lambda: x @ w)

# 实际吞吐率
actual_flop_per_sec = (2 * 16384 * 32768 * 8192) / actual_time
```
将实际吞吐率与硬件芯片规格表中的理论峰值性能（[H100 spec sheet](https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306)）进行对比，就能了解我们的代码对硬件的榨取程度。

---

## Slide 13: Model FLOPs Utilization (模型算力利用率 MFU)

### 讲解

在评估大规模训练系统时，我们通常使用 **MFU (Model FLOPs Utilization)** 来衡量系统效率：
```math
\text{MFU} = \frac{\text{Actual FLOP/s}}{\text{Promised FLOP/s}}
```
- $`\text{Actual FLOP/s}`$ 是通过理论公式计算每步的 FLOPs 除以单步运行时间得到的计算吞吐速度。
- $`\text{Promised FLOP/s}`$ 是 GPU 标称的理论峰值计算速度。
- 为什么 MFU 不可能达到 1？因为 GPU 在训练过程中，除了核心的矩阵乘法计算外，还需要花费大量时间在 GPU 显存与计算单元（SM）之间传输数据（内存搬运）、GPU 卡间通信（通信延迟）、非融合内核的启动开销等。
- 在分布式训练中，**MFU $`\ge 0.5`$（$`50\%`$ 以上）** 已经是极其优秀、经过深度优化的工程表现了。为了提高 MFU，我们需要深入分析硬件的算术强度瓶颈。

---

# Part 4: Arithmetic Intensity & Roofline Analysis (算术强度与 Roofline 模型分析)

## Slide 14: Compute vs Memory Bottlenecks (计算受限与内存受限)

![Compute Memory Architecture](images/compute-memory.png)

### 讲解

GPU 芯片内部包含两大部分：
1. **计算核心 (SM)**：负责高速浮点数学运算（算力性能限制，FLOP/s）。
2. **显存 (HBM)**：负责存储所有的张量（带宽性能限制，Bytes/s）。

每一次计算，硬件都需要经历：
1. 将输入数据从显存（HBM）搬运到计算核心（寄存器/SRAM）。
2. 核心进行浮点运算。
3. 将计算完成的输出写回显存（HBM）。

#### **系统运行时间的决定因素**
根据完美重叠（Overlap）的理想假设，一项计算任务的总时间取决于传输时间和计算时间的最大值：
```math
\text{Time} = \max\left(\frac{\text{Bytes transferred}}{\text{Memory Bandwidth}}, \frac{\text{FLOPs}}{\text{Peak FLOP/s}}\right)
```
由此我们将任务归为两类：
- **内存受限 (Memory-Bound)**：$`\frac{\text{Bytes}}{\text{Bandwidth}} > \frac{\text{FLOPs}}{\text{Peak FLOP/s}}`$。此时计算核心在空转，大部分时间都浪费在等待数据搬运上。
- **计算受限 (Compute-Bound)**：$`\frac{\text{Bytes}}{\text{Bandwidth}} < \frac{\text{FLOPs}}{\text{Peak FLOP/s}}`$。数据搬运速度充沛，硬件性能的极限取决于计算核心的运算速率。

为了量化这两种状态，我们引入**算术强度（Arithmetic Intensity）**。

---

## Slide 15: Arithmetic Intensity of Elementwise Operations (ReLU & GeLU 的算术强度分析)

### 讲解

**算术强度（Arithmetic Intensity, AI）**定义为：一次计算任务中，平均每搬运一个字节（Byte）数据，能进行多少次浮点运算（FLOPs）：
```math
\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes transferred}}
```

与此对应，硬件本身也有一个**硬件强度阈值（Accelerator Intensity）**：
```math
\text{Accelerator Intensity} = \frac{\text{Peak FLOP/s}}{\text{Memory Bandwidth (Bytes/s)}}
```
- **H100 SXM 标称值**：峰值算力约 $`989.5 \text{ TFLOP/s}`$，内存带宽约 $`3.35 \text{ TB/s}`$。
- **H100 硬件强度阈值**：
  ```math
  \text{Accelerator Intensity}_{H100} = \frac{989.5 \times 10^{12}}{3.35 \times 10^{12}} \approx 295.4 \text{ FLOP/Byte}
  ```
- 结论：如果某项计算的算术强度 **低于 295.4**，它在 H100 上就是**内存受限（Memory-bound）**的；只有 **高于 295.4**，才能使 GPU 达到**计算受限（Compute-bound）**并榨干算力。

我们先来核算逐元素操作（Elementwise Operations）的算术强度。

#### **1. ReLU 激活函数**
对 $`N`$ 维的 BF16 向量 $`x`$ 执行 $`y = \text{ReLU}(x)`$：
- **数据搬运量**：读取 $`x`$（$`2N`$ 字节），写入 $`y`$（$`2N`$ 字节），共 $`4N`$ 字节（BF16 精度下为 2 字节/浮点数）。
- **浮点运算量**：$`N`$ 次比较/赋值运算，计为 $`N`$ FLOPs。
- **算术强度**：
  ```math
  \text{AI}_{\text{ReLU}} = \frac{N}{4N} = 0.25 \text{ FLOP/Byte}
  ```
- 显然 $`0.25 \ll 295.4`$。因此，**ReLU 运算是极度内存受限的**。GPU 核心大部分时间都在无所事事地等待数据。

#### **2. GeLU 激活函数**
其数学形式为 $`\text{GELU}(x) = 0.5 x \left(1 + \tanh\left(\sqrt{2/\pi} (x + 0.044715 x^3)\right)\right)`$：
- **数据搬运量**：同样是读取 $`x`$，写入 $`y`$，共 $`4N`$ 字节。
- **浮点运算量**：由于包含乘法、加法、三次方和 $`\tanh`$ 等，大约需要 $`20N`$ FLOPs（$`\tanh`$ 一般使用多项式逼近）。
- **算术强度**：
  ```math
  \text{AI}_{\text{GELU}} = \frac{20N}{4N} = 5 \text{ FLOP/Byte}
  ```
- 尽管 GeLU 的算术强度是 ReLU 的 20 倍，但 $`5`$ 依然远小于 $`295.4`$。所以 **GeLU 依然是内存受限的**。在单算子执行中，GeLU 并不比 ReLU 慢，因为它们的执行时间均取决于显存搬运速度。

---

## Slide 16: Arithmetic Intensity of Vector/Matrix Operations (点积、矩阵-向量乘、矩阵乘的算术强度分析)

### 讲解

接着我们来分析更复杂的向量与矩阵操作在 BF16 精度下的算术强度：

#### **1. 向量点积 (Dot Product)**
计算长度为 $`N`$ 的向量 $`x, w`$ 的点积 $`y = x^T w`$：
- **数据搬运量**：读取 $`x`$（$`2N`$ 字节），读取 $`w`$（$`2N`$ 字节），写入标量 $`y`$（可忽略），共计约 $`4N`$ 字节。
- **浮点运算量**：$`N`$ 次乘法与 $`N-1`$ 次加法，约为 $`2N`$ FLOPs。
- **算术强度**：
  ```math
  \text{AI}_{\text{Dot}} = \frac{2N}{4N} = 0.5 \text{ FLOP/Byte}
  ```
- 结论：$`\text{AI}_{\text{Dot}} \ll 295.4`$，**点积也是内存受限的**。

#### **2. 矩阵-向量乘积 (Matrix-Vector Product)**
计算 $`y = x W`$，其中 $`x \in \mathbb{R}^{1 \times N}, W \in \mathbb{R}^{N \times N}, y \in \mathbb{R}^{1 \times N}`$：
- **数据搬运量**：读取 $`x`$（$`2N`$ 字节），读取 $`W`$（$`2N^2`$ 字节），写入 $`y`$（$`2N`$ 字节），共 $`2N^2 + 4N`$ 字节。
- **浮点运算量**：$`N`$ 次点积，每次点积 $`2N`$ FLOPs，共 $`2N^2`$ FLOPs。
- **算术强度**：
  ```math
  \text{AI}_{\text{MatVec}} = \frac{2N^2}{2N^2 + 4N} \approx 1 \text{ FLOP/Byte}
  ```
- 结论：$`\text{AI}_{\text{MatVec}} \ll 295.4`$，**依然是内存受限的**。这也是大模型**推理阶段（Autoregressive Decoding）**的痛点——解码时每次仅生成一个 Token，其本质就是矩阵-向量乘法，所以推理是严重的内存受限任务。

#### **3. 矩阵-矩阵乘法 (Matrix-Matrix Multiplication)**
计算 $`Y = X W`$，其中 $`X \in \mathbb{R}^{N \times N}, W \in \mathbb{R}^{N \times N}, Y \in \mathbb{R}^{N \times N}`$：
- **数据搬运量**：读取 $`X`$（$`2N^2`$），读取 $`W`$（$`2N^2`$），写入 $`Y`$（$`2N^2`$），共 $`6N^2`$ 字节。
- **浮点运算量**：$`N^2`$ 次点积，每次点积 $`2N`$ FLOPs，共 $`2N^3`$ FLOPs。
- **算术强度**：
  ```math
  \text{AI}_{\text{MatMul}} = \frac{2N^3}{6N^2} = \frac{N}{3} \text{ FLOP/Byte}
  ```
- 当 $`N`$ 足够大时（例如 $`N=1024`$ 时 $`\text{AI} \approx 341.3 > 295.4`$），**矩阵乘法终于成为了计算受限（Compute-bound）的任务**！
- 这就是为什么大模型**预训练阶段（Training）**和**推理的 Prefill 阶段**能够实现极高的 MFU，因为它们是一次性输入整批或整句数据进行矩阵乘法。

---

## Slide 17: Roofline Plots (Roofline 瓦顶图与 MFU 关系)

### 讲解

我们可以使用 **Roofline（瓦顶图）** 模型来直观展示算法性能极限与算术强度的关系。（参考 [Roofline Reference](https://jax-ml.github.io/scaling-book/roofline/)）

![Roofline Plot](https://jax-ml.github.io/scaling-book/assets/img/roofline-improved-1400.webp)

- **横轴**：算术强度（Arithmetic Intensity, FLOP/Byte）。
- **纵轴**：实际运算速度（Performance, FLOP/s）。
- **斜线斜率**：对应硬件的显存带宽（Memory Bandwidth）。在此区域内，性能受带宽限制，呈现线性上升，属于 **Memory-bound 区域**。
- **水平天花板**：对应硬件的理论峰值性能（Peak FLOP/s）。在此区域内，性能受算力限制，无法继续超越，属于 **Compute-bound 区域**。
- **拐点（Kink）**：即为该硬件的 Accelerator Intensity。

#### **结合 MFU 关系式**
基于 Roofline 模型，模型的最大理论算力利用率可以表示为：
```math
\text{MFU} = \min\left(1, \frac{\text{Arithmetic Intensity}}{\text{Accelerator Intensity}}\right)
```

---

# Part 5: Backward Pass & Gradient Compute (反向传播与梯度计算开销)

## Slide 18: Gradient Basics (自动求导与反向传播基础)

### 讲解

在训练语言模型时，我们不仅需要执行前向传播（Forward Pass）来计算 Loss，更需要通过反向传播（Backward Pass）计算梯度，以更新参数。

我们以一个极简的一维线性回归为例，定义损失函数：
```math
y = 0.5 (x \cdot w - 5)^2
```

在 PyTorch 中，我们可以将参数的 `requires_grad` 设置为 `True`，当前向计算结束后，调用 `loss.backward()` 自动求导：
```python
x = torch.tensor([1.0, 2.0, 3.0])
w = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)

# 1. 前向传播
pred_y = x @ w           # 1*1 + 2*1 + 3*1 = 6
loss = 0.5 * (pred_y - 5).pow(2)  # 0.5 * (6 - 5)^2 = 0.5

# 2. 反向传播
loss.backward()

# 验证梯度 dw = d loss / d w
# 根据链式法则：d loss / d w = (pred_y - 5) * x = (6 - 5) * x = x
assert torch.equal(w.grad, torch.tensor([1.0, 2.0, 3.0]))
```

---

## Slide 19: FLOPs for Gradients (反向传播的 FLOPs 计数：为什么是前向的 2 倍？)

### 讲解

为了精确量化训练阶段所需的 FLOPs，我们需要拆解反向传播在网络单层中的具体数学计算。

![Deep Network](images/deep-network.png)

设网络单层的计算为 $`h_2 = h_1 W_2`$。其中输入激活值 $`h_1 \in \mathbb{R}^{B \times D}`$，参数权重 $`W_2 \in \mathbb{R}^{D \times D}`$，输出 $`h_2 \in \mathbb{R}^{B \times D}`$：
- **前向传播计算**：
  ```math
  h_2 = h_1 W_2
  ```
  对应的前向 FLOPs 为：
  ```math
  \text{FLOPs}_{\text{forward}} = 2 \times B \times D \times D
  ```

- **反向传播计算**：
  在反向传播中，我们会从上一层接收到对于 $`h_2`$ 的梯度 $`\frac{\partial \mathcal{L}}{\partial h_2} \in \mathbb{R}^{B \times D}`$。根据链式法则，为了继续向后传播并更新当前层参数，我们需要计算两个矩阵乘法：
  1. **计算关于输入的梯度**以继续向后传播：
     ```math
     \frac{\partial \mathcal{L}}{\partial h_1} = \frac{\partial \mathcal{L}}{\partial h_2} W_2^T
     ```
     该矩阵乘法的形状为 $`(B \times D) \times (D \times D)`$，对应的 FLOPs 数量为：
     ```math
     \text{FLOPs}_{\text{grad\_input}} = 2 \times B \times D \times D
     ```
  2. **计算关于权重的梯度**以更新当前层参数：
     ```math
     \frac{\partial \mathcal{L}}{\partial W_2} = h_1^T \frac{\partial \mathcal{L}}{\partial h_2}
     ```
     该矩阵乘法的形状为 $`(D \times B) \times (B \times D)`$，对应的 FLOPs 数量为：
     ```math
     \text{FLOPs}_{\text{grad\_weight}} = 2 \times D \times B \times D = 2 \times B \times D \times D
     ```

因此，反向传播的总 FLOPs 为：
```math
\text{FLOPs}_{\text{backward}} = \text{FLOPs}_{\text{grad\_input}} + \text{FLOPs}_{\text{grad\_weight}} = 4 \times B \times D \times D
```

这便得出了深度学习系统中的黄金定律：**反向传播的计算量（FLOPs）是前向传播的 2 倍**。

---

## Slide 20: Total Training FLOPs (训练阶段的总 FLOPs 估算：6ND 规律)

### 讲解

将前向传播与反向传播的计算量相加，对于任何多层感知机（MLP）或 Transformer 层，单步训练（一次迭代）所需的总计算量为：
```math
\text{FLOPs}_{\text{step}} = \text{FLOPs}_{\text{forward}} + \text{FLOPs}_{\text{backward}} = 6 \times B \times N
```
其中：
- $`B`$ 是该批次中包含的全部 Token 数量（Batch Size $`\times`$ Sequence Length）。
- $`N`$ 是模型的总参数量（Number of Parameters）。

由此，对于一个总 Token 数为 $`D`$ 的数据集，将模型训练完所需的总 FLOPs 估算为：
```math
\text{Total Training FLOPs} \approx 6 \times N \times D
```

这个简短的经验公式（**$`6ND`$**）是整个大模型算力资源核算的核心，不论是估算训练时长还是配置集群，都是首要计算的底公式。（参考 [Transformer Memory](https://erees.dev/transformer-memory/) 与 [Transformer FLOPs](https://www.adamcasson.com/posts/transformer-flops)）

---

# Part 6: Training Memory & Optimization (训练过程的内存开销与优化)

## Slide 21: Model Memory Breakdown (张量内存开销分解：参数、梯度、优化器状态、激活值)

### 讲解

在训练一个拥有 $`N`$ 个参数、层数为 $`L`$、隐藏维度为 $`D`$、批次大小为 $`B`$ 的深度神经网络时，其显存（Memory）占用主要由以下四部分组成：

#### **1. 参数 (Parameters)**
- 在混合精度训练中，前向计算使用 BF16 存储。
- 占用显存：$`M_{\text{param}} = 2N \text{ 字节}`$

#### **2. 梯度 (Gradients)**
- 同样使用 BF16 存储。
- 占用显存：$`M_{\text{grad}} = 2N \text{ 字节}`$

#### **3. 优化器状态 (Optimizer States)**
- 为了数值稳定性，Adam/AdamW 优化器需要在主内存中使用 FP32（4 字节）保存参数的一阶矩（Momentum）和二阶矩（Variance）状态。
- 对于 AdamW：
  ```math
  M_{\text{opt}} = 4N \text{ (一阶矩)} + 4N \text{ (二阶矩)} = 8N \text{ 字节}
  ```
- 对于 AdaGrad（仅存二阶矩）：
  ```math
  M_{\text{opt}} = 4N \text{ 字节}
  ```

#### **4. 激活值 (Activations)**
- 前向传播计算中每一层产生的所有中间状态，必须保留在显存中，直到反向传播对应层计算梯度时被使用。
- 激活值内存大小与批次大小 $`B`$ 成正比：
  ```math
  M_{\text{act}} = 2 \times B \times D \times L \text{ 字节}
  ```

---

## Slide 22: AdaGrad Optimizer State & Memory (AdaGrad 优化器原理与内存开销)

### 讲解

我们通过一个简化的 AdaGrad 优化器的 PyTorch 实现，来观察优化器状态是如何占用显存的。

```python
class AdaGrad(torch.optim.Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 0.01):
        super(AdaGrad, self).__init__(params, dict(lr=lr))

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                # 访问该参数的优化器状态
                state = self.state[p]
                grad = p.grad.data

                # 获取平方梯度的累加和 g2 = sum_{i<t} g_i^2
                # 若没有，初始化为与梯度形状一致的零矩阵
                g2 = state.get("g2", torch.zeros_like(grad))

                # 更新状态：累加平方梯度
                g2 += torch.square(grad)
                state["g2"] = g2

                # 参数更新公式
                p.data -= lr * grad / torch.sqrt(g2 + 1e-5)
```
- **显存消耗分析**：对于参数 $`p`$，我们需要保存一个相同形状的 `g2` 状态张量。由于 `g2` 经常累加小数值，它必须以更高精度的 FP32（4 字节）形式存储。因此，AdaGrad 为每个参数额外增加了 4 字节的显存开销。如果换成 Adam，由于要同时存储一阶矩，开销将翻倍为 8 字节。

---

## Slide 23: Gradient Accumulation (梯度累加技术)

### 讲解

在大模型训练中，较大的 Batch Size 对训练稳定性至关重要，但激活值显存 $`M_{\text{act}} = 2BDL`$ 与 $`B`$ 呈线性增长，这会导致显存溢出（OOM, Out Of Memory）。

#### **梯度累加 (Gradient Accumulation)**
- **原理**：我们将一个大 Batch 拆分为若干个微批次（Micro-batches）。
- 在前向传播和反向传播时，我们只计算一个 Micro-batch 的梯度，并且**不清除**（`optimizer.zero_grad()`）梯度，而是让它们累加。
- 只有当运行了指定的累加步数（Accumulation Steps）后，我们才调用 `optimizer.step()` 更新参数，随后清空梯度。
- **显存节省**：
  原本 $`B=64`$ 时激活值显存为 $`2 \times 64 \times D \times L`$。
  拆分为 Micro-batch Size $`= 16`$ 运行 4 步累加，那么在任意时刻，显存中只需要保留一个 Micro-batch 的激活值，显存直接降为原来的四分之一，从而用有限的硬件训练超大 Batch。

---

## Slide 24: Activation Checkpointing (激活值重计算/检查点技术)

### 讲解

当模型层数 $`L`$ 极深时，即使 Micro-batch 很小，累积的激活值显存依然是 OOM 的主要来源。

#### **激活值检查点（Activation Checkpointing / Rematerialization）**
- **核心思想**：用计算（Compute）换显存（Memory）。
- **前向传播**：我们不保存每一层产生的激活值，而是只在某些“检查点”（Checkpoints，如每隔几层）保存激活值。
- **反向传播**：当我们需要某层未保存的激活值来计算梯度时，我们从最近的检查点出发，重新运行前向传播计算出这部分的激活值。

![Activation Checkpointing](images/deep-network.png)

```python
class DeepNetworkCheckpointed(nn.Module):
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([Block(dim) for i in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            # 关键：只在检查点层保留激活值，其余部分反向传播时再重新计算
            x = torch.utils.checkpoint.checkpoint(layer, x)
        return x
```

#### **重计算的权衡策略（Trade-off）**
- **全部保存**：激活值显存为 $`O(L)`$，无重计算。
- **完全不保存**：激活值显存降为 $`O(1)`$，但需要从头开始重计算，计算开销呈 $`O(L^2)`$ 级。
- **平方根保存**：每隔 $`\sqrt{L}`$ 层设立一个检查点。激活值显存将降至 $`O(\sqrt{L})`$，而重计算的计算开销仅需额外的 $`O(L)`$（相当于前向传播多运行 1 次）。这是目前工业界最普遍的权衡做法。
