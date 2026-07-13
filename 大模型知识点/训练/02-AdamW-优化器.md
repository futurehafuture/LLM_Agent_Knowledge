---
tags:
  - 面试
  - 大模型
  - 优化器
  - AdamW
created: 2026-05-11
updated: 2026-05-11
---

# AdamW 优化器：解耦权重衰减

> 核心问题：为什么 LLM 训练几乎都用 AdamW 而不是 Adam？Adam 和 AdamW 的区别到底是什么？

---

## 0. 前置理解：优化器到底在干什么

训练神经网络的目标是找到一个最优的权重 $`\theta`$，让损失函数 $`f(\theta)`$ 尽可能小。

优化器的工作就是**每次迭代时，根据当前梯度 $`g = \nabla f(\theta)`$ 决定如何更新权重**：

```math
\theta_{\text{新}} = \theta_{\text{旧}} - \text{更新量}
```

不同的优化器，区别就在于**这个"更新量"怎么算**。

---

## 1. 从 SGD 到 Adam：一步步演进

### 1.1 SGD（随机梯度下降）：最简单的方案

```math
\theta_t = \theta_{t-1} - \alpha \cdot g_t
```

**直觉**：梯度 $`g_t`$ 指向 loss 上升最快的方向，所以往反方向走一步。步长由学习率 $`\alpha`$ 控制。

**问题**：$`\alpha`$ 对每个参数都一样。如果某个参数的梯度很小（比如只有 0.001），另一个参数的梯度很大（比如 100），用同一个 $`\alpha`$ 很难同时照顾两头。

---

### 1.2 SGD + Momentum：让更新有"惯性"

```math
\begin{aligned}
m_t &= \beta \cdot m_{t-1} + (1 - \beta) \cdot g_t \quad\text{（$\beta \approx 0.9$）} \\
\theta_t &= \theta_{t-1} - \alpha \cdot m_t
\end{aligned}
```

**直觉**：不再是只看当前这一步的梯度，而是看**近期梯度的加权平均** $`m_t`$。

打个比方：
- SGD：一个人每一步都只看脚下，方向容易反复横跳。
- SGD + Momentum：像一个有惯性的球滚下山，之前滚的方向会保留下来，不会因为局部小坑就改变方向。

---

**为什么公式长这样？**

$`m_t`$ 是梯度的**指数移动平均（EMA）**。一句话解释：

> 我不需要记住之前每一步的梯度。我只需要记住**一个数** $`m_{t-1}`$（它概括了所有历史梯度），然后用当前梯度 $`g_t`$ 去更新它。

公式拆解：

```math
m_t = \underbrace{\beta \cdot m_{t-1}}_{\text{旧记忆保留 }90\%} + \underbrace{(1-\beta) \cdot g_t}_{\text{新梯度吸收 }10\%}
```

- $`\beta + (1-\beta) = 1`$，这是加权平均的必须条件（系数之和等于 1，结果才不会被放大或缩小）
- $`\beta=0.9`$ 意味着：每走一步，保留 90% 的旧记忆，只吸收 10% 的新信息 → $`m_t`$ 变化缓慢、方向稳定
- 如果 $`\beta=0`$，退化为 $`m_t = g_t`$，就是纯 SGD
- 如果 $`\beta`$ 接近 1，$`m_t`$ 变化极慢，几乎跟不上梯度的变化

把递推式展开，能看到"指数"的由来：

```math
\begin{aligned}
m_t &= (1-\beta) g_t + \beta m_{t-1} \\
    &= (1-\beta) g_t + \beta\big[(1-\beta) g_{t-1} + \beta m_{t-2}\big] \\
    &= (1-\beta) g_t + (1-\beta)\beta \cdot g_{t-1} + (1-\beta)\beta^2 \cdot g_{t-2} + \cdots
\end{aligned}
```

**$`k`$ 步之前的梯度 $`g_{t-k}`$，现在的权重是 $`(1-\beta)\beta^k`$。** $`\beta^k`$ 随 $`k`$ 增大指数衰减——越远的梯度影响越小，但永远不会变成 0。这就是"指数移动平均"的数学本质。

> 更详细的 EMA 拆解（数值示例、有效窗口、$`\beta`$ 怎么选）见 [1.5 节](#15-深入理解为什么-m_t-和-v_t-的公式长这样)。

---

### 1.3 RMSProp：给每个参数自己的学习率

> **RMSProp = Root Mean Square Propagation**，由 Geoff Hinton 在 2012 年 Coursera 课程中提出。
>
> 拆开看：
> - **RMS（均方根）**：对梯度平方做平均再开根号，即 $`\sqrt{\text{平均}(g^2)}`$
> - **Prop（传播）**：用这个 RMS 值来缩放每个参数的更新步长
>
> 合起来就是：**用梯度均方根来缩放学习率**。

```math
\begin{aligned}
v_t &= \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2 \quad\text{（$\beta_2 \approx 0.999$）} \\
\theta_t &= \theta_{t-1} - \alpha \cdot \frac{g_t}{\sqrt{v_t} + \epsilon}
\end{aligned}
```

**直觉**：$`v_t`$ 跟踪的是**梯度大小的历史平均值**（梯度平方的指数移动平均）。

- 如果某个参数历史上梯度一直很大 → $`v_t`$ 很大 → $`\sqrt{v_t}`$ 很大 → 分母大 → **更新量自动变小**（防止震荡）
- 如果某个参数历史上梯度一直很小 → $`v_t`$ 很小 → $`\sqrt{v_t}`$ 很小 → 分母小 → **更新量自动变大**（加速学习）

这就是"自适应学习率"的核心：**梯度大的参数步长自动缩小，梯度小的参数步长自动放大**。

---

### 1.4 Adam = Momentum + RMSProp

Adam 把两者合并：

- 用 $`m_t`$（一阶矩）提供**方向惯性**（Momentum）
- 用 $`v_t`$（二阶矩）提供**自适应步长**（RMSProp）

```math
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad\text{（方向平滑）} \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad\text{（步长自适应）} \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \quad\text{（初始偏差校正，见下文）} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
```

**$`v_t`$ 到底是什么？**

$`v_t`$ 是**梯度平方的指数移动平均**。逐词拆解：

- "梯度**平方**"：对 $`g_t`$ 逐元素取平方 $`g_t^2`$。平方的作用是**去掉正负号**——梯度可以是正（往上）或负（往下），但它的**大小**（幅度）才是我们关心的。比如 $`g_t = -3`$ 和 $`g_t = +3`$，平方后都是 $`9`$，表示"这个参数最近变动幅度很大"
- "**指数移动平均**"：和 $`m_t`$ 一样的 EMA 机制，$`\beta_2 = 0.999`$ 意味着 $`v_t`$ 极端平滑——窗口约 1000 步，变化极慢

**$`v_t`$ 在更新公式里怎么起作用？** 看最后一行：

```math
\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\underbrace{\sqrt{\hat{v}_t} + \epsilon}_{\text{分母由 $v_t$ 决定}}}
```

- $`v_t`$ 大 → $`\sqrt{v_t}`$ 大 → 分母大 → $`\frac{\hat{m}_t}{\sqrt{\hat{v}_t}}`$ 小 → **这个参数的更新量自动缩小**
- $`v_t`$ 小 → $`\sqrt{v_t}`$ 小 → 分母小 → $`\frac{\hat{m}_t}{\sqrt{\hat{v}_t}}`$ 大 → **这个参数的更新量自动放大**

一个具体例子：

> 参数 A 的梯度一直很大（比如 ±10 的范围震荡），$`v_A \approx 100`$，$`\sqrt{v_A} = 10`$，更新量被缩小到 $`\frac{1}{10}`$
>
> 参数 B 的梯度一直很小（比如 ±0.01），$`v_B \approx 0.0001`$，$`\sqrt{v_B} = 0.01`$，更新量被放大到 $`\frac{1}{0.01} = 100`$ 倍

这样**每个参数自动获得了适合自己的学习率**，不需要手工给每个参数设不同的 $`\alpha`$。

**$`\epsilon`$ 是干什么的？** 纯数值稳定性。防止某一步 $`\sqrt{v_t} = 0`$ 导致除零错误。通常 $`\epsilon = 10^{-8}`$，不影响正常情况。

**为什么 $`\beta_2 = 0.999`$（比 $`\beta_1 = 0.9`$ 大很多）？**

$`m_t`$（方向）需要相对快速地响应梯度变化，窗口 ~10 步就够了。但 $`v_t`$（步长缩放）需要**极度稳定**——如果每一步的步长都剧烈波动，优化就会震荡。所以用 $`\beta_2 = 0.999`$，窗口 ~1000 步，$`v_t`$ 变化非常缓慢。

**$`m_0`$ 和 $`v_0`$ 是什么？为什么需要偏差校正？**

训练开始前（$`t=0`$），还没有任何梯度信息，所以：

```math
m_0 = 0,\quad v_0 = 0
```

就是说：在没有见过任何梯度之前，动量是 0，梯度平方的估计也是 0。

这导致什么问题？看 $`t=1`$ 时发生了什么：

```math
\begin{aligned}
m_1 &= \beta_1 \cdot m_0 + (1-\beta_1) \cdot g_1 = \beta_1 \cdot 0 + 0.1 \cdot g_1 = 0.1 \cdot g_1 \\
v_1 &= \beta_2 \cdot v_0 + (1-\beta_2) \cdot g_1^2 = \beta_2 \cdot 0 + 0.001 \cdot g_1^2 = 0.001 \cdot g_1^2
\end{aligned}
```

$`m_1`$ 只有真实梯度的 10%，$`v_1`$ 只有真实梯度平方的 0.1%。**无论梯度多大，前几步的 $`m_t`$ 和 $`v_t`$ 都严重偏小。**

打个比方：你刚认识一个人（$`t=0`$），对他一无所知（$`m_0 = 0`$）。第一次他迟到 1 小时（$`g_1 = 1`$），但因为你 90% 相信自己的空白记忆、只吸收 10% 的新信息，你只更新为"他平均迟到 0.1 小时"——这显然不对。

**偏差校正**就是把这部分拉回来：

```math
\hat{m}_t = \frac{m_t}{1 - \beta_1^t},\quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
```

| $`t`$ | $`1 - \beta_1^t`$ | 校正效果 |
|---|---|---|
| 1 | $`1 - 0.9 = 0.1`$ | $`m_1`$ 除以 0.1 → **完全恢复**到 $`g_1`$ |
| 2 | $`1 - 0.81 = 0.19`$ | 除以 0.19，校正仍然明显 |
| 10 | $`1 - 0.35 = 0.65`$ | 除以 0.65，校正减弱 |
| 100 | $`\approx 1`$ | 基本不校正了 |

随着训练进行，$`t`$ 增大，$`\beta^t \to 0`$，$`1 - \beta^t \to 1`$，校正自然消失——这正是我们想要的，因为后期 $`m_t`$ 和 $`v_t`$ 已经积累了足够的历史信息，不再需要校正。

### 1.5 深入理解：为什么 $`m_t`$ 和 $`v_t`$ 的公式长这样？

看完上面，你可能有一个疑问：

```math
m_t = \beta m_{t-1} + (1-\beta) g_t
```

**这个公式到底是什么意思？为什么前面乘 $`\beta`$、后面乘 $`(1-\beta)`$？为什么两边系数加起来恰好是 1？**

这个公式叫做**指数移动平均（Exponential Moving Average，EMA）**。下面把它拆开讲清楚。

---

#### 1.5.1 从"简单移动平均"说起

假设你每天记录体重：70, 71, 69, 72, 68 kg。想知道"最近 3 天的平均体重"：

```math
\text{平均} = \frac{69 + 72 + 68}{3} = 69.67
```

这就是**简单移动平均（SMA）**。它有两个缺点：

1. **硬截断**：窗口外的值（第 4 天之前的数据）权重直接变成 0，太粗暴
2. **存储开销**：需要保存窗口内所有数据（存 3 天的体重）

EMA 就是为了解决这两个问题而设计的。

---

#### 1.5.2 EMA 的核心思想：把"存所有数据"变成"只存一个值"

EMA 公式：

```math
v_t = \beta \cdot v_{t-1} + (1-\beta) \cdot x_t
```

关键解读：

| 成分 | 含义 |
|---|---|
| $`v_{t-1}`$ | 上一步的 EMA 值（**一个数就概括了之前所有数据的"记忆"**） |
| $`x_t`$ | 当前时刻的新数据 |
| $`\beta`$ | 旧记忆保留的比例（通常 0.9~0.999） |
| $`(1-\beta)`$ | 新数据吸收的比例 |

**为什么 $`\beta + (1-\beta) = 1`$？** 因为这本质上是一个**加权平均**，加权平均的系数之和必须等于 1（否则结果会被放大或缩小）。

---

#### 1.5.3 拆解：从递推式到展开式

把递推式一路展开，就能看到"指数"的来源：

```math
\begin{aligned}
v_t &= (1-\beta) x_t + \beta v_{t-1} \\
    &= (1-\beta) x_t + \beta \big[(1-\beta) x_{t-1} + \beta v_{t-2}\big] \\
    &= (1-\beta) x_t + (1-\beta)\beta x_{t-1} + \beta^2 v_{t-2} \\
    &= \cdots \\
    &= (1-\beta) \cdot \big[x_t + \beta \cdot x_{t-1} + \beta^2 \cdot x_{t-2} + \beta^3 \cdot x_{t-3} + \cdots \big]
\end{aligned}
```

**规律**：$`k`$ 步之前的数据 $`x_{t-k}`$，权重是 $`(1-\beta)\beta^k`$。

因为 $`\beta < 1`$，$`\beta^k`$ 随着 $`k`$ 增大**指数级衰减**——越远的数据权重越小，但永远不会变成 0。这就是"指数移动平均"名称的由来。

---

#### 1.5.4 一个具体数值示例

令 $`\beta = 0.9`$，序列为 [1.0, 2.0, 3.0, 4.0, 5.0]，$`v_0 = 0`$：

- $`v_1 = 0.9 \times 0 + 0.1 \times 1.0 = 0.100`$
- $`v_2 = 0.9 \times 0.100 + 0.1 \times 2.0 = 0.290`$
- $`v_3 = 0.9 \times 0.290 + 0.1 \times 3.0 = 0.561`$
- $`v_4 = 0.9 \times 0.561 + 0.1 \times 4.0 = 0.905`$
- $`v_5 = 0.9 \times 0.905 + 0.1 \times 5.0 = 1.314`$

> 真实均值是 3.0，但 EMA 在第 5 步只有 1.31。为什么？因为 $`v_0=0`$，前几步被"拉向零"了。**这就是偏差校正存在的原因**。

---

#### 1.5.5 $`\beta`$ 的选择与"有效窗口"

$`\beta`$ 决定了记忆的"长度"。经验法则：有效窗口大小 $`\approx \frac{1}{1 - \beta}`$。

| $`\beta`$ | 有效窗口 | 含义 |
|---|---|---|
| 0.9 | ~10 步 | 跟踪短期趋势（Momentum 用） |
| 0.99 | ~100 步 | 中等平滑 |
| 0.999 | ~1000 步 | 长期平滑（Adam 二阶矩用） |

**原理**：当 $`k = \frac{1}{1-\beta}`$ 时，$`\beta^k \approx e^{-1} \approx 0.368`$，即历史数据的权重衰减到约 37%，之后影响显著变小。

---

#### 1.5.6 回到 Adam：$`m_t`$ 和 $`v_t`$ 分别做了什么？

现在再看 Adam 的两个核心公式：

```math
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad (\beta_1 = 0.9) \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad (\beta_2 = 0.999)
\end{aligned}
```

| | $`m_t`$（一阶矩） | $`v_t`$（二阶矩） |
|---|---|---|
| 平滑的对象 | 梯度 $`g_t`$ | 梯度的平方 $`g_t^2`$ |
| $`\beta`$ 值 | 0.9（窗口 ~10 步） | 0.999（窗口 ~1000 步） |
| 作用 | 平滑梯度**方向**，提供"惯性" | 平滑梯度**大小**，提供自适应步长 |
| 直觉 | 近期梯度往哪走，大方向就往哪走 | 近期梯度有多大，步长就调多大 |

**为什么 $`v_t`$ 的 $`\beta`$ 更大（0.999 vs 0.9）？** 因为自适应学习率需要更稳定的估计——如果把某个参数的每次梯度大小剧烈波动，分母 $`\sqrt{v_t}`$ 跟着震荡，优化就不稳定了。所以用更大的 $`\beta`$ 让 $`v_t`$ 变化更缓慢、更平滑。

---

## 2. Adam 的问题：Weight Decay 被"污染"了

### 2.1 Weight Decay 是干什么的

Weight decay（权重衰减）的作用是**防止参数变得太大**，从而防止过拟合。它的想法很简单：每次更新时，额外把参数往 0 的方向拉一点。

### 2.2 原始 Adam 是怎么做的（错误的做法）

原始 Adam 实现中，weight decay 是通过在 loss 里加一项 $`\frac{\lambda}{2}\|\theta\|^2`$ 实现的。这等价于：

```math
g_t = \nabla f(\theta_{t-1}) + \lambda \theta_{t-1}
```

也就是说，**把 $`\lambda\theta`$ 直接加到了梯度里**。

### 2.3 这有什么问题？

**关键问题**：加上去的 $`\lambda\theta_{t-1}`$ 和正常梯度一起，被后面的自适应学习率 $`\frac{1}{\sqrt{\hat{v}_t}}`$ 缩放了。

用一个具体例子来说明：

> 假设两个参数的梯度大小差异很大：
> - 参数 A：梯度很大，$`\sqrt{\hat{v}_A} \approx 10`$ → 自适应步长 $`\frac{1}{10}`$，**缩小了 10 倍**
> - 参数 B：梯度很小，$`\sqrt{\hat{v}_B} \approx 0.1`$ → 自适应步长 $`\frac{1}{0.1}`$，**放大了 10 倍**
>
> Weight decay 项 $`\lambda\theta`$ 被一起缩放后：
> - 参数 A 的正则化效果被**削弱了 10 倍**
> - 参数 B 的正则化效果被**加强了 10 倍**
>
> 结果：你设的 $`\lambda`$ 对不同参数的实际效果差了 100 倍！

**这就是"耦合"的含义**：学习率（及自适应缩放）和正则化强度搅在一起了，你调 $`\alpha`$ 会间接改变正则化效果。

---

## 3. AdamW 的解决方案：分两步走

AdamW 的做法极其简单：**把 weight decay 从梯度通道里拿出来，直接对参数操作**。

```math
\theta_t = \underbrace{\theta_{t-1} - \alpha\lambda\theta_{t-1}}_{\text{权重衰减（独立于梯度）}} - \underbrace{\alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}}_{\text{Adam 更新（和以前一样）}}
```

等价于先做 weight decay，再做 Adam 更新：

1. $`\theta \leftarrow \theta \cdot (1 - \alpha\lambda)`$ — 所有参数统一缩小一点
2. $`\theta \leftarrow \theta - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}`$ — 正常的 Adam 更新

**效果**：weight decay 不再经过自适应学习率的缩放，每个参数受到的正则化强度是**完全一样**的（都是 $`1-\alpha\lambda`$ 的衰减比例）。

### 一句话总结

| | Weight decay 路径 | 被自适应 lr 缩放？ | 正则化均匀？ |
|---|---|---|---|
| **Adam + L2** | 走梯度通道 → 混入 $`m_t`$, $`v_t`$ | 是 | 否 |
| **AdamW** | 走参数通道 → 直接乘 $`1 - \alpha\lambda`$ | 否 | 是 |

---

## 4. PyTorch 使用

```python
from torch.optim import AdamW

optimizer = AdamW(
    model.parameters(),
    lr=1e-3,           # 学习率
    betas=(0.9, 0.999), # β₁（动量）, β₂（RMS）
    eps=1e-8,           # 数值稳定
    weight_decay=1e-2,  # λ，LLM 预训练常用 0.01~0.1
)
```

### 常见做法：LayerNorm 和 bias 不加 weight decay

```python
# 把参数分成两组：需要 decay 的和不需要的
decay_params = []
no_decay_params = []
for name, param in model.named_parameters():
    if param.ndim <= 1 or 'layernorm' in name.lower():
        no_decay_params.append(param)   # bias, LayerNorm → 不加 decay
    else:
        decay_params.append(param)      # 权重矩阵 → 加 decay

optimizer = AdamW([
    {'params': decay_params, 'weight_decay': 0.1},
    {'params': no_decay_params, 'weight_decay': 0.0},
], lr=1e-3)
```

---

## 5. 手撕 AdamW（教育用实现）

```python
import numpy as np


class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.params = params
        self.m = [np.zeros_like(p) for p in params]   # 一阶矩（动量）
        self.v = [np.zeros_like(p) for p in params]   # 二阶矩（自适应步长）

    def step(self, grads):
        self.t += 1
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # ① 解耦的 weight decay：直接衰减参数
            param -= self.lr * self.weight_decay * param

            # ② 更新动量（梯度平滑）
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # ③ 更新自适应步长（梯度平方平滑）
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # ④ 偏差校正
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # ⑤ Adam 参数更新
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ===== 验证：最小化 (x-3)² + (y+2)² =====
if __name__ == "__main__":
    np.random.seed(42)
    params = [np.array([0.0]), np.array([0.0])]
    optimizer = AdamW(params, lr=0.1, weight_decay=0.01)

    for step in range(500):
        x, y = params[0], params[1]
        grads = [2 * (x - 3), 2 * (y + 2)]
        optimizer.step(grads)
        if (step + 1) % 100 == 0:
            loss = (x - 3) ** 2 + (y + 2) ** 2
            print(f"Step {step+1:4d}: x={x.item():.6f}, y={y.item():.6f}, loss={loss.item():.8f}")

    print(f"\n最终: x={params[0].item():.4f} (最优=3), y={params[1].item():.4f} (最优=-2)")
```

---

## 6. 面试高频追问

### Q1: Adam 和 AdamW 哪个泛化更好？为什么？

AdamW。Adam 的 weight decay 被自适应学习率缩放，导致不同参数的正则化强度不均匀（梯度大的参数正则弱，梯度小的参数正则强）。AdamW 对所有参数施以统一的衰减比例，正则化更合理，泛化更好。

### Q2: 为什么 LLM 训练几乎都用 AdamW 而不是 SGD？

1. **对学习率不敏感**：自适应学习率自动调整每个参数的步长，不需要精心设计 lr schedule
2. **训练稳定**：大模型不同参数梯度量级差异极大，Adam 的自适应性天然适配
3. **收敛快**：早期迭代就能有较好的 progress
4. **解耦**：lr schedule 和 weight decay 可以独立调

### Q3: Weight decay 一般设多少？

- 预训练：0.01~0.1（LLaMA 用 0.1，GPT 系列用 0.01）
- 微调：0.01~0.05
- Bias 和 LayerNorm 参数通常设为 0

### Q4: 偏差校正有什么作用？

初始化时 $`m_0 = v_0 = 0`$，早期估计严重偏小。除以 $`1-\beta^t`$ 做校正。$`t=1`$ 时校正量最大（完全恢复），训练后期 $`\beta^t \to 0`$，校正几乎无效。

---

## 参考资料

**论文：**
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019 ([arXiv:1711.05101](https://arxiv.org/abs/1711.05101))
- Kingma & Ba, *Adam: A Method for Stochastic Optimization*, ICLR 2015

**知乎优质讲解：**
- [十分钟速通优化器原理，通俗易懂（从 SGD 到 AdamW）](https://zhuanlan.zhihu.com/p/686410423)
- [通俗易读·理解 L2 Regularization 和 Weight Decay 和 AdamW](https://zhuanlan.zhihu.com/p/713804721)
- [从 SGD 到 AdamW，优化器详解](https://zhuanlan.zhihu.com/p/1928857996655588384)
- [AdamW（一）| 指数移动平均 EMA](https://zhuanlan.zhihu.com/p/2008673874859017390)

**官方文档：**
- [PyTorch AdamW 文档](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
