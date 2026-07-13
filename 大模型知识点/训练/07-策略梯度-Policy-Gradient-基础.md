---
tags:
  - 面试
  - 大模型
  - 强化学习
  - 策略梯度
  - REINFORCE
  - Actor-Critic
created: 2026-06-12
updated: 2026-07-13
---

# 策略梯度（Policy Gradient）从零入门

> 核心问题：PPO、GRPO 这些 RLHF 算法到底建立在什么数学基础上？策略梯度是什么？为什么不直接用监督学习而要用强化学习？

---

## 0. 面试怎么答（30 秒版）

> "策略梯度是让模型**直接对目标函数（期望累计奖励）求梯度**的一类强化学习算法。
>
> 关键等式是**策略梯度定理**：$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \cdot G_t]$。
>
> 直觉是：好的动作（$G_t > 0$）就提高它的概率，坏的动作（$G_t < 0$）就降低它的概率。
>
> 最朴素的实现叫 **REINFORCE**：跑一条轨迹，算完整回报，用策略梯度定理更新参数。
>
> REINFORCE 方差很大，所以引入 **baseline**（用 Critic 来估计），这就变成了 **Actor-Critic**。
>
> PPO 是在 Actor-Critic 基础上加入**概率比率（ratio）与 Clip 代理目标**，允许对同一批近似 on-policy 数据做有限轮更新，同时抑制策略骤变。"

---

## 1. 为什么要用强化学习？

### 1.1 监督学习的局限

SFT（监督微调）告诉模型"这个回答是对的"，用交叉熵直接拟合。但有些问题做不到：

| 问题 | 监督学习的局限 |
|---|---|
| 回答质量是**整体好坏**，不是逐 token 对错 | CE 要求每个 token 都有标准答案 |
| **探索更好回答**的能力 | SFT 只会模仿，不会尝试新路径 |
| 根据**人类偏好**调整（哪种风格更好） | 偏好是相对比较，不是绝对正误 |

强化学习可以让模型**自己采样回答 → 获得反馈 → 更新**，不需要完美的 token-level 监督标签。

### 1.2 强化学习的基本框架：MDP

```
              采取动作
智能体 ──────────────────→ 环境
(Agent)         a_t           (Environment)
   ↑                              │
   │ 观察状态                      │ 给出
   │ s_{t+1}                      │ 奖励 r_t
   └──────────────────────────────┘
```

**五个核心概念**：

| 概念 | 定义 | LLM 里对应 |
|---|---|---|
| **状态** $s_t$ | 当前环境的完整描述 | prompt + 已生成的前 $t-1$ 个 token |
| **动作** $a_t$ | 智能体在状态下的选择 | 第 $t$ 步生成的 token |
| **奖励** $r_t$ | 这一步得到的即时反馈 | RM 打分（通常只在最后一步非零） |
| **策略** $\pi_\theta(a|s)$ | 给定状态下选动作的概率分布 | LLM 的输出 softmax 概率 |
| **轨迹** $\tau$ | 一条完整的状态-动作序列 $(s_0, a_0, s_1, a_1, \dots, s_T)$ | 一段完整的 response |

> 关键认知：LLM 生成 = 在 MDP 里按策略 $\pi_\theta$ 逐步采样动作（token）。每生成一个 token，就从状态 $s_t$ 转移到 $s_{t+1}$（状态就是"到目前为止生成了什么"）。

---

## 2. 优化目标：最大化期望累计奖励

### 2.1 即时奖励、回报与目标函数不是同一个量

时刻 $t$ 的即时奖励记作 $r_t$。从时刻 $t$ 开始，未来所有奖励的折扣和叫作**回报**（return）：

```math
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
    = \sum_{k=t}^{T}\gamma^{k-t}r_k .
```

- $r_t$：只描述当前一步得到多少奖励；
- $G_t$：描述从当前一步向后看，整段未来一共得到多少；
- $J(\theta)$：对策略可能生成的所有轨迹取平均后得到的优化目标。

$\gamma\in[0,1]$ 是折扣因子。有限长度的 LLM 生成通常取 $\gamma=1$；经典的持续控制任务常取 $0.99$ 一类的值。

例如，奖励模型只在回答结束时给出 $1.5$ 分：

```text
时刻：  0        1        2        3
动作： token₁   token₂   token₃   <eos>
奖励：  0        0        0        1.5
```

于是 $G_3=1.5$，$G_2=\gamma\cdot1.5$，$G_0=\gamma^3\cdot1.5$。当 $\gamma=1$ 时，每个 token 最终看到的序列级回报都是 $1.5$。

### 2.2 轨迹是什么

把一次完整交互写成：

```math
\tau=(s_0,a_0,r_0,s_1,a_1,r_1,\ldots,s_T,a_T,r_T,s_{T+1}).
```

为了简化符号，也常把一条轨迹的总回报写成：

```math
R(\tau)=\sum_{t=0}^{T}\gamma^t r_t.
```

在 LLM 中，prompt 可以看作初始状态；动作 $a_t$ 是第 $t$ 个生成 token；状态 $s_t$ 是“prompt 加上此前已经生成的 token”；追加 token 后的状态转移通常是确定的。

### 2.3 目标函数的三种等价写法

我们的目标是让平均回报最大：

```math
J(\theta)=\mathbb E_{\tau\sim p_\theta(\tau)}[R(\tau)].
```

若轨迹空间是离散的，期望就是加权求和：

```math
J(\theta)=\sum_{\tau}p_\theta(\tau)R(\tau).
```

若轨迹含连续变量，则写成积分：

```math
J(\theta)=\int p_\theta(\tau)R(\tau)\,\mathrm d\tau.
```

三种写法表达的是同一件事。$\mathbb E$ 并没有“把 $p_\theta$ 弄没”，而是把“概率乘以取值再求和或积分”缩写起来了。

---

## 3. 策略梯度：从积分逐行推到可采样公式

### 3.1 对积分求导，不能直接把积分号去掉

从连续形式开始：

```math
J(\theta)=\int p_\theta(\tau)R(\tau)\,\mathrm d\tau.
```

在满足常见的可微与可积条件时，可以把求导移进积分号：

```math
\nabla_\theta J(\theta)
=\nabla_\theta\int p_\theta(\tau)R(\tau)\,\mathrm d\tau
=\int\nabla_\theta\!\left[p_\theta(\tau)R(\tau)\right]\mathrm d\tau.
```

注意，**积分号还在**。求导是在问“每一条轨迹的加权项怎样随参数变化”，积分则是在把所有轨迹的贡献汇总；二者不是可以互相抵消的运算。

对乘积使用求导法则：

```math
\nabla_\theta[p_\theta(\tau)R(\tau)]
=R(\tau)\nabla_\theta p_\theta(\tau)
+p_\theta(\tau)\nabla_\theta R(\tau).
```

标准策略梯度假设环境和奖励函数固定，因此在**固定同一条轨迹 $\tau$** 时，$R(\tau)$ 不显含策略参数：

```math
\nabla_\theta R(\tau)=0.
```

所以：

```math
\nabla_\theta J(\theta)
=\int R(\tau)\nabla_\theta p_\theta(\tau)\,\mathrm d\tau.
```

### 3.2 奖励明明由模型生成结果决定，为什么说它与 $\theta$ 无关

这里最容易混淆的是“直接依赖”和“通过采样分布间接依赖”。

- 策略参数 $\theta$ 改变后，模型更可能生成某些轨迹，因此 $p_\theta(\tau)$ 会变；
- 但如果先把轨迹固定为“模型回答 2”，固定的奖励规则对这条轨迹打多少分，并不会因为策略参数变了而变。

因此 $J(\theta)$ **当然依赖 $\theta$**，但依赖关系通常在 $p_\theta(\tau)$ 中：

```math
\theta\longrightarrow p_\theta(\tau)
\longrightarrow\text{更常采到某些 }\tau
\longrightarrow\mathbb E[R(\tau)]\text{ 改变}.
```

可以把它类比为掷一枚可调偏置的硬币：奖品规则“正面得 1 分，反面得 0 分”是固定的；参数只改变正面出现的概率，最终平均得分仍然随参数变化。

> 如果奖励模型与策略共享参数，或者奖励公式本身写成 $R_\theta(\tau)$，那么 $\nabla_\theta R_\theta(\tau)$ 这一项不能丢。经典 RLHF 训练通常把奖励模型冻结，所以使用标准推导。

### 3.3 Log-derivative trick 从哪里来

由对数函数的链式法则：

```math
\nabla_\theta\log p_\theta(\tau)
=\frac{1}{p_\theta(\tau)}\nabla_\theta p_\theta(\tau).
```

两边乘以 $p_\theta(\tau)$：

```math
\nabla_\theta p_\theta(\tau)
=p_\theta(\tau)\nabla_\theta\log p_\theta(\tau).
```

把它代回上一节：

```math
\begin{aligned}
\nabla_\theta J(\theta)
&=\int R(\tau)p_\theta(\tau)
  \nabla_\theta\log p_\theta(\tau)\,\mathrm d\tau\\
&=\mathbb E_{\tau\sim p_\theta}
  \left[R(\tau)\nabla_\theta\log p_\theta(\tau)\right].
\end{aligned}
```

最后一步只是重新认出了期望的定义：$\int p(x)f(x)\,\mathrm dx=\mathbb E[f(x)]$。概率密度没有消失，而是被期望符号收进去了。

这个技巧的重要价值是：我们不需要对不可微的“采样动作”本身反向传播，只需采样轨迹，然后计算可微的 $\log p_\theta(\tau)$。

### 3.4 把轨迹概率拆开

一条轨迹的联合概率可由链式法则分解：

```math
p_\theta(\tau)
=\rho_0(s_0)\prod_{t=0}^{T}
\pi_\theta(a_t\mid s_t)
P(s_{t+1}\mid s_t,a_t).
```

其中：

- $\rho_0(s_0)$ 是初始状态分布；
- $\pi_\theta(a_t\mid s_t)$ 是策略，含有待训练参数；
- $P(s_{t+1}\mid s_t,a_t)$ 是环境转移概率，标准设定下不含 $\theta$。

先取对数，把连乘变成连加：

```math
\log p_\theta(\tau)
=\log\rho_0(s_0)
+\sum_{t=0}^{T}\log\pi_\theta(a_t\mid s_t)
+\sum_{t=0}^{T}\log P(s_{t+1}\mid s_t,a_t).
```

再对 $\theta$ 求梯度。初始分布和环境转移不含 $\theta$，其梯度为零：

```math
\nabla_\theta\log p_\theta(\tau)
=\sum_{t=0}^{T}\nabla_\theta
  \log\pi_\theta(a_t\mid s_t).
```

于是得到整条轨迹回报版本的 REINFORCE 公式：

```math
\nabla_\theta J(\theta)
=\mathbb E_{\tau\sim p_\theta}
\left[
R(\tau)\sum_{t=0}^{T}
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right].
```

### 3.5 为什么整条轨迹回报 $R(\tau)$ 可以换成 $G_t$

先把总回报拆成动作前与动作后的两部分：

```math
R(\tau)
=\underbrace{\sum_{k=0}^{t-1}\gamma^k r_k}_{\text{动作 }a_t\text{ 之前}}
+\underbrace{\sum_{k=t}^{T}\gamma^k r_k}_{\text{动作 }a_t\text{ 之后}}.
```

时刻 $t$ 的动作不可能改变已经发生的奖励。给定历史 $s_t$ 后，过去奖励只是常数，而 score function 的条件期望为零：

```math
\begin{aligned}
\mathbb E_{a_t\sim\pi_\theta}
\left[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right]
&=\sum_{a_t}\pi_\theta(a_t\mid s_t)
  \nabla_\theta\log\pi_\theta(a_t\mid s_t)\\
&=\sum_{a_t}\nabla_\theta\pi_\theta(a_t\mid s_t)\\
&=\nabla_\theta\sum_{a_t}\pi_\theta(a_t\mid s_t)\\
&=\nabla_\theta 1=0.
\end{aligned}
```

因此“过去奖励 × 当前 score”的期望为零，删掉它不会改变梯度期望，只会减少噪声。保留从 $t$ 开始的 reward-to-go，并把公共折扣因子按约定整理后：

```math
\boxed{
\nabla_\theta J(\theta)
=\mathbb E_{\tau\sim p_\theta}
\left[
\sum_{t=0}^{T}
\nabla_\theta\log\pi_\theta(a_t\mid s_t)G_t
\right]
}.
```

这一步体现了**因果性**：只让动作对它之后可能影响的奖励负责。

### 3.6 从 $G_t$ 到 $Q^\pi(s_t,a_t)$

$G_t$ 是单次采样得到的随机回报。在固定 $(s_t,a_t)$ 后对所有可能未来取期望，就得到动作价值函数：

```math
Q^\pi(s_t,a_t)
=\mathbb E_\pi[G_t\mid s_t,a_t].
```

利用条件期望，可把 $G_t$ 换成 $Q^\pi$：

```math
\nabla_\theta J(\theta)
=\mathbb E_\pi\left[
\sum_{t=0}^{T}
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
Q^\pi(s_t,a_t)
\right].
```

这就是 episodic 场景下常用的策略梯度定理形式。更一般的写法会把时间求和吸收到策略诱导的状态访问分布 $d^\pi(s)$ 中：

```math
\nabla_\theta J(\theta)
\propto
\mathbb E_{s\sim d^\pi,\,a\sim\pi_\theta}
\left[
\nabla_\theta\log\pi_\theta(a\mid s)Q^\pi(s,a)
\right].
```

### 3.7 一个只有两个动作的可验证例子

假设只有动作 A、B，$\pi_\theta(A)=\sigma(\theta)=p$，$\pi_\theta(B)=1-p$；选 A 得 2 分，选 B 得 0 分。那么：

```math
J(\theta)=2p=2\sigma(\theta),
\qquad
\frac{\mathrm dJ}{\mathrm d\theta}=2p(1-p).
```

再用策略梯度公式计算：

```math
\begin{aligned}
\mathbb E\left[R(a)\frac{\mathrm d}{\mathrm d\theta}
\log\pi_\theta(a)\right]
&=p\cdot2\cdot\frac{\mathrm d}{\mathrm d\theta}\log p
 +(1-p)\cdot0\cdot\frac{\mathrm d}{\mathrm d\theta}\log(1-p)\\
&=p\cdot2\cdot(1-p)\\
&=2p(1-p).
\end{aligned}
```

两种方法完全一致。这说明 score-function 估计不是“拍脑袋的近似”；它在期望上就是目标函数的真实梯度。实际训练中的误差来自有限样本的蒙特卡洛估计。

### 3.8 梯度方向的直觉

```text
∇θ log πθ(a_t|s_t) × G_t
          ↑                 ↑
  增大该动作对数概率     这次结果有多好
```

- $G_t>0$：沿梯度上升方向提高所采动作的概率；
- $G_t<0$：降低所采动作的概率；
- $G_t=0$：这条样本不提供更新信号。

严格说，“取 log”的首要原因是恒等式 $\nabla p=p\nabla\log p$ 能把梯度改写成可采样的期望；它还把轨迹概率的连乘变成逐步 log-prob 的求和，因而非常适合自回归模型。

---

## 4. REINFORCE：最朴素的策略梯度算法

### 4.1 算法流程

策略梯度定理给出了梯度的**期望形式**，用蒙特卡洛（采样均值）来近似：

```
REINFORCE 算法：

循环 (每个 iteration):
  1. 用当前策略 π_θ 采样 N 条轨迹
     τ_1, τ_2, ..., τ_N

  2. 对每条轨迹 τ_i，计算每一步的回报
     G_{i,t} = Σ_{k≥t} γ^{k-t} r_{i,k}

  3. 计算策略梯度
     ∇_θ J ≈ (1/N) Σ_{i=1}^{N} Σ_{t} ∇_θ log π_θ(a_{i,t}|s_{i,t}) · G_{i,t}

  4. 梯度上升更新参数（最大化 J）
     θ ← θ + α · ∇_θ J
```

### 4.2 具体例子：LLM 生成数学题

```
Prompt: "1 + 1 = ?"

采样的轨迹 1: "2"       → reward = +1  (对)
  G = +1
  梯度：提高生成"2"的概率

采样的轨迹 2: "3"       → reward = -1  (错)
  G = -1
  梯度：降低生成"3"的概率

采样的轨迹 3: "答案是 2" → reward = +1  (对)
  G = +1
  梯度：提高生成"答"、"案"、"是"、"2"的概率
```

### 4.3 PyTorch 伪代码

```python
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

for iteration in range(max_iter):
    # Step 1: 采样轨迹
    trajectories = []
    for _ in range(N):
        response = policy.generate(prompt)   # 自回归采样
        reward = reward_model(prompt, response)
        trajectories.append((response, reward))

    # Step 2 & 3: 计算梯度
    loss = 0.0
    for response, G in trajectories:
        log_probs = policy.log_probs(prompt, response)  # 每个 token 的 log prob
        # 策略梯度定理：-log_prob × G（负号是因为 PyTorch 做梯度下降，我们想上升）
        loss += -(log_probs.sum() * G)

    loss = loss / N

    # Step 4: 更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

> **符号注意**：PyTorch 做的是梯度下降，所以把要最大化的采样目标加负号。这里的 loss 是标量，不是梯度：
>
> ```math
> \mathcal L_{\mathrm{PG}}(\theta)
> =-\sum_t\log\pi_\theta(a_t\mid s_t)G_t,
> \qquad
> \nabla_\theta\mathcal L_{\mathrm{PG}}
> =-\widehat{\nabla_\theta J}.
> ```
>
> 代码中应把采样得到的 $G_t$ 当作常数权重（通常会 `detach`），梯度只流过 log-prob。

---

## 5. REINFORCE 的痛点：高方差

### 5.1 问题根源

REINFORCE 用**单次轨迹的回报 $G$** 来估计**期望回报 $\mathbb{E}[G]$**，是蒙特卡洛估计，**方差极大**：

```
同一个 prompt "写一首关于春天的诗"：

采样第 1 次 → G = 3.2  （RM 觉得很好）
采样第 2 次 → G = 0.1  （RM 觉得很一般）
采样第 3 次 → G = 2.8  （RM 觉得不错）

三次估计的方差很大，导致梯度更新方向忽左忽右，训练不稳。
```

### 5.2 Baseline：减去基线降低方差

**关键数学性质**：给回报减去一个**不依赖于动作**的 baseline $b(s)$，不改变期望，但降低方差：

```math
\mathbb{E}_{a \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)\right] = 0
```

**证明**（展开积分）：

```math
\mathbb{E}_a[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)] = b(s) \sum_a \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s) = b(s) \sum_a \nabla_\theta \pi_\theta(a|s) = b(s) \nabla_\theta \underbrace{\sum_a \pi_\theta(a|s)}_{=1} = 0
```

因此，修改后的梯度估计量**期望不变，但方差可以显著降低**：

```math
\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \underbrace{(G_t - b(s_t))}_{\text{Advantage } A_t}\right]
```

这就是 **Advantage** 的来源：**$A_t = G_t - b(s_t)$，即"这个动作比基线（平均水平）好多少"**。

### 5.3 最佳 Baseline 是什么？

最常用的 baseline 是状态值函数 $V^\pi(s_t)$，即“在状态 $s_t$ 下，按当前策略行动时的平均回报”：

```math
b(s_t) = V^\pi(s_t)
= \mathbb{E}_{a \sim \pi}[Q^\pi(s_t, a)]
= \mathbb{E}_\pi\left[G_t \mid s_t\right].
```

此时：

```math
A_t = G_t - V(s_t) = Q(s_t, a_t) - V(s_t)
```

严格来说，使梯度估计方差最小的标量 baseline 还会受到 score 向量模长的影响：

```math
b^*(s)
=\frac{\mathbb E_{a\sim\pi}
\left[Q^\pi(s,a)\left\|\nabla_\theta\log\pi_\theta(a\mid s)\right\|^2\right]}
{\mathbb E_{a\sim\pi}
\left[\left\|\nabla_\theta\log\pi_\theta(a\mid s)\right\|^2\right]}.
```

当 score 的模长与动作关系不大时，它近似为 $V^\pi(s)$。工程上 $V^\pi(s)$ 易解释、易学习，因此是 Actor-Critic 的标准选择。

**直觉**：

```
V(s_t) = "在这个状态下，什么都不知道，期望能拿多少"
        = 所有可能动作的平均水平

Q(s_t, a_t) = "在这个状态下，具体选 a_t 这个动作，期望能拿多少"

A_t = Q - V = "选 a_t 比随便选一个动作 好/差 多少"
```

好的动作 $A_t > 0$ → 提高概率
差的动作 $A_t < 0$ → 降低概率

---

## 6. Actor-Critic：用神经网络估计 Baseline

### 6.1 问题：V(s) 怎么得到？

$V(s_t)$ 需要对**所有可能的未来轨迹取期望**，显然无法精确计算。

解决方案：**用另一个神经网络来学习 $V$**，这个网络叫 **Critic**（评论家），而原本的策略网络叫 **Actor**（演员）。

```
              ┌──────────────────────────────────┐
              │         Actor-Critic             │
              │                                  │
  状态 s_t  ──┤──→ Actor π_θ(a|s)  ──→ 动作 a_t │
              │                                  │
              │──→ Critic V_φ(s)   ──→ 价值 V    │
              └──────────────────────────────────┘
```

### 6.2 双网络的分工

| 角色 | 网络 | 学什么 | 损失函数 |
|---|---|---|---|
| **Actor**（演员） | $\pi_\theta(a|s)$ | 怎么选动作（策略） | 策略梯度损失 $-\nabla_\theta J$ |
| **Critic**（评论家） | $V_\phi(s)$ | 每个状态值多少钱 | 均方误差（MSE）$\|V_\phi(s) - R_t\|^2$ |

### 6.3 完整 Actor-Critic 算法

```
循环 (每个 iteration):
  1. Actor 采样轨迹：π_θ 生成 response
  2. 计算 returns R_t（真实累计奖励）
  3. Critic 预测 V_φ(s_t)
  4. 计算 advantage：A_t = R_t - V_φ(s_t)
  5. 更新 Actor：θ ← θ + α · Σ_t ∇_θ log π_θ(a_t|s_t) · A_t
  6. 更新 Critic：φ ← φ - β · Σ_t 2(V_φ(s_t) - R_t) · ∇_φ V_φ(s_t)
```

> 在 LLM RLHF 里：
> - Actor = 待训练的 LLM（参数 $\theta$）
> - Critic = 一个和 Actor 架构相同的 LLM，输出一个标量 $V$（参数 $\phi$）
> - 这就是为什么 PPO 需要同时在内存里装两个 LLM

---

## 7. 从 Actor-Critic 到 PPO

到这里，Actor-Critic 已经是一个可工作的 RLHF 算法，但还有两个实践问题：

### 7.1 问题 1：on-policy 样本效率低

Actor-Critic 是 **on-policy** 的：每次采样只能用一次。采样完 → 更新 → 旧数据扔掉 → 重新采样。这很浪费，因为采样（LLM 生成）很慢。

**解决方案：重要性采样（Importance Sampling）**

把"在新策略 $\pi_\theta$ 下的期望"转化为"在旧策略 $\pi_{\theta_\text{old}}$ 下的期望"：

```math
\mathbb{E}_{a \sim \pi_\theta}[f(a)] = \mathbb{E}_{a \sim \pi_{\theta_\text{old}}}\left[\underbrace{\frac{\pi_\theta(a)}{\pi_{\theta_\text{old}}(a)}}_{\text{ratio } r_t} \cdot f(a)\right]
```

这个 ratio $r_t = \frac{\pi_\theta(a_t)}{\pi_{\theta_\text{old}}(a_t)}$ 就是 PPO 里的**重要性比率**。PPO 通常只对刚由旧策略采出的同一批数据做少量 epoch，因此仍被归类为 on-policy / near-policy 方法，而不是任意复用历史经验的通用 off-policy 算法。

### 7.2 问题 2：ratio 太大导致更新不稳定

如果 $r_t$ 可以无限大，一次更新可能让策略崩掉。PPO 用 **Clip** 限制 ratio 的范围：

```math
\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t A_t,\ \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t\right)\right]
```

这就是 PPO-Clip。$\epsilon=0.2$ 表示代理目标在比率越过 $[0.8,1.2]$ 且继续朝“有利方向”变化时不再给额外收益。它**不等于**参数最多变化 20%，也不严格保证所有动作概率都只变化 20%；实践中还会监控 KL、设置较小学习率或提前停止。

### 7.3 演化路线总结

```
REINFORCE
  │
  ├── 太高方差 → 引入 Baseline（V 函数）
  │             → Actor-Critic
  │
  └── Actor-Critic
        │
        ├── 样本效率低 → 重要性采样（ratio）
        │
        └── 更新不稳定 → Clip 限制
                        → PPO
```

---

## 8. 两个值函数：V、Q、Advantage 的关系

面试里这三个概念经常被问，统一澄清：

### 8.1 状态值函数 $V^\pi(s)$

```math
V^\pi(s) = \mathbb{E}_\tau\left[G_t \mid s_t = s\right] = \mathbb{E}\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k} \mid s_t = s\right]
```

**含义**：在状态 $s$ 下，按策略 $\pi$ 往后走，**期望能拿多少总回报**。不指定动作，用所有可能动作的平均。

### 8.2 动作值函数 $Q^\pi(s, a)$

```math
Q^\pi(s, a) = \mathbb{E}_\tau\left[G_t \mid s_t = s, a_t = a\right]
```

**含义**：在状态 $s$ 下，**明确选 $a$ 这个动作**，然后按策略 $\pi$ 往后走，期望能拿多少。

### 8.3 Advantage $A^\pi(s, a)$

```math
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
```

**含义**：选 $a$ 这个动作，**比平均水平好多少**。

```
例子（数学题生成）：
  状态 s = "1 + 1 = ?"
  V(s) = 0.6   ← "这道题平均能拿 0.6 分"

  动作 a1 = "2" → Q(s, a1) = 1.0 → A = +0.4  （好于平均）
  动作 a2 = "3" → Q(s, a2) = 0.0 → A = -0.6  （差于平均）
  动作 a3 = "答案" → Q(s, a3) = 0.8 → A = +0.2  （略好）
```

### 8.4 三者关系

```math
V(s) = \mathbb{E}_{a \sim \pi}[Q(s, a)] = \mathbb{E}_{a \sim \pi}[A(s, a) + V(s)]
```

（期望 advantage = 0，这也是为什么 advantage 是好的梯度信号——有正有负，零均值）

---

## 9. GAE：估计 Advantage 的工程实现

上面的 Advantage 在理论上是 $Q - V$，但实践中 $Q$ 也不知道。**GAE（广义优势估计）** 是 PPO 用的工程近似方案。

### 9.1 TD 残差（单步估计）

不知道真实 $Q(s_t, a_t)$，但可以用 TD bootstrapping 近似：

```math
Q(s_t, a_t) \approx r_t + \gamma V(s_{t+1})
```

"这步得了 $r_t$，然后剩下的按 $V(s_{t+1})$ 估计"。

代入 advantage：

```math
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \quad \text{（TD 残差）}
```

- 优点：方差小（只看一步）
- 缺点：有偏（$V$ 本身不准）

### 9.2 蒙特卡洛（多步估计）

也可以用真实采样的整条轨迹：

```math
A_t \approx G_t - V(s_t) = \sum_{k \geq t} \gamma^{k-t} r_k - V(s_t)
```

- 优点：无偏
- 缺点：方差大

### 9.3 GAE：参数 $\lambda$ 做权衡

```math
A_t^{GAE} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l} = \delta_t + \gamma\lambda\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \dots
```

| $\lambda$ | 退化到 | 特点 |
|---|---|---|
| $\lambda = 0$ | 单步 TD 残差 $\delta_t$ | 低方差、有偏 |
| $\lambda = 1$ | 蒙特卡洛 $G_t - V(s_t)$ | 无偏、高方差 |
| $\lambda = 0.95$ | 二者折中 | 实践中最常用 |

递推公式（代码实现倒序扫描）：

```math
A_t^{GAE} = \delta_t + \gamma\lambda \cdot A_{t+1}^{GAE}
```

> GAE 的完整代码实现见 [[09-PPO损失函数详解]]。

---

## 10. LLM RLHF 里的对应关系全景

```
LLM 自回归生成 = MDP 策略采样

  Prompt:   "1+1=?"
             ↓
  状态 s_0 = [prompt tokens]

  Actor 生成 token_1 = "答"  （动作 a_0）
             ↓
  状态 s_1 = [prompt + "答"]

  Actor 生成 token_2 = "案"  （动作 a_1）
             ↓
  ...

  Actor 生成 token_4 = <eos>  （动作 a_3，episode 结束）
             ↓
  RM 打分 r_3 = 0.8           （稀疏奖励，只有最后一步有）

  Critic 对每个 s_t 估计 V(s_t)
  GAE 计算每个位置的 advantage A_t
  PPO 用 clip + ratio 更新 Actor
```

---

## 11. 四种算法的策略梯度视角对比

| 算法 | Advantage 来源 | 是否需要 Critic | 是否 on-policy |
|---|---|---|---|
| **REINFORCE** | $G_t$（原始回报） | ❌ | ✅ |
| **REINFORCE+Baseline** | $G_t - b$ | ❌（baseline 可以是均值） | ✅ |
| **Actor-Critic** | $G_t - V_\phi(s_t)$ | ✅ | ✅ |
| **PPO** | GAE（用 $V_\phi$ 算 TD 残差） | ✅ | ✅（+重要性采样复用） |
| **GRPO** | 组内 z-score（不需要 $V$） | ❌ | ✅ |

GRPO 是把"Critic 估计 baseline"替换成了"同一 prompt 的 G 个回答的均值作为 baseline"，省掉了 Critic 这个大模型。

---

## 12. 面试高频追问

### Q1：策略梯度和监督学习的梯度有什么区别？

| | 监督学习（SFT）| 策略梯度（RL）|
|---|---|---|
| 梯度 | $-\nabla_\theta \log \pi_\theta(y^*\|x)$（标签固定） | $-\nabla_\theta \log \pi_\theta(a_t\|s_t) \cdot A_t$（回报加权） |
| 标签来源 | 人工标注的真实答案 | 模型自己采样 + RM 打分 |
| 所有正样本梯度符号 | 一样（都是正向优化） | 根据 $A_t$ 正负可能反向 |

本质上，SFT 是让模型学"这个回答是对的"，策略梯度是让模型学"这个回答比平均好/差"。

### Q2：为什么强化学习比 SFT 更难训练？

1. **高方差**：梯度估计依赖蒙特卡洛采样，噪声大
2. **非平稳性**：策略更新后，下一次的采样分布就变了（on-policy 的本质）
3. **稀疏奖励**：LLM 的 reward 只在最后一步，信号传播困难
4. **正反馈循环风险**：一旦某种回答偶然拿到高分，就被不断加强，可能跑偏（reward hacking）

### Q3：Advantage 为什么要做归一化？

batch 内减均值除标准差（$A \leftarrow (A - \mu) / \sigma$）。

- 不改变**方向**（只是线性缩放），不影响哪个动作好哪个差
- 稳定梯度的**尺度**：不同 batch 的 reward 范围可能差很多，归一化后 loss 尺度统一
- 等价于自适应地调整了这个 batch 的有效学习率

### Q4：V 和 Q 哪个更容易学？为什么 Critic 学的是 V 而不是 Q？

**V 只依赖状态**，Q 还依赖动作。

- 对于 LLM，动作空间是词表（50K+），不可能把每个 (s, a) 对都学一个 Q 值
- V 网络只需要吃 `state = [prompt + 已生成 tokens]`，输出一个标量，可以直接在 LLM 上加一个回归头
- Q 通过 TD bootstrapping 近似：$Q(s_t, a_t) \approx r_t + \gamma V(s_{t+1})$，不需要显式建模

### Q5：REINFORCE 能直接用于 LLM 训练吗？

原理上可以，并且可以用一个 batch 的多条轨迹求均值；但朴素版本仍有以下问题：
- 没有良好 baseline 时，序列级回报带来的方差很大；
- LLM 奖励通常稀疏，难以判断具体哪些 token 应负责；
- 需要大量在线生成，样本与算力效率较低。

GRPO 本质上是"用 G 个 response 的均值作 baseline"的 REINFORCE，已经比朴素 REINFORCE 好很多。PPO 则通过 Critic + GAE + Clip 做了更彻底的稳定化。

### Q6：on-policy 和 off-policy 的本质区别？

| | On-Policy | Off-Policy |
|---|---|---|
| 定义 | 用**当前策略**采样的数据训练 | 用**其他策略**采样的数据训练 |
| 优点 | 梯度无偏 | 数据可复用，效率高 |
| 缺点 | 数据用完就丢，效率低 | 需要重要性采样校正，可能有偏 |
| 代表算法 | REINFORCE、A2C、PPO、GRPO | Q-Learning、DQN、SAC |
| PPO 的定位 | 用旧策略刚采出的数据做有限轮更新，通常仍归入 on-policy | — |

> DPO 使用固定的离线偏好数据，但它是偏好优化目标，不是经典意义上通过行为策略数据学习价值函数的 off-policy RL 算法，最好单独分类。

### Q7：为什么不用 Q-Learning（如 DQN）来训练 LLM？

DQN 需要对**每个离散动作**维护一个 Q 值。LLM 词表有 50K+ token，无法直接套用。策略梯度直接在策略参数空间梯度上升，不需要显式的 Q 值表。

---

## 13. 关联阅读

- [[08-PPO-DPO-GRPO-DAPO-GSPO演进对比]] —— 策略梯度在大模型 RLHF 里的全景演进
- [[09-PPO损失函数详解]] —— ratio / advantage / GAE / returns 深入拆解
- [[06-SFT和RL区别]] —— 从监督学习视角理解 RL 训练的差异

## 参考资料

- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed. — 教材最权威
- Williams, *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning*, 1992 — REINFORCE
- Sutton et al., *Policy Gradient Methods for Reinforcement Learning with Function Approximation*, NeurIPS 1999 — 策略梯度定理与函数逼近
- Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, ICLR 2016 — GAE 原始论文
- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017 — PPO 原始论文
- Ouyang et al., *InstructGPT / RLHF*, NeurIPS 2022 — LLM RLHF 第一篇大成
