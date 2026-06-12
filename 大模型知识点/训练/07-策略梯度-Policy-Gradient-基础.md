---
tags:
  - 面试
  - 大模型
  - 强化学习
  - 策略梯度
  - REINFORCE
  - Actor-Critic
created: 2026-06-12
updated: 2026-06-12
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
> PPO 是在 Actor-Critic 基础上再加**重要性采样（ratio）+ Clip 约束**，让 off-policy 数据可以安全复用。"

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

### 2.1 回报 G（累计折扣奖励）

一条轨迹从时刻 $t$ 往后的**累计折扣奖励**：

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

- $\gamma \in [0, 1]$ 是**折扣因子**：越近的奖励权重越大，越远的衰减越快
- $\gamma = 1$：不折扣，所有步的奖励同等重要（LLM RLHF 常用）
- $\gamma = 0.99$：更看重短期奖励（经典 RL 任务常用）

**具体例子**：

```
轨迹：  token1  token2  token3  <eos>
奖励：    0       0       0     1.5   ← RM 只在最后给分

G_0 = 0 + γ·0 + γ²·0 + γ³·1.5 = γ³ × 1.5  （从第一步看）
G_3 = 1.5                                    （从最后一步看）
```

### 2.2 目标函数 J(θ)

强化学习的优化目标：**最大化所有可能轨迹的期望回报**

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G_0] = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right]$$

注意：$\tau \sim \pi_\theta$ 意味着轨迹是**用当前策略 $\pi_\theta$ 采样出来的**。参数 $\theta$ 既影响轨迹分布，也出现在期望里，这就是为什么不能直接求导。

---

## 3. 策略梯度定理：核心数学

### 3.1 直接求导的困难

$$\nabla_\theta J(\theta) = \nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[G(\tau)] = \nabla_\theta \sum_\tau P(\tau|\theta) G(\tau)$$

问题：**$P(\tau|\theta)$ 里包含 $\theta$，但我们没法枚举所有轨迹**（LLM 的词表 50K+，序列可以无限长）。

### 3.2 Log-Derivative Trick（对数导数技巧）

用一个数学技巧绕过枚举：

$$\nabla_\theta P(\tau|\theta) = P(\tau|\theta) \cdot \frac{\nabla_\theta P(\tau|\theta)}{P(\tau|\theta)} = P(\tau|\theta) \cdot \nabla_\theta \log P(\tau|\theta)$$

代入期望：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\nabla_\theta \log P(\tau|\theta) \cdot G(\tau)\right]$$

### 3.3 轨迹概率的分解

一条轨迹的概率：

$$P(\tau|\theta) = p(s_0) \prod_{t=0}^{T} \underbrace{\pi_\theta(a_t|s_t)}_{\text{策略（含 θ）} } \cdot \underbrace{p(s_{t+1}|s_t, a_t)}_{\text{状态转移（不含 θ）}}$$

取对数：

$$\log P(\tau|\theta) = \log p(s_0) + \sum_{t=0}^{T} \log \pi_\theta(a_t|s_t) + \sum_{t=0}^{T} \log p(s_{t+1}|s_t, a_t)$$

对 $\theta$ 求梯度，$\log p(s_0)$ 和状态转移项与 $\theta$ 无关，消失：

$$\nabla_\theta \log P(\tau|\theta) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t)$$

### 3.4 策略梯度定理

代入得到**策略梯度定理**：

$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]}$$

**直觉解释**（非常重要）：

```
∇_θ log π_θ(a_t|s_t)  ×  G_t
        ↑                    ↑
  "往提高 a_t 概率         "这个动作有
   的方向走"              多好（回报）"
```

- **$G_t > 0$（好的结果）**：梯度方向 = 提高生成 $a_t$ 的概率 → **好结果出现的动作概率上升**
- **$G_t < 0$（差的结果）**：梯度方向 = 降低生成 $a_t$ 的概率 → **坏结果出现的动作概率下降**
- **$G_t = 0$**：不更新

> **为什么取 log？** $\log \pi_\theta(a|s)$ 的梯度 $= \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)}$，分母自动归一化了概率的绝对大小，让梯度只反映相对变化，而不是让"本来就高概率的 token 因为概率大而获得更大梯度"。

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

> **符号注意**：PyTorch 做的是**梯度下降**（minimize loss），所以要加负号把"最大化 J"变成"最小化 -J"。
> 即：loss = $-\nabla_\theta J = \sum_t -\log \pi_\theta(a_t|s_t) \cdot G_t$

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

$$\mathbb{E}_{a \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)\right] = 0$$

**证明**（展开积分）：

$$\mathbb{E}_a[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)] = b(s) \sum_a \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s) = b(s) \sum_a \nabla_\theta \pi_\theta(a|s) = b(s) \nabla_\theta \underbrace{\sum_a \pi_\theta(a|s)}_{=1} = 0$$

因此，修改后的梯度估计量**期望不变，但方差可以显著降低**：

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \underbrace{(G_t - b(s_t))}_{\text{Advantage } A_t}\right]$$

这就是 **Advantage** 的来源：**$A_t = G_t - b(s_t)$，即"这个动作比基线（平均水平）好多少"**。

### 5.3 最佳 Baseline 是什么？

可以证明，**最优 baseline 是状态值函数 $V(s_t)$**，即"在状态 $s_t$ 下，不管选什么动作，期望能拿到的平均回报"：

$$b^*(s_t) = V(s_t) = \mathbb{E}_{a \sim \pi}[Q(s_t, a)] = \mathbb{E}_\tau\left[G_t | s_t\right]$$

此时：

$$A_t = G_t - V(s_t) = Q(s_t, a_t) - V(s_t)$$

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

$$\mathbb{E}_{a \sim \pi_\theta}[f(a)] = \mathbb{E}_{a \sim \pi_{\theta_\text{old}}}\left[\underbrace{\frac{\pi_\theta(a)}{\pi_{\theta_\text{old}}(a)}}_{\text{ratio } r_t} \cdot f(a)\right]$$

这个 ratio $r_t = \frac{\pi_\theta(a_t)}{\pi_{\theta_\text{old}}(a_t)}$ 就是 PPO 里的**重要性比率**，让旧策略采样的数据可以**多次复用**。

### 7.2 问题 2：ratio 太大导致更新不稳定

如果 $r_t$ 可以无限大，一次更新可能让策略崩掉。PPO 用 **Clip** 限制 ratio 的范围：

$$\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t A_t,\ \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

这就是 PPO-Clip。$\epsilon = 0.2$ 意味着每次更新 policy 最多漂移 ±20%。

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

$$V^\pi(s) = \mathbb{E}_\tau\left[G_t \mid s_t = s\right] = \mathbb{E}\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k} \mid s_t = s\right]$$

**含义**：在状态 $s$ 下，按策略 $\pi$ 往后走，**期望能拿多少总回报**。不指定动作，用所有可能动作的平均。

### 8.2 动作值函数 $Q^\pi(s, a)$

$$Q^\pi(s, a) = \mathbb{E}_\tau\left[G_t \mid s_t = s, a_t = a\right]$$

**含义**：在状态 $s$ 下，**明确选 $a$ 这个动作**，然后按策略 $\pi$ 往后走，期望能拿多少。

### 8.3 Advantage $A^\pi(s, a)$

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

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

$$V(s) = \mathbb{E}_{a \sim \pi}[Q(s, a)] = \mathbb{E}_{a \sim \pi}[A(s, a) + V(s)]$$

（期望 advantage = 0，这也是为什么 advantage 是好的梯度信号——有正有负，零均值）

---

## 9. GAE：估计 Advantage 的工程实现

上面的 Advantage 在理论上是 $Q - V$，但实践中 $Q$ 也不知道。**GAE（广义优势估计）** 是 PPO 用的工程近似方案。

### 9.1 TD 残差（单步估计）

不知道真实 $Q(s_t, a_t)$，但可以用 TD bootstrapping 近似：

$$Q(s_t, a_t) \approx r_t + \gamma V(s_{t+1})$$

"这步得了 $r_t$，然后剩下的按 $V(s_{t+1})$ 估计"。

代入 advantage：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \quad \text{（TD 残差）}$$

- 优点：方差小（只看一步）
- 缺点：有偏（$V$ 本身不准）

### 9.2 蒙特卡洛（多步估计）

也可以用真实采样的整条轨迹：

$$A_t \approx G_t - V(s_t) = \sum_{k \geq t} \gamma^{k-t} r_k - V(s_t)$$

- 优点：无偏
- 缺点：方差大

### 9.3 GAE：参数 $\lambda$ 做权衡

$$A_t^{GAE} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l} = \delta_t + \gamma\lambda\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \dots$$

| $\lambda$ | 退化到 | 特点 |
|---|---|---|
| $\lambda = 0$ | 单步 TD 残差 $\delta_t$ | 低方差、有偏 |
| $\lambda = 1$ | 蒙特卡洛 $G_t - V(s_t)$ | 无偏、高方差 |
| $\lambda = 0.95$ | 二者折中 | 实践中最常用 |

递推公式（代码实现倒序扫描）：

$$A_t^{GAE} = \delta_t + \gamma\lambda \cdot A_{t+1}^{GAE}$$

> GAE 的完整代码实现见 [[06-PPO损失函数详解]]。

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

原理上可以，但实践中：
- 每次只用一条轨迹估计梯度，方差过大，几乎无法收敛
- 需要大量采样才能降低方差，效率太低

GRPO 本质上是"用 G 个 response 的均值作 baseline"的 REINFORCE，已经比朴素 REINFORCE 好很多。PPO 则通过 Critic + GAE + Clip 做了更彻底的稳定化。

### Q6：on-policy 和 off-policy 的本质区别？

| | On-Policy | Off-Policy |
|---|---|---|
| 定义 | 用**当前策略**采样的数据训练 | 用**其他策略**采样的数据训练 |
| 优点 | 梯度无偏 | 数据可复用，效率高 |
| 缺点 | 数据用完就丢，效率低 | 需要重要性采样校正，可能有偏 |
| 代表算法 | REINFORCE, Actor-Critic, GRPO | DPO（离线偏好数据），Q-Learning |
| PPO 的定位 | 基于 on-policy 采样，但用 IS ratio 允许**有限次** off-policy 复用 | — |

### Q7：为什么不用 Q-Learning（如 DQN）来训练 LLM？

DQN 需要对**每个离散动作**维护一个 Q 值。LLM 词表有 50K+ token，无法直接套用。策略梯度直接在策略参数空间梯度上升，不需要显式的 Q 值表。

---

## 13. 关联阅读

- [[05-PPO-DPO-GRPO-DAPO-GSPO演进对比]] —— 策略梯度在大模型 RLHF 里的全景演进
- [[06-PPO损失函数详解]] —— ratio / advantage / GAE / returns 深入拆解
- [[07-SFT和RL区别]] —— 从监督学习视角理解 RL 训练的差异

## 参考资料

- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed. — 教材最权威
- Sutton et al., *Policy Gradient Methods for Reinforcement Learning with Function Approximation*, NIPS 1999 — 策略梯度定理原始论文
- Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, ICLR 2016 — GAE 原始论文
- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017 — PPO 原始论文
- Ouyang et al., *InstructGPT / RLHF*, NeurIPS 2022 — LLM RLHF 第一篇大成
