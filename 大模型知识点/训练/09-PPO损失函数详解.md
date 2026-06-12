---
tags:
  - 面试
  - 大模型
  - PPO
  - RLHF
  - 损失函数
  - GAE
created: 2026-05-24
updated: 2026-05-24
---

# PPO 损失函数详解：ratio / advantage / returns / GAE 在大模型里到底是什么

> 核心问题：PPO 损失里的每一项是什么含义？大模型 RLHF 里的 advantage 怎么算？returns 是什么？GAE 是什么？为什么用 MSE 训练 Critic？

---

## 0. 面试怎么答（一句话拆解）

> "PPO 的损失分**三部分**：
>
> ① **Policy loss**：$-\min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)$，其中 $r_t = \pi_\theta / \pi_{\theta_\text{old}}$ 是新旧策略的概率比，$A_t$ 是 advantage（这个动作比基线好多少），clip 防止单步更新太激进。
>
> ② **Value loss**：$(V_\phi(s_t) - R_t)^2$ 这是 MSE，让 critic 拟合真实 return $R_t$。
>
> ③ **KL 正则**：$\beta \cdot \text{KL}(\pi_\theta \| \pi_\text{ref})$ 限制 policy 不要离 SFT 模型太远。
>
> 在大模型 RLHF 里：
> - **Reward** 通常只在整段回答结束时由 RM 给一个标量；
> - **Returns** $R_t = \sum_{t'\ge t} \gamma^{t'-t} r_{t'}$ 是从 $t$ 开始的折扣回报；
> - **Advantage** 用 **GAE**（广义优势估计）算：$A_t^{GAE} = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}$，其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是 TD 残差。$\lambda$ 控制偏差-方差权衡。"

---

## 1. 先把 RL 术语对到 LLM 上

在标准 RL 里：
- **state** $s_t$ = 环境状态
- **action** $a_t$ = 智能体在状态下的动作
- **reward** $r_t$ = 这一步获得的回报

在大模型 RLHF 里：

| RL 概念 | LLM 里对应 |
|---|---|
| state $s_t$ | prompt + 已生成的前 $t-1$ 个 token |
| action $a_t$ | 第 $t$ 个生成的 token |
| policy $\pi_\theta(a_t\|s_t)$ | LLM 给出的下一 token 概率 |
| reward $r_t$ | 通常**只在最后一个 token** 由 RM 给一个标量（稀疏 reward） |
| episode | 一段完整的 response |
| trajectory | (prompt, $o_1, o_2, \dots, o_T$) |

> 关键：LLM 的 reward 是**稀疏的、句子级的**，这是后面 advantage 设计的重要前提。

---

## 2. PPO 完整损失函数

PPO-Clip 的总损失（要**最小化**这个，注意符号）：

$$
\mathcal{L}^{PPO}(\theta, \phi) = -\mathcal{L}^{CLIP}(\theta) + c_1 \cdot \mathcal{L}^{VF}(\phi) - c_2 \cdot \mathcal{H}[\pi_\theta] + \beta \cdot \text{KL}(\pi_\theta \| \pi_\text{ref})
$$

四部分：

1. **Clip Policy loss** $\mathcal{L}^{CLIP}$ ← 主要的 RL 信号
2. **Value loss** $\mathcal{L}^{VF}$（MSE） ← 训练 Critic
3. **Entropy bonus** $\mathcal{H}[\pi_\theta]$ ← 鼓励探索（系数 $c_2$ 在 LLM RLHF 里常设为 0）
4. **KL penalty** ← 防止 policy 偏离 SFT reference 太远

下面逐项拆。

---

## 3. Policy Loss：ratio + advantage + clip

### 3.1 公式

$$
\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t\Bigl[\min\bigl(r_t(\theta) A_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\bigr)\Bigr]
$$

### 3.2 ratio $r_t(\theta)$ 是什么

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}
$$

**直观理解**：新策略生成这个 token 的概率，比旧策略大多少倍。
- $r_t = 1$：新旧策略一样
- $r_t = 2$：新策略对这个 token 的偏好翻倍
- $r_t = 0.5$：新策略对这个 token 减半

为什么要 ratio？因为 PPO 想用**旧策略采样的数据**多更新几次新策略（提高样本效率），就需要做 importance sampling 修正：

$$
\mathbb{E}_{a \sim \pi_\theta}[f(a)] = \mathbb{E}_{a \sim \pi_{\theta_\text{old}}}\left[\frac{\pi_\theta(a)}{\pi_{\theta_\text{old}}(a)} f(a)\right]
$$

ratio 就是这个 importance sampling 因子。

### 3.3 advantage $A_t$ 是什么

**Advantage = 这个动作比"平均"好多少**：

$$
A_t = Q(s_t, a_t) - V(s_t)
$$

- $Q(s_t, a_t)$：在状态 $s_t$ 选动作 $a_t$ 的预期未来回报
- $V(s_t)$：在状态 $s_t$ 不指定动作的预期未来回报（baseline）

如果 $A_t > 0$：这个动作好于平均 → 想增大 $\pi_\theta(a_t|s_t)$
如果 $A_t < 0$：这个动作差于平均 → 想减小 $\pi_\theta(a_t|s_t)$

**为什么用 advantage 而不是直接用 reward？** 减 baseline 可以**降低梯度估计的方差**而不引入偏差。

> 大模型里的 advantage 怎么算？两条主流路线：① **PPO+Critic**：用 GAE（下一节）；② **GRPO**：用组内 reward 的 z-score。

### 3.4 Clip 的作用

```
            ratio r_t
              │
   1+ε  ──────┼──────────── 上限（advantage > 0 时被裁掉）
              │     ╱
              │   ╱
       1  ────┼─╱─────────
              │╱
            ╱ │
          ╱   │
   1-ε  ──────┼──────────── 下限（advantage < 0 时被裁掉）
              │
```

把 ratio 强制限制在 $[1-\epsilon, 1+\epsilon]$（典型 $\epsilon = 0.2$）。

**为什么取 min**？

$$
\min(r_t A_t,\ \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)
$$

这个 min 形成"**悲观下界**"：
- $A_t > 0$ 时：限制 $r_t$ 上限到 $1+\epsilon$（防止过度上升）
- $A_t < 0$ 时：限制 $r_t$ 下限到 $1-\epsilon$（防止过度下降）

整体效果：**单步更新被约束在信任域内**，policy 不会被单次更新打飞。

---

## 4. Value Loss：MSE 训练 Critic

### 4.1 公式

$$
\mathcal{L}^{VF}(\phi) = \mathbb{E}_t\left[\bigl(V_\phi(s_t) - R_t\bigr)^2\right]
$$

—— 标准 **MSE 损失**。

### 4.2 直觉

Critic $V_\phi(s_t)$ 是一个回归头（通常在 LLM 上加一个标量输出 head），任务是"**给定 prompt + 已生成的 token，预测剩余的累计 reward 是多少**"。

监督信号 $R_t$ 是**真实采样到的 return**（接下来会讲怎么算）。

MSE 让 critic 的预测尽量接近真实 return。

### 4.3 为什么用 MSE 不用别的？
- 回归问题的天然选择
- 梯度光滑、对称
- 部分实现会加 value clip（类似 policy clip）进一步稳定

> 与"分类用交叉熵、回归用 MSE"的一般规则一致。详见 [[01-交叉熵损失详解]]。

---

## 5. Returns 是什么

**Return** = 从某个时刻起的**累计折扣 reward**：

$$
R_t = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'} = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots
$$

- $\gamma$ = discount factor，典型 0.99（LLM RLHF 里也常用 1.0，因为 episode 短）
- $r_t$ = 第 $t$ 步的即时 reward

### 5.1 大模型里 returns 长什么样

由于 LLM 的 reward 是**只在最后一个 token 才有**的稀疏信号：

$$
r_t = \begin{cases} r_\text{RM}(\text{response}) - \beta \log\frac{\pi_\theta}{\pi_\text{ref}}(a_t|s_t) & \text{if } t = T \\ -\beta \log\frac{\pi_\theta}{\pi_\text{ref}}(a_t|s_t) & \text{otherwise} \end{cases}
$$

注意：**每一步都有 KL penalty 作为 token-level reward 的一部分**（这是 LLM RLHF 的常见做法，把 KL 摊到 token 上）。

所以 returns $R_t$ 实际上是：

```
R_t = (剩余 token 的 KL 惩罚之和) + γ^(T-t) × r_RM
```

### 5.2 Returns 用在哪
- **训练 Critic** 的 MSE 监督信号：$(V_\phi(s_t) - R_t)^2$
- **GAE** 中作为 advantage 的真值参照（在 $\lambda=1$ 时 advantage 就退化为 $R_t - V(s_t)$）

---

## 6. GAE：广义优势估计

### 6.1 为什么不直接用 $R_t - V(s_t)$

最简单的 advantage 估计：

$$
A_t = R_t - V(s_t) \quad \text{（蒙特卡洛 advantage）}
$$

- 优点：**无偏**
- 缺点：**方差大**（依赖整条 trajectory 的随机 reward）

另一个极端：

$$
A_t = r_t + \gamma V(s_{t+1}) - V(s_t) \quad \text{（TD residual, 单步）}
$$

- 优点：**方差小**
- 缺点：**有偏**（V 本身不准）

### 6.2 GAE 是 TD-λ 的 advantage 版

GAE 通过 $\lambda \in [0, 1]$ 做**指数加权平均**：

$$
A_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

其中 TD 残差：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

**两个极端**：
- $\lambda = 0$：$A_t = \delta_t$（单步 TD，低方差高偏差）
- $\lambda = 1$：$A_t = R_t - V(s_t)$（蒙特卡洛，高方差无偏）
- 中间值（典型 0.95）：方差-偏差权衡

### 6.3 GAE 的递推形式（实现用）

倒着递推：

$$
A_t = \delta_t + \gamma \lambda \cdot A_{t+1}
$$

```python
def compute_gae(rewards, values, gamma=1.0, lam=0.95):
    """
    rewards: [T]  每一步的即时 reward（LLM 里包含 KL）
    values:  [T+1]  包括 bootstrap value V(s_T)
    """
    T = len(rewards)
    advantages = [0.0] * T
    last_gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae
    returns = [a + v for a, v in zip(advantages, values[:-1])]
    return advantages, returns
```

### 6.4 LLM RLHF 里 GAE 的注意事项

- $\gamma$ 常设为 1.0（response 短，不需要 discount）
- $\lambda$ 典型 0.95
- 算完 advantage 通常会**做 batch 归一化**：减均值除标准差，进一步降低方差
- terminal value：episode 结束时 $V(s_{T+1}) = 0$

---

## 7. KL Penalty

### 7.1 公式

$$
\text{KL}(\pi_\theta \| \pi_\text{ref}) = \mathbb{E}_{a \sim \pi_\theta}\left[\log\frac{\pi_\theta(a|s)}{\pi_\text{ref}(a|s)}\right]
$$

### 7.2 直觉

防止 RL 训练把模型推到"reward 很高但已经不再是合理语言"的角落（reward hacking）。Reference model 通常就是 SFT 完成后冻结的那个模型。

### 7.3 两种实现方式

| 方式 | 怎么加 |
|---|---|
| **作为 token-level reward** | $r_t \leftarrow r_t - \beta \log\frac{\pi_\theta}{\pi_\text{ref}}$，KL 直接进入 advantage |
| **作为 loss 项** | 在总 loss 里加 $\beta \cdot \text{KL}$，与 policy loss 并列 |

OpenAI / TRL 实现一般用第一种。

---

## 8. PPO 训练循环伪代码（LLM RLHF 版）

```python
for iteration in range(N_iterations):
    # === 1. Rollout ===
    prompts = sample_prompts(batch_size)
    responses = policy.generate(prompts)              # 用 π_old 采样

    # === 2. 计算 reward（包含 KL）===
    scores = reward_model(prompts, responses)         # 标量 RM 打分
    log_probs = policy.log_probs(prompts, responses)
    ref_log_probs = ref_policy.log_probs(prompts, responses)
    kl = log_probs - ref_log_probs                    # token-level KL
    rewards = -beta * kl
    rewards[:, -1] += scores                          # 末尾加 RM 打分

    # === 3. 用 Critic 算 value & GAE ===
    values = critic(prompts, responses)               # [B, T]
    advantages, returns = compute_gae(rewards, values, gamma=1.0, lam=0.95)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # === 4. PPO 多 epoch 更新 ===
    for epoch in range(K):                            # 通常 K = 1~4
        for minibatch in shuffle(rollouts):
            new_log_probs = policy.log_probs(minibatch.prompts, minibatch.responses)
            ratio = (new_log_probs - minibatch.old_log_probs).exp()

            # Clip Policy loss
            surr1 = ratio * minibatch.advantages
            surr2 = ratio.clamp(1 - eps, 1 + eps) * minibatch.advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (MSE)
            value_loss = ((critic(minibatch.prompts, minibatch.responses) - minibatch.returns) ** 2).mean()

            loss = policy_loss + c1 * value_loss
            loss.backward()
            optimizer.step()
```

---

## 9. 各种 Loss 在大模型里"怎么算"对照

### 9.1 MSE Loss

```python
loss = ((pred - target) ** 2).mean()
```
- 大模型里**只用在 Critic** 上：`((V(s_t) - R_t) ** 2).mean()`
- 主 LLM head 用的是交叉熵（[[01-交叉熵损失详解]]）

### 9.2 GAE "Loss"

GAE **不是 loss**，是一个 **advantage 估计算法**。它产出 advantage 和 returns 这两个张量，分别给 policy loss 和 value loss 当输入。

> 面试里如果有人问"GAE loss 怎么算"，先澄清：GAE 是优势估计，不是 loss；它产生的 advantage 进入 PPO 的 clip 损失，它产生的 returns 进入 Critic 的 MSE 损失。

### 9.3 CE Loss

- 预训练 / SFT 的标准损失
- 在 RLHF 阶段**不直接出现**（但 ratio 的 log 本质上是 log-probability 之差）

---

## 10. 面试高频追问

### Q1：ratio 为什么是新旧策略的概率比？
ratio = importance sampling 校正因子。PPO 想用旧策略采样的数据更新多次新策略，每次更新都要修正"采样分布 ≠ 目标分布"的偏差。

### Q2：clip 的 $\epsilon$ 调大调小有什么影响？
- $\epsilon$ 调大：单步更新允许更激进 → 学得快但可能崩
- $\epsilon$ 调小：保守 → 慢但稳

### Q3：advantage 算完为什么要归一化？
batch 内减均值除标准差，本质是把 advantage 的尺度稳定下来。**这是降方差技巧，不改变方向，等价于给 policy loss 整体乘一个常数。**

### Q4：returns 和 advantage 的关系？
$R_t = A_t + V(s_t)$。换句话说，**Critic 学的是 returns，policy 学的方向是 advantage = returns - baseline**。

### Q5：大模型 RLHF 里 reward 是稀疏的，怎么把它分配到每个 token？
两条路线：
- **PPO/GAE**：靠 Critic 学一个 token-level value，advantage 自然分到每个 token
- **GRPO**：不要 Critic，整段 response 共享一个 outcome-level advantage（详见 [[05-PPO-DPO-GRPO-DAPO-GSPO演进对比]]）

### Q6：为什么 Critic 用 MSE 不用 Huber？
原则上都可以。MSE 更常见因为：① 简单；② 梯度光滑；③ 大模型场景 value 范围不会特别 outlier。Huber 在 value 有极端值的环境里更稳。

### Q7：Entropy bonus 在 LLM RLHF 里要不要？
- 经典 RL：要，鼓励探索
- LLM RLHF：常**不加**或系数很小。因为生成温度 + KL 已经隐式提供随机性

### Q8：discount factor $\gamma$ 该怎么选？
- 普通 RL：0.99
- LLM RLHF：常用 1.0，因为 episode 长度有限且没有"早结束有利"的偏好

### Q9：PPO 的 K-epoch 多次更新是怎么回事？
一次 rollout 之后，对采样的 minibatch 做 K 个 epoch 的梯度更新（K 通常 1-4）。这是 PPO 比 vanilla policy gradient 样本效率高的关键。**clip 保证多次更新不会让 policy 和采样分布偏离太多。**

---

## 关联阅读

- [[05-PPO-DPO-GRPO-DAPO-GSPO演进对比]] —— RL 算法演进全景
- [[01-交叉熵损失详解]] —— 训练阶段的核心 loss
- [[02-AdamW-优化器]] —— 优化器选择

## 参考资料

- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017
- Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, ICLR 2016
- Ouyang et al., *InstructGPT*, NeurIPS 2022 —— RLHF 第一篇大成
- HuggingFace TRL 文档 & DeepSpeed-Chat 实现
