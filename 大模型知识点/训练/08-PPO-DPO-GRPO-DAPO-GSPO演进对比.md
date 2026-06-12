---
tags:
  - 面试
  - 大模型
  - RLHF
  - PPO
  - DPO
  - GRPO
  - DAPO
  - GSPO
created: 2026-05-24
updated: 2026-05-24
---

# 大模型对齐算法演进：PPO → DPO → GRPO → DAPO → GSPO

> 核心问题：为什么 RLHF 一开始用 PPO？后来为什么出现 DPO、GRPO、DAPO、GSPO？它们解决了 PPO 的什么痛点？

---

## 0. 面试怎么答（30 秒电梯版）

> "RLHF 早期标准做法是 **PPO**：训一个 Reward Model 给出标量打分，然后用 actor-critic 框架的 PPO 来最大化 reward 同时用 KL 限制 policy 漂移。
>
> 但 PPO 有三个痛点：① 需要训单独的 Critic（Value Model），显存翻倍；② Reward Model 训练贵且容易 hack（钻空子）；③ 工程复杂、调参困难。
>
> **DPO** 把整个 RL 阶段干掉，用一个闭式推导把"偏好对"直接当作监督信号，做对比学习式的离线训练，简单稳定。
>
> **GRPO**（DeepSeek-R1 用）保留 PPO 框架但去掉 Value Model，用同一个 prompt 采样 G 个 response，**组内的 reward 标准化作为 advantage**，省一半显存。
>
> **DAPO**（字节 Seed）针对长 CoT RL 训练做了四个工程改进：Clip-Higher、动态采样、Token-Level Loss、Overlong Reward Shaping。
>
> **GSPO**（Qwen3 用）发现 GRPO 的 token-level 重要性比 ratio 在长序列下方差爆炸不稳定，改成 **sequence-level ratio**（对整段 response 做一次 ratio），训练更稳。"

记忆口诀：**PPO 是基线，DPO 砍掉 RL，GRPO 砍掉 Critic，DAPO 修工程坑，GSPO 修长序列稳定性**。

---

## 1. 背景：RLHF 三阶段

```
阶段 1：SFT (Supervised Fine-Tuning)
        用高质量人工标注的 (prompt, answer) 监督训练
        ↓
阶段 2：RM (Reward Model)
        训练一个打分模型：给定 (prompt, response)，输出标量 reward
        训练数据是人工排序的偏好对 (prompt, chosen, rejected)
        ↓
阶段 3：RL (Reinforcement Learning)
        用 RM 当作环境，让 LLM (policy) 学习生成 reward 更高的回答
        经典选择：PPO
```

PPO、DPO、GRPO、DAPO、GSPO 解决的都是**阶段 3**或者**阶段 2 + 阶段 3**的问题。

---

## 2. PPO：经典 RLHF 起点

### 2.1 核心组件

PPO 在 RLHF 里需要 **4 个模型同时在内存里**：

| 模型 | 作用 | 是否更新 |
|---|---|---|
| **Actor / Policy** $\pi_\theta$ | 待训练的 LLM，产生回答 | ✅ 训练目标 |
| **Critic / Value Model** $V_\phi$ | 给每个 token 估计未来 return | ✅ 同步训练 |
| **Reference Model** $\pi_\text{ref}$ | SFT 后的冻结快照，用来算 KL 防漂 | ❌ 冻结 |
| **Reward Model** $r_\psi$ | 给整段回答打分 | ❌ 冻结 |

> 这就是 PPO 的第一个痛点：**4 个模型的显存几乎等于 4 倍 SFT 训练**。

### 2.2 PPO 目标函数（简化版）

$$
\mathcal{L}^{PPO}(\theta) = \mathbb{E}_t\left[\min\left(\underbrace{\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}}_{\text{ratio } r_t(\theta)} A_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]
$$

- $r_t(\theta)$ = **新旧 policy 的概率比**，新策略想生成这个 token 比旧策略愿意多多少
- $A_t$ = **Advantage**，这个 token 的动作有多好（比"平均"好多少）
- $\text{clip}(\cdot, 1-\epsilon, 1+\epsilon)$ = 把 ratio 限制在 $[1-\epsilon, 1+\epsilon]$（典型 $\epsilon = 0.2$），防止单步更新太激进

完整 LLM RLHF 损失还会加上：
- **Critic loss**：$\frac{1}{2} (V_\phi(s_t) - R_t)^2$ —— 让 critic 学习真实 return
- **KL penalty**：$\beta \cdot \text{KL}(\pi_\theta \| \pi_\text{ref})$ —— 防止 policy 偏离 SFT 模型太远

#### PPO 的 Advantage 计算：GAE

PPO 的 $A_t$ 不是直接拿 reward 减 baseline 这么简单，而是用 **GAE（Generalized Advantage Estimation）** 算出来的。GAE 是 PPO 能稳定训练的关键。

**第一步：TD 误差**

每个 token 位置 $t$ 有一个即时 reward $r_t$（来自 Reward Model），Critic 输出当前状态的价值 $V(s_t)$ 和下一个状态的价值 $V(s_{t+1})$。TD 误差：

$$\delta_t = r_t + \gamma \cdot V(s_{t+1}) - V(s_t)$$

- $\delta_t > 0$：这一步比 Critic 预期的好（惊喜）
- $\delta_t < 0$：这一步比 Critic 预期的差（失望）
- $\gamma$：折扣因子（典型 0.95~0.99），让近期 reward 权重更大

**第二步：GAE 加权累积**

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l} = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \dots$$

其中 **$\lambda \in [0,1]$** 是 GAE 的核心超参，控制 bias-variance 权衡：

| $\lambda$ | 等价于 | 特点 |
|---|---|---|
| $\lambda=0$ | $\hat{A}_t = \delta_t$（只看一步 TD 误差） | 低方差、高偏差（只用到即时 reward） |
| $\lambda=1$ | $\hat{A}_t = \sum \gamma^l r_{t+l} - V(s_t)$（蒙特卡洛 return） | 高方差、低偏差（用到所有未来 reward） |
| 典型 $\lambda=0.95$ | 加权折中 | 最近的 $\delta$ 权重更大，远的衰减 |

**第三步：Returns（Critic 的训练目标）**

Advantage 算出后，Critic 要学的 Returns 是：

$$R_t = \hat{A}_t + V(s_t)$$

—— "这个状态实际拿了多少 return = baseline 预测 + advantage 修正"。

**完整计算链**：

```
Reward Model → {r_0, r_1, r_2, ..., r_T}   (每个 token 一个标量 reward)
    ↓
TD Error    → δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
    ↓
GAE         → A_t = Σ (γλ)^l · δ_{t+l}       (每个 token 一个 advantage)
    ↓
PPO Loss    → min(ratio·A_t, clip(ratio)·A_t)
```

> **对比 GRPO**：PPO 每个 token 有独立的 $A_t$（因为 GAE 对每个位置单独累加），而 GRPO 同一 response 所有 token 共享同一个 $\hat{A}_i$。这就是 PPO 需要 Critic 的根源——要算出每个位置的 $V(s_t)$ 才能做 GAE。

> **但是——Reward Model 本身只给一个句子级别的分，PPO 凭什么有 token 级 advantage？**
>
> 这才是问题的核心。在 RLHF 中，Reward Model 只在 **EOS token** 位置给一个标量分数，其余所有位置的 $r_t = 0$：
>
> ```
> Token:    "The"  "cat"  "sat"  "<eos>"
> r_t:        0      0      0     r=0.8     ← 实际只有最后一个 token 有分
> ```
>
> 如果只看 reward，这就是纯句子级信号。**PPO 能把它变成 token 级 advantage 的关键就是 Critic（$V(s_t)$）**。
>
> 具体来说，虽然 $r_t = 0$（$t < T$），但 $V(s_t)$ 在不同位置**不一样**——Critic 学会了在"已经说出关键内容后"给更高的 V 值，于是即使没有即时 reward，TD 误差也非常：
>
> ```
>                      "The"      "cat"      "sat"      "<eos>"
> r_t:                   0          0          0         r_T=0.8
> V(s_t):              0.1        0.3        0.6        0.0    ← Critic 估计的"后续能拿多少"
> δ_t = 0 + γ·V(s_{t+1}) - V(s_t):
>                    γ·0.3-0.1  γ·0.6-0.3  0.8-0.6     —      ← 每个位置 δ 不同！
> ```
>
> GAE 从不同起点累加这些 $\delta_t$，得到的 $\hat{A}_0, \hat{A}_1, \hat{A}_2, \hat{A}_3$ 各不相同——**一句话的分被 Critic 拆成了每个 token 的贡献度**。
>
> **PPO vs GRPO 的本质区别就在这里**：
>
> | | PPO | GRPO |
> |---|---|---|
> | Reward 信号 | 句子级（RM 在 EOS 输出一个分） | 句子级（verifier 给整段 response 打一个分） |
> | 如何变成 token 级？ | **Critic 学习 $V(s_t)$**，通过 TD 误差将 reward 分配到每个 token | **不拆分**，所有 token 直接用同一个组内 z-score $\hat{A}_i$ |
> | Advantage 粒度 | Token 级（$A_0, A_1, ..., A_T$ 各不同） | Outcome 级（$A_i$ 一个标量整段共享） |
> | 代价 | 多一个 Critic 模型，显存翻倍 | 信号粗，但省一半显存 |
>
> **一句话**：PPO 的 reward 也是句子级的，但 Critic 学会了"功劳分配"——哪个 token 对最终高分贡献大、哪个在拖后腿。GRPO 放弃了这个分配，换来省掉 Critic。

> 详细的 ratio、advantage、GAE 计算见 [[06-PPO损失函数详解]]。

### 2.2.1 深度解析：新旧 Policy 概率比到底是什么？

**一句话回答**：这个 ratio 比的不是模型参数，而是**同一个 token 在新旧两个 policy 下的输出概率**。参数的"新旧"只是手段——我们实际上关心的是训练过程中 policy 输出的概率变化有多大。

---

#### ① 比的是什么？（不是参数，是概率）

假设当前 token 是 `"cat"`，输入上下文是 `"The quick brown fox jumps over the lazy"`：

- 旧 policy $\pi_{\theta_{\text{old}}}$ 在这个上下文下输出 `"cat"` 的概率是 **0.02**
- 新 policy $\pi_{\theta}$（经过几步更新后）在同样上下文输出 `"cat"` 的概率变成了 **0.025**

那么：
$$r_t(\theta) = \frac{0.025}{0.02} = 1.25$$

> 这意味着新 policy 比旧 policy 更愿意生成 `"cat"` 这个 token，意愿提升了 25%。

**这里的 "新旧" 指的不是参数值本身相除，而是两个参数快照分别做 forward pass 得到的 logits → softmax → 取该 token 位置的概率 → 再相除。**

> **关键细节：取的是哪个 token 的概率？**
>
> 取的是 **$a_t$——采样时实际生成的那个 token** 的概率，不是整个词表上的某个统计量。
>
> 举例：当前序列是 `"The quick brown fox"`，采样阶段旧 policy 在这个上下文下输出了一个 token 概率分布（词表 50K 个选项各有各的概率），它**实际选中了** `"jumps"`。那么：
> - 旧概率 $P_{\text{old}}$ = 旧 policy 在"fox"后面分配给 `"jumps"` 的 softmax 概率（比如 0.15）
> - 新概率 $P_{\text{new}}$ = 新 policy 在**完全相同上下文**下分配给 `"jumps"` 的 softmax 概率（比如变成了 0.18）
>
> ratio = 0.18 / 0.15 = 1.2，意思是：新 policy 比旧 policy 更愿意**重复旧 policy 当时的选择**。
>
> **如果旧 policy 当时选的是 `"leaps"` 而不是 `"jumps"`**，那我们比的就是 `"leaps"` 的概率。**a_t 是"已经发生的动作"，不是"最优动作"**——ratio 衡量的是 policy 对自己过去行为的"态度变化"。
>
> **那 a_t 是谁选的？就是旧策略 $\pi_{\theta_{\text{old}}}$ 自己在 rollout 阶段按自身概率分布采样出来的。** 这就是 PPO 做 importance sampling 的经典模式：
> 1. **Rollout（采样）**：用旧策略跑一遍，按它的 softmax 分布逐 token 采样，生成一条完整 response，同时记下每个位置旧策略给被选中 token 的概率。
> 2. **Train（更新）**：用新策略对同一条数据做 forward，重新算这串 token 的概率，跟旧策略当时记下的概率相比得到 ratio。
>
> 本质是：**行为来自旧策略，评价来自新策略。** 新策略问自己："如果当时是我（新参数）面对同样的上下文，我会多大概率做同样的选择？"——ratio > 1 说明新策略更认同旧策略的选择，ratio < 1 说明新策略觉得旧策略选得不好。

---

#### ② 计算流程（具体怎么比）

```
输入：同一个 prompt 的前缀序列 s_t（已生成的 token 序列）

Step 1：用旧参数 θ_old 做一次 forward
        s_t → model(θ_old) → logits → softmax → 取出 a_t 位置的概率 = P_old
Step 2：用当前参数 θ 做一次 forward
        s_t → model(θ) → logits → softmax → 取出 a_t 位置的概率 = P_new
Step 3：ratio = P_new / P_old
```

**实现细节**：
- 前面 rollout 采样阶段用 $\theta_{\text{old}}$ 生成一条完整 response，同时记录下每个 token 位置旧 policy 给出的概率值。
- 后面训练阶段用当前 $\theta$ 对同一条数据再做 forward，取对应位置的概率值。
- ratio 逐 token 相除，得到每个位置的重要性权重。

```python
# PyTorch 伪代码
log_probs_old = old_log_probs  # 采样时记录的，每个 token 的对数概率
log_probs_new = current_model(seq).log_softmax(-1).gather(-1, token_ids)
ratio = torch.exp(log_probs_new - log_probs_old)  # exp(log p_new - log p_old) = p_new / p_old
```

---

#### ③ 两个维度：Ratio 粒度 vs Advantage 粒度

这里有一个极易混淆的点——Ratio 和 Advantage 是**两个独立维度**，不能混为一谈：

| 维度 | 问的是什么 | PPO | GRPO / DAPO | GSPO |
|---|---|---|---|---|
| **Ratio 粒度** | 新旧 policy 在**什么单位**上比概率？ | Token | Token | **Sequence** |
| **Advantage 粒度** | "好坏"信号在**什么单位**上给？ | Token（GAE） | **Outcome**（整段共享） | Outcome（整段共享） |

---

##### 维度一：Ratio 粒度（概率比的单位）

这就是前面讲的——每个 token 独立算 ratio，还是整段共享一个 ratio。

```
PPO / GRPO / DAPO（Token-Level Ratio）:
  Response: "The  cat  sat"
            r1   r2   r3    ← 3 个独立 ratio

GSPO（Sequence-Level Ratio）:
  Response: "The  cat  sat"
            └─── S ────┘    ← 1 个 ratio（geometric mean）
```

##### 维度二：Advantage 粒度（好坏信号的单位）

这才是你问的关键问题。**Advantage 回答的是"这个动作比平均好多少"，不同算法对这个"动作"的定义粒度不同。**

**PPO — Token-Level Advantage（GAE）**

PPO 有一个独立的 Critic（Value Model）为**每个 token 位置**估计一个 V(s_t)（"从这个位置往后预期能拿多少 return"），然后用 GAE 算出每个 token 的 advantage $A_t$。同一段 response 里，不同位置的 token 有不同的 $A_t$。

```
Response:  "The"    "cat"    "sat"
GAE A_t:   +0.1     +0.5     -0.2    ← 3 个不同的 advantage
```

> 这就是 PPO 需要 Critic 的根本原因：token 级的 advantage 需要 token 级的 value 估计，而 Critic 正是用来输出每个 token 的 V(s_t) 的。

**GRPO / DAPO / GSPO — Outcome-Level Advantage（组内 z-score）**

GRPO 没有 Critic。它对同一个 prompt 采样 $G$ 个 response，每个 response 得到一个标量 reward $r_i$（比如数学题做对 1 分、做错 0 分），然后在组内做 z-score 标准化：

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\})}$$

**同一个 response 里的所有 token 共享同一个 $\hat{A}_i$**。

```
Prompt: "1+1=?"

  Response 1: "2"           reward=1  →  Â₁ = +1.2
             Token: "2"
             Adv:   +1.2    ← 只有 1 个 token，无所谓

  Response 2: "答案是2"      reward=1  →  Â₂ = +1.2
             Token: "答" "案" "是" "2"
             Adv:   +1.2 +1.2 +1.2 +1.2  ← 4 个 token 共享同一个 Â
```

所以 GRPO 那节写的"同一组里的所有 token 共享同一个 $\hat{A}_i$"是对的——GRPO 的 advantage 确实是 outcome-level（一段 response 一个标量），不是 token-level。

##### 两个维度交叉 = 四种组合

| Ratio 粒度 | Advantage 粒度 | 谁在用 | 特点 |
|---|---|---|---|
| Token | Token | **PPO** | 最细粒度，需要 Critic，显存大 |
| Token | Outcome | **GRPO, DAPO** | Ratio 细粒度 + Advantage 粗粒度，去掉 Critic |
| Sequence | Outcome | **GSPO** | Ratio 和 Advantage 都粗，长序列最稳 |
| Sequence | Token | （暂无主流方法） | Ratio 粗但 Advantage 细，不太有意义 |

> **关键结论**：GRPO 砍掉 Critic 的代价就是 advantage 从 token-level 退化到 outcome-level——每个 token 不再有独立的"好坏"判断，整段 response 共享一个分数。对于数学/代码这种结果导向的任务（只看最终答案对不对），这完全够用；对于开放式对话（每句话的流畅度都重要），outcome-level 信号太粗。

#### ④ 两个容易被问倒的追问

**追问 1："ratio > 1 和 ratio < 1 分别意味着什么？"**

- $r_t > 1$：新 policy 比旧 policy 更倾向这个 token → 如果 advantage 为正，鼓励这种变化
- $r_t < 1$：新 policy 比旧 policy 更不倾向这个 token → 如果 advantage 为负，鼓励这种变化（少犯错误）
- $r_t \approx 1$：两个 policy 对这个 token 的态度差不多

**追问 2："为什么一定要和旧 policy 比，直接优化 reward 不行吗？"**

不行。如果只优化 reward 不从旧 policy 起步：
- 模型会**忘记基础语言能力**（reward hack，比如输出全变成标点符号因为 RM 觉得标点无害）
- 训练会**不稳定**——同一个 prompt 两次采样的 response 不同但 reward 不同，梯度方向乱跳
- ratio 的本质是**信任域（Trust Region）约束**：只允许 policy 在旧 policy 附近小步移动，防止"一步改太大 → 崩了 → 不可恢复"

---

#### ⑤ Ratio 粒度对梯度的影响

Token-Level 和 Sequence-Level 两种 ratio 设计直接影响梯度更新：

| | Token-Level (PPO/GRPO/DAPO) | Sequence-Level (GSPO) |
|---|---|---|
| Ratio 定义 | $r_t = \frac{\pi_\theta(a_t)}{\pi_{\text{old}}(a_t)}$ | $s = \left(\prod_{t=1}^{T}\frac{\pi_\theta(a_t)}{\pi_{\text{old}}(a_t)}\right)^{1/T}$ |
| 每个 token 贡献 | 独立，各有自己的 clip | 共享同一个 ratio 和 clip |
| 长序列方差 | $\propto T$（乘性累积） | $\propto 1$（平均化，受控） |
| 梯度更新 | 每 token 独立缩放 | 所有 token 统一缩放 |

> 注意这里只对比 **Ratio 粒度**。Advantage 粒度是另一个独立维度（见 ③），PPO 用 token-level advantage（GAE），GRPO/DAPO/GSPO 都用 outcome-level advantage（组内 z-score，整段共享）。

**一句话总结**：ratio 是一个**概率层面的相对变化率**，不是参数比值。它回答了"这个 token 在新策略里比旧策略里变得多重要了"这个问题。Token-level ratio 给细粒度但长序列方差炸，Sequence-level ratio 给稳定性但稍粗。**而 Advantage 是另一个东西——PPO 用 Critic 算 token-level advantage，GRPO 系用组内 reward 算 outcome-level advantage，两者独立。**

### 2.3 PPO 的痛点

| 痛点 | 具体表现 |
|---|---|
| 💸 **显存爆炸** | Actor + Critic + Reward + Ref = 4 个模型 |
| 🐛 **调参敏感** | 学习率、$\epsilon$、$\beta$、GAE 的 $\lambda$ 都要调 |
| 🎭 **Reward Hack** | RM 是模型，policy 会找 RM 的漏洞（过长、加感叹号等） |
| 🐢 **样本效率低** | On-policy，旧数据不能直接复用 |

---

## 3. DPO：跳过 RL 阶段

### 3.1 关键洞察

DPO 论文（NeurIPS 2023）的核心数学结论：

> RLHF 的最优策略 $\pi^*$ 与奖励 $r$ 之间存在**闭式关系**：$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)} + \text{const}$
>
> 所以可以反过来：**直接用偏好数据训练 policy，不需要显式的 RM 和 PPO**。

### 3.2 DPO 损失函数

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l) \sim D}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\right]
$$

其中：
- $y_w$ = chosen（被偏好的回答）
- $y_l$ = rejected（被拒绝的回答）
- $\sigma$ = sigmoid
- $\beta$ = 类似 PPO 里的 KL 温度系数

**直观理解**：让模型对 $y_w$ 的概率（相对于参考模型）提升，对 $y_l$ 的概率下降，提升和下降的"程度差"用 sigmoid 包起来做对比学习。

### 3.3 DPO 的优缺点

✅ **优点**：
- **只需要 2 个模型**：policy + reference（reference 可以冻结一次预算）
- 训练像 SFT 一样稳定
- 不需要 reward model 训练流程
- 不需要在线采样

❌ **缺点**：
- **离线训练**，没有 exploration，能力上限受限于偏好数据集
- 容易过拟合偏好对里的 superficial pattern
- 对 OOD 数据泛化弱

> **演进**：基于 DPO 出现 IPO、KTO、cDPO、ORPO 等变体修正不同的问题。

---

## 4. GRPO：DeepSeek R1 让 RL 飞回来

### 4.1 动机：把 Critic 砍掉

DeepSeekMath 论文（2024）提出 GRPO（Group Relative Policy Optimization）。

> PPO 需要 Critic 是因为要给每个 token 算 advantage。但 Critic 本身就是一个 LLM 大小的模型，显存吃不消。
>
> **GRPO 的做法**：对同一个 prompt 采样 $G$ 个回答（一组），用**组内 reward 的均值/方差**当作 baseline，不需要单独的 Critic 来估计 baseline。

### 4.2 GRPO 公式

对每个 prompt $q$，采样 $G$ 个回答 $\{o_1, o_2, \dots, o_G\}$，得到 reward $\{r_1, \dots, r_G\}$。

**组内归一化的 advantage**：

$$
\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\})}
$$

> 同一组里的所有 token 共享同一个 $\hat{A}_i$（outcome-level advantage）。

GRPO 损失（沿用 PPO 的 clip 结构）：

$$
\mathcal{L}_{\text{GRPO}} = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\min\left(r_{i,t}(\theta)\hat{A}_i,\ \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_i\right)\right] - \beta \cdot \text{KL}(\pi_\theta \| \pi_\text{ref})
$$

其中 $r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_\text{old}}(o_{i,t}|q, o_{i,<t})}$ 仍然是 **token-level ratio**。

> **为什么 GRPO 公式有 $\frac{1}{G}$ 和 $\frac{1}{|o_i|}$，而 PPO 公式没有？**
>
> PPO 的公式写的是 $\mathbb{E}_t[\dots]$，$\mathbb{E}_t$ 是**对"所有 timestep + 所有样本"取期望**。在代码实现中，PPO 也是把所有 token 的 loss 加起来除以总 token 数，只不过这些操作全被 $\mathbb{E}_t$ 一个符号"藏"了。写成显式就是：
>
> $$\text{PPO 实际代码：} \quad \frac{1}{\text{total\_tokens}} \sum_{\text{batch}} \sum_{t} \min(r_t A_t,\ \text{clip}(\dots))$$
>
> GRPO 不能用一个 $\mathbb{E}$ 糊弄过去，因为引入了一个新结构——**每个 prompt 有 G 个 response 的"组"**，必须说明：① 组间怎么聚合（除以 G），② 同一个 response 里 token 怎么聚合（除以 $|o_i|$）。
>
> 具体拆解：
>
> | 符号 | 含义 | 为什么 PPO 不需要 |
> |---|---|---|
> | $\frac{1}{G}$ | 对同一 prompt 的 $G$ 个 response **取平均** | PPO 每个 prompt 只采 1 个 response（$G=1$），不需要这个维度 |
> | $\frac{1}{\vert o_i\vert}$ | 对第 $i$ 个 response 内的所有 token **取平均**（长度归一化） | PPO 的 $\mathbb{E}_t$ 隐含按 token 平均；写成显式也是除以总 token 数 |
>
> **$\frac{1}{|o_i|}$ 为什么重要？** 如果不除以 $|o_i|$，长 response 的 token 数量多，对梯度的贡献就天然更大。GRPO 用 per-response 平均来保证**每个 response 权重相等**——不管长短。这也引出了后续 DAPO 的争议（见第 5 节）：DAPO 认为 per-response 平均会让长 CoT 里每个 token 的梯度被稀释，改回了直接按总 token 数平均。

### 4.3 GRPO 的优势与问题

✅ **优势**：
- **去掉 Value Model**，显存省一半
- 在数学/代码这种**有可验证 reward** 的任务上效果极好（DeepSeek-R1、Kimi-K2）
- 组内对比天然鼓励 exploration

⚠️ **问题**：
- 当 reward 全 0（全错）或全 1（全对）时 $\hat{A}_i$ 退化（std = 0）
- token-level ratio 在**长序列下方差爆炸**（GSPO 要修的痛点）
- 组采样开销随 $G$ 线性增长

---

## 5. DAPO：字节 Seed 的工程优化

### 5.1 背景

DAPO（Decoupled Clip and Dynamic Sampling Policy Optimization，字节 Seed 2025）针对 **R1 复现路线下长 CoT 训练**遇到的实际问题，提了 4 个改进。

### 5.2 四大改进

#### ① Clip-Higher（解耦 clip 上下界）

**问题**：PPO/GRPO 用对称 clip $[1-\epsilon, 1+\epsilon]$，导致**正向 ratio 被压制 = 探索不足**。

**做法**：

$$
\text{clip}(r_t, 1-\epsilon_\text{low}, 1+\epsilon_\text{high}),\quad \epsilon_\text{high} > \epsilon_\text{low}
$$

典型 $\epsilon_\text{low} = 0.2, \epsilon_\text{high} = 0.28$。给"低概率好 token"更多上升空间。

#### ② Dynamic Sampling（动态采样）

**问题**：当一组 $G$ 个回答 reward 全对或全错时，advantage = 0，这组数据无贡献。

**做法**：**持续采样直到这一组里同时有对的和错的**才进入梯度。避免无效 batch。

#### ③ Token-Level Policy Gradient Loss（token 级损失）

**问题**：GRPO 的损失把"先平均到 response 再平均到 batch"，**长 response 里的每个 token 权重被稀释**。

**做法**：

$$
\mathcal{L}_{\text{DAPO}} = \mathbb{E}\left[\frac{\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}\min(\dots)}{\sum_{i=1}^{G}|o_i|}\right]
$$

—— 直接对 token 数加权平均，让长 CoT 的每个 token 都被公平对待。

#### ④ Overlong Reward Shaping（超长惩罚平滑）

**问题**：长度超过 context 限制的样本被截断 → reward = 0，给模型很噪的信号。

**做法**：对超长样本不是粗暴 0，而是**软惩罚**（线性递减），让信号更稳。

### 5.3 DAPO 性价比

DAPO 在 Qwen2.5-32B 上 AIME 2024 从 30% 提到 50%+，关键是**工程层修复**而不是损失函数大改。

---

## 6. GSPO：Qwen3 修长序列稳定性

### 6.1 动机：Token-Level Ratio 的方差爆炸

> Qwen 团队（2025 年中）观察到：GRPO 在长 CoT 训练后期**容易崩**，原因是 token-level importance ratio $r_{i,t}(\theta)$ 在长序列下乘起来方差极大。

具体说：每个 token 的 ratio 是一个有噪声的随机量，长度 $T$ 越长，**采样误差累积越严重**。

### 6.2 GSPO 的核心改动：Sequence-Level Ratio

GSPO（Group Sequence Policy Optimization）把 token-level ratio 换成 **整段 response 的 geometric mean ratio**：

$$
s_i(\theta) = \left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_\text{old}}(o_i|q)}\right)^{1/|o_i|} = \exp\left(\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\log\frac{\pi_\theta(o_{i,t}|\cdot)}{\pi_{\theta_\text{old}}(o_{i,t}|\cdot)}\right)
$$

—— 对整段序列做**长度归一化的 likelihood ratio**，从一个 token 一个 ratio 变成一段一个 ratio。

GSPO 损失：

$$
\mathcal{L}_{\text{GSPO}} = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\min\left(s_i(\theta)\hat{A}_i,\ \text{clip}(s_i(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_i\right)\right]
$$

### 6.3 为什么稳

- token-level ratio 方差随长度 $T$ **乘性累积**
- sequence-level ratio 是 log-likelihood 的平均，方差随 $T$ **平均化** → 长 CoT 训练更稳
- 还有一个 GSPO-token 变体把 token 级也保留下来，兼容 token-wise advantage

> Qwen3-Coder / Qwen3-235B 后期 RL 阶段就是用 GSPO。

---

## 7. 五种算法对照表

| 算法 | 是否要 Critic | 是否要 RM | 是否在线采样 | Advantage 来源 | Ratio 粒度 | 主要修复 |
|---|---|---|---|---|---|---|
| **PPO** | ✅ | ✅ | ✅ | GAE 估计 | Token | 基线 |
| **DPO** | ❌ | ❌ | ❌（离线偏好对） | 隐式（闭式） | — | 砍掉 RL |
| **GRPO** | ❌ | ✅ | ✅ | 组内 z-score | Token | 砍掉 Critic |
| **DAPO** | ❌ | ✅ | ✅ + 动态 | 组内（+ token loss） | Token | 工程稳定性 |
| **GSPO** | ❌ | ✅ | ✅ | 组内 | **Sequence** | 长序列方差 |

---

## 8. 演化主线

```
PPO (经典 RLHF, 4 模型重)
 │
 ├─ DPO ──── 砍掉 RL，跳到偏好对监督学习
 │
 └─ GRPO ─── 砍掉 Critic，组内 baseline
       │
       ├─ DAPO ─── 工程层修：clip-higher / 动态采样 / token-level loss / 超长平滑
       │
       └─ GSPO ─── 把 token-level ratio 换成 sequence-level，长 CoT 不崩
```

---

## 9. 面试高频追问

### Q1：为什么 RLHF 一开始选 PPO 而不是 TRPO/A2C？
PPO 在 OpenAI 内部已经成熟，clip 机制工程上比 TRPO 的二阶约束更容易实现；A2C 没有 clip，单步更新易爆，PPO 更稳定。

### Q2：DPO 和 PPO 谁更好？
看场景：
- **数据是离线偏好对、追求工程简单**：DPO；
- **有 verifier / 强化推理（数学/代码）能在线生成 + 打分**：PPO/GRPO 系列上限更高（能 exploration）。

### Q3：GRPO 为什么能去掉 Critic？
PPO 用 Critic 是为了给每个 token 估计 baseline 来减小 advantage 的方差。GRPO 通过**同一 prompt 采样多个 response，用组内 reward 的均值当 baseline**，把 baseline 估计从一个学习问题降为一个简单的统计。

### Q4：GRPO 里 advantage 怎么算？
**outcome reward**（一段回答只有一个 reward）→ 减均值除标准差得到组内 z-score → 这个标量作为 response 内**所有 token 共享的 advantage**。

### Q5：DAPO 解决了 GRPO 的什么问题？
不是替换 GRPO，是 4 个工程修复：① clip 上下界解耦给探索更多空间；② reward 全对/全错的组动态采样跳过；③ token 级 loss 避免长 CoT 被稀释；④ 超长 response 不再 reward = 0。

### Q6：GSPO 比 GRPO 强在哪？
长 CoT（几千 token）训练时，token-level ratio 方差累积导致训练崩。GSPO 把 ratio 改成 sequence-level geometric mean，方差不再累积，训练稳定性显著提升。

### Q7：为什么 GSPO 用 geometric mean 而不是 arithmetic mean？
因为 ratio 的本质是 $\exp(\sum \log p_\theta - \sum \log p_\text{old})$，对 log 求平均 = geometric mean，与 likelihood ratio 的数学结构天然对齐。

### Q8：能不能去掉 Reward Model？
- DPO 已经去掉了
- 对于**可验证 reward 的任务**（数学有标准答案、代码有单测），可以用 rule-based reward 完全跳过 RM —— 这是 DeepSeek-R1 / Kimi-K2 的做法
- 一般对话场景仍需 RM 提供软打分

### Q9：on-policy vs off-policy 的区别？
- on-policy（PPO/GRPO/GSPO）：用当前 policy 采样的数据来更新当前 policy，数据用一次就丢
- off-policy（DPO）：用别人（或别的 checkpoint）采样的数据训练当前 policy

---

## 10. 关联阅读

- [[06-PPO损失函数详解]] —— ratio / advantage / GAE / returns 完整推导
- [[02-AdamW-优化器]] —— RL 训练优化器选择
- [[07-Qwen3-模型架构]] —— Qwen3 用的就是 GSPO

## 参考资料

- Schulman et al., *Proximal Policy Optimization*, 2017
- Rafailov et al., *DPO: Your Language Model is Secretly a Reward Model*, NeurIPS 2023
- Shao et al., *DeepSeekMath / GRPO*, 2024
- Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*, 字节 Seed, 2025
- Zheng et al., *GSPO: Group Sequence Policy Optimization*, Qwen 团队, 2025
