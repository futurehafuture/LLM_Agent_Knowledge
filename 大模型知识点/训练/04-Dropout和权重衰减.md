# Dropout 和权重衰减

> 面经问题：dropout 和权重衰减分别是什么？二者有什么区别？为什么 AdamW 里经常强调“解耦权重衰减”？

## 1. 一句话回答

**Dropout** 和 **权重衰减（weight decay）** 都是缓解过拟合的正则化方法，但它们约束模型的方式不同：

- Dropout 在训练时随机把一部分神经元输出置零，相当于让模型每次只使用一个随机子网络，迫使特征不要过度依赖某几个神经元。
- 权重衰减直接约束参数大小，让模型倾向于使用较小的权重，从而降低有效复杂度。

更面试化地说：**dropout 是对激活/网络结构加随机噪声；权重衰减是对参数范数加约束或在优化步骤中收缩参数。**

---

## 2. 为什么需要它们：过拟合与模型复杂度

深度网络参数很多，表达能力很强。如果训练数据有限，模型可能不仅学习真实规律，也记住训练集里的噪声。正则化的核心思想是：

> 不只追求训练损失最低，还要限制模型用“过于复杂、过于脆弱”的方式拟合训练集。

Dropout 和权重衰减就是两种常见限制方式：

- Dropout 限制的是 **特征之间的共适应（co-adaptation）**。
- 权重衰减限制的是 **参数规模（weight norm）**。

---

## 3. Dropout：训练时随机丢神经元

### 3.1 基本机制

设某一层的激活为 $h$，dropout 概率为 $p$，保留概率为 $q = 1 - p$。训练时采样一个伯努利掩码：

$$
m_i \sim \mathrm{Bernoulli}(q)
$$

常见框架使用 **inverted dropout**，训练时直接缩放保留下来的激活：

$$
\tilde{h}_i = \frac{m_i}{q} h_i
$$

这样有：

$$
\mathbb{E}[\tilde{h}_i] = h_i
$$

因此推理时不需要再缩放，dropout 层等价于恒等映射。PyTorch 的 `nn.Dropout` 文档也说明：训练时输出会按 $1/(1-p)$ 缩放，eval 阶段直接计算恒等映射。

### 3.2 直观理解

每次前向传播时，dropout 都随机删除一部分神经元及其连接，所以模型不能把某个判断完全压在少数特征上。它会被迫学习更分散、更鲁棒的表示。

从另一个角度看，dropout 近似训练了指数数量的子网络，并在测试时做一种参数共享的模型平均。JMLR 的 Dropout 原论文将它描述为：训练时随机丢弃神经元，减少神经元之间复杂的共适应。

### 3.3 使用位置

常见经验：

- 在 MLP、分类头、Transformer 的 attention/FFN/residual dropout 中常见。
- CNN 早期常在全连接层后使用；现代 CNN/ResNet 中因为 BatchNorm、数据增强、残差结构已经提供较强正则，dropout 使用更谨慎。
- 如果 dropout 放在 BatchNorm 前后，可能引入训练/推理统计不一致。CVPR 2019 的一篇论文将 dropout 与 BN 组合效果变差解释为 variance shift。

---

## 4. 权重衰减：让参数不要长太大

### 4.1 L2 正则化形式

给原始训练目标 $L(w)$ 加上 L2 惩罚：

$$
L_{reg}(w) = L(w) + \frac{\lambda}{2} \lVert w \rVert_2^2
$$

对参数求梯度：

$$
\nabla_w L_{reg}(w) = \nabla_w L(w) + \lambda w
$$

如果用普通 SGD 更新：

$$
w_{t+1} = w_t - \eta (\nabla_w L(w_t) + \lambda w_t)
$$

整理得：

$$
w_{t+1} = (1 - \eta \lambda) w_t - \eta \nabla_w L(w_t)
$$

这就是“衰减”的来源：每一步都会把权重乘上一个小于 1 的系数，再沿损失梯度下降。

### 4.2 直观理解

权重越大，模型往往越容易形成非常陡峭、复杂、对输入扰动敏感的决策边界。权重衰减通过惩罚大权重，让模型倾向于更平滑、更简单的解。

D2L 对权重衰减的解释很直接：它不直接减少参数数量，而是限制参数可以取得的数值范围；在小批量 SGD 下，它通常等价于 L2 正则化。

---

## 5. L2 正则化和 weight decay 一定等价吗？

这是面试高频追问。

### 5.1 在普通 SGD 中基本等价

如上面的推导，SGD 中把 $\lambda w$ 加到梯度里，等价于每步对参数做一次比例收缩。因此很多材料会把 L2 regularization 和 weight decay 混用。

### 5.2 在 Adam 这类自适应优化器中不等价

Adam 会维护一阶矩和二阶矩估计。如果把 L2 正则项直接加进梯度：

$$
g_t = \nabla_w L(w_t) + \lambda w_t
$$

那么 $\lambda w_t$ 会进入 Adam 的动量和二阶矩统计，被自适应学习率重新缩放。这样就不再是“对参数做统一收缩”。

AdamW 的核心思想是：**把权重衰减从梯度更新里解耦出来**。先用 Adam 根据损失梯度更新，再单独对参数做衰减。简化写法是：

$$
w_{t+1} = w_t - \eta \cdot \mathrm{AdamStep}(\nabla L(w_t)) - \eta \lambda w_t
$$

Loshchilov 和 Hutter 的 AdamW 论文指出：L2 正则化和权重衰减在 SGD 中等价，但在 Adam 等自适应梯度算法中并不等价；解耦后，weight decay 的最优取值也更少和学习率纠缠。PyTorch 的 `AdamW` 文档也明确写到：AdamW 中 weight decay 不会累积到 momentum 或 variance 中。

---

## 6. Dropout vs 权重衰减：核心区别

| 维度 | Dropout | 权重衰减 |
|---|---|---|
| 作用对象 | 激活值、神经元输出、子网络结构 | 模型参数 $w$ |
| 数学形式 | 乘以随机 mask：$\tilde{h}=\frac{m}{1-p}h$ | 惩罚 $\lVert w \rVert_2^2$ 或直接衰减参数 |
| 随机性 | 有随机噪声，训练和推理行为不同 | 通常确定性，每步收缩参数 |
| 主要效果 | 防止特征共适应，近似子网络集成 | 限制参数范数，偏向平滑/简单解 |
| 常见超参 | dropout rate $p$ | weight decay 系数 $\lambda$ |
| 与优化器关系 | 主要体现在前向传播 | 与优化器实现强相关，AdamW 要解耦 |
| 推理阶段 | 关闭 dropout | 没有单独推理行为 |

---

## 7. 能不能一起用？

可以，而且很多模型会同时使用。例如 Transformer 训练里常见：

- attention dropout / residual dropout：约束中间激活。
- AdamW weight decay：约束参数规模。

但要注意：

1. **两者不是越大越好**。dropout 太大会欠拟合，weight decay 太大会让参数过小、训练不足。
2. **有 BatchNorm 时谨慎使用 dropout**。尤其是 dropout 放在 BN 前可能导致 BN 统计量和推理分布不一致。
3. **不要对所有参数都 weight decay**。工程中通常不对 bias、LayerNorm/BatchNorm 的 scale 和 shift 做 weight decay，因为这些参数本身不承担同样的复杂函数拟合角色，强行衰减可能伤害训练。
4. **Adam 优先考虑 AdamW**。如果你想要真正的 weight decay 行为，现代深度学习中通常使用 AdamW，而不是 Adam + L2。

---

## 8. 面试回答模板

可以这样回答：

> Dropout 和权重衰减都是正则化方法。Dropout 作用在激活上，训练时随机把一部分神经元输出置零，并对保留部分做缩放，使期望不变；推理时关闭。它的直觉是减少神经元之间的共适应，也可以看成在训练很多共享参数的子网络。权重衰减作用在参数上，常见形式是给 loss 加 $\frac{\lambda}{2}\lVert w\rVert^2$，在 SGD 下等价于每一步把权重乘上 $(1-\eta\lambda)$，让模型偏向较小权重和更简单的函数。二者区别在于 dropout 是随机结构/激活噪声，weight decay 是参数范数约束。还要注意，L2 和 weight decay 在 SGD 下等价，但在 Adam 中不等价，因为 L2 项会进入一阶、二阶矩；AdamW 通过把 weight decay 从梯度里解耦，才更接近真正的参数衰减。

---

## 9. 常见追问

### 9.1 为什么 dropout 推理时不用随机丢神经元？

因为推理阶段希望输出稳定。如果训练时使用 inverted dropout，保留下来的激活已经乘以 $1/(1-p)$，保证期望与原激活一致，所以推理时直接使用完整网络即可。

### 9.2 dropout rate 怎么选？

常见经验：

- MLP 隐藏层可从 0.1 到 0.5 试起。
- Transformer 中常见 0.0 到 0.1 或 0.2，取决于数据规模和模型大小。
- 数据越少、模型越容易过拟合，dropout 可以适当增大；如果已经欠拟合，应减小或关闭。

### 9.3 weight decay 怎么选？

常见经验：

- AdamW 中常从 $10^{-2}$、$10^{-3}$、$10^{-4}$ 量级搜索。
- 大模型预训练中 weight decay 常与学习率、warmup、训练 token 数一起调。
- bias、LayerNorm/BatchNorm 参数通常放到 no_decay 参数组。

### 9.4 dropout 和数据增强有什么区别？

二者都可以看成加噪声，但位置不同：

- 数据增强给输入加变化，鼓励模型对输入扰动不敏感。
- dropout 给内部表示加随机 mask，鼓励模型不要依赖某些隐藏单元。

---

## 10. 参考资料

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting, JMLR 2014](https://jmlr.org/papers/v15/srivastava14a.html)
- [PyTorch `torch.nn.Dropout` 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html)
- [Dive into Deep Learning: Dropout](https://d2l.ai/chapter_multilayer-perceptrons/dropout.html)
- [Dive into Deep Learning: Weight Decay](https://d2l.ai/chapter_linear-regression/weight-decay.html)
- [Decoupled Weight Decay Regularization, arXiv / ICLR 2019](https://arxiv.org/abs/1711.05101)
- [PyTorch `torch.optim.AdamW` 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
- [Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift, arXiv / CVPR 2019](https://arxiv.org/abs/1801.05134)
- [知乎：一文搞透 Dropout、L1/L2 正则化/权重衰减](https://zhuanlan.zhihu.com/p/147029320)
