# Stanford AI 课程 第九讲：Scaling Laws 详细讲义

> 课程截图拍摄日期：2026-05-10（PPT 1-25）、2026-05-11（PPT 26-55）  
> 主题：Scaling Laws——从历史背景到 LLM 工程实践与 Chinchilla 深度解析  
> 内容来源：斯坦福 AI 课程讲义截图（共 55 张 PPT）

新增内容（PPT 26–55）：

Part 2 — 模型工程 Scaling

（PPT 26–38） 
- 架构比较：Transformer vs LSTM，用小模型曲线外推即可判断 
- 跨架构普适性（Tay 2022） 
- 优化器：Adam vs SGD 斜率相同，截距略异 
- 深度/宽度：1层 vs 2层差距大，宽深比对性能影响极小 
- Embedding 参数的特殊性（不遵循同样的 Scaling 规律） 
- MoE 中参数价值分析 
- Critical Batch Size：定义、公式、与 Loss 的关系 
- μP 参数化：使最优学习率对模型宽度尺度不变 
- 警告：下游任务的 Scaling 远不如预训练可预测 

Part 3 — 联合 Scaling 与 Chinchilla（PPT 39–55） 
- 联合 data-model scaling laws（Rosenfeld、Kaplan 两种公式） 
- Chinchilla 三种拟合方法详解（Method 1/2/3） 
- Kaplan vs Chinchilla 差距根源（参数计数方式 + warmup 问题） 
- Chinchilla Method 3 的数据错误（Besiroglu 2024 修正） 
- 关键实践结论：训练最优 ≠ 部署最优，Llama 3 70B 用了 215 tokens/param - IsoFLOP 方法的普适性（适用于 LM、Diffusion、MoE）

---

## PPT 1 — Sample Complexity and Rates（样本复杂度与收敛速率）

![](slide01.png)

### 核心内容

本页是整节课的引入页，展示了理论研究者对"规模扩展（scaling）"问题的早期思考。

**关键公式 1：有限假设集的泛化误差界**

$$\epsilon(\hat{h}) \leq \left(\min_{h \in H} \epsilon(h)\right) + 2\sqrt{\frac{1}{m} \log \frac{2k}{\delta}}$$

**逐项解读：**

- $\hat{h}$：从训练集上学习到的最优假设
- $\min_{h \in H} \epsilon(h)$：假设集 $H$ 中最优假设的真实误差（即"贝叶斯误差"下界，模型本身的极限）
- 第二项 $2\sqrt{\frac{1}{m} \log \frac{2k}{\delta}}$：**估计误差**——由有限样本引入的额外误差
  - $m$：训练样本数，越大误差越小
  - $k$：假设集大小，候选假设越多需要越多样本
  - $\delta$：置信度参数（失败概率）
- 整个不等式说明：**总误差 = 近似误差 + 估计误差**，估计误差以 $O(1/\sqrt{m})$ 的速率随数据量下降

**关键公式 2：非参数密度估计的收敛速率（Hall 1989）**

在定理 1.5 的假设下，密度估计器 $\hat{p}_n(x_0)$ 的收敛速率为 $\psi_n = n^{-\frac{\beta}{2\beta+1}}$，即：

$$\sup_{p \in \mathcal{P}(\beta, L)} \mathbb{E}_p\left[(\hat{p}_n(x_0) - p(x_0))^2\right] \leq C \psi_n^2$$

- $\beta$：密度函数的光滑度阶数（越光滑收敛越快）
- 这个结果针对**生成式建模（smooth densities）**
- 指数 $-\frac{\beta}{2\beta+1}$ 是理论最优的收敛速率

**底部提示：** 这些都是**上界（upper bounds）**，不是实际观察到的误差值。理论界往往是保守的。

---

## PPT 2 — Part 1. Scaling Laws: History and Background（第一部分：历史与背景）

![](slide02.png)

### 核心内容

本页是第一部分的章节标题页，列出两个核心主题：

1. **Data scaling as empirical sample complexities**  
   将**数据规模扩展**理解为**经验版的样本复杂度理论**——即实际观察到的数据量与性能的关系曲线

2. **Initial forays into understanding neural scaling with data**  
   早期研究者尝试将**神经网络的规模**与**数据量**的关系规律化的探索

这一部分的逻辑脉络：先从统计学理论出发，再过渡到神经网络时代的经验观察。

---

## PPT 3 — Earliest (Data) Scaling Law Paper – 1993

![](slide03.png)

### 核心内容

**论文：Learning Curves: Asymptotic Values and Rate of Convergence**  
作者：Corinna Cortes, L.D. Jackel, Sara A. Solla, Vladimir Vapnik, John S. Denker（AT&T Bell Labs）

这是**已知最早的数据规模扩展定律论文**，早在 1993 年，来自贝尔实验室的团队（包括 Vapnik，SVM 的发明者）就在研究这个问题。

### 图表解读

PPT 右侧展示了一张**学习曲线图**：
- **X 轴**：训练集大小（training set size, $i$），从 2560 到 15360
- **Y 轴**：误差（error）
- **实线（test error）**：测试误差，始终高于训练误差
- **虚线（training error）**：训练误差，随样本增加单调下降
- 两条曲线最终趋向同一个渐近值（asymptotic value）

**关键数学模型：**

$$\mathcal{E}_{test} = a + \frac{c}{i^{\alpha}} \quad \text{and} \quad \mathcal{E}_{train} = a - \frac{c}{i^{\beta}}$$

这是最早对学习曲线进行幂律（power-law）建模的工作，说明误差以数据量的幂次方下降。

### 为什么训练误差上升、测试误差下降？

这是图中最直观但也最容易误解的现象，公式直接给出了答案：

$$\mathcal{E}_{test} = a + \frac{c}{i^{\alpha}} \xrightarrow{i \to \infty} a \quad \text{（从上方收敛）}$$

$$\mathcal{E}_{train} = a - \frac{c}{i^{\beta}} \xrightarrow{i \to \infty} a \quad \text{（从下方收敛）}$$

两条曲线从两侧趋向同一渐近值 $a$（不可减误差 / 贝叶斯误差）。

**训练误差为何随数据量上升：**
- 样本极少时，模型参数数量远超样本量，可以**完美记住**每个训练点 → 训练误差 ≈ 0
- 样本增多后，各样本之间存在噪声与矛盾，模型无法再记住所有点，只能学习规律 → 训练误差被迫上升，趋向 $a$

**测试误差为何随数据量下降：**
- 样本极少时，模型记住了训练集的噪声，**过拟合**严重，在测试集上表现差 → 测试误差很高
- 样本增多后，模型被迫学习真实的数据分布而非噪声，**泛化能力提升** → 测试误差下降，趋向 $a$

| 数据量 | 训练误差 | 测试误差 | 本质状态 |
|--------|---------|---------|---------|
| 极少 | ≈ 0（记忆） | 很高（过拟合） | High variance |
| 中等 | 上升中 | 下降中 | 泛化能力增强 |
| 极多 | ≈ $a$ | ≈ $a$ | 两者收敛，差距消失 |

**一句话：** 少量数据时模型处于过拟合状态，训练误差和测试误差之间存在巨大的"泛化差距"；随着数据增多，这个差距从两侧向不可减误差 $a$ 收拢直至消失。

### 历史意义

这篇论文表明，**数据规模扩展定律并非 2020 年 GPT 时代的新发明**，而是有着 30 年历史的理论传统。可以追溯到 Bell Labs 和 Corinna Cortes。

---

## PPT 4 — Early History of Scaling Laws: Data Scaling（Banko & Brill 2001）

![](slide04.png)

### 核心内容

**论文引用：Banko and Brill '01**  
标题：Log-linear scaling with data  
任务：Confusion Set Disambiguation（混淆词消歧义，NLP 任务）

### 图表解读

**Figure 1 — Learning Curves for Confusion Set Disambiguation**

- **X 轴**：训练数据规模（Millions of Words），从 0.1 到 1000（对数刻度）
- **Y 轴**：测试准确率（Test Accuracy）
- **多条曲线**代表不同的模型：Memory-Based、Winnow、Perceptron、Naive Bayes
- 关键观察：**所有模型在数据足够多时准确率仍在上升，没有饱和**

**右侧文字框的核心结论：**

> "these results suggest that we may want to reconsider the trade-off between spending time and money on algorithm development versus spending it on corpus development. At least for the problem of confusable disambiguation, none of the learners tested is close to asymptoting in performance at the training corpus size commonly employed by the field."

**中文翻译：** 结果提示我们应重新考虑在算法开发与语料库建设之间的投入取舍。至少对于混淆词消歧义任务，现有的**训练数据规模**远未达到**任何模型的性能瓶颈**。

### 深层含义

这篇 2001 年的论文得出了一个在当时颇具革命性的结论：**更多数据 > 更好算法**。这与后来 OpenAI Scaling Laws 的哲学一脉相承。

---

## PPT 5 — Early History of Scaling Laws: Functional Forms（Kolachina 2012）

![](slide05.png)

### 核心内容

**论文：Kolachina et al. 2012**  
任务：机器翻译（Machine Translation），使用 BLEU 分数评估  
核心探索：**数据量与性能关系的函数形式**

### 图表解读

**左图：拟合曲线（Curve Fit using Recursive Least Squares）**

- X 轴：训练样本数（Training sample size），0 到 1,200,000
- Y 轴：BLEU 分数（0.14 到 0.22）
- 图中展示了**多条候选拟合曲线**，以及实际观测值（x 标记）

**右表：Table 1 — Curve Families（候选函数族）**

| 模型 | 公式 |
|------|------|
| Exp₃ | $y = c - e^{-ax+b}$ |
| Exp₄ | $y = c - e^{-ax^{\alpha}+b}$ |
| ExpP₃ | $y = c - e^{(x-b)^{\alpha}}$ |
| Pow₃ | $y = c - ax^{-\alpha}$ |
| Pow₄ | $y = c - (-ax+b)^{-\alpha}$ |
| ILog₂ | $y = c - (a/\log x)$ |

**关键结论：**
- **幂律函数（Power Law, Pow₃/Pow₄）** 拟合效果最好
- 这正是后来所有 Scaling Laws 论文的标准假设

**底部结论：**  
"Early tests of functional forms — Kolachina et al 2012 — power law relation between data and downstream performance"

---

## PPT 6 — Hestness et al. 2017（首个大规模神经网络 Scaling 研究）

![](slide06.png)

### 核心内容

**论文：Hestness et al. 2017**  
这是**最早的大规模神经网络 Scaling 研究**，在多个任务（机器翻译 MT、语言模型 LM、语音识别 Speech）上系统研究了数据量与性能的关系。

### 图表解读

**左图：各独立模型的学习曲线**
- 对每个任务分别拟合，得到 $\varepsilon_{opt}(n) = \alpha m^{\beta} + \gamma$ 的参数
- 两条曲线：$\varepsilon_{opt}(n) = 41.2 m^{-0.38}$（某任务）和 $\varepsilon_{opt}(n) = 21.5 m^{-0.32}$（另一任务）

**中图：最优模型的复合学习曲线**
- 综合多个任务：$\varepsilon(n) = 3.67 m^{-0.68}$
- 说明随数据规模增长，**选择最优模型**后的性能提升速率

**右图：学习曲线的三个区域（Generalization Error vs Training Data Set Size，双对数坐标）**

这个图极为重要，总结了数据规模扩展的三段式结构：

1. **Small Data Region（小数据区）**：误差在"Best Guess Error"附近，高于渐近值，模型不足以学到规律
2. **Power-law Region（幂律区）**：误差随数据量以幂律下降，呈 log-log 线性——这是 Scaling Laws 发挥作用的核心区间
3. **Irreducible Error Region（不可减误差区）**：误差趋近于"Predictable Error"红线，继续增加数据不再有益，瓶颈在于模型容量或数据噪声

**核心贡献：**
- 提出了预测性的 Scaling 形状（Hypothesized Scaling Shape）
- 在多个任务上验证了幂律的普适性

---

## PPT 7 — Hestness II：超前于时代的洞察

![](slide07.png)

### 核心内容

本页展示 Hestness et al. 2017 论文中几个至今仍有争议的重要概念，这些概念在论文发表时远超时代。

### 三个关键概念框

**1. "Emergence"（涌现）**

> "Although small data set testing may be possible, it can be difficult to ensure that training data is large enough to see the power-law learning curve region. We have found that models with poor optimizer parameterization or model priors/initialization show accuracy cliffs, where accuracy is only as good as best guessing, but the model trains on enough data to be in the power-law region."

**中文解释：** 当模型在小数据区时，性能可能突然从"随机猜测"跳跃到有意义的学习——这就是"涌现（emergence）"现象的早期描述。后来 GPT-4 时代的"能力涌现"讨论正是这个现象的延伸。

**2. Scaling by Compute（算力规模扩展）**

> "The compute requirements to reach a particular accuracy level. The compute requirements could inform decisions about how to scale computational capacity to unlock these compute-limited applications."

**中文解释：** 如果找到了理想的 Scaling 模型，就可以根据算力来预测性能，从而指导算力投入决策。这是后来 Chinchilla 和 GPT-4 训练预算分配的理论基础。

**3. Speed = Accuracy（速度-精度权衡）**

> "The Performance-Accuracy Trade-off: Many DL software and hardware techniques impose a trade-off between model accuracy and the speed of computation... For example, low-precision computation/quantization and sparse models give up some model accuracy (e.g., up to 20%) in order to improve compute throughput."

**中文解释：** 量化（quantization）、稀疏化（sparsification）等工程技术通过**牺牲少量精度**换取**大幅提速**，而 Scaling Laws 可以帮助量化这个权衡是否值得。

---

## PPT 8 — Part 2. Neural (LLM) Scaling Behaviors（第二部分：LLM 的规模扩展行为）

![](slide08.png)

### 核心内容

第二部分章节标题页，提出三个核心问题：

**1. Data vs Performance（数据量 vs 性能）**
> "Are there simple rules that determine how data affects performance?"

这是最基础的问题：给定更多数据，性能如何定量提升？

**2. Data vs Model Size（数据量 vs 模型大小）**
> "Do we train on more data or bigger models?"

这是工程实践中最关键的权衡：在固定计算预算下，该优先扩大数据集还是模型规模？

**3. Hyper-parameters vs Performance（超参数 vs 性能）**
> "How should we set hyperparameters on the big model?"

大模型的超参数（学习率、批量大小、架构细节）如何随规模变化而调整？

---

## PPT 9 — Scaling Laws: Power Law Relationships for Many Factors

![](slide09.png)

### 核心内容

**引用：Kaplan et al. 2020（OpenAI Scaling Laws 经典论文）**

### 核心主张

Scaling Laws 在**许多不同现象**上都成立：

**上方三图（来自 Kaplan 2020）：**

$$L = (C_{min}/2.3 \times 10^8)^{-0.048}$$

（Compute 维度）

$$L = (D/5.4 \times 10^{13})^{-0.095}$$

（Dataset Size 维度）

$$L = (N/8.8 \times 10^{13})^{-0.076}$$

（Parameters 维度）

**图表结构：**
- 三张双对数（log-log）图，分别展示：
  - **Test Loss vs Compute**（PF-days，非 embedding 参数）
  - **Test Loss vs Dataset Size**（tokens）
  - **Test Loss vs Parameters**（非 embedding）
- 所有图均呈现出完美的线性关系——证明幂律成立

**下方三图（来自 Kaplan+ 2020 的非标准任务）：**
- Word Unscrambling（词语重排）
- Persian QA（波斯语问答）
- Arithmetic question-solving（算术求解）
- 结论：**即使在非标准任务上，Scaling Laws 仍然成立**

**核心信息：** 幂律规律的普适性是 Scaling Laws 最令人惊讶的特性之一。

---

## PPT 10 — Data vs Performance（数据量与性能的关系）

![](slide10.png)

### 核心内容

**定义：什么是 Data Scaling Law？**

> **Data scaling laws**: simple formula that maps dataset size (n) to error

即：一个从数据集大小到误差的简单映射函数。

**预期形状：**

引用 Hestness+ 2017 的三区域图（同 PPT 6 右图）：
- Small Data Region
- Power-law Region（核心关注区）
- Irreducible Error Region

**关键描述：** "Monotonic, logistic-like curves"

学习曲线是**单调的、类 Logistic 形状的曲线**：
- 初始快速下降（幂律主导）
- 最终趋于水平（不可减误差）

这种形状在双对数坐标下的中间段呈**直线**，就是 Scaling Laws 的核心表现。

---

## PPT 11 — Data Scaling Laws for Language Models（语言模型的数据 Scaling 定律）

![](slide11.png)

### 核心内容

**经验观察：Loss 与数据集大小在 log-log 图上是线性关系**

### 图表解读

- **X 轴**：Dataset Size（tokens），从 $10^8$ 到 $10^9$（对数刻度）
- **Y 轴**：Test Loss，从 2.7 到 4.2
- **拟合直线**：$L = (D/5.4 \times 10^{13})^{-0.095}$
- **结论**：log-log 线性 → **幂律关系**

**数学含义：**

$$L = C \cdot D^{-0.095}$$

其中 $C = (5.4 \times 10^{13})^{0.095}$ 是常数。

**术语解释：**
- "Scale-free" 或 "Power law"：无尺度规律，即放大或缩小数据规模，规律的斜率保持不变

**数据来源：** Kaplan+ 2020，语言建模任务

---

## PPT 12 — Conceptual Foundations of Data Scaling Laws（数据 Scaling 定律的理论基础）

![](slide12.png)

### 核心内容

**核心问题：为什么 Scaling Laws 是幂律？**

提问框：
> Q: Why do scaling laws show up?  
> We know error should be monotone. But why is it a power law / linear in log-log?

**答案（框内）：**
> A (?): Estimation error naturally decays polynomially.

**估计误差天然以多项式速率衰减**——这是幂律出现的根本原因。

右侧附有三区域图（Small Data / Power-law / Irreducible Error）辅助直觉。

**引导示例：**
> Example: If our task is to estimate the mean of a dataset, what's the scaling law?

这个例子将在下一张 PPT 详细推导，是理解 Scaling Laws 理论基础的关键桥梁。

---

## PPT 13 — Toy Example: Mean Estimation（玩具例子：均值估计）

![](slide13.png)

### 核心内容

这是整节课**最重要的推导**，用一个最简单的例子说明为什么 Scaling Laws 必然是幂律。

### 完整推导

**设定：**

$$\text{Input: } x_1 \ldots x_n \sim \mathcal{N}(\mu, \sigma^2)$$
$$\text{Task: 用 } \hat{\mu} = \frac{\sum_i x_i}{n} \text{ 估计均值}$$

**误差计算：** By standard arguments（标准统计论证）

$$\mathbb{E}[(\hat{\mu} - \mu)^2] = \frac{\sigma^2}{n}$$

**这就是一个 Scaling Law！**

取对数：

$$\log(\text{Error}) = -\log n + 2\log \sigma$$

在 log-log 坐标下，这是一条**斜率为 $-1$、截距为 $2\log\sigma$ 的直线**。

**一般化：**
> More generally, any polynomial rate $1/n^\alpha$ is a scaling law

只要误差以 $n^\alpha$ 的多项式速率下降，在 log-log 坐标下就会呈现直线，即 Scaling Law。

### 深层含义

这个例子揭示了 Scaling Laws 的统计本质：
- **样本均值的 MSE 是 $\sigma^2/n$**，这是一个幂律（指数 $\alpha=1$）
- 神经网络学习本质上也是某种"估计"问题
- 因此，**幂律是统计估计的内在属性**，而不是神经网络的特殊性质

---

## PPT 14 — Scaling Law Exponents: An Intriguing Mystery（指数之谜）

![](slide14.png)

### 核心内容

**已知事实：** 类似的论证表明大多数"经典"模型（回归等）都有 $\frac{1}{n}$ 的缩放规律

这意味着 log-log 图上斜率应为：

$$y = -x + C$$

**实际观察到的是什么？**

### 三张图的对比

**左图：Machine Translation（机器翻译）**
- 拟合：$\varepsilon(n) = 3.87 m^{-0.38}$
- 斜率 ≈ $-0.38$（远小于理论预测的 $-1$）

**中图：Speech（语音识别）**
- 拟合：$\varepsilon(n) = 2.26 m^{-0.40}$
- 斜率 ≈ $-0.40$

**右图：Language Modeling（语言建模）**
- $L = (D/5.4 \times 10^{13})^{-0.095}$
- 斜率 ≈ $-0.095$（极小！）

**核心困惑：**

> Very different from predictions. Why might this be?

理论预测斜率为 $-1$，实际观测斜率在 $-0.1$ 到 $-0.4$ 之间。这个差异意味着：**神经网络的数据利用效率远低于理论最优的经典估计量**。这是 Scaling Laws 领域一个尚未完全解决的理论谜题。

---

## PPT 15 — Detour: Scaling Laws for Nonparametric Learning（非参数学习的 Scaling 定律）

![](slide15.png)

### 核心内容

**神经网络是非参数学习器**（可以逼近任意函数），让我们用非参数视角来理解 Scaling 指数。

### 疑问：神经网络不是有参数（weights）吗，为何叫"非参数"？

这里"参数"一词在统计学和深度学习中含义不同：

| | 参数模型（Parametric） | 非参数模型（Nonparametric） |
|--|--|--|
| 统计学定义 | 参数个数固定，不随数据量变化 | 模型复杂度可随数据量增长 |
| 典型例子 | 线性回归（永远 $d+1$ 个权重） | k-NN、核方法（存储所有训练点） |
| 对函数的假设 | 限定在某个固定的函数族（如线性） | 对函数形式几乎不做假设 |

神经网络虽然有权重参数（weights），但在统计学意义上被视为**非参数模型**，原因有两点：

1. **万能近似（Universal Approximation）**：足够大的神经网络可以逼近任意光滑函数，不限于线性、多项式等固定函数族
2. **实践中复杂度随数据增长**：数据越多，我们往往用越大的网络，模型复杂度并非固定

因此"非参数"描述的是**函数类的灵活性**，不是"没有权重"。正是这种灵活性，使得神经网络的 Scaling 行为服从非参数估计理论，而非参数估计（线性回归等）的 $n^{-1}$ 规律。

### 推导过程

**设定：**

$$\text{Input: } x_1 \ldots x_n \text{ uniform in 2D unit box}, \quad y_i = f(x_i) + \mathcal{N}(0,1)$$
$$\text{Task: 估计函数 } f(x)$$
$$\text{Approach: 将 2D 空间切成边长为 } n^{-1/4} \text{ 的小格子}$$

**误差估计：**

- 切分后有 $\sqrt{n}$ 个格子（因为 $n^{-1/4}$ 边长 → $(n^{1/4})^2 = \sqrt{n}$ 个格子）
- 每个格子平均得到 $\sqrt{n}$ 个样本
- 在每个格子内用均值估计，误差 $\approx \frac{1}{\sqrt{\sqrt{n}}} = n^{-1/4}$

$$\text{Error} \approx \frac{1}{\sqrt{n}} + \text{(other smoothness terms)}$$

**推广到 $d$ 维：**

$$\text{Error} = n^{-1/d} \implies \text{scaling law: } y = -\frac{1}{d}x + C$$

### 核心 Takeaway

> Flexible 'nonparametric' learning has **dimension dependent** scaling laws.

**维度越高，Scaling 指数越小（绝对值越小），性能随数据量改善越慢。**

这就解释了为什么语言模型（极高维任务）的 Scaling 指数（$-0.095$）远小于理论最优值（$-1$）：**自然语言的有效维度极高**。

---

## PPT 16 — Intrinsic Dimensionality Theory of Data Scaling Laws（内在维度理论）

![](slide16.png)

### 核心内容

**Bahri 2021 的论断：**

1. Scaling Laws 源于 $\frac{1}{n^\alpha}$ 的多项式学习速率
2. 斜率 $\alpha$ 与数据的**内在维度（intrinsic dimensionality）**密切相关

### 图表解读

**散点图：**
- **X 轴**：Dimension（数据集的估计内在维度），从 2 到 26
- **Y 轴**：$4/\alpha_0$（缩放因子，越大说明收敛越慢）
- **数据点来源**：Teacher-Student、CIFAR-10、CIFAR-100、SVHN、FashionMNIST、MNIST
- **两条参考线**：$4/\alpha_0$（实线）和 $2/\alpha_0$（虚线）
- **观察**：大体上，**维度越高，$4/\alpha_0$ 越大，即 Scaling 指数越小**——与非参数理论预测一致

### 重要警告

**底部注释：**
> But estimators of intrinsic dimension are sketchy, and this is not airtight..

**中文解释：** 内在维度的估计方法本身并不可靠，这个理论框架目前仍不够严密。这说明 Scaling Laws 的理论基础仍是开放性研究问题。

---

## PPT 17 — Other Data Scaling Laws（其他数据 Scaling 定律）

![](slide17.png)

### 核心内容

**前文回顾：** Data Scaling Laws 研究的是**数据集大小**如何影响性能

**新问题：数据集的组成（composition）如何影响性能？**

三个子问题：

1. **Picking optimal data mixture using small scale models**  
   用小规模模型实验来确定最优的数据混合比例，然后迁移到大模型——这是数据配比（data mixture）研究的核心

2. **Deciding whether to repeat data or not**  
   有限数据情况下，是否值得多次训练同一批数据（data repetition / epoch 数的选择）？

3. **Combining the two and balancing quality with repetition rate**  
   如何同时优化数据质量与重复训练次数之间的平衡？

---

## PPT 18 — Other Advanced Data Scaling Law: Distribution Shift（分布偏移下的 Scaling）

![](slide18.png)

### 核心内容

**引用：Hashimoto 2021**

**核心结论：**
> Data composition affects the **offset**, not the slope.

数据组成影响的是 log-log 图上的**截距**，而不是**斜率**。

**实际意义：** 收集更多样化的数据可以降低误差的基线（offset），但不能改变 Scaling 的速率（slope）。

### 图表解读

**左图：Excess Error vs Training Data Size**

- X 轴：训练数据量（$10^2$ 到 $10^3$，对数刻度）
- Y 轴：Excess error（$10^{-2}$ 到 $10^0$，对数刻度）
- 三条曲线对应不同的 $q$ 值（数据源比例）：$q=0.00$（蓝）、$q=0.22$（绿）、$q=0.56$（黄）
- **所有曲线斜率相同，但截距不同**——验证了"组成影响 offset"的结论

**右图：Expected Error Intercept vs Data Source Proportion**

- X 轴：Data source proportion（0 到 1.0）
- Y 轴：$\log C(q)$（截距的对数）
- 呈现 U 型曲线，说明存在**最优数据混合比例**（中间某处误差截距最小）
- 极端情况（只用单一来源：$q=0$ 或 $q=1$）都不是最优

**结论：**
> These 'distribution shift' scaling laws can tell us about the importance of collecting diverse data!

多样化的数据来源对于降低整体误差至关重要。

---

## PPT 19 — In Practice: Data Mixture Selection via Scaling is Hard（实践中数据配比很难）

![](slide19.png)

### 核心内容

本页介绍两篇试图系统化数据配比选择的论文，并指出其困难所在。

### 论文 1：Data Mixing Laws（数据混合定律）

**思路：** 用小规模实验预测大规模训练的最优数据混合

**三步流程（左图）：**
1. Small Steps, Small Models, Seen Mixture → 训练多个小模型，扫描数据比例
2. Large Steps, Small Models, Seen Mixture → 训练更久的小模型
3. Large Steps, Large Models, Unseen Mixture → **外推**到大模型和未见过的数据配比

**核心挑战：**
- 小模型的最优配比不一定迁移到大模型
- "Unseen Mixture" 意味着需要**泛化**而非内插

### 论文 2：DataDecide — How to Predict Best Pretraining Data with Small Experiments

**思路：** 纯经验方法——直接用小实验评估哪个数据集最好，然后外推

**图示（右侧）：**
- 用小规模（$10^{18}$ 到 $10^{19}$ FLOPs）实验预测目标规模（$10^{22}$ FLOPs）的最优数据集
- 成功率约 75-80%（约 23/25 个实验预测正确）

**底部对比：**
- **Natural idea**：建立数据 Scaling Laws（理论驱动）
- **Empirical eval**：直接取最优小数据集（经验驱动）

---

## PPT 20 — Scaling Laws Under Data Repetition（数据重复下的 Scaling 定律）

![](slide20.png)

### 核心内容

**实际问题：** 我们的数据是有限的——重复训练相同的样本会如何影响 Scaling？

### 图表解读

**左图：Return on Compute When Repeating**

- X 轴：Tokens（训练 token 量），标注了 Epochs（轮次）
- Y 轴：Final test loss（2.0 到 3.4）
- **关键观察：**
  - 重复 1-4 轮（epoch）与同计算量新数据表现**几乎相同**（"Up to 4 epochs, repeating is almost as good as new data"）
  - 重复轮次过多时，性能明显下降（"Rapidly diminishing returns for more repetitions"）

**右图：Allocating Compute When Repeating**

- X 轴：Tokens（显示 Epochs 数）
- Y 轴：Parameters（模型参数量）
- 颜色深浅表示 FLOPs 等级（$10^{21}$ FLOPs）
- **最优轨迹（Efficient Frontier）**：两条曲线对应不同的 Loss 目标（2.376 vs 2.359）

### 核心公式（底部）

来自论文 *Scaling Data-Constrained Language Models*：

$$D' = U_D + U_D R_D^* \left(1 - e^{-R_D/R_D^*}\right)$$

**变量解释：**

| 符号 | 含义 |
|------|------|
| $D'$ | 有效数据量（等效不重复数据量） |
| $U_D$ | 唯一 token 数（Unique tokens） |
| $R_D$ | 实际重复次数 |
| $R_D^*$ | 最优重复次数（约为 4） |

**公式行为：**
- $R_D \ll R_D^*$ 时：$D' \approx U_D(1 + R_D)$，近似线性增长
- $R_D \gg R_D^*$ 时：$D' \to 2U_D$，有效数据饱和
- 说明**数据重复的边际收益递减**，可用此公式精确量化

---

## PPT 21 — Scaling Laws in Compute Unbounded Settings（无限算力下的 Scaling 定律）

![](slide21.png)

### 核心内容

**背景：** 当算力不受限（Pre-training under infinite compute）时，Scaling Laws 有哪些注意事项？

**两条重要警告：**

1. **Scaling laws can 'break' if you blindly apply it**  
   盲目套用 Scaling Laws 可能会失效——这不是一个无限外推的工具

2. **Scaling laws are lower bounds so you can always potentially do better**  
   Scaling Laws 给出的是**下界**，实际表现可以比预测更好

### 图表解读（三张图）

**左图：Increasing Epoch Count**
- 不同学习率下，随 epoch 增加损失的变化
- 调参不当会破坏 Scaling 的规律性

**中图：Increasing Parameter Count**
- 固定 epoch 数下，模型越大未必越好
- 需要数据量与模型规模协同调整

**右图：Varying Seed Token Count**
- 比较 Standard scope、Regularized ensembles、Tunable ensembles
- **集成方法（ensemble）可以突破标准 Scaling Laws 预测的上界**

**引用：** Kiran Ravi\*, Sohee Rother\*, Orca Zang, Tanya Chattopadhyay（Stanford University）

---

## PPT 22 — Data Selection Scaling and Accounting for Finiteness（有限数据下的自适应数据选择）

![](slide22.png)

### 核心内容

**核心洞察：**
> Given that repeated data is less valuable...  
> Data selection should then be **adaptive to scale**!

重复数据的价值递减 → 数据选择策略应该随着算力规模动态调整

### 图表解读

**中间图：Quality-Quantity Tradeoff (QQT) for Data Filtering**

数据质量从 E（最高）到 A（最低），随算力规模增加，纳入的数据质量阈值逐步降低：

| 算力规模 | 最优数据池 |
|---------|-----------|
| 小算力 | Pool 1：仅 E 类（最高质量） |
| 中算力 | Pool 2：E + D + C |
| 大算力 | 全部：E + D + C + B + A |

**右图：Estimated Scaling Curves（ImageNet-1k）**

- X 轴：Millions of Total Training Samples Seen（$10^2$ 到 $10^3$）
- Y 轴：ImageNet-1k Estimated Error（0.60 到 0.95）
- 各曲线（Bucket E only / E+D / E+D+C / E+D+C+B+A）在不同规模下交叉
- **没有一种固定数据策略在所有规模下都最优**——数据选择必须 Scaling 感知（compute-aware）

**论文引用：**  
*Scaling Laws for Data Filtering — Data Curation cannot be Compute Agnostic*  
（CMU / Bosch Center for AI）

---

## PPT 23 — Recap: Data Scaling Laws（数据 Scaling 定律总结）

![](slide23.png)

### 核心内容

本页是数据 Scaling 部分的总结，四个核心要点：

1. **Remarkably linear relationship between log-data size and log-error**  
   log 数据量与 log 误差之间呈现惊人的线性关系——这是 Scaling Laws 最核心的经验规律

2. **Holds across domains and models**  
   跨领域（NLP、CV、语音）、跨模型架构（Transformer、LSTM、CNN）都成立

3. **Theory understanding: similar to generalization bounds; mean estimation example**  
   理论解释类似泛化界（generalization bounds）；均值估计例子（PPT 13）是最清晰的直觉来源

4. **Applications: data collection / curation**  
   实践应用：指导数据收集和数据筛选策略（何时收集更多数据 vs 筛选高质量数据）

**讲师评价（底部字幕）：**
> This is the part that I think is most straightforward to understand.

讲师认为数据 Scaling Laws 是整节课最直观的部分。

---

## PPT 24 — Scaling Laws for Model Engineering（模型工程的 Scaling 定律）

![](slide24.png)

### 核心内容

进入第三部分：如何用 Scaling Laws 指导**模型设计决策**。

**核心动机：**
> How can we efficiently design huge LMs?

**两类具体问题：**

**框 1：架构与优化器选择**
- LSTMs vs Transformers（哪种架构更 Scaling 友好？）
- Adam vs SGD（哪种优化器更高效？）

**框 2：资源分配策略**
- Train models longer vs train bigger models（训练更久 vs 训练更大模型？）
- Collect more data vs get more GPUs（收集数据 vs 采购计算资源？）

**答案：**
> Scaling laws provide a simple procedure to answer these.

Scaling Laws 提供了一个**量化的、数据驱动的答案**，而不是靠经验猜测。

**历史意义：** 这正是 Chinchilla（DeepMind 2022）论文的核心贡献——用 Scaling Laws 证明 GPT-3 等模型是严重"训练不足"的（数据太少，模型太大）。

---

## PPT 25 — Hyperparameter Questions（超参数问题）

![](slide25.png)

### 核心内容

**背景：** 在经典 Kaplan Scaling 论文框架下，考虑以下超参数的 Scaling 行为

四个核心超参数问题：

1. **Architecture（架构）**  
   什么样的架构设计在大规模下表现最优？Transformer 的哪些变体更符合 Scaling？

2. **Optimizer（优化器）**  
   随着模型规模扩大，Adam 的行为如何变化？学习率如何调整？

3. **Aspect Ratio / Depth（宽度与深度的比例）**  
   是宽而浅的模型（多头数、大 FFN）还是窄而深的模型在相同参数量下更高效？

4. **Batch Size（批量大小）**  
   批量大小对 Scaling 的影响是什么？存在最优批量大小吗？

**讲师说明：**
> We'll consider some of these choices in the context of the classic Kaplan scaling paper

这些问题将在 Kaplan 2020 论文的框架内展开分析——后续 PPT 将继续（截图未完成）。

---

---

## PPT 26 — Architecture: Transformers vs LSTMs（架构比较：Transformer vs LSTM）

![](slide26.png)

### 核心内容

**问题：** Transformer 比 LSTM 更好吗？

**暴力验证方法：** 花费数千万美元训练一个 LSTM 版的 GPT-3

**Scaling Law 方法：** 用小模型建立两者的 Scaling 曲线，然后外推预测

### 图表解读

**X 轴：** Parameters (non-embedding)，从 $10^5$ 到 $10^9$  
**Y 轴：** Test Loss（2.4 到 5.4）

图中有两组曲线：
- **红色（LSTMs）**：1 层、2 层、4 层，Scaling 曲线斜率较缓
- **蓝色（Transformers）**：斜率更陡，在相同参数量下 Loss 更低

**核心结论：**
- 在相同参数量下，Transformer 始终优于 LSTM
- 两条曲线的差距随规模增大而拉大
- 因此**无需训练大模型**即可确认 Transformer 更优——小模型的 Scaling 曲线已经说明问题

**引用：** Kaplan+ 2021

---

## PPT 27 — Many Architectures（多种架构的跨架构 Scaling）

![](slide27.png)

### 核心内容

**引用：** Tay et al. 2022（Cross-architecture scaling）

### 图表解读

**左图：** 散点图（Negative Log-Perplexity vs FLOPs）
- 每个点代表一种架构配置
- 点的颜色/形状区分不同架构族

**右侧 11 个小图：** 展示不同架构的 Scaling 曲线
- ALBERT、DCconv、Evolved、Funnel
- Transformer-GLU、LConv、MLP-Mixer、MoS Transformer
- Performer、Switch Transformer、Universal Transformer

**核心观察：**
- 几乎所有架构都表现出 Scaling 规律（log-log 线性）
- 这表明 Scaling Laws 是**跨架构普适的**，而不是 Transformer 专属
- 讲师引用："people kind of use scaling laws as like a really almost paradigmatic way"——Scaling Laws 已成为架构评估的**范式性工具**

---

## PPT 28 — Optimizer Choice: Adam vs SGD（优化器选择）

![](slide28.png)

### 核心内容

**问题：** Adam 和 SGD 哪个更好？

**引用：** Hestness+ 2017（注意：这是 2017 年，Transformer 之前的时代，使用的是 RHN — Recurrent Highway Networks）

### 图表解读

**X 轴：** Training Data Set Size, Number of Chars（字符数，对数刻度，$2^{19}$ 到 $2^{27}$）  
**Y 轴：** Minimum Validation Loss（对数刻度）

四条曲线：
- **蓝色实线（Depth-10 RHNs, SGD）**：$\varepsilon(m) = 5.37 m^{-0.094}$
- **橙色实线（Depth-10 RHNs, Adam）**：$\varepsilon(m) = 5.25 m^{-0.095}$
- 两条虚线为对应的趋势线

**关键结论：**
- Adam 和 SGD 的 Scaling **斜率几乎相同**（$-0.094$ vs $-0.095$）
- Adam 的**截距略低**（$5.25$ vs $5.37$），说明 Adam 在每个规模下都稍好
- 但两者的 Scaling 行为基本一致——**优化器选择不影响 Scaling 指数**

**方法论意义：** 可以用小模型的 Scaling 曲线比较优化器，而无需运行大规模实验。

---

## PPT 29 — Depth/Width: Number of Layers（深度与宽度：层数的影响）

![](slide29.png)

### 核心内容

**问题：** 深度还是宽度对性能影响更大？

### 图表解读

**X 轴：** Parameters (non-embedding)，从 $10^3$ 到 $10^9$  
**Y 轴：** Test Loss（2 到 7）

五条曲线：1 Layer、2 Layers、3 Layers、6 Layers、>6 Layers

**关键结论：**

1. **1 层 vs 2 层：差距巨大。** 1 层模型在所有规模下都显著差于 2 层，说明最小深度是必要的
2. **更多层的收益递减：** 当参数量低于 $10^7$ 时，3 层、6 层、>6 层的曲线几乎重合
3. **深度的收益集中在大参数量区间：** 只有在参数量 $> 10^7$ 时，更深的模型才表现出明显优势

**实践意义：** 不需要无限加深——超过一定深度后，增加宽度（参数量）比增加深度更高效。

---

## PPT 30 — Depth/Width: Aspect Ratio and Other Transformer Hypers（宽深比与其他超参数）

![](slide30.png)

### 核心内容

**问题：** 宽深比（aspect ratio）等超参数依赖于规模吗？

**Figure 5 的核心结论（来自 Kaplan+ 2020）：**

> Performance depends very mildly on model shape when the total number of non-embedding parameters $N$ is held fixed. The loss varies only a few percent over a wide range of shapes.

在参数量固定的条件下，模型形状（宽深比）对性能的影响**极小**。

### 三张子图解读

**左图（Feed-Forward Ratio, $d_{ff}/d_{model}$）：**
- X 轴：宽度比，覆盖 50M 到 1.3B 参数
- Loss 变化仅在几个百分点以内
- 宽泛的架构形状都可以达到相近性能

**中图（Aspect Ratio, $d_{model}/n_{head}$）：**
- 标注："A wide range of architectures achieve similar performance"
- 说明架构比例对性能不敏感

**右图（Attention Head Dimension）：**
- "22% additional compute compensates for 1% loss increase"
- 宽深比变化 40 倍，Loss 仅变化 3%

**关键术语：** "Scale-invariant hyperparameters"——这些超参数与规模无关，可以用小模型确定后直接迁移到大模型。

---

## PPT 31 — Not All Parameters Are Equal（参数并非生而平等）

![](slide31.png)

### 核心内容

**关键发现：** Embedding 层的参数与非 Embedding 层参数行为不同！

### 图表解读

**左图：** Parameters (with embedding) vs Test Loss
- 包含 Embedding 参数时，不同深度（0-6+ 层）的曲线**差距很大**，1 层模型明显更差

**右图：** Parameters (non-embedding) vs Test Loss
- 去掉 Embedding 参数后，曲线**收敛得更整齐**，各深度的差距减小

**核心结论：**
> Embedding layer parameters don't behave the same!

Embedding 层参数不遵循与 Transformer 层相同的 Scaling 规律。这解释了 Kaplan 论文中**排除 Embedding 参数**的原因。

**延伸话题：**
> Related: recent papers on scaling laws for mixtures of experts.

MoE（Mixture of Experts）模型中，"激活参数"与"总参数"的区分进一步复杂化了 Scaling 分析——这是当前研究热点。

---

## PPT 32 — Side Note: Value of Parameters in MoE（MoE 中参数的价值）

![](slide32.png)

### 核心内容

**背景：** 在 MoE 模型中，参数 Scaling 的规律会发生变化

**标题论文：** *Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models*  
作者：Apple、MIT 等机构

### 图表解读

**左上图（Optimal Total Parameters $N^*$）：**
- X 轴：Total Parameters，从 178M 到 25B
- Y 轴：Pre-training Loss
- 不同颜色曲线代表不同稀疏度（MoE Sparsity S）
- 观察：最优总参数量随稀疏度变化而改变

**左下图（Optimal Active Parameters $N_a^*$）：**
- 在不同稀疏度下，最优**激活**参数量不同
- 说明在 MoE 中需要同时优化总参数量和激活参数量

**右图（3D IsoFLOP surface）：**
- X 轴：Active Parameters $N_{active}$
- Y 轴：MoE Sparsity
- Z 轴：Loss
- 颜色映射 Loss 值
- 展示了在固定 FLOPs 下，最优（稀疏度，激活参数）组合形成一个曲面

**核心结论：** MoE 中参数 Scaling 更复杂，最优策略同时依赖总参数、激活参数和稀疏度三个维度。

---

## PPT 33 — Batch Size: Critical Batch Size（关键批量大小）

![](slide33.png)

### 核心内容

**背景：** Batch Size 已知在某个临界点后收益急剧递减

**定义：**
> **Critical batch** = min number of examples before diminishing returns

**两组图：**

**左图（等高线图）：**
- 横轴：Steps（训练步数）
- 纵轴：Examples（样本总数）
- 等高线代表不同的 Loss 水平（0.02 到 1.00）
- **红叉（×）**：小 batch 的路径（更多步数，更少总样本）
- **蓝叉（×）**：大 batch 的路径（更少步数，更多总样本）
- **最优区域**在中间：既不需要极多步数，也不需要极多样本

**右图（Predicted Training Speed）：**
- X 轴：$B / \mathcal{B}$（批量大小 / Noise Scale，即 Critical Batch Size 的倍数）
- Y 轴：$c_{opt}(B) / c_{max}$（归一化效率）
- 对数刻度，从 $10^{-2}$ 到 $10^2$
- **Perfect scaling 区**：$B \ll \mathcal{B}$ 时，速度与批量线性正比（每增加一倍 batch，速度加倍）
- **Ineffective scaling 区**：$B \gg \mathcal{B}$ 时，速度几乎不再随 batch 增大而提升

**引用：** *An Empirical Model of Large-Batch Training*（McCandlish, Kaplan, Amodei 等）

---

## PPT 34 — Critical Batch Size: Formal Definition（关键批量大小：正式定义）

![](slide34.png)

### 核心内容

**精确定义 Critical Batch Size 的步骤：**

1. 选定目标 Loss，分别记录：
   - $S$：达到目标所需的**训练步数**
   - $E$：达到目标所需的**样本总数**
2. 扫描不同批量大小，得到 $(S, E)$ 曲线
3. 该曲线满足以下关系：

$$\frac{S}{S_{min}} - 1 = \left(\frac{E}{E_{min}} - 1\right)^{-1}$$

4. 拟合 $S_{min}$ 和 $E_{min}$，定义：

$$\mathcal{B}_{crit} = \frac{E_{min}}{S_{min}}$$

### 物理含义

$\mathcal{B}_{crit}$ 平衡了步数效率和样本效率：
- 使用 $B = \mathcal{B}_{crit}$ 时，步数约为最优的 **2 倍**（即"大约在两侧各给出 2x 的步数/passes 最优点"）

**理论联系（括号注）：**
> This is claimed to be close to the ratio of the trace of the gradient covariance and squared norm of the gradient.

即 $\mathcal{B}_{crit} \approx \frac{\text{tr}(\text{Cov}(\nabla L))}{\|\mathbb{E}[\nabla L]\|^2}$，这是梯度噪声的一种度量。

---

## PPT 35 — Batch Size: Critical Batch Size vs Performance（关键批量大小与性能的关系）

![](slide35.png)

### 核心内容

**核心图表：** Critical Batch Size vs WebText2 Train Loss

**X 轴：** WebText2 Train Loss（从 $3 \times 10^0$ 到 $10^1$，对数刻度）  
**Y 轴：** Critical Batch Size（Tokens，从 $10^3$ 到 $10^6$）

两条曲线：
- **蓝色：** Empirical $\mathcal{B}_{crit}$，$N = 3M$ 参数
- **橙色：** Empirical $\mathcal{B}_{crit}$，$N = 85M$ 参数
- **虚线：** $\mathcal{B}_{crit} = 2.1 \times 10^8 \cdot L^{-4.8}$（拟合曲线）

**关键规律：**
> The smaller the loss target, the bigger the batch

$$C_{min}(C) \equiv \frac{C}{1 + B/\mathcal{B}_{crit}(L)} \quad \text{（minimum compute, at } B \ll \mathcal{B}_{crit}\text{）}$$

**解读：**
- Loss 越低（模型越好），Critical Batch Size **越大**
- 意味着：训练越接近收敛，可以用越大的 batch，因为梯度信号更稳定
- 反过来，早期训练时用小 batch 更高效

---

## PPT 36 — Learning Rates: μP and Scale-Aware LR（μP 与尺度感知学习率）

![](slide36.png)

### 核心内容

**问题：** 如果我们朴素地扩大模型规模，最优学习率（LR）也会随之变化——这让超参数迁移变得困难

### 图表解读

**左图（Standard Practice）：**
- X 轴：$\log_2$ Learning Rate（从 $-20$ 到 $-10$）
- Y 轴：Training Loss（3.5 到 7.0）
- 多条曲线代表不同宽度（Width：128 到 8192）
- **最优 LR 随宽度变化而漂移**（"optimum shifts"）

**右图（Our Work = μP）：**
- 采用 μP 参数化后，**所有宽度的最优 LR 相同**（"optimum stable"）
- 这意味着可以在小模型上调好 LR，然后直接迁移到大模型

**引用：** Yang et al. 2022（μP 论文）、Yao et al. 2024

### μP 参数化规则表

| 超参数 | 标准模型 $M$ | $M' = r$ 倍宽度 |
|--------|------------|----------------|
| AdamW LR（矩阵型） | $l$ | $l/r$ |
| AdamW LR（其他） | $l$ | $l$ |
| 初始化方差（矩阵型） | $\sigma$ | $\sigma/r$ |
| 初始化方差（其他） | $\sigma$ | $\sigma$ |
| Multiplier（output） | $\tau$ | $\tau/r$ |
| Multiplier（others） | $\tau$ | $\tau$ |

**核心要点：**
- μP 提供了一套**随宽度缩放的超参数规则**
- 实践中，**始终需要调整的超参数是 batch size**

---

## PPT 37 — Caution: Scaling Behaviors Differ Downstream（警告：下游任务的 Scaling 不可预测）

![](slide37.png)

### 核心内容

**已知规律（预训练）：** Scaling 可预测，主要由参数量决定

**警告（下游任务）：** Downstream Scaling 往往可预测性差得多

### 图表解读

**引用：** Tay et al. 2023

**左图（Negative Log-Perplexity vs Params）：**
- 在**预训练困惑度**维度，各模型（NL6-XXXL、XL、NL12-XXL、NL32-LG 等）随参数量单调提升
- 排序清晰：更大 → 更好

**右图（SuperGlue Accuracy vs Params）：**
- 在**下游 SuperGlue 任务**上，模型排序**完全不同**
- NL32-XL（参数量中等）反而超过 NL12-XXL（参数量更大）
- 曲线出现**交叉**，排序不再单调

**核心结论：**
- 预训练 Scaling 不能直接预测下游任务表现
- 架构差异（attention head 设计、FFN 变体等）在下游任务上影响更显著
- 这是 Scaling Laws **局限性**的重要案例

---

## PPT 38 — Surprising Takeaways: Scaling Law Design Procedure（惊人的结论：基于 Scaling Law 的设计流程）

![](slide38.png)

### 核心内容

**最惊人的结论：**

> The effect of hyperparameters on big LMs can be predicted *before* training!

大语言模型的超参数效果可以**在训练之前**预测出来！

可以预测的超参数包括：
- **Optimizer choice**（优化器选择）
- **Model depth**（模型深度）
- **Architecture choice**（架构选择）

### 标准设计流程

**The scaling law based design procedure:**

1. **Train a few smaller models**：训练若干个小规模基准模型
2. **Establish a scaling law**（e.g. ADAM vs SGD scaling law）：建立 Scaling 曲线
3. **Select optimal hyperparam based on the scaling law prediction**：根据外推选择最优超参数

**深层含义：** 这将超参数调优从"黑盒实验"变成了**有理论依据的预测问题**，极大降低了大模型研发成本。

---

## PPT 39 — One Important Use: Joint Data-Model Scaling Laws（联合数据-模型 Scaling 定律）

![](slide39.png)

### 核心内容

**关键问题：** 我们需要更多数据还是更大的模型？

**经验观察：** 小模型上有大量数据被浪费了（小模型无法充分利用海量数据）

**Joint data-model scaling laws** 描述两者如何共同决定性能

### 两个公式

**Rosenfeld+ 2020：**

$$\text{Error} = n^{-\alpha} + m^{-\beta} + C$$

**Kaplan+ 2020：**

$$\text{Error} = [m^{-\alpha} + n^{-1}]^{\beta}$$

两者形式不同，但都试图联合建模数据量 $n$ 和模型参数量 $m$ 对误差的影响。

### 图表解读

**右上图（Loss vs Model and Dataset Size）：**
- X 轴：Tokens in Dataset
- Y 轴：Loss（2.5 到 4.5）
- 多条曲线按参数量（Params）分组
- 可见：数据量增加在小模型上收益递减，在大模型上收益更持续

**右下图（3D error landscape）：**
- 展示 WikiText-103 error（cross entropy）关于模型参数和数据量的联合曲面
- 标注：(a) WikiText-103 error (cross entropy) landscape

---

## PPT 40 — Model-Data Joint Scaling Is Accurate（联合 Scaling 准确有效）

![](slide40.png)

### 核心内容

**引用：** Rosenfeld 的工作——在小数据、小模型上拟合 Scaling 指数，预测大规模情况

### 图表解读（三张图）

**(a) Illustration（示意图）：**
- 左侧散点矩阵：行 = model fraction（$\log_2(m/N)$），列 = data fraction（$\log_2(n/N)$）
- 绿色点 = 拟合（fit），红色点 = 外推（extrapolated）
- 验证哪些组合能被准确外推

**(b) Extrapolation on ImageNet（ImageNet 外推）：**
- X 轴：measured top-1 error（真实值）
- Y 轴：estimated top-1 error（预测值）
- 数据点分两组：fit 和 extrapolated（model fraction 1/16, data fraction 1/8）
- 精度：$\mu = 4.5\%$，$\sigma = 4.681\%$
- 预测误差在 5% 以内

**(c) Extrapolation on WikiText-103：**
- 精度：$\mu = 0.5\%$，$\sigma = 1.689\%$
- 语言建模任务的外推精度更高

**底部结论：**
> Trading off data size and model size: optimize $n^{-\alpha} + m^{-\beta} + C$ with your costs.

在固定计算预算下，最优化这个联合公式即可得到最优的数据量与模型规模分配。

---

## PPT 41 — Optimal Compute-Data Tradeoffs: Kaplan vs Chinchilla（Kaplan vs Chinchilla 的算力-数据权衡）

![](slide41.png)

### 核心内容

**Kaplan 的预测：**

$$N_{opt} = C^{0.73}, \quad D_{opt} = C^{0.27}$$

意味着：随算力增加，**模型规模**增长远快于**数据量**（token 数量随算力增加而减少）

**Chinchilla 的反驳：**
> Chinchilla [Hoffman et al] argue these fits are quite off.

### 图表解读

**X 轴：** FLOPs（$10^{17}$ 到 $10^{25}$）  
**Y 轴：** Parameters（$10M$ 到 $1T$）

四条曲线：
- **Approach 1、2、3**（Chinchilla 的三种拟合方法，后续 PPT 详述）
- **Kaplan et al. 2020**（虚线）

**标注的真实模型：**
- ☆ Chinchilla（70B）
- ☆ Gopher（280B）
- ☆ GPT-3（175B）
- ☆ Megatron-Turing NLG（530B）

**关键观察：**
- Kaplan 预测的曲线斜率更陡（模型增长快，数据增长慢）
- Chinchilla 三种方法给出的斜率约为 **0.5**（即 $N_{opt} \propto C^{0.5}$，$D_{opt} \propto C^{0.5}$）
- 意味着：**数据量应与模型参数量同步增长**

---

## PPT 42 — Chinchilla in Depth: 3 Methods（Chinchilla 深度解析：三种方法）

![](slide42.png)

### 核心内容

**Chinchilla 论文提出三种拟合方法，结论基本一致（除方法 3 外）：**

| Approach | $a$（$N_{opt} \propto C^a$，置信区间） | $b$（$D_{opt} \propto C^b$，置信区间） |
|----------|--------------------------------------|--------------------------------------|
| 1. Minimum over training curves | 0.50 (0.488, 0.502) | 0.50 (0.501, 0.512) |
| 2. IsoFLOP profiles | 0.49 (0.462, 0.534) | 0.51 (0.483, 0.529) |
| 3. Parametric modelling of the loss | 0.46 (0.454, 0.455) | 0.54 (0.542, 0.543) |
| **Kaplan et al. (2020)** | **0.73** | **0.27** |

**核心结论：**
- 方法 1 和 2 一致给出约 $N_{opt} \propto C^{0.5}$，$D_{opt} \propto C^{0.5}$
- 方法 3 结果与前两者有些偏差（后来发现方法 3 数据处理有问题）
- Kaplan 的 0.73/0.27 与 Chinchilla 的 ~0.5/0.5 差距悬殊，说明 **GPT-3 等大模型严重训练不足**

---

## PPT 43 — Method 1: Minimum Over Runs（方法一：最小化训练曲线包络）

![](slide43.png)

### 核心内容

**思路：** 类似 Kaplan 的 FLOP 图——对所有训练曲线取包络（最小值），该包络呈幂律

**Figure 2: Training curve envelope**

三张图：
- **左图（Training Curves）：** 展示 70M 到 10B 参数、各种 cosine cycle 长度的所有训练曲线（色谱从绿到黄代表参数量从小到大）
- **中图（Optimal Parameters vs FLOPs）：** 从左图提取每个 FLOP 预算下的最小 Loss 点，投影得到最优参数量曲线
- **右图（Optimal Tokens vs FLOPs）：** 对应的最优训练 token 数曲线

**绿色点：** Gopher 模型（$5.76 \times 10^{23}$ FLOPs）的投影位置

**核心逻辑：** 对每个 FLOP 预算，找到 Loss 最小的（参数量，token 数）组合，这些点构成的曲线服从幂律。

---

## PPT 44 — Method 2: IsoFLOPs（方法二：等算力曲线）

![](slide44.png)

### 核心内容

**思路：** 选定一组 FLOP 预算，在每个 FLOP 预算下扫描不同参数量，取最小 Loss 点

**Figure 3: IsoFLOP curves**

三张图：
- **左图（IsoFLOP曲线）：** 每条曲线对应一个固定 FLOP 预算，X 轴为参数量，Y 轴为 Loss。每条曲线呈 U 型（有最优参数量），U 型谷底即最优点
- **中图（Optimal Parameters vs FLOPs）：** 将每条曲线的谷底连接，得到 $N_{opt}$ vs FLOPs 的幂律关系
- **右图（Optimal Tokens vs FLOPs）：** 同理得到 $D_{opt}$ vs FLOPs 的幂律关系

**方法特点：**
- cosine cycle 设置为与目标 FLOP 一致（训练步数 = FLOP/参数量）
- 绿色点为 Gopher 的估计

---

## PPT 45 & 46 — Method 3: Joint Fits（方法三：联合参数化拟合）

![](slide45.png)

### 核心内容

**思路：** 在参数量-数据量网格上运行大量模型，用最小二乘法联合拟合 Loss 函数

**Figure 4: Parametric fit**

**左图（IsoLoss contours）：**
- X 轴：Training FLOPs
- Y 轴：Model size
- 颜色等高线：Loss 水平
- **蓝色曲线（Efficient Frontier）：** 在相同 Loss 下 FLOPs 最少的路径
- **绿色点（Empirical data）：** 实际观测的最优点
- **橙色线（IsoFLOP slice）：** 固定 FLOP 下的截面

**右图（IsoFLOP slices）：**
- X 轴：Model Size（100M 到 40B）
- Y 轴：Loss（2.0 到 5.0）
- 每条曲线为不同 FLOP 预算下的 Loss-参数量曲线
- Gopher 预算对应最优点为 40B 参数

**核心公式（被拟合的目标）：**

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

其中 $E$ 为不可减误差，$A/N^\alpha$ 为模型规模项，$B/D^\beta$ 为数据量项。

---

## PPT 47 — Why Such a Big Difference?（为什么 Kaplan vs Chinchilla 差这么大？）

![](slide47.png)

### 核心内容

**关键问题：** Kaplan 和 Chinchilla 都在拟合联合 Scaling 定律，为什么结果差这么远？

- **Kaplan 预测：** $N_{opt} \propto C^{0.73}$（模型主导）
- **Chinchilla 预测：** $N_{opt} \propto C^{0.5}$（数据与模型并重）

图表重新展示了 PPT 41 的四条曲线，标注了 Chinchilla、GPT-3、Gopher、Megatron-Turing NLG 的实际训练点。

**核心困惑：** 两者都在做合理的拟合，差距从何而来？

---

## PPT 48 — Explanation 1: Methodology Differences（解释一：方法论差异）

![](slide48.png)

### 核心内容

**引用：** *Resolving Discrepancies in Compute-Optimal Scaling of Language Models*（Wortsmann, Ilharco, Schmidt, Carmon 等）

**五步分析流程（图示）：**

(a) **Reproducing Kaplan** → 复现 Kaplan 的结果（$N(C) = 0.498 C^{0.64}$，cf. $2^{0.68} C^{0.68}$）

(b) **Counting last layer FLOPs** → 将最后一层的 FLOPs 计入（Kaplan 排除了这部分）→ 指数从 0.68 降至约 0.64

(c) **Correcting warmup** → 修正 Warmup 策略（Kaplan 在极小算力预算时 warmup 过高）→ 指数进一步调整

(d) **Cosine decay (no tuning)** → 使用不调 decay 的 cosine LR 调度 → 结果接近 Hoffmann（Chinchilla）

(e) **Optimizer tuning (no decay)** → 只调优化器不调 decay → 结果稳定

**三个具体原因：**
1. **Kaplan 将最后一层参数从计数中移除**（低估了有效参数量）
2. **极小算力时 warmup 过高**（导致小模型的 Loss 偏高，拟合偏向大模型）
3. **Decay 本身不是关键因素**，但 batch/LR 若正确调整则 decay 影响不大

---

## PPT 49 — Explanation 2: Non-embedding vs Total Params（解释二：参数定义的差异）

![](slide49.png)

### 核心内容

**引用：** *Reconciling Kaplan and Chinchilla Scaling Laws*（Tim Pearce, Microsoft Research；Jinyeop Song, MIT；reviewed on OpenReview）

**核心发现：** Kaplan 使用**非 Embedding 参数**，Chinchilla 使用**总参数**——这一差异加上轻微的非线性，足以解释两者的指数差距

**四步推导流程（图示）：**

1. **从 Chinchilla 的训练曲线出发** → $\text{Loss}(N_T, D) = \frac{492}{N_T^{0.37}} + \frac{2085}{D^{0.57}} + 1.82$（总参数 $N_T$）

2. **生成 Kaplan 研究中的模型尺寸范围**（1M 到 1B 非 Embedding 参数）

3. **用总参数 $N_T$ 找 Chinchilla 的 compute-optimal frontier** → 局部幂律指数 $\approx 0.51$，接近 Chinchilla 的 0.50

4. **用非 Embedding 参数 $N_e$ 重新拟合** → 局部幂律指数 $\approx 0.78$，接近 Kaplan 的 0.73

**结论：**
> Non-embedding vs total param choice + small nonlinearities

参数定义（是否包含 Embedding 参数）加上 Loss 函数的轻微非线性，共同导致了两个指数（0.73 vs 0.50）的巨大表观差距。

---

## PPT 50 — Fun Addendum: Errors in Chinchilla Method 3（附录：Chinchilla 方法 3 的数据错误）

![](slide50.png)

### 核心内容

**发现：** Chinchilla 论文中**方法 3 很可能存在数据错误**

**过程：** 有研究者做了"数据取证"（data forensics）——恢复原始数据，重新拟合，结果与方法 1 和 2 更一致

### 图表解读

**左图（Residuals）：**
- 比较 Hoffmann et al. 与修正后的拟合残差
- 修正后残差更小，分布更紧

**右图（Optimal tokens per parameters ratio）：**
- X 轴：Training compute（FLOP，对数刻度）
- Y 轴：Optimal tokens per parameters ratio
- 绿色曲线（Ours）：修正后的最优比例，随算力单调增加
- 橙色曲线（Hoffmann et al.）：原始 Chinchilla 方法 3 的结果，形状异常
- 黑点：Chinchilla 模型本身

**结论：**
- 方法 3 的原始数据处理有误
- 修正后，三种方法的结论重新趋于一致
- **引用：** Besiroglu et al. 2024

---

## PPT 51 — Chinchilla Summary（Chinchilla 总结）

![](slide51.png)

### 核心内容

**三种方法的最终汇总（同 PPT 42 的表格）：**

| Approach | $a$（$N_{opt} \propto C^a$） | $b$（$D_{opt} \propto C^b$） |
|----------|-------------------------------|-------------------------------|
| 1. Minimum over training curves | 0.50 | 0.50 |
| 2. IsoFLOP profiles | 0.49 | 0.51 |
| 3. Parametric modelling of the loss | 0.46 | 0.54 |
| **Kaplan et al. (2020)** | **0.73** | **0.27** |

**关键启示：**
> There should be a fixed ratio between the two.

即：在固定 FLOPs 下，最优的**参数量 ∶ Token 数**比例是固定的。

根据 Chinchilla 的结论，这个比例大约是 **20 tokens/param**（Chinchilla 70B 用了约 1.4T tokens）。

---

## PPT 52 — Important Note: Train-Optimal ≠ What You Want（训练最优 ≠ 你真正想要的）

![](slide52.png)

### 核心内容

**关键区分：**
> Chinchilla aims to tell you what gives the best model for fixed training compute.  
> But most of the compute in a real deployment is inference.  
> So we should 'over' train.

Chinchilla 最优化的是**训练效率**，但实际部署中主要成本是**推理**。因此应该**过度训练**（比 Chinchilla 建议使用更多数据训练更小的模型）。

### 实际案例数据

| 模型 | Tokens/Param |
|------|-------------|
| GPT-3 | 2 tokens/param |
| Chinchilla | 20 tokens/param |
| LLaMA 65B | 22 tokens/param |
| Llama 2 70B | 29 tokens/param |
| Mistral 7B | 110 tokens/param |
| Llama 3 70B | 215 tokens/param |

**趋势解读：**
- GPT-3 严重训练不足（2 tokens/param）
- Chinchilla 是当时的"训练最优"基准（20 tokens/param）
- Llama 系列模型持续过度训练（22 → 215 tokens/param）
- **原因：** 推理成本巨大，值得用更多训练数据换取更小但更强的模型

**结论：**
> The more usage we expect, the more it becomes worth it to pay the upfront cost.

预期使用量越大，越值得在训练时投入更多数据（前期成本），以换取更低的推理成本。

---

## PPT 53 — IsoFLOPs Everywhere（等算力曲线无处不在）

![](slide53.png)

### 核心内容

**IsoFLOP 方法的普适性：** 几乎所有类型的模型都可以用 IsoFLOP 分析

**Figure 5: IsoFLOP profiles**

**上方两图（Autoregressive models）：**
- 左图：non-embedding parameters vs NLL（Negative Log-Likelihood）
- 右图：non-embedding autoregressive parameters vs NLL
- 多条 IsoFLOP 曲线（$10^{18}$ 到 $10^{23}$ FLOPs）均呈清晰的 U 型

**右上图（Diffusion vs Autoregressive）：**
- 比较 Diffusion 模型和 Autoregressive 模型的 Scaling 曲线
- 引用：Gulrajani 2023
- 结论：两者都遵循 Scaling Laws

**下方两图（MoE models）：**
- (a) 总参数 $N$ vs MoE Sparsity $S$ vs Loss 的 IsoFLOP 曲面
- (b) 激活参数 $N_{active}$ vs MoE Sparsity $S$ vs Loss 的 IsoFLOP 曲面
- 引用：Abnar+ 2025

**核心信息：** IsoFLOP 方法适用于语言模型、扩散模型、MoE 模型等各类架构，是分析最优算力分配的通用工具。

---

## PPT 54 — Scaling Laws for Models and Compute（模型与算力的 Scaling 定律总结）

![](slide54.png)

### 核心内容

**Log-linearity extends to model parameters and compute!**

对数线性关系同样适用于模型参数量和计算量。

**两个核心用途：**

**框 1：用小模型确定训练配置**
- Pick optimizer（选择优化器）
- Pick architecture and model sizes（选择架构和模型规模）

**框 2：做出智能资源决策**
- Big models vs more data?（更大模型 vs 更多数据？）
- 通过 IsoFLOP 或 Joint Scaling 给出定量答案

**关键思路：** 先在小模型上建立 Scaling 曲线，再外推到目标规模——这是现代大模型研发的标准方法论。

---

## PPT 55 — Recap: Scaling Laws — Surprising and Useful!（总结）

![](slide55.png)

### 核心内容

三条核心结论：

1. **Data scaling**: understand how data affects models, clean theory  
   数据 Scaling：理解数据如何影响模型，有清晰的理论基础

2. **Model scaling**: dramatically reduce costs for training  
   模型 Scaling：大幅降低训练成本（通过最优分配算力）

3. **Scaling as prediction**: understand what problems can be 'brute forced'  
   Scaling 作为预测工具：判断哪些问题可以通过堆算力"暴力解决"

**讲师结语（字幕）：**
> Hopefully, you remember, I think, all the different components here. Data scaling...

---

## 总体知识脉络总结

```
Scaling Laws 完整知识脉络
│
├── Part 1: 数据 Scaling Laws
│   ├── 历史起源（1993-2017）
│   │   ├── 统计理论基础（Cortes 1993）：幂律学习曲线
│   │   ├── NLP 实证（Banko & Brill 2001）：数据 > 算法
│   │   ├── 函数形式探索（Kolachina 2012）：幂律最佳拟合
│   │   └── 神经网络 Scaling（Hestness 2017）：三区域结构
│   │
│   ├── LLM 时代（Kaplan 2020）
│   │   ├── 多维度幂律：Compute / Data / Parameters 各维度
│   │   ├── 理论解释
│   │   │   ├── 均值估计：Error ~ σ²/n（最简 Scaling Law）
│   │   │   └── 非参数：维度决定指数 α ≈ 1/d
│   │   └── 指数之谜（实测 0.095-0.40 vs 理论 1.0）
│   │
│   └── 高级专题
│       ├── 数据组成（composition affects offset, not slope）
│       ├── 分布偏移（Hashimoto 2021）
│       ├── 数据重复（D' = Ud + Ud·Rd*·(1-e^{-Rd/Rd*})，约4轮最优）
│       └── 自适应数据选择（QQT：小算力用高质量数据）
│
├── Part 2: 模型工程 Scaling
│   ├── 架构选择
│   │   ├── Transformer vs LSTM：小模型曲线外推即可判断
│   │   └── 跨架构普适（Tay 2022）：几乎所有架构都遵循 Scaling
│   │
│   ├── 优化器选择
│   │   └── Adam vs SGD：斜率相同，截距略异，可用小模型预测
│   │
│   ├── 深度/宽度
│   │   ├── 1层 vs 2层差距大；>6层收益递减
│   │   ├── 宽深比（aspect ratio）对性能影响极小
│   │   └── Embedding 参数 ≠ 非Embedding 参数（Scaling 行为不同）
│   │
│   ├── Batch Size
│   │   ├── Critical Batch Size: B_crit = E_min/S_min
│   │   └── Loss 越低 → B_crit 越大；早期训练用小batch
│   │
│   └── 学习率
│       └── μP：通过参数化使最优LR对宽度尺度不变
│
└── Part 3: 联合 Scaling（Kaplan vs Chinchilla）
    ├── 联合公式
    │   ├── Rosenfeld: Error = n^{-α} + m^{-β} + C
    │   └── Kaplan: Error = [m^{-α} + n^{-1}]^β
    │
    ├── Kaplan (2020): N_opt ∝ C^0.73, D_opt ∝ C^0.27（模型主导）
    ├── Chinchilla (2022): N_opt ∝ C^0.5, D_opt ∝ C^0.5（数据模型并重）
    │   ├── Method 1: 最小化训练曲线包络
    │   ├── Method 2: IsoFLOP U型曲线谷底
    │   └── Method 3: 联合参数化拟合（有数据错误，Besiroglu 2024修正）
    │
    ├── 差异根源
    │   ├── Explanation 1: 参数计数方式 + warmup 设置
    │   └── Explanation 2: 非Embedding参数 vs 总参数 + 非线性
    │
    └── 实践推论
        ├── "训练最优" ≠ "部署最优"：推理成本更大→应过度训练
        ├── Tokens/Param 趋势：GPT-3(2) → Chinchilla(20) → Llama3(215)
        └── IsoFLOP 方法普适：适用于LM、Diffusion、MoE等各类模型
```

**核心公式一览：**

| 场景 | 公式 | 含义 |
|------|------|------|
| 泛化界 | $\epsilon(\hat{h}) \leq \epsilon^* + 2\sqrt{\frac{\log 2k/\delta}{m}}$ | 有限假设集误差上界 |
| 均值估计 | $\mathbb{E}[(\hat{\mu}-\mu)^2] = \sigma^2/n$ | 最简单的 Scaling Law |
| 非参数（d维） | $\text{Error} = n^{-1/d}$ | 维度决定收敛速率 |
| 语言模型数据 | $L = (D/5.4\times10^{13})^{-0.095}$ | 实测数据 Scaling |
| 数据重复 | $D' = U_D + U_D R_D^*(1-e^{-R_D/R_D^*})$ | 有效数据量公式（约4轮最优） |
| Critical Batch | $\mathcal{B}_{crit} = E_{min}/S_{min}$ | 批量大小临界点 |
| Joint Scaling | $\text{Error} = n^{-\alpha} + m^{-\beta} + C$ | 数据-模型联合误差（Rosenfeld） |
| Chinchilla LM Loss | $L(N,D) = E + A/N^\alpha + B/D^\beta$ | 联合 Loss 建模 |
| Kaplan optimal | $N_{opt} = C^{0.73}$，$D_{opt} = C^{0.27}$ | 模型主导（已被修正） |
| Chinchilla optimal | $N_{opt} \propto C^{0.5}$，$D_{opt} \propto C^{0.5}$ | 数据模型并重 |
