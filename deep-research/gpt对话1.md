## webgpt

**WebGPT** 的核心方法是把开放域长问答建模成“模型操作文本浏览器”的任务：模型可以搜索、打开网页、滚动、查找、摘录引用，最后生成带引用的答案。训练上先收集人类在浏览器中回答问题的轨迹，用 **behavior cloning / imitation learning** 让 GPT-3 学会搜索和引用；再用人类对两个答案的偏好比较训练 **reward model**；最终推理时对同一问题采样多个候选答案，用 reward model 做 **best-of-n / rejection sampling** 选出最优答案。评测主要在 **ELI5** 上做人工 pairwise preference，比较 WebGPT、人类示范者和 Reddit 高赞答案；最佳 175B best-of-64 模型相对人类示范者被偏好 56%，相对 Reddit 高赞答案被偏好 69%；另外也在 TruthfulQA 上评估真实性。([arXiv](https://arxiv.org/abs/2112.09332?utm_source=chatgpt.com "WebGPT: Browser-assisted question-answering with human feedback"))

这篇是 **WebGPT: Browser-assisted question-answering with human feedback**。它的评估不是用 BLEU/ROUGE 这类自动文本相似度，而是核心依赖**人类偏好评测**，并辅以 TruthfulQA 的真实性评测。

主要评估方式如下：

### 1. 在 ELI5 上做人工两两偏好比较

论文主要评估 3 个 WebGPT 模型：**760M best-of-4、13B best-of-16、175B best-of-64**。这些模型都是先用 behavior cloning 训练，再用 reward model 做 rejection sampling；推理时 temperature=0.8，最多 100 次浏览动作。

在 **ELI5 test set** 上，他们做了两类对比：

第一类是 **WebGPT vs. 人类示范者**。也就是把模型答案和使用同一个浏览器环境写出来的人类答案放在一起，让标注员判断哪个更好。评价维度包括 overall usefulness、coherence、factual accuracy。最终 175B best-of-64 的答案相对人类示范者被偏好 **56%**。

第二类是 **WebGPT vs. ELI5 数据集原始参考答案**。ELI5 的参考答案是 Reddit 上最高赞答案。为了公平，作者去掉了 WebGPT 答案里的引用和 references，并请不熟悉详细标注规则的新标注员用更简化的标准评估。175B best-of-64 相对 ELI5 最高赞答案被偏好 **69%**。

论文里把 tie 按 **50% preference** 处理，而不是丢掉。图中的误差条是 **±1 standard error**。

### 2. 标注员具体怎么判断“哪个答案更好”

对比评测不是简单问“你喜欢哪个”，而是让标注员先读问题、读两个答案及其 references，再判断引用是否可靠、答案里的 claims 是否被支持，最后给 overall usefulness 的偏好评分。评分使用 5 点 Likert scale：A much better、A better、Equally good、B better、B much better。

他们特别强调不要求标注员独立做完整 research 来查事实，因为这太难且主观；而是判断答案中的 claim 是否有可靠 reference 支持，或是否属于 common knowledge。最终综合判断时，优先级大致是：是否有 unsupported information、是否回答了核心问题、是否提供了额外有用信息、是否连贯和引用是否出错、是否有无关内容。

### 3. 在 TruthfulQA 上评估真实性

除了 ELI5，论文还在 **TruthfulQA** 上评估。TruthfulQA 是一个故意包含误导性问题的数据集，用来测试模型是否会给出常见但错误的回答。指标是两个：**truthful percentage** 和 **truthful + informative percentage**。

对 GPT-3 baseline，他们使用 TruthfulQA 论文里的 automated metric；但对 WebGPT，因为它的答案分布和 GPT-3 baseline 不同，自动指标不适合，所以用人工评估。由于 TruthfulQA 是短回答数据集，他们把 WebGPT 的答案截断到 50 tokens，并去掉末尾不完整句子。

结果是所有 WebGPT 模型在 truthfulness 和 truthful+informative 上都超过 GPT-3 baseline；其中最好模型大约达到 **75% truthful**、**54% truthful and informative**，但仍低于人类水平。

### 4. 还做了训练方法消融

论文还比较了 **BC、RL、best-of-n rejection sampling** 的效果。结果显示 rejection sampling 带来的提升很明显：175B best-of-64 BC 相比普通 175B BC 被偏好 **68%**；RL 也有提升，但较小，175B RL 相比 175B BC 被偏好 **58%**。

### 一句话总结

这篇论文的评估核心是：**让模型带引用回答开放问题，然后用人工 pairwise preference 来比较答案质量**，主指标是人类更偏好哪个答案；同时用 TruthfulQA 检查真实性。它更看重“答案是否有用、连贯、事实是否被可靠引用支持”，而不是和标准答案的 n-gram 重合度。

## webglm
可以把上一段里的 **WebGLM** 补充成这样：

**WebGLM** 的核心方法是面向真实部署改进 WebGPT，目标是在较小模型和较低成本下实现类似的 web-enhanced QA。它不是让模型完整模拟浏览器操作，而是拆成更工程化的三部分：**LLM-augmented retriever、bootstrapped generator、human preference-aware scorer**。训练上，首先训练/增强检索器：用搜索引擎拿到网页候选，再用 LLM 生成或蒸馏出的相关性信号来训练细粒度 retriever，使它能从网页中选出更相关的段落；然后训练生成器：以 GLM-10B 为基础，用长问答数据进行监督微调，并通过 bootstrapping 扩充训练样本，让模型学会基于检索到的网页内容生成带引用的长答案；最后训练偏好打分器：收集人类对不同答案的偏好比较，用来学习哪些答案更符合人类偏好，并在推理时对候选答案进行排序或筛选。整体流程更像“检索器负责找证据，生成器负责写答案，打分器负责选好答案”的高效 RAG 系统，而 WebGPT 更像“会一步步浏览网页的 agent”。WebGLM 论文也明确说它围绕这三个模块实现效率、准确性和成本优势，并公开了 code、demo 和 data。([arXiv](https://arxiv.org/abs/2306.07906?utm_source=chatgpt.com "WebGLM: Towards An Efficient Web-Enhanced Question Answering System with Human Preferences"))

评测上，WebGLM 提出了多维人工评估标准，不只看答案是否流畅，还看事实性、相关性、引用支持、完整性等维度；同时做了消融实验，分别验证 retriever、generator、scorer 各模块的贡献。论文报告称，**WebGLM-10B** 在人工评估中优于相近规模的 **WebGPT-13B**，并且接近 **WebGPT-175B** 的表现。它和 WebGPT 的主要区别在于：WebGPT 通过人类浏览轨迹做 imitation learning，再用 reward model 做 best-of-n rejection sampling；WebGLM 则把训练拆成检索、生成、偏好排序三个模块，更偏工程化和部署友好。([arXiv](https://arxiv.org/abs/2306.07906?utm_source=chatgpt.com "WebGLM: Towards An Efficient Web-Enhanced Question Answering System with Human Preferences"))