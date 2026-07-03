# Lecture 15: "After Pretraining" (Mid/Posttraining) 深度笔记

本笔记基于斯坦福 CS336 (Language Modeling from Scratch) 第十五讲的课堂内容整理。在前几讲中，我们详细探讨了语言模型的预训练（Pre-training）阶段（如分词、Transformer 架构、训练优化与数据清洗等）。预训练使模型能够以高表达能力预测下一个 Token，并获得了强大的语言表征与世界知识。然而，预训练基座模型（Base Model）通常无法直接供普通用户流畅交互，它需要进一步通过**后期训练（Post-training）与对齐（Alignment）**，包括**监督微调（Supervised Fine-Tuning, SFT）**与**人类反馈强化学习（Reinforcement Learning from Human Feedback, RLHF）**。

> **课程信息**：CS336 · Spring 2026 · 主题："After Pretraining" (SFT & RLHF)

---

# Part 1: Supervised Fine-Tuning (SFT) 监督微调部分

 SFT 是对齐流程的第一步，其核心在于通过高质量的“指令-回答”对，训练模型学会如何理解人类用户的提问（Instruction），并以合适的格式和语气输出回答（Response）。

---

## Slide 1: Title Slide (标题页)

![Slide 1](images/page_1.png)

### 讲解
本节课的主题是 **"After Pretraining"（预训练之后的工作）**。随着大语言模型（LLM）的参数规模和预训练 Token 数不断增大，业界越来越发现，模型表现不仅由预训练决定，预训练之后的 **Mid-training（中期训练）** 和 **Post-training（后期训练）** 在让模型具备交互性、安全性和特定专业任务（如代码、工具调用、复杂推理）的解决能力上起到了决定性的作用。

---

## Slide 2: The class thus far (迄今为止的学习进度)

![Slide 2](images/page_2.png)

### 讲解
迄今为止，我们已经探讨了预训练模型（如 GPT-3）的构建。虽然预训练模型是强力的通用概率续写器，但它与用户所期望的对话助手之间有很大的鸿沟。
- **左图 (GPT-3 时代)**：直接将网页文本作为续写输入（如 Copy.ai 提取广告词）。
- **右图 (ChatGPT 时代)**：用户希望以多轮对话、问答或辅助解题的形式，让模型表现出如人类般的理解力和交互能力（即 InstructGPT 到 ChatGPT 的跃升）。
- **核心问题**：我们应该如何跨越预训练基座（GPT-3）到对齐对话模型（InstructGPT / ChatGPT）之间的技术鸿沟？

---

## Slide 3: Instruction following is a remarkable form of control (指令遵循是卓越的控制机制)

![Slide 3](images/page_3.png)

### 讲解
引自微软经典论文 *[Bubeck et al. 2023] (Sparks of Artificial General Intelligence)*：
用户给出了一个极其复杂且约束繁多的提示词（Prompt）：要求用 `matplotlib` 绘制四个关于数据变化的图表、添加误差线、平滑曲线、绘制饼图并要求饼图随时间动态动画显示，还要保持 vertical line 动画等。
- **结果**：GPT-4 能够严格遵循所有的空间布局、数学计算和编程指令，直接生成能运行的 Python 绘图代码，并渲染出符合所有约束的动态可视化图表。
- **结论**：指令遵循（Instruction Following）不仅是简单的问答，它是对语言模型输出行为的一种**深度、紧密的控制机制**。

---

## Slide 4: Goal today: enable better, tighter controls over LM output (今天的目标：使模型输出得到更紧密的控制)

![Slide 4](images/page_4.png)

### 讲解
- **痛点**：预训练数据（Raw Web Data）虽具有大规模扩展性，但并不符合我们对模型的最终行为期望。
- **方案**：我们能否收集特定期望行为的数据来对 LM 进行专项训练？
- 为了实现这一目标，我们需要解答三个核心问题：
  1. **这些数据（对齐/指令数据）具体长什么样？** (What does that data look like?)
  2. **我们该如何最大化利用这些数据进行训练？** (How do we best make use of that data?)
  3. **我们在对齐阶段也需要像预训练那样庞大的算力与数据规模吗？** (Do we need scale for this?)

---

## Slide 5: Caveat: post-training information is pretty sparse! (警告：后期训练的公开技术细节极少)

![Slide 5](images/page_5.png)

### 讲解
在 LLM 业界，预训练的数据细节是相对公开的（有很多开源的 WET、Dolma 数据集论文），但**对齐与后期训练的数据配方被各大闭源厂商（如 OpenAI、Google、Anthropic）视为核心商业机密（Secret Sauce）**。
- **ChatGPT 诞生前（2020-2022）**：有相对详实的研究成果。如 OpenAI 的 *[Stiennon et al. 2020] (RLHF 摘要论文)* 包含了极其详细的标注员准则（Annotation Guidelines）和过滤机制；Anthropic 的 *[Bai et al. 2022]* 详述了安全对齐数据集（Helpful & Harmless）的构建细节。
- **现代（ChatGPT 爆发后）**：公开细节极少。开源模型也大多只在发布说明中给出模糊的比例，或直接使用未披露细节的蒸馏数据。

---

## Slide 6: Where today’s lecture fits in (今日讲座的定位)

![Slide 6](images/page_6.png)

### 讲解
本节课聚焦于经典的对齐三阶段流程（引自 *[Ouyang et al. 2022] (InstructGPT 论文)*）中的 **Part 1：监督微调 (Supervised Fine-Tuning, SFT)**。
- **Step 1 (SFT)**：采样用户 Prompt，让人类标注员（或高能力模型）写出期望的回答（Demonstration Data），在此数据集上微调预训练模型。
- **Step 2 & 3**：收集对比数据训练奖励模型（Reward Model），并基于该 RM 使用 PPO 等强化学习算法更新策略网络。我们今天首先关注 Step 1 SFT。

---

## Slide 7: What are the ingredients in SFT? (SFT 包含哪些元素？)

![Slide 7](images/page_7.png)

### 讲解
SFT 的核心配方由两部分组成：
1. **训练数据（Training Data）**：
   - 各种任务的数据集：自然指令集（Natural Instructions）、Muffin 任务集、TO-SF 通用微调数据集。
   - 推理数据：思维链（CoT, Chain-of-Thought Reasoning）数据集。
   - 人机对话数据：如 OpenAssistant。
2. **训练方法（The Method）**：
   - 如何混合这些不同任务、不同来源的数据。
   - 下方饼图展示了模型在**稳定阶段（Stable Stage）**和**衰退阶段（Decay Stage）**预训练数据与 SFT 数据的配比演进（例如预训练后期逐步掺入 SFT 数据）。

---

## Slide 8: Training data (对齐训练数据的两大探讨点)

![Slide 8](images/page_8.png)

### 讲解
我们重点关注指令微调数据集的两个核心细节：
1. **这些数据集中具体包含什么内容？** (What's actually inside these datasets?)
   - 我们将剖析 FLAN、Alpaca、OpenAssistant 和 Nemotron 的典型样本。
2. **在构建“高性能”指令微调数据时，什么因素起到了关键作用？** (What matters in building 'high performance' instruction tuning data?)
   - 我们将探讨回复长度、格式样式、事实性、规模与安全性的影响。

---

## Slide 9: Progression of SFT data in the open world (开源世界中 SFT 数据集的发展史)

![Slide 9](images/page_9.png)

### 讲解
开源社区在过去几年中见证了 SFT 数据从“NLP 任务模板化”向“多轮对话与工具调用”的快速演进：
1. **FLAN** (Google)：将传统的 NLP 数据集（分类、摘要、翻译等）用模版改写成指令形式。
2. **Self-Instruct**：利用大模型自动扩增指令的先驱尝试。
3. **Stanford Alpaca**：使用 GPT-3.5（text-davinci-003）生成了 52K 条廉价的指令遵循数据。
4. **ShareGPT/Vicuna**：收集人类与 ChatGPT 的真实对话，极大地提升了模型的对话流畅度（“Chattiness”）。
5. **Open Assistant**：由社区众包构建的、完全人工撰写的优质多轮对话数据。
6. **WizardLM**：通过 Evol-Instruct（复杂化演进提示）方法，使用 LLM 自动生成具有不同难度层级的指令。
7. **Tulu 3**：艾伦人工智能研究所（AI2）发布的最新、最全面的系统性 Post-training 开源框架。
8. **Nemotron (NVIDIA)**：专注于复杂的工具调用（Tool Use）、Agent 多步执行与长上下文对齐。

---

## Slide 10: FLAN – random examples (FLAN 数据集样本)

![Slide 10](images/page_10.png)

### 讲解
FLAN 是学术型 NLP 任务指令化的典型代表：
- **例1（写邮件主题）**：输入一段邮件正文，指令要求“为这封邮件撰写一个主题行”。
- **例2（单项选择）**：输入一段企业收购的新闻，提问“这段文本是关于什么的？选项：世界、体育、商业、科学/技术”。
- **例3（提取关键摘要）**：给出一篇关于海牙旅游的详细游记，要求“写出这篇文章的关键点”。
- **例4（结构化数据转文本）**：给定 `name = Aromi, eatType = coffee shop, food = English`，指令要求“根据以下餐厅数据写一句话”。
- **特点**：单轮、格式严谨、非常注重传统的 NLP 评测性能（如分类和短提取），但缺乏日常对话所需的“助手感”和温和语气。

---

## Slide 11: Alpaca – random examples (Alpaca 数据集样本)

![Slide 11](images/page_11.png)

### 讲解
Alpaca 是使用 LLM 蒸馏自动生成数据的早期标杆：
- **例1（生活建议）**：“给出三条保持健康的建议。”（生成回答：均衡饮食、定期运动、充足睡眠）。
- **例2（定义解释）**：“‘算法’是什么意思？”（生成回答：解决问题或实现目标的步骤说明……）。
- **例3（代码编写）**：“找出列表中的平均值。”（生成回答：一段计算平均数的 Python 简易函数）。
- **特点**：任务风格非常契合普通用户的日常求助，但回答偏向单一的列表或简明陈述，缺乏深度的论证和细节。

---

## Slide 12: OpenAssistant – random examples (OpenAssistant 数据集样本)

![Slide 12](images/page_12.png)

### 讲解
OpenAssistant 是完全由人类志愿者在社区平台撰写和审核的精细数据：
- **例1（学术事实与引用）**：要求写一篇关于经济学中“买方垄断（monopsony）”的简短介绍，并引用相关研究（人类标注的回答给出了详尽的概念解释，并在末尾给出了标准的学术参考文献，如 Bivens & Mishel, 2013）。
- **例2（生活与教育建议）**：要求为小学低年级孩子推荐便宜有趣的科学小实验（回答给出了制作自制熔岩灯、太阳能烤箱等的详细步骤和材料清单）。
- **特点**：**信息量极高、语气温和专业、具备真正的“人类专家感”**。

---

## Slide 13: Nemotron-SFT-OpenCode-v1 (英伟达代码与工具调用数据集)

![Slide 13](images/page_13.png)

### 讲解
Nemotron 展示了现代 Agentic 应用所需的指令对齐格式：
- **例1（编程概念对比）**：提问 `async/await` 与 `.then()` 在 JS 中处理 Promise 的区别。
  - 模型的回复不仅包含文本，还包括了具体的 **`tool_calls`**（如调用 `bash-skills` 来运行测试）。
- **例2（工程决策决策）**：构建计算器应用时选择整数还是浮点数的权衡。
  - 回复中嵌入了多步计划和工具调用结构：
    ```json
    "tool_calls": [{"id": "...", "type": "function", "function": {"name": "todowrite", "arguments": "..."}}]
    ```
- **特点**：这标志着 SFT 数据已经从“生成纯文本”进化到了“**为 Agent 规划多步执行计划并生成符合 JSON 格式的 Tool Calling API 结构**”。

---

## Slide 14: What’s varying across these? (这些数据集之间有什么维度的差异？)

![Slide 14](images/page_14.png)

### 讲解
以上不同代际的 SFT 数据集，在以下三个维度上呈现显著差异：
1. **“话痨度” (Chattiness)**：
   - 早期 FLAN 几乎没有多余的废话，直接给出 NLP 任务的简明预测（因为人类不希望在 NLP 评测基准上看到废话）。
   - 现代数据集（如 ShareGPT）则倾向于输出更长、更详细、语气更委婉的自然回复。
2. **细节程度 (Detail)**：
   - OpenAssistant 包含了大量关于事实知识的极具深度的细节描述。
   - **双刃剑**：细节多说明模型表现好；但如果模型没有在预训练中记住该知识，强行进行细节 SFT 会诱发模型严重的“幻觉（Hallucinations）”。
3. **工具调用 (Tool use)**：
   - 近一两年的 SFT 数据极大地向 Agentic Workflow 倾斜，包含大量工具调用格式、API 参数生成与多步推理链的对齐。

---

## Slide 15: What did we notice across the datasets? (我们能从这些数据集的变化中总结出什么？)

![Slide 15](images/page_15.png)

### 讲解
SFT 数据集在表层和深层存在着许多不同参数的变体：
- **表层（显性）变化**：
  - 回复的长度（Length）。
  - 是否包含无序列表和要点（Bullet Points，风格化差异）。
  - 是否包含文献引用（References）和复杂的逻辑推理知识。
- **深层（隐性）变化**：
  - 数据集的规模（Scale）。
  - 安全过滤与对齐标准（Safety）。
- **核心追问**：**这些设计要素（特别是长度、风格和知识深度）到底是如何在数学上和行为上影响最终训练出来的模型的？**

---

## Slide 16: Style variations in data and models (数据与模型中的风格变化)

![Slide 16](images/page_16.png)

### 讲解
引自论文 *[Wang+ 2023]* 的实验：
该表对比了主流指令数据在以下几个指标上的表现：实例数量（# Instances）、平均多轮对话数（$\bar{N}_{\text{rounds}}$）、平均 Prompt 长度（$\bar{L}_{\text{prompt}}$）、平均回复 Completion 长度（$\bar{L}_{\text{completion}}$）。
- **观察**：
  - 传统的学术数据（如 Flan V2）的 Completion 平均长度仅为 **31.2** 个 Token，极其短促。
  - Dolly 和 Alpaca 的 Completion 长度上升到了 **60-90** 个 Token。
  - 人工撰写的 OpenAssistant 1 的 Completion 平均长度暴涨至 **212.5** 个 Token。
  - 来自 ChatGPT 真实对话提取的 ShareGPT，平均 Completion 长度更是高达 **357.8** 个 Token，且包含平均 3.2 轮的对话。
- **结论**：对齐模型的“话痨度”几乎完全是由其 SFT 训练数据中的平均回复长度直接决定的。

---

## Slide 17: When evaluating by preferences, style matters. (偏好评估中的风格陷阱)

![Slide 17](images/page_17.png)

### 讲解
引自论文 *[Dubois+ 2023] (AlpacaFarm)* 的研究：
无论是在人类评测（Human Evaluation）还是在大模型充当裁判（GPT-based Evaluation）的偏好测试中，都存在极其明显的**“长度与格式偏见”（Length/Style Bias）**：
- **例1（列表偏好）**：如果模型的回复中使用了无序列表（Lists）或排版工整的要点，人类和 GPT-4 裁判对其打出高分的偏好比例显著超过 $60\%$。
- **例2（长度偏好）**：如果一个模型的 Completion 长度明显更长，被判定为“胜出（Win）”的概率随长度增加而飙升（偏好比例高达 $70\%-80\%$）。
- **警示**：这意味着**模型只需要在 SFT 数据中学会如何“把话写长、多用列表”，就能在各种 Chat 评估排行榜上刷出高分**，而其核心逻辑和解题能力并没有实质提升。

---

## Slide 18: What about benchmarks? (这些风格因素对基础评测基准有何影响？)

![Slide 18](images/page_18.png)

### 讲解
根据对比实验（以 Vanilla LLaMA 13B 为基座掺入不同 SFT 数据）：
- 虽然在主观对话指标（如 AlpacaEval）上，使用 ShareGPT（Win % 达 70.5%）、GPT4-Alpaca 等长 Completion、高话痨度的数据能获得压倒性胜利。
- 但在**硬性客观指标（如 MMLU 事实性测试、GSM8K 数学推理、BBH 逻辑测试）上，这些表层的风格变化几乎没有带来正向的收益**。
  - 例如，LLaMA-13B 基座在 MMLU 上的评分为 42.3%。
  - 加入极度“话痨”的 ShareGPT 后，MMLU 仅有微弱提升（49.3%），而 GSM 数学推理和 BBH 逻辑测试的表现几乎与加入低话痨度数据集一致甚至更低。
- **结论**：**SFT 只是改变了模型的表层行为风格（如学会礼貌、学会列表、学会长篇大论），并没有赋予模型新的推理内核。**

---

## Slide 19: References, complex knowledge, and factuality (文献引用、复杂知识与事实性之间的矛盾)

![Slide 19](images/page_19.png)

### 讲解
回到 OpenAssistant 关于“买方垄断”并要求引用文献的例子：
- **表面上看**：这个例子非常棒，教导了模型如何专业地回答经济学问题并在末尾给出标准的引用文献（Bivens & Mishel, 2013）。
- **深层拷问**：这个例子到底在教导模型什么？
  1. 它是在教模型关于“Bivens J & Mishel, L 在 2013 年写了某篇论文”的**具体事实**吗？
  2. 还是仅仅教模型**“当被用户要求引用时，应该伪造一些看起来极其逼真的文献格式”**？
- **机制探讨**：由于预训练知识存储在模型的权重参数中，SFT 只有极少的 Token 步数，模型根本无法在 SFT 阶段真正掌握这篇论文的真实细节。模型最终学会的只是“模仿文献引用的文字外观”。

---

## Slide 20: Knowledge extraction and alignment (知识提取与对齐的关系)

![Slide 20](images/page_20.png)

### 讲解
在对齐领域有一个著名的行业共识（Folklore）：**在模型未掌握的知识上强行进行微调，是导致大模型产生幻觉（Hallucinations）的核心元凶。**

- **Schulman (2023) [OpenAI RLHF 负责人] 的行为克隆（Behavior Cloning）观点 (左图)**：
  - 预训练使模型参数中建立了一个“知识图谱（Knowledge Graph）”，对每个知识边有不同的置信度。
  - SFT 的本质是“行为克隆”。如果我们给出的 SFT 目标回复中包含了基底模型知识图谱中本不存在（或置信度极低）的“冷门事实 $A$”，SFT 梯度会强行修正模型参数，逼迫模型“假装自己知道 $A$ 并自信地说出来”。
  - 这种训练等同于**训练模型去“撒谎和过度自信”**。当模型在推理阶段面对其他不确定的长尾知识时，它会习惯性地输出极度逼真的假事实（幻觉）。

- **Gekhman (2023) 实验验证 (右图)**：
  - 在“模型已知的事实（Train Known）”上微调，模型的开发集准确率会稳步上升。
  - 在“模型未知的事实（Train Unknown）”上微调，仅在 10 个 Epochs 后，模型在开发集（Dev Accuracy）上就开始发生严重的过拟合和衰退，并开始在生成中高频出现胡说八道。

---

## Slide 21: Takeaways on knowledge extraction and alignment (知识提取与对齐的总结)

![Slide 21](images/page_21.png)

### 讲解
1. **避免在长尾/冷门事实（Tail Knowledge）上做 SFT**：即便用户的最终用例要求模型掌握这些知识，我们也不应该在 SFT 数据中加入冷门事实，而应该通过 RAG（检索增强生成）或长上下文输入来解决。SFT 应该仅用来对齐那些在预训练阶段模型已经深刻掌握的高频常识。
2. **强化学习（RL）风格的正确性反馈理论上能缓解这一问题**：通过给撒谎行为施加负向奖励罚项，能促使模型说出“我不知道（I don't know）”。
3. 语言模型中知识的存储与提取机制极其混乱、微妙，目前尚无完美的解析理论。

---

## Slide 22: Safety (安全性对齐)

![Slide 22](images/page_22.png)

### 讲解
大语言模型被部署给成千上万的终端用户，必须具备极强的安全边界。
- **左图 [Goldstein+ 2023] (误导信息模型)**：如果不对模型做安全对齐，恶意攻击者可以通过 Model Access 制造大范围的假新闻、社交操纵、垃圾邮件 filter 绕过，并进行大规模的社会工程（Social Engineering）攻击，导致用户做出错误的决策。
- **右图 [Kang+ 2023] (钓鱼与垃圾邮件)**：展示了恶意用户诱导模型撰写紧急社会安全号索取钓鱼邮件的案例。经过安全对齐的模型必须识别该意图，并主动输出“此内容可能违反我们的内容政策……（This content may violate our content policy...）”予以拒绝。

---

## Slide 23: Safety SFT in the wild (现实中的安全 SFT)

![Slide 23](images/page_23.png)

### 讲解
各大模型厂商对安全数据的披露非常吝啬：
- **数据规模**：以 Llama 2 为例，其安全 SFT 数据仅包含几千个（a few thousand）高敏感性、经过人工红队（Red Teaming）精心撰写的反面提示词和标准安全拒绝模板。
- **难点（矛盾点）**：
  - **违规率 (Violation Rate, VR)**：安全防护不足，模型容易被越狱（Jailbroken）输出有害信息。
  - **误拒率 (False Refusal Rate, FRR)**：安全防护过度，导致模型对正常、合法的敏感词提问（如“如何杀死一个 Linux 进程”）也粗暴拒绝（俗称“安全脑残”）。
  - 右图展示了 Llama 3 8B 和 70B 模型在违规率（VR）与误拒率（FRR）之间的帕累托前沿（Pareto Front）权衡。模型规模越大，区分恶意攻击和正常合理提问的辨别力越好。

---

## Slide 24: Pipeline with most details (Tulu 3 的安全对齐细节)

![Slide 24](images/page_24.png)

### 讲解
在目前开源对齐框架中，Tulu 3 (AI2) 提供了最详细的安全训练配方。其安全 SFT 混合了以下细分数据集：
- **Safety & Non-Compliance (安全与合规防御)**：
  - **Tulu 3 CoCoNot** (10,983 个样本)：Brahman et al. 2024 构建的安全拒绝与合规性对齐集。
  - **Tulu 3 WildJailbreak** (50,000 个样本)：Jiang et al. 2024 构建的对抗性越狱与防御数据集。
  - **Tulu 3 WildGuardMix** (50,000 个样本)：Han et al. 2024 构建的安全护栏分类与防御数据集。

---

## Slide 25: Main safety approach – extract scenarios from users (核心安全方法：从真实日志中提取对抗场景)

![Slide 25](images/page_25.png)

### 讲解
目前最前沿的安全对齐理念是**从野外真实对话中提炼越狱模式**。
- **WildChat**：包含了来自真实用户与 ChatGPT 交互的 100 万条公开日志。由于是真实日志，里面充斥着各种极具创造性的对抗性话题和越狱尝试。
- **WildTeaming 框架 (右图)**：
  - **Step 1 (MINE)**：在 WildChat 中，使用 LlamaGuard 等安全分类模型，自动挖掘出人类写出来的最新越狱战术（ITW Jailbreak Tactics）。
  - **Step 2 (COMPOSE)**：使用 GPT-4 自动将挖掘出的“越狱战术 $PR_h$”与“常规有害查询 $M_{\text{target}}$”融合，合成出数万个难度极高的对抗性 Prompt。用这批合成的对抗 Prompt 训练模型进行拒绝，能极大提高模型在野外的防御力。

---

## Slide 26: Safety-tuning with just a little data (极少的数据即可带来显著的安全性提升)

![Slide 26](images/page_26.png)

### 讲解
- **令人惊讶的实验结论**：安全对齐对于数据量的依赖低得超乎想象。
- **观察图表**：仅加入 **500 个 Alpaca 风格的安全正样本（Added Safety Data = 500）**进行微调，模型在恶意指令（I-MaliciousInstructions）、仇恨言论（I-CoNa）以及对抗询问（Q-Harm）上的不良输出均分（Mean Score）就发生了断崖式下降，迅速逼近完美安全线。
- **原理**：安全性对齐主要是改变模型的表层行为作风（学会说“No”的触发词），不需要模型去学复杂的知识，因此极少的高质量模板数据即可完成知识泛化。

---

## Slide 27: Putting it together – SFT Data (SFT 数据核心法则总结)

![Slide 27](images/page_27.png)

### 讲解
1. **SFT 是“行为唤醒”而非“知识灌输”**：SFT 最适合用来**提取（Extract）**模型在预训练阶段已经学到的行为和常识，不适合强行注入新知识（否则会引发幻觉）。
2. **加入看似正确的事实数据有时反而有害**：如果模型在预训练中没有记住这些事实，SFT 会训练模型生成假引用、编造细节并导致过度自信。
3. **“少即是多” (Less is More)**：在安全性、格式样式、对话礼貌性等风格特征上，极少量（如 500-1000 个）的黄金数据就能带来巨大跃升。但如果要彻底覆盖各种长尾的专业问答场景，依然需要庞大、多样的长尾数据集来帮助泛化。

---

## Slide 28: How to fine-tune (如何进行 SFT 训练？)

![Slide 28](images/page_28.png)

### 讲解
在学术界或小规模实验中，SFT 在代码实现上与普通的预训练并无本质区别，只需对“提示词-回复”对进行标准的自回归梯度下降（Supervised Autoregressive Loss）：
- **实现代码**：如 PyTorch 的简易训练 Loop。
- **细节差异（关键）**：在计算交叉熵损失（Cross-Entropy Loss）时，**通常会将 Prompt 对应的 Token 的 Loss Mask 掉（权重设为 0）**，只对 Completion 对应的 Token 计算梯度。
- **工业界痛点**：对于拥有极度充裕算力和数据的头部大模型厂商，简单的单次 SFT 微调依然不够。由于微调数据量相对预训练极小，模型容易遭遇**灾难性遗忘（Catastrophic Forgetting）**，导致基础推理能力发生退化。

---

## Slide 29: Turning instruction tuning into pretraining (将指令微调融入预训练)

![Slide 29](images/page_29.png)

### 讲解
为了在大规模对齐时不破坏基座模型原有的强大知识面与泛化推理能力，业界摸索出了一套融合的“三步走”方案：
1. **Phase 1**：在海量网页/无监督数据上进行纯粹的预训练。
2. **Phase 2 (Mid-training)**：在预训练后期，**将一定比例的指令微调（SFT）数据直接混入无监督的网页数据中**进行混合训练（将 SFT 视作预训练数据的一种高阶形式）。
3. **Phase 3**：最后，进行一个周期极短、数据量极小的纯 SFT/RLHF 轮次，激活最终的助手语调。
- **好处**：极大地减缓了模型在对齐时的灾难性遗忘。

---

## Slide 30: ‘Midtraining’ / ‘Two-phase training’ (中期训练 / 两阶段训练方案)

![Slide 30](images/page_30.png)

### 讲解
这种将对齐挪到预训练后期（即 Midtraining）的做法在工业界是广泛使用的共识（但在学术论文中很少被完整提及）。
- **左图：稳定阶段（Stable Stage）数据配比**：包含了常规的中文网页（CommonCrawl_Chn 25%）、代码（Code Pretrain 25%）、英文网页（Dolma 24%）、C4、The Pile 等。
- **右图：衰退阶段（Decay Stage）数据配比**：在预训练的最后阶段，学习率降温（Decay）的同时，将 **Math_SFT、Stack Exchange QA、Book Chinese、UltraChat、Knowledge_SFT** 等大量的对齐数据以不同的百分比强势混入。
- 该方案在 **miniCPM**、**jetMoE** 等中文背景的小型模型中被公开提及并验证，能确保小模型在对齐后依然保持强悍的客观 benchmark 性能。

---

# Part 2: RLHF Data & Human Annotation 强化学习数据与人工标注部分

 虽然 SFT 能让模型说出流畅得体的话，但它本质上只是一种“模仿行为（Imitation）”，无法引导模型解决逻辑与事实偏好的深层对齐。我们需要通过强化学习来优化模型行为。

---

## Slide 31: The second part of RLHF (RLHF 的第二阶段)

![Slide 31](images/page_31.png)

### 讲解
现在，我们进入经典三阶段的后两个步骤：
- **Step 2**：采样 Prompt，让模型生成多个不同的输出，让人类对其从好到坏进行**排序（Rank）**。用这个对比数据训练一个**奖励模型 (Reward Model, RM)**。
- **Step 3**：利用 RM 提供在线奖励（Reward），通过 **PPO 算法**以在线强化学习的形式，最大化模型在 RM 上的期望得分，从而微调 Policy。
- 我们本节首先剖析 RLHF 中最核心的燃料：**偏好标注数据（Pairwise Preference Data）**。

---

## Slide 32: From imitation to optimization (从模仿学习走向最优控制)

![Slide 32](images/page_32.png)

### 讲解
这页给出了 SFT 与 RLHF 在数学建模上的根本区别：
- **Imitation (SFT) 监督微调**：
  $$\text{Fit } \hat{p}(y \mid x) \approx p^*(y \mid x) \text{ for some reference distribution } p^*(y \mid x)$$
  - **本质**：拟合一个由高水平人类或强模型表现出的参考概率分布 $p^*$。这纯粹是一个生成式建模问题，模型能达到的高度完全受限于参考分布 $p^*$ 的质量。
- **Optimization (RLHF) 强化学习**：
  $$\text{Find } \hat{p}(y \mid x) \text{ such that } \max_{p} \mathbb{E}_p [R(y, x)] \text{ for a reward } R(y, x)$$
  - **本质**：将大模型视作一个决策策略（Policy），在给定的输入 $x$ 下，动作（生成 Token 序列 $y$）必须**最大化奖励函数 $R(y, x)$ 的期望值**。
  - **突破**：这从根本上摆脱了“模仿”的限制，模型可以通过在强化学习环境中自行搜索与优化，产出比 SFT 样本甚至比人类标注员写得更好的回复。

---

## Slide 33: Why optimize? G-V gap (为什么要进行优化？G-V 鸿沟)

![Slide 33](images/page_33.png)

### 讲解
为什么我们不直接用 SFT 模仿好作文，非要大费周折搞 RLHF 优化？因为存在 **G-V Gap (Generation-Verification Gap, 生成-检验鸿沟)**。

> **👨‍🏫 小白导读**：
> 我们可以把大模型想象成一个学生，把人类标注员想象成老师。
> - **SFT 是“抄作业”**：让学生一字不差地模仿老师写出来的范文。但问题是，很多时候**“写出一篇完美的范文极其艰难，但判定两篇作文哪篇写得更好却非常容易”**。比如在数学证明、长文摘要等任务中，让人类写一份完美的证明或摘要可能要耗时一小时，但要从两篇现有的生成结果中挑出一个更通顺、更正确的，只需要花一分钟。
> - **RLHF 是“判作业”**：我们只收集人类“二选一”的评判反馈（这非常廉价且精确），然后用算法去**逼迫模型自己去探索和寻优**。这能让模型自发产出连人类标注员都写不出来的精妙解法。

- **右图 [Zhang et al. 2023] (新闻摘要评估)**：
  - 实验表明：在新闻摘要任务中，虽然人类在“自己动笔写（Generation）”时更喜欢外包写手（Freelance writers）的风格；但在“对比挑选（Verification）”时，人类会以绝对优势偏好大模型（Instruct Davinci）生成的、结构更紧密、要素更全的摘要。这充分说明了生成偏好与选择偏好之间的巨大鸿沟。

---

## Slide 34: Overview (偏好对齐的授课大纲)

![Slide 34](images/page_34.png)

### 讲解
本章接下来的内容将分为三个关键方面：
1. **数据 (Data)**：
   - 偏好对齐数据是如何收集的？
   - 在众包偏好数据时有哪些令人头疼的陷阱和法律道德约束？
2. **如何运行 RLHF？** (How do we RLHF?)
   - **PPO**：经典的在线强化学习方案。
   - **DPO**：创新的离线直接偏好优化方案。
3. **RLHF 会给模型带来哪些未曾预料的副作用（Side-effects）？**

---

## Slide 35: RLHF Data (RLHF 数据收集链路)

![Slide 35](images/page_35.png)

### 讲解
在三阶段架构中，要获取高质量的 pairwise feedback（成对偏好反馈），我们需要回答：
- 我们该提供什么样的对比界面？
- 我们应该如何引导人类做出客观、一致（Good Agreement）的偏好判断？

---

## Slide 36: RLHF and data – standard setups (标准的成对偏好标注界面)

![Slide 36](images/page_36.png)

### 讲解
展示了在 Amazon Mechanical Turk (MTurk) 或 ScaleAI 等标注平台上标准的 RLHF 标注界面：
- 给定用户 Prompt：*“告诉我关于自动驾驶汽车的信息（Tell me about self driving cars）”*。
- 同时展示两个模型的匿名输出：**AI Response 1** 和 **AI Response 2**。
- 让人类标注员在下方进行单选打分：
  - Response 1 is better. (输出 1 更好)
  - Response 1 is only slightly better. (输出 1 仅略微好一点，需双方极接近才选)
  - Response 2 is only slightly better. (输出 2 仅略微好一点)
  - Response 2 is better. (输出 2 更好)

---

## Slide 37: RLHF and data – instruct GPT guideline (InstructGPT 标注准则)

![Slide 37](images/page_37.png)

### 讲解
这页展示了 OpenAI 为 InstructGPT 标注团队定制的、长达数十页的偏好对齐准则Excerpt（3H 标准）：
- **Helpful (有帮助)**：模型必须清晰、准确地遵循人类意图。若提问模糊，应主动澄清；必须具备跨国界与文化视角的敏感度。
- **Truthful (真实性)**：严禁模型输出虚假信息，特别是“无中生有”或“胡编乱造”。在摘要等任务中，模型只能使用输入中已有的信息。
- **Harmless (无害性)**：模型绝不能输出包含人身攻击、煽动暴力、鼓吹非法活动或带有群体歧视色彩的内容。在遇到极端越狱提问时，应保持中立并进行合规拒绝。

---

## Slide 38: Another old example – bard annotations (谷歌 Bard 的标注准则与等级)

![Slide 38](images/page_38.png)

### 讲解
展示了谷歌当年为 Bard 标注团队提供的偏好对齐等级划分：
- **Helpfulness (帮助度评估量表)**：
  - *Not at All Helpful*：完全无用，甚至包含垃圾、虚假或带有极大偏见的内容。
  - *Slightly Helpful*：回答与 Prompt 略微相关，但错漏严重。
  - *Somewhat Helpful*：回答了一半，但遗漏了关键细节或包含部分过时信息。
  - *Very Helpful*：意图回答清晰，逻辑完整，对用户非常实用。
  - *Extremely Helpful*：完美回答，信息翔实，展现出了高度的专业素养。
- **Presentation (陈述展现度评估量表)**：评估格式排版是否易读、是否重复啰嗦、语法是否通顺、语气是否过于自满或虚伪。

---

## Slide 39: Modern worker distribution (现代标注工人群体的分布)

![Slide 39](images/page_39.png)

### 讲解
偏好数据并非来自于虚拟空间，而是由现实中的人一条条点出来的。这页展示了主流标注平台（如 Outlier, ScaleAI）的工人群体统计分布（来自 Oxford Economics 的调查报告）：
- **年龄分布**：主要集中在 **25-44** 岁之间（占约 60%），属于年轻的劳动力群体。
- **学历分布**：拥有本科以上学历（Bachelor / Master / PhD）的比例高达 **85%**，属于高知群体。
- **专业细分**：技术性（Coding、数学、数据科学）占了很大的比例，此外还涵盖创意写作、法律、历史、生物等诸多专业。
- **结论**：对齐标注已经完全脱离了早期的“低成本体力劳动”（如图片拉框），转向了**需要专业领域专家（Subject Matter Experts）参与的高薪智力服务**。

---

## Slide 40: Large variation in compensation (标注行业薪资的巨大落差与 expert 对齐热潮)

![Slide 40](images/page_40.png)

### 讲解
- **专家标注（Expert Annotation）的爆发式增长**：
  大模型厂商之间在专业推理上的竞争极其激烈，促使他们以高薪招募各行各业的顶尖学者、程序员与法学专家。
- **左图 Business Insider 报道**：在 OpenAI 内部代号为 "Project Stagecraft" 的项目下，数据标注初创公司 Handshake AI 为各行业 freelancer 提供**至少每小时 50 美元**的报酬，专门撰写用于微调 ChatGPT 的专业领域数据。
- **右图 薪资分布图**：展示了美国 Specialist Data Annotator 的薪资水平。对于物理、工程、金融、CS、医学等核心理科领域的 Expert，其小时薪资上限被拉升到了 **100-120 美元**。

---

## Slide 41: RLHF and data - crowdsourcing (众包标注的复杂性与陷阱)

![Slide 41](images/page_41.png)

### 讲解
虽然众包能提供规模化数据，但它面临三个致命的技术和工程管理痛点：
1. **高素质且可验证的标注员极其难求**：标注平台（如 Outlier）中充斥着虚假简历和作弊注册，如何建立有效的背景审核和能力测试是大难题。
2. **标注员极难真正核对内容的正确性**：当面对复杂代码、量子物理或高等数学等高精尖任务时，大模型经常输出“看起来极其通顺且高级、但包含细微逻辑逻辑谬误”的欺骗性回复。普通标注员往往被漂亮的外观蒙蔽，随手给错误解法点赞，导致模型学会狡辩。
3. **标注员自身使用 AI 作弊**：部分外包工友为了追求高效率和多拿报酬，使用 ChatGPT 来替自己写标注反馈，导致数据发生污染（用 AI 生成偏好去训练 AI，导致逻辑同质化）。

---

## Slide 42: RLHF and data – crowdsourcing ethics (众包标注的伦理与剥削黑历史)

![Slide 42](images/page_42.png)

### 讲解
- **数据背后的“AI 劳工底层” (America's AI Underclass / Kenay Workers)**：
  大模型光鲜亮丽的安全回答背后，是全球廉价劳动力的残酷剥削与心理折磨。
- **TIME 独家报道**：OpenAI 曾外包给 Sama 公司（一家位于旧金山的数据标注公司，在肯尼亚招聘员工），付给这些工人的时薪**低于 2 美元**。
- **工作内容**：这些肯尼亚工人每天要阅读和分类海量包含虐待儿童、自残、谋杀、强奸和酷刑等极度令人不适和恶心的血腥暴力文本日志，以帮助 ChatGPT 建立有毒内容的防御库。许多工人因此患上了严重的创伤后应激障碍（PTSD）等心理疾病。

---

## Slide 43: RLHF and data - demographics (标注工人群体人口统计特征的行为偏差传导)

![Slide 43](images/page_43.png)

### 讲解
引自论文 *[Santurkar+ 2023]*：
**标注工人群体的人口统计学特征分布，会以数学偏好概率的形式，直接、深刻地塑造大模型的价值取向。**
- **数据分析**：
  - 标注群体往往以特定国家（如美欧）、特定信仰（Protestant/Catholic 等）、特定学历（高学历群体）为主（左侧表）。
  - 右侧表测试了不同模型（如 AI21, OpenAI）的输出结果与不同宗教背景的人群的契合度。
  - **观察**：OpenAI 模型的偏好极度向美欧的 Protestant/Catholic 群体倾斜（契合度超 $0.75$），而与 Buddhist、Hindu 等非西方宗教背景人群的偏好契合度显著偏低。
- **启示**：模型并没有绝对的“普世真理偏好”，你雇佣什么人点赞，大模型就会继承这个群体的世界观和道德偏好。

---

## Slide 44: RLHF and style – annotators matter (a lot) (众包工友 vs 领域专家：标注风格偏差矩阵)

![Slide 44](images/page_44.png)

### 讲解
引自论文 *[Hosking, Blunsom, Bartolo 2024]*：
该图用热力图（Heatmap）展示了普通众包工人（Crowdsourced Annotators）与领域专家（Expert Annotators）在面对同一批模型回复时的**纠错误差率（Error Rate Difference）**：
- **核心发现**：
  - **在 Assertiveness（自信/说服力）维度**：当模型的回复表现得极度自信、语气斩钉截铁（Assertiveness++）时，普通众包工人会发生严重的判断失误。他们会低估模型的不一致性（Inconsistency $-16.9\%$）和事实性错误（Factuality $-22.3\%$）。也就是说，**模型只要语气足够强硬自信，普通工人就会偏心判定其回答正确，即便里面全是胡说八道。**
  - **在 Complexity（复杂性）维度**：当回复的逻辑非常复杂冗长时，普通工人发现事实性错误的能力会暴跌 $14.3\%$。
- **结论**：为了对齐深度推理，我们必须依赖专业的 Expert，而不能依赖普通外包工。

---

## Slide 45: RLFH and data - LM-generated (大模型生成的 AI 反馈, RLAIF)

![Slide 45](images/page_45.png)

### 讲解
为了彻底摆脱人类标注的高昂成本、偏见和低效，研究人员开始用强模型（如 GPT-4）代替人类作为偏好裁判：
- **左图（系统级相关性）**：
  GPT-4 对模型胜率的评估（Simulated Win-rate）与人类评估（Human Win-rate）呈现出**近乎完美的线性相关性**（皮尔逊/斯皮尔曼相关系数高达 **0.98**）。这表明在宏观系统评测上，GPT-4 完全可以平替人类。
- **右图（微观一致性）**：
  横轴为每千次偏好评判的美元成本，纵轴为与人类 mode 判定的一致性。GPT-4 充当裁判的一致性率（约 $64\%-65\%$）已经完美达到了人类标注员之间的互信度上限（Inter-annotator Agreement），且**成本比招募 Expert 便宜了数个数量级**。

---

## Slide 46: RLHF and data - LM-generated in the wild (野外大规模使用的 AI 反馈)

![Slide 46](images/page_46.png)

### 讲解
目前，AI 反馈（RLAIF）已经成为开源与闭源前沿模型最核心的偏好获取方式。
- **代表性应用**：
  - **UltraFeedback**：开源的偏好标注数据集，完全由大模型对多个开源回复从 Helpful, Honest, Harmless 维度进行细分打分并二值化。
  - **Zephyr 7b** (Hugging Face)：其主要贡献之一正是利用 RLAIF 蒸馏替代传统的人工 RLHF 对齐，证明了 AI 反馈的高效性（引自 Lewis et al. 采访）。
  - **Tulu 3**：在 Prompt 选择后，利用包含 Llama-3 8B SFT、Tulu 3 8B 等在内的 22 个混合能力模型生成多路 Completion，最后使用 `GPT-4o-2024-08-06` 充当裁判从“有帮助”、“指令遵循”、“真实性”、“得体性”四维度打分，并将结果 binarize（二值化）成 DPO 训练对。

---

## Slide 47: RLHF and data – Self-training (宪法 AI 与模型的自我迭代对齐)

![Slide 47](images/page_47.png)

### 讲解
引自 Anthropic 著名的 **Constitutional AI (宪法 AI) [Bai et al. 2022]** 流程：
1. **监督阶段（SL-CAI）**：
   - 引导模型对有毒提问输出 Harmful 回复。
   - 提供一套“人类宪法原则”（如：请让回复尽可能温和、且不要危害人类安全）。
   - 让模型自己写出对上述回复的批评（Critique），并根据批评写出修正版（Revision）回复。用这批 Revision 数据微调模型。
2. **强化阶段（RL-CAI）**：
   - 模型对越狱提问生成 Response 1 & 2。
   - 用 SL-CAI 训练出来的“宪法判官”根据宪法准则，给两个输出打分，构建 Preference 模型。
   - 最终使用这个自我评判的 PM 对 Policy 运行 RLAIF 强化微调。
- **意义**：模型不需要看任何人类写的有害/无害范文，完全通过自省（Self-training）和规则约束实现了完美的无害化对齐。

---

## Slide 48: RLHF and style – Length effects (RLHF 的长度膨胀副作用)

![Slide 48](images/page_48.png)

### 讲解
虽然 RLHF 和 RLAIF 极为成功，但它们带来了一个极其显著的副作用——**长度失控膨胀（Length Explosion）**。
- **左图 [Chen et al. 2024]**：横轴为回复长度，纵轴为 AlpacaEval Win %。无论使用 PPO 还是 DPO 优化，随着迭代步数增加，模型的 Win % 提升在很大程度上是由于**模型学会了输出越来越长（从 200 个 Token 一路飙升至 280 个 Token 以上）**的废话。
- **右图 [Singhal et al. 2024]**：
  - 用户问题：*“为什么成年人睡觉时不容易掉下床？”*
  - **SFT 模型回答（59 个 Token）**：因为成年人发育出了肌肉记忆，能在睡觉时维持体态……（精简且正确）。
  - **RLHF 优化后回答（243 个 Token）**：大段废话，前两段重复 SFT 内容，然后强势灌水加入：“另外，很多成年人会觉得在睡梦中翻身是一件不舒服甚至痛苦的事……如果掉下床会受伤……”。
- **本质**：RM 奖励模型由于人类/GPT-4 裁判的偏见，学会了给“长废话”提供高 Reward 分数。这导致强化学习策略被诱导向“生成大段无意义细节”方向狂奔。

---

# Part 3: PPO & DPO Algorithms PPO 与 DPO 对齐算法部分

 拥有了偏好标注数据后，我们该如何改造和更新模型参数？我们将探讨经典的 PPO（强化学习派）与 DPO（离线直接优化派）的数学原理。

---

## Slide 49: How do we RLHF? (我们如何运行 RLHF？)

![Slide 49](images/page_49.png)

### 讲解
这页提出了 RLHF 算法演进的两大核心路线：
- **Part 1: PPO (Proximal Policy Optimization)**：最初由 OpenAI 在 InstructGPT 中使用的、也是强化学习领域极其经典的在线策略算法。但它极度不稳定，对超参数敏感，且训练工程极其繁琐。
- **Part 2: DPO (Direct Preference Optimization)**：由斯坦福团队于 2023 年提出的革命性替代方案，将强化学习巧妙规避，转化为极其稳定的离线监督对比分类损失，极大地降低了对齐的工程门槛。

---

## Slide 50: From imitation to optimization (从模仿学习走向最优控制 - 数学重温)

![Slide 50](images/page_50.png)

### 讲解
再次重温数学定义，为 PPO 和 DPO 的推导做铺垫：
- SFT 只要求模型模仿参考策略 $p^*$ 的分布。
- RLHF 要求我们在给定的 Prompt 空间下，找到一个策略分布 $\hat{p}(y \mid x)$，它能最大化由奖励模型 $R(y, x)$ 评估的期望 Reward。
- 难点在于，大模型的动作空间（Vocabulary 词汇表大小 $V$，输出序列长度 $T$）是指数级庞大的，直接做全空间期望积分是不可行的，必须使用蒙特卡洛策略梯度方法进行优化。

---

## Slide 51: PPO in language modeling (语言模型中的 PPO 数学目标函数)

![Slide 51](images/page_51.png)

### 讲解
在语言模型中运行 PPO（以 InstructGPT 为例），其优化的核心目标函数（Objective）设计如下：

$$\text{objective}(\phi) = \mathbb{E}_{(x,y) \sim D_{\pi_\phi^{\text{RL}}}} \left[ r_\theta(x, y) - \beta \log \left( \frac{\pi_\phi^{\text{RL}}(y \mid x)}{\pi^{\text{SFT}}(y \mid x)} \right) \right] + \gamma \mathbb{E}_{x \sim D_{\text{pretrain}}} [\log(\pi_\phi^{\text{RL}}(x))]$$

这虽然看起来无害，但它在数学上融合了三个极其重要的约束设计：
1. **$r_\theta(x, y)$**：当前 RL 策略网络对 Prompt $x$ 生成的 Completion $y$ 从奖励模型（Reward Model）获得的原始 Reward 得分。
2. **KL 散度惩罚项（中括号内）**：
   $$\text{KL Penalty} = -\beta \log \left( \frac{\pi_\phi^{\text{RL}}(y \mid x)}{\pi^{\text{SFT}}(y \mid x)} \right)$$
   - **作用**：随着强化学习不断推进，策略网络 $\pi^{\text{RL}}$ 会不断调整参数以刷高奖励分。如果不加限制，模型很快会找到奖励模型参数在未覆盖区域的漏洞（Reward Hacking），输出满是漏洞的乱码废话。
   - **约束**：在每一个 Token 上施加 KL 惩罚，约束当前策略 $\pi^{\text{RL}}$ 不要偏离初始 SFT 基底模型 $\pi^{\text{SFT}}$ 太远，从而保护模型的自然语言生成逻辑不崩溃。
3. **预训练梯度混入项（Ptx Term，最后一项）**：
   $$\gamma \mathbb{E}_{x \sim D_{\text{pretrain}}} [\log(\pi_\phi^{\text{RL}}(x))]$$
   - 在计算强化学习梯度的同时，混入预训练自回归的负对数似然梯度。
   - **作用**：防止模型陷入严重的“对齐退化”，强行稳住模型在常规 NLP objective 上的语言功底。当 $\gamma$ 设为 0 时，该项被取消。

---

## Slide 52: More details and background from Stiennon (关于 RM 损失与 KL 惩罚的细节探讨)

![Slide 52](images/page_52.png)

### 讲解
这页给出了更底层的理论支撑（引自 *[Stiennon et al. 2020]*）：
- **奖励模型 (RM) 的训练损失函数**：
  偏好训练数据为三元组 $(x, y_0, y_1, i)$，其中 $x$ 为 Prompt，$y_0$ 和 $y_1$ 为两个模型回复，$i \in \{0, 1\}$ 表示哪个更好（假设 $y_i$ 为胜出者，$y_{1-i}$ 为失败者）。
  奖励模型的优化采用经典的 Bradley-Terry 偏好模型，其交叉熵 Loss 为：
  $$\text{loss}(r_\theta) = -\mathbb{E}_{(x, y_0, y_1, i) \sim D} [\log(\sigma(r_\theta(x, y_i) - r_\theta(x, y_{1-i})))]$$
  - **直觉**：拉大胜出者的得分 $r_\theta(x, y_i)$ 与失败者的得分 $r_\theta(x, y_{1-i})$ 之间的分差。
- **关于 KL 散度约束的另外两个核心功能**：
  1. 它作为一种**熵的奖励（Entropy Bonus）**，鼓励 Policy 去积极探索更多样化的输出空间，防止模型在 RL 后期出现严重的“模式塌陷（Mode Collapse）”退化为只复读几句话。
  2. 确保 Policy 生成的输出概率与 RM 的分布距离在置信度以内，防止评估失准。

---

## Slide 53: PPO – at a conceptual level (PPO 的概念与技术演进史)

![Slide 53](images/page_53.png)

### 讲解
大语言模型中的 PPO 极难调试，我们需要在概念上理清它的三个演进尝试：
- **Attempt 1: Policy Gradients (策略梯度法 / REINFORCE)**：
  其计算公式为：
  $$\nabla_\theta \mathbb{E}_{p_\theta} [R(z)] = \mathbb{E}_{p_\theta} [R(z) \nabla_\theta \log p_\theta(z)]$$
  - **致命弱点**：**方差极高**（High Variance）。如果环境返回的 Reward $R(z)$ 的尺度波动很大，或者模型采样概率有一丝扰动，梯度方向就会发生剧烈抖动，导致训练直接发散崩溃。
- **Attempt 2: TRPO (Trust Region Policy Optimization, 信赖域策略优化)**：
  - 为了稳住梯度更新，强行对每一次参数更新前后的策略分布设定置信限界：
    $$\max_\theta \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \hat{A}_t \right]$$
    $$\text{subject to } \hat{\mathbb{E}}_t [\text{KL}[\pi_{\theta_{\text{old}}}(\cdot \mid s_t), \pi_\theta(\cdot \mid s_t)]] \le \delta$$
  - **弱点**：计算二阶 Hessian 矩阵倒数和二阶导数极度耗费显存与计算时间，不适合百亿参数的 LLM。
- **Attempt 3: PPO (Proximal Policy Optimization)**：
  - 放弃二阶计算，改用最简易的“裁剪代理目标”（Clipped Surrogate Objective）来硬性截断更新比例，限制在 $[1-\epsilon, 1+\epsilon]$ 之间：
    $$L(s, a, \theta_k, \theta) = \min \left( \frac{\pi_\theta(a \mid s)}{\pi_{\theta_k}(a \mid s)} A^{\pi_{\theta_k}}(s, a), \text{clip} \left( \frac{\pi_\theta(a \mid s)}{\pi_{\theta_k}(a \mid s)}, 1 - \epsilon, 1 + \epsilon \right) A^{\pi_{\theta_k}}(s, a) \right)$$
  - **虽然 PPO 较 TRPO 简单，但在 LLM 训练中它依然非常 finicky（繁琐且不稳定）**，因为：
    - 它需要在显存中同时存放四个超大模型：**Actor Policy（需要训练的策略网络）、Critic Value Network（需要优化的价值评估网络）、Reference SFT Model（需要冻结以计算 KL 的参考网络）和 Reward Model（冻结的奖励模型）**。
    - 四大模型的存在对多卡显存和通信带宽造成了极致榨干，且 Actor 和 Critic 之间的学习率同步配合极难对齐。

---

## Slide 54: Can we get rid of PPO? (能否摆脱 PPO？)

![Slide 54](images/page_54.png)

### 讲解
面对 PPO 的噩梦级工程负担，业界想方设法尝试避开在线强化学习：
- **方案 1：Control Token 机制**：将偏好对（Chosen/Rejected）拼接在一起进行常规 SFT，在 Chosen 样本前添加 `[GOOD]` Token，在 Rejected 前加 `[BAD]`。推理阶段，强行输入 `[GOOD]` 以引导生成高质量回复。
- **方案 2：只在 Preferred Output 上做 SFT**：直接抛弃坏样本，直接用高赞回答做常规自回归微调（但容易丧失防越狱的拒绝能力）。
- **方案 3：Best-of-N 拒绝对比采样（Rejection Sampling / Best-of-1024）**：
  - 首先，基于 SFT 训练一个优秀的 Reward Model。
  - 在推理阶段，对一个 Prompt 连续让模型采样 1024 个不同的 Completion。
  - 用 RM 给这 1024 个回复打分，**只挑选得分最高的那一个输出给用户**。
  - **效果**：这在推理阶段增加了巨额计算量（Test-Time Compute），但能在不进行 RL 训练的前提下，输出极其强悍的对齐品质。

---

## Slide 55: DPO – RLHF without tears? (DPO：无强化学习眼泪的对齐？)

![Slide 55](images/page_55.png)

### 讲解
斯坦福团队提出的 **DPO (Direct Preference Optimization, 直接偏好优化) [Rafailov et al. 2023]** 是一次极其惊艳的**数学降维打击**：

- **PPO 的痛点**：在线强化学习之所以慢且不稳定，是因为需要从策略网络中实时生成回复（Rollouts），经过奖励模型打分，再更新策略，构成一个高代价的在线外循环。
- **DPO 的革命性构想**：
  1. **消灭独立的 Reward Model**。
  2. **消灭任何在线 Rollouts（外循环）**。
- **替代方案**：直接在现有的偏好数据集上，对胜出 Completion 执行正向梯度更新（提升 log-likelihood），对失败 Completion 执行负向梯度更新，一步到位完成对齐。

---

## Slide 56: DPO – derivation from the RLHF formula (DPO 的数学推导 1： implied reward 的闭式解)

![Slide 56](images/page_56.png)

### 讲解
这页给出了 DPO 优雅的数学起点。我们的目标依然是优化经典的 KL 散度约束最大化期望 Reward 问题：

$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(y \mid x)} [r(x, y)] - \beta \mathbb{D}_{\text{KL}} [\pi(y \mid x) \parallel \pi_{\text{ref}}(y \mid x)]$$

1. 我们对动作概率和 KL 散度的概率分布展开写出：
   $$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}} \left[ \mathbb{E}_{y \sim \pi(y \mid x)} \left[ r(x, y) - \beta \log \left( \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right) \right] \right]$$
2. 做无约束优化，在非参数（Non-parametric）分布假设下，由于这是一个典型的凸优化问题，我们可以求出其全局最优解策略 $\pi_r(y \mid x)$ 的闭式解（Closed-form Solution）：
   $$\pi_r(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp \left( \frac{1}{\beta} r(x, y) \right)$$
   其中 $Z(x) = \sum_{y} \pi_{\text{ref}}(y \mid x) \exp \left( \frac{1}{\beta} r(x, y) \right)$ 是归一化的配分函数（Partition Function）。
3. 关键步骤：**将最优策略等式取对数，反向求出 implied reward（隐含的奖励函数表达式）**：
   $$r(x, y) = \beta \log \frac{\pi_r(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$
   - **核心意义**：**任何一个对齐策略网络 $\pi_r$，都唯一、等价地对应着一个在数学上被最优化隐含的奖励函数 $r(x, y)$！**

---

## Slide 57: DPO derivation 2 (DPO 推导 2：DPO 损失函数)

![Slide 57](images/page_57.png)

### 讲解
我们将推导出的 implied reward 代入之前 Bradley-Terry 偏好模型训练 RM 时的交叉熵损失中：

$$\text{loss}(r_\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} [\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))]$$

1. 代入 implied reward：
   $$r_\theta(x, y_w) - r_\theta(x, y_l) = \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} + \beta \log Z(x) \right) - \left( \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} + \beta \log Z(x) \right)$$
2. 我们惊人地发现，**常数项 $\beta \log Z(x)$ 在做差时被完全消去（Cancel out）了！** 也就是说，我们根本不需要去计算那个在工程上无法穷举的归一化求和 $Z(x)$！
3. 代入后直接得到最终的 **DPO Objective 损失函数**：
   $$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

- **DPO 的关键步骤总结**：
  1. 采用非参数化假设，以闭式解链接策略 $\pi_\theta$ 和隐含奖励 $r$。
  2. 用策略网络概率比值参数化奖励 $r$。
  3. 将参数化后的隐含奖励直接套入监督偏好分类损失中训练。这在概念上就是**直接在成对偏好数据上执行极大似然估计（MLE）**。

---

## Slide 58: DPO updates and components (DPO 梯度与动力学机制分析)

![Slide 58](images/page_58.png)

### 讲解
为了在直觉上理解 DPO，我们对其损失函数求梯度（Gradient）：

$$\nabla_\theta \mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\beta \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \underbrace{\sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w))}_{\text{隐含的预测误差权重}} \left[ \underbrace{\nabla_\theta \log \pi_\theta(y_w \mid x)}_{\text{拉高好回复概率}} - \underbrace{\nabla_\theta \log \pi_\theta(y_l \mid x)}_{\text{拉低坏回复概率}} \right] \right]$$

其中 $\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$ 为隐含奖励。
- **机制剖析**：
  - 右侧项：直接对好回答 $y_w$ 做正向梯度更新，对坏回答 $y_l$ 做负向更新。
  - **隐含的预测误差权重项 (左侧 $\sigma$ 项)**：当模型对当前偏好对判定错误（即模型误以为坏回答得分高 $\hat{r}(y_l) > \hat{r}(y_w)$）时，分差为正，$\sigma$ 输出大，赋予极大的更新权重，惩罚模型；当模型判定正确且拉大分差时，$\sigma$ 输出小，几乎不产生梯度更新。这类似于**主动把学习焦点放在“错题集”上**。

---

## Slide 59: DPO in LLaMA (and other open models) (DPO 在 Llama 及开源模型中的落地工作流)

![Slide 59](images/page_59.png)

### 讲解
展示了 DPO 配合**拒绝采样（Rejection Sampling / Expert Iteration）**的现代工业级落地流程：
1. 从 Prompt 库中提取数据，利用上一轮迭代的最佳模型（Best models）进行采样生成。
2. 通过冻结的 Reward Model 运行拒绝采样（Rejection Sampling），过滤出胜出与失败的 Pairwise 数据。
3. 将这批对比数据喂给 DPO 进行监督梯度对齐，产生新的 Policy。
4. 这种组合极大地融合了“RM 的高检验效率”与“DPO 的离线训练稳定性”，被 Llama 3 采用。

---

## Slide 60: Variants (DPO 的衍生变体：SimPO 与 Length Normalized DPO)

![Slide 60](images/page_60.png)

### 讲解
SFT 与 RLHF 发展至今，涌现出许多 DPO 变体，以 Tulu 3 使用的两个最有名的变体为例：
- **SimPO (Simple Preference Optimization) [Meng et al. 2024]**：
  - **痛点**：DPO 依然需要加载一个冻结的 $\pi_{\text{ref}}$ 作为分母参考，这还是占用了两倍的显存。
  - **方案**：SimPO 彻底去除了 $\pi_{\text{ref}}$，改用长度归一化和硬边界间隔（Margin $\gamma$）惩罚：
    $$\mathcal{L}_{\text{SimPO}}(\pi_\theta) = -\mathbb{E} \left[ \log \sigma \left( \frac{\beta}{|y_w|} \log \pi_\theta(y_w \mid x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l \mid x) - \gamma \right) \right]$$
- **Length Normalized DPO**：
  - 在 DPO 比值上除以回复的 Token 长度（$|y_c|$ 和 $|y_r|$），以此来**强行遏制和抵消 RLHF 训练中模型为了刷分自发产生的“话痨膨胀效应”**。

---

## Slide 61: But PPO does too (and sometimes better?) (PPO 真的被 DPO 淘汰了吗？)

![Slide 61](images/page_61.png)

### 讲解
引自论文 *[Unpacking DPO and PPO] (AI2, 2024)* 与 Tulu 3 评测报告：
- **学术界观点**：DPO 简单好用，PPO 应该进垃圾堆。
- **现实的无情反转**：
  - **右侧 Tulu 3 实测表**：在相同的基底模型上进行偏好训练。
    - 初始 SFT 基底均分：55.7。
    - 优化后的 SimPO 得分：51.8 - 52.9（甚至不如 SFT）。
    - 优化后的 DPO 均分：55.2。
    - **PPO 优化后均分：54.5 - 55.5（PPO-ptx）**。
    - **Length-Normalized DPO 均分：57.3**。
  - **结论**：PPO 作为在线 RL 算法，其自发进行在线探索寻优的能力（即能探索出超出 SFT 数据限制的精妙解）是 DPO 这种纯离线拟合无法匹敌的。DPO 的极限往往被死死锁在 SFT 数据的上限里。

---

# Part 4: RLHF Side-Effects & Challenges RLHF 的副作用与挑战部分

 强化学习是强大的优化器，但它会以损害模型的其他隐性特性的形式，进行无情的“作弊式”刷分。

---

## Slide 62: Things to watch out for in RLHF (RLHF 运行中的两大警示)

![Slide 62](images/page_62.png)

### 讲解
在运行 RLHF 时，我们必须时刻监控并预防以下两大核心危机：
1. **过度优化（Overoptimization）/ 奖励劫持（Reward Hacking）**：模型发现了 RM 的漏洞。
2. **模式塌陷（Mode Collapse）/ 熵减退（Entropy Collapse）**：模型丧失了多样性，只会生成固定模板的无趣内容。

---

## Slide 63: Things to watch out for - Overoptimization (过度优化危机)

![Slide 63](images/page_63.png)

### 讲解
这页给出了极其经典的过度优化帕累托变化曲线图：
- **横轴**：当前 RL tuned policy 与初始 SFT policy 的 KL 散度（代表更新幅度、探索距离）。
- **纵轴**：不同评估器打出的模型 Win-rate 胜率得分。
- **观察**：
  - **(a) 人类偏好评分** 与 **(b) 嘈杂的 AlpacaFarm 评分**：在 KL 散度增加的初期（KL $\le 20$ 左右），模型表现稳步提升。但一旦越过拐点（KL 散度过大），**模型在人类和裁判眼中的实际表现发生断崖式下跌！** 模型进入了“Reward Hacking”荒原，开始通过拼凑能取悦 RM 神经网络但毫无逻辑的乱码来获得虚高分。
  - **(c) 无噪声的 GPT-4 评分（右图）**：曲线没有发生下降。这是因为 GPT-4 裁判与 RM 分类器具有相同的系统性系统偏见（Sytematic Bias），因此无法识别该越界行为。

---

## Slide 64: Things to watch out for - mode collapse (模式塌陷与校准度丧失)

![Slide 64](images/page_64.png)

### 讲解
这页展示了 RLHF 带来的另外两个严重退化指标：
- **校准度丧失（Loss of Calibration, 右上图）**：
  - 预训练和 SFT 模型在概率预测上是高度校准的（左图 model=pre-train，Accuracy 与预测概率呈完美的 45 度对角线）。
  - 经过 PPO 强化对齐后，**模型彻底丧失了概率校准度（右图 model=ppo）**。在很多高难度提问上，即使模型的真实准确率仅为 $50\%$，其输出首个 Token 的置信概率依然高达 $99\%$。这导致我们无法通过 Token Logits 的概率分布来判定模型的真实信心。
- **熵崩溃（Entropy Collapse, 下图）**：
  - 随着对齐进行，模型的概率熵值（Entropy）迅速向 0 偏移（深蓝色 text-davinci-003 高度集中在 0 附近），说明模型的多样性彻底崩溃。在相同的提问下，模型无论采样多少次，都只会给出那一套高度雷同的无趣回答。

---

## Slide 65: Recap of the lecture (课程全景总结)

![Slide 65](images/page_65.png)

### 讲解
这页对大模型对齐流程进行了核心回顾：
1. **偏好数据收集是极其复杂的工程**：面临话痨长度偏见、冷门事实幻觉、外包众包质量、人口统计学偏差以及肯尼亚 AI 底层道德剥削等多重物理与伦理挑战。
2. **对齐算法的实现非常具有挑战性**：在线强化学习算法 PPO 极其复杂且不易收敛。即使是看似简单稳定的离线 DPO 算法，其实际训练的上限也深受 SFT 数据的制约。
3. **时刻警惕“过度优化”与“熵减”**：模型在 RM 上的高分不代表真正变聪明，我们需要通过 KL 散度约束、长度归一化、SimPO 机制以及多样化的评估集来精心维持模型的综合素质。
