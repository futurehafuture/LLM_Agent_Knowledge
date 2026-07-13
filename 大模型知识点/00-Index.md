---
tags: [面试, 大模型, 索引]
created: 2026-05-09
updated: 2026-07-13
---

# 大模型岗位面试准备 · 索引

> 文档按 **模型架构 / 训练 / 推理 / 算法题** 四大类组织到子文件夹。Obsidian `[[PageName]]` 链接不受文件夹影响。

---

## 一、模型架构 · `模型架构/`

> 总览见 [[模型架构/00-模型架构-总览]]

### 通用架构

- **[[01-MHA-多头注意力]]** —— Self-Attention 本质（Q/K/V 数据库类比）；为什么多头；完整计算流程；$`\sqrt{d_k}`$ 缩放原理；PyTorch 手撕；KV Cache 引出动机。
- **[[02-GQA-分组查询注意力]]** —— MHA→MQA→GQA 演进；KV Cache 推理瓶颈；三种机制参数/Cache/性能对比；分组逻辑与实现；各模型方案选型。
- **[[03-MLA-多头潜在注意力]]** —— DeepSeek-V2/V3 用的注意力；低秩压缩 K/V；吸收矩阵 trick；RoPE 解耦设计；KV Cache 比 GQA 还小，性能不降反升。
- **[[04-RoPE-旋转位置编码]]** —— Attention 置换不变性；Sinusoidal PE→RoPE 演进；2D 旋转矩阵推导（$`R(a)^T R(b)=R(b-a)`$）；高维分组旋转；base 频率与长文本外推；代码实现。
- **[[05-位置编码全景]]** —— Sinusoidal / Learned / Transformer-XL / T5 RB / ALiBi / RoPE / RoPE+PI/NTK/YaRN/DCA 完整演进；面试"位置编码有哪些"的标准答案。
- **[[06-ViT与多模态位置编码]]** —— ViT 1D PE；二维位置编码；Swin 相对位置 bias；RoPE 2D；MRoPE / Interleaved MRoPE（Qwen2-VL / Qwen3-VL）；动态分辨率与视频时序对齐。

### 具体模型

- **[[07-Qwen3-模型架构]]** —— Dense 与 MoE 双线；核心改进（去 QKV Bias / QK-Norm / RoPE base 1M / YaRN+DCA）；MoE 128 专家+Top-K 路由+无共享专家；Thinking Mode 软硬切换；与 Qwen2.5 对比。

### 多模态

- **[[08-Qwen3-VL核心创新点]]** —— Qwen-VL 系列演进；Interleaved-MRoPE / DeepStack / Text-Timestamp Alignment；Naive Dynamic Resolution、Window Attention ViT、Dense+MoE 双线；训练流程；与 GPT-4o/Gemini/Claude/InternVL 对比。

---

## 二、训练 · `训练/`

> 总览见 [[训练/00-训练-总览]]

### 基础训练

- **[[01-交叉熵损失详解]]** —— 信息论解释；CE = KL + 熵；softmax + CE 梯度推导；为什么分类不用 MSE；大模型 next-token CE；手撕代码（含数值稳定 log-sum-exp）；fp32 cast、ignore_index、shift 一位等工程细节。
- **[[02-AdamW-优化器]]** —— SGD→Momentum→RMSProp→Adam 演进；EMA 公式深入拆解（$`m_t`$/$`v_t`$ 为什么长这样、$`m_0`$/$`v_0`$ 初始化问题）；Adam 的 weight decay 与自适应 lr 耦合；AdamW 解耦方案；手撕代码 + 面试追问。
- **[[03-BF16-FP16-FP32精度详解]]** —— 三种浮点格式的位宽/指数/尾数对比；动态范围 vs 精度权衡；混合精度训练原理；loss scaling；为什么 LLM 训练用 BF16。
- **[[04-Dropout和权重衰减]]** —— Dropout 训练/推理差异；Inverted Dropout；L1/L2 weight decay 原理；与 AdamW 的关系；适用场景对比。

### 微调

- **[[05-LoRA微调与量化技术]]** —— LoRA 低秩分解原理；$`\Delta W = AB`$ 为什么能减少参数；QLoRA（4-bit+LoRA）；量化基础（GPTQ/AWQ）；PEFT 选型对比。

### RLHF / 对齐

- **[[06-SFT和RL区别]]** —— 监督微调 vs 强化学习的核心区别；训练信号来源；适用场景；进入 RLHF 的概念铺垫。
- **[[07-策略梯度-Policy-Gradient-基础]]** —— 策略梯度从零入门：MDP 框架、策略梯度定理推导、REINFORCE、Baseline 与 Advantage、Actor-Critic、GAE、到 PPO 的演化路线；全程附具体例子，适合 RL 零基础。
- **[[08-PPO-DPO-GRPO-DAPO-GSPO演进对比]]** —— RLHF 算法演进主线：PPO 经典 → DPO 砍掉 RL → GRPO 砍掉 Critic（DeepSeek-R1） → DAPO 工程改进（字节 Seed） → GSPO sequence-level ratio（Qwen3）；对照表 + 每种算法的核心痛点与修复。
- **[[09-PPO损失函数详解]]** —— PPO 损失四部分（clip / value / entropy / KL）；ratio、advantage、returns、GAE 是什么；大模型 RLHF 里怎么算（稀疏 reward + token-level KL）；Critic MSE；伪代码完整训练循环。

---

## 三、推理 · `推理/`

> 总览见 [[推理/00-推理-总览]]

### 基础与性能模型

- **[[01-KV-Cache-为什么不缓存Q]]** —— causal attention、Prefill/Decode、KV 显存公式与只缓存 K/V 的数学原因。
- **[[02-LLM推理完整流程与性能指标]]** —— 请求全流程；TTFT、TPOT、ITL、吞吐、Goodput 与 SLO。
- **[[03-Prefill与Decode性能模型]]** —— Roofline、算术强度；Prefill compute-bound 与 Decode memory-bound。
- **[[04-LLM解码与采样算法]]** —— Greedy、Temperature、Top-k/Top-p、Beam Search 与采样系统开销。
- **[[05-FlashAttention原理与演进]]** —— IO-aware exact attention、online softmax 与 FA1—FA4。
- **[[06-PagedAttention与KV内存管理]]** —— KV 分页、Block Table、引用计数与 Copy-on-Write。

### 调度与加速

- **[[07-Continuous-Batching与推理调度]]**、**[[08-Chunked-Prefill与长Prompt调度]]** —— token budget、动态 batch、长 Prefill 干扰与调度权衡。
- **[[09-Prefix-Caching与RadixAttention]]** —— 前缀 KV 复用、块哈希、Radix Tree 与缓存淘汰。
- **[[10-大模型量化推理]]** —— GPTQ、AWQ、SmoothQuant、FP8/FP4 与 KV 量化。
- **[[11-投机解码-Speculative-Decoding]]** —— 分布保持证明、Draft、Medusa、EAGLE 与 MTP。
- **[[12-分布式推理与模型并行]]**、**[[13-Prefill-Decode分离部署]]**、**[[14-长上下文推理优化]]** —— 多卡并行、KV 传输、PD 分离与长上下文。

### 框架与生产系统

- **[[15-vLLM内部架构详解]]**、**[[16-SGLang与RadixAttention]]**、**[[17-TensorRT-LLM推理优化]]** —— 三类主流推理运行时的架构与核心优化。
- **[[18-CUDA-Graph与算子融合]]**、**[[19-MoE模型推理优化]]** —— Kernel launch、融合、Expert Parallel 与 All-to-All。
- **[[20-结构化输出与约束解码]]**、**[[21-Multi-LoRA推理服务]]**、**[[22-多模态模型推理]]** —— 约束生成、多租户 adapter 和多模态专项。
- **[[23-推理服务架构与生产部署]]**、**[[24-推理性能评测与Benchmark]]** —— 路由、限流、扩缩容、故障、Benchmark 与 Goodput。

---

## 四、算法题 · `算法题/`

> 总览见 [[算法题/00-算法题-总览]]

- **[[面试准备/大模型知识点/算法题/32-最长有效括号]]** —— LC Hard。栈解法（哨兵 -1）/ DP 解法 / 双向扫描 O(1) 空间，三种方法对比；变体题。
- **[[面试准备/大模型知识点/算法题/33-搜索旋转排序数组]]** —— LC Medium。二分关键性质"任意切点必有一段有序"；边界 `<=` vs `<`；含重复变体（LC 81）；找旋转点变体（LC 153/154）。

---

## 待补充（占位）

| 分类    | 待补充主题                                                                                          |
| ----- | ---------------------------------------------------------------------------------------------- |
| 模型架构  | MoE（Switch Transformer / GShard / DeepSeek MoE）、Layer Norm vs RMS Norm、激活函数 GeLU/SwiGLU        |
| 训练    | Mixed Precision / ZeRO / FSDP / Megatron TP+PP、Flash Attention 1/2/3、梯度累积 / 梯度检查点              |
| 推理    | CPU/端侧推理、异构加速器与能耗优化可作为后续专题 |
| 多模态   | CLIP / BLIP / LLaVA / Flamingo / Q-Former                                                      |
| Agent | ReAct / Reflexion / Tool Use / Function Calling / MCP                                          |
| 算法题   | 高频 hot 100、动态规划专题、二叉树专题、滑动窗口专题                                                                 |
