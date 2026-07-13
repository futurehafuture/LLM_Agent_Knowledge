---
tags: [大模型, 推理, SGLang, RadixAttention, Structured-Generation]
created: 2026-07-13
updated: 2026-07-13
---

# SGLang 与 RadixAttention

> SGLang 面向的不只是单次 completion，而是包含多次生成、分支、工具调用和共享前缀的语言模型程序。

## 1. 两层结构

- **Frontend**：描述生成、选择、并行、结构化输出和控制流；
- **Runtime**：调度请求、复用 KV、执行模型、管理缓存和分布式 worker。

这样可以让运行时看到整个 LM program 的结构，而不是把每次调用当作毫无关系的请求。

## 2. 为什么 Agent 工作流有大量重复

```text
System Prompt
  ├─ 工具结果 A → 继续推理
  ├─ 工具结果 B → 继续推理
  └─ 多候选回答 → verifier
```

各分支共享很长的前缀。若每条路径都重新 prefill，计算和 KV 显存会重复。

## 3. Radix Tree

RadixAttention 把 token 序列作为 radix tree 的路径，节点对应可复用 KV 段。插入新请求时找到最长公共前缀：

```math
L=\mathrm{LCP}(x^{new},\ tree).
```

前 $`L`$ 个 token 复用 KV，只计算后缀。请求结束后，未被引用的节点可以留在 LRU cache 中供未来请求命中。

## 4. 与块哈希缓存对比

| Radix tree | Block hash |
|---|---|
| 显式保存前缀层次关系 | 用父 hash 链接块 |
| 易表达分支工作流 | 易与 paged block allocator 结合 |
| 树操作与节点拆分较复杂 | hash lookup 简洁 |

二者本质都利用相同 token prefix 的 KV 等价性，工程数据结构不同。

## 5. Cache-aware Scheduling

调度器可以优先选择与当前 GPU cache 重合更多的请求，减少 prefill；但过分追求命中率可能让早到请求或冷 prefix 饥饿。因此要在：

- prefix 命中收益；
- 请求等待时间；
- SLO；
- KV 容量与淘汰成本

之间权衡。

## 6. Structured Generation

SGLang runtime 可将 JSON Schema/正则/CFG 编译为约束状态机，并在每步 mask 非法 token。压缩状态机、GPU-friendly mask 和与 batch 调度融合，能避免纯 Python 逐 token 处理成为瓶颈。

## 7. 多模态与多模型能力

现代 SGLang runtime 还涉及：

- 多模态 encoder 输入与缓存；
- TP/DP/EP；
- speculative decoding；
- quantization；
- PD disaggregation；
- multi-LoRA；
- reasoning/tool-call parser。

学习时应把 RadixAttention 视为核心切入点，而不是它的全部能力。

## 8. 与 vLLM 的理解方式

不要简单比较“谁更快”。应比较：

- 具体版本和 backend；
- 模型与量化格式；
- prefix 共享程度；
- structured output/Agent 工作流；
- 输入输出长度和并发；
- TTFT/TPOT SLO。

相同框架在不同工作负载下排名可能逆转。

## 9. 最小实践路线

1. 部署同一个模型；
2. 先测随机 prompt 基线；
3. 再构造共享 system prompt；
4. 观察 cache hit rate 和 TTFT；
5. 构造分支生成或多轮 Agent；
6. 加入 JSON Schema，测约束解码开销。

## 10. 面试回答

> SGLang 将多次 LM 调用视为可优化的程序。RadixAttention 用 radix tree 组织 token prefix 和对应 KV，自动复用多轮、分支和 few-shot 请求的公共前缀，并用 LRU 管理未被引用的缓存。它特别适合 Agent 和 structured generation，但调度器仍需在缓存命中、公平性和 SLO 之间权衡。

## 参考资料

- [SGLang Paper](https://arxiv.org/abs/2312.07104)
- [SGLang Documentation](https://docs.sglang.ai/)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
