---
tags: [大模型, 推理, LoRA, Multi-LoRA, Punica, S-LoRA]
created: 2026-07-13
updated: 2026-07-13
---

# Multi-LoRA 推理服务

> Multi-LoRA 让多个租户或任务共享一份基础模型权重，只动态选择不同 LoRA adapter，避免每个 adapter 部署一整份模型。

## 1. LoRA 推理公式

对基础权重 $`W`$ 和 rank 为 $`r`$ 的 adapter：

```math
y=x(W+BA)=xW+(xB)A,
```

其中 $`A\in\mathbb R^{r\times d_{in}}`$、$`B\in\mathbb R^{d_{out}\times r}`$。通常 $`r\ll d`$。

基础项 $`xW`$ 可由 batch 中所有请求共享执行，但不同请求的 $`A,B`$ 不同。

## 2. 为什么普通 Batching 失效

若 batch 内每个请求使用不同权重 $`W_i=W+B_iA_i`$，不能直接用一个标准 GEMM 处理全部样本。逐请求启动 LoRA GEMM 又会产生大量 tiny kernels。

## 3. Segmented Gather Matrix-Vector Multiplication

Punica 等系统将 batch 按 adapter id 分段，对多个不同 LoRA 权重执行 grouped/segmented kernel：

```math
y_j=x_jW+(x_jB_{a(j)})A_{a(j)}.
```

$`a(j)`$ 表示第 $`j`$ 个 token 的 adapter。kernel 读取映射表，为每段选择对应低秩矩阵。

## 4. Adapter Cache

Adapter 比基础模型小，但数量可能成千上万。分层存储：

```text
对象存储/NVMe → CPU RAM → GPU adapter cache
```

调度器需要考虑：

- adapter 是否已在 GPU；
- 加载延迟；
- 热度与 LRU/LFU；
- adapter 大小/rank；
- 等待队列中相同 adapter 的请求。

## 5. Cache-aware Batching

优先合并相同 adapter 可提高 kernel 效率，却可能增加其他请求等待。调度目标要结合 adapter locality、SLO 和公平性。

## 6. 与 Prefix Cache 的交互

LoRA 会改变每层 hidden state 与 KV，因此 prefix cache key 必须包含 adapter id 和版本。不同 adapter 即使 prompt 相同，也不能复用同一 KV。

若 adapter 被热更新，旧 KV cache 必须隔离或失效。

## 7. 与量化的交互

常见组合是低精度基础权重 + FP16/BF16 LoRA。执行时要处理：

- 基础 GEMM 的反量化/低精度累加；
- LoRA 分支的高精度 GEMM；
- 两路输出的 scale 和 dtype；
- merged adapter 是否破坏动态切换。

## 8. 合并还是动态加载

- **Merge**：把 $`BA`$ 合进 $`W`$，单 adapter 性能简单，但切换需要新权重副本；
- **Dynamic**：基础权重共享，适合多租户，但需要专用 kernel 与 cache；
- **预热副本**：对极热门 adapter 建独立 merged replica，其余走动态路径。

## 9. 安全与隔离

- adapter 文件来源校验；
- 租户访问控制；
- adapter id 不可由用户任意映射到其他租户；
- 基础模型与 adapter 兼容性；
- 避免通过缓存时间泄露其他租户热度。

## 10. 评测

除了 TPS，还应扫：

- adapter 数量与 Zipf 热度；
- cache hit rate；
- adapter load P99；
- 同/异 adapter batch 比例；
- 不同 rank；
- TTFT/TPOT 与显存占用。

## 11. 面试回答

> Multi-LoRA 共享基础 GEMM $`xW`$，再按请求 adapter id 计算不同的低秩增量。难点是 batch 内权重不同，普通 GEMM 无法直接合并，因此使用 segmented/grouped LoRA kernel，并用 GPU/CPU/磁盘多级 adapter cache。Prefix KV 必须按 adapter 隔离，调度还要平衡 adapter locality 与请求 SLO。

## 参考资料

- [Punica](https://arxiv.org/abs/2310.18547)
- [S-LoRA](https://arxiv.org/abs/2311.03285)
- [vLLM LoRA](https://docs.vllm.ai/en/latest/features/lora.html)
