---
tags: [大模型, 推理, Chunked-Prefill, Sarathi, Scheduling]
created: 2026-07-13
updated: 2026-07-13
---

# Chunked Prefill 与长 Prompt 调度

## 1. Prefill 与 Decode 为什么互相干扰

Prefill 是大矩阵乘，单次 kernel 时间长；decode 每个请求工作量小，但用户希望 token 均匀到达。如果将 32K prompt 的完整 prefill 插进 decode 流，所有活跃请求都可能在这段时间停止输出。

```text
无切分：Decode Decode [========== Long Prefill ==========] Decode
```

这会显著抬高 ITL/TPOT 尾延迟。

## 2. Chunked Prefill

把长度 $`N`$ 的 prompt 拆成大小最多为 $`C`$ 的片段：

```math
K=\left\lceil\frac{N}{C}\right\rceil.
```

每处理一个片段，就把它产生的 KV 写入缓存；下一片段可以把前面 KV 作为 prefix 继续计算。

```text
切分后：Decode [P1] Decode [P2] Decode [P3] Decode
```

## 3. 数学上为什么成立

第 $`k`$ 个 chunk 的 query 需要看到此前所有 token 的 K/V：

```math
O_k=\operatorname{softmax}\left(Q_k[K_{<k};K_k]^\top+M_k\right)[V_{<k};V_k].
```

历史 $`K_{<k},V_{<k}`$ 已在 cache 中，因此不需要重算历史 hidden state。注意当前 chunk 内仍要使用正确的 causal mask。

## 4. Chunk Size 的取舍

### Chunk 太大

- 单次 prefill 阻塞 decode 更久；
- TPOT/ITL 尾延迟恶化；
- 但矩阵更大，GPU 效率通常更高。

### Chunk 太小

- 更容易与 decode 穿插；
- TTFT 与吞吐可能因 kernel launch、调度和重复元数据处理变差；
- 小 GEMM 无法充分利用 Tensor Core。

所以 $`C`$ 是吞吐与尾延迟之间的控制旋钮。

## 5. Piggyback / Stall-free Batching

Decode 通常带宽受限，计算单元没有完全用满。可以把一部分 prefill token 与 decode token 合在同一 iteration，尝试利用空闲计算资源，同时限制每轮 prefill 量，避免长时间阻塞。

这不意味着两类工作没有干扰：它们仍共享 HBM、SM、KV allocator 和通信链路。

## 6. 与 Prefix Cache 的关系

若前 $`H`$ 个 token 已命中 prefix cache，只需对剩余 $`N-H`$ 个 token 做 chunked prefill：

```math
N_{compute}=N-H.
```

缓存命中减少总工作量；chunking 控制剩余工作的调度粒度。二者解决不同问题。

## 7. 与 PD 分离的关系

- 共置系统：chunking 用来减少 prefill 对 decode 的阻塞；
- PD 分离系统：prefill 不再占用 decode GPU，但仍可能需要 chunking 来控制 TTFT、公平性和 KV 传输粒度。

## 8. 调度例子

假设每轮 token budget 为 2048：

- 256 个活跃 decode 请求消耗 256 token；
- 剩余 1792 token 可用于一个或多个 prefill chunk；
- 若有高优先级 decode，可降低本轮 prefill 预算；
- 若 decode batch 很小，可扩大 chunk 提高算力利用率。

## 9. 正确评测

扫描不同 chunk size，并同时记录：

- TTFT P50/P99；
- TPOT/ITL P50/P99；
- input/output TPS；
- Goodput；
- GPU 利用率和 KV 占用。

只看吞吐通常会偏向过大的 chunk。

## 10. 面试回答

> Chunked Prefill 把长 prompt 拆成多个片段，每片生成的 KV 缓存供后续片段使用，因此数学结果不变。它让调度器在片段之间插入 decode，减轻长 prefill 对 TPOT 的阻塞。大 chunk 吞吐高但干扰强，小 chunk 延迟平滑但 kernel 效率低，需要按 SLO 和负载调节。

## 参考资料

- [Sarathi-Serve: Taming Throughput-Latency Tradeoff in LLM Inference](https://arxiv.org/abs/2403.02310)
- [vLLM Chunked Prefill](https://docs.vllm.ai/en/latest/configuration/optimization.html)
- [DeepSpeed-FastGen Dynamic SplitFuse](https://arxiv.org/abs/2401.08671)
