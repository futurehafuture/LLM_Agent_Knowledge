---
tags: [大模型, 推理, Prefill, Decode, Roofline, Arithmetic-Intensity]
created: 2026-07-13
updated: 2026-07-13
---

# Prefill 与 Decode 性能模型

> 目标：不用先跑 benchmark，也能判断一个推理阶段主要受算力、显存容量还是显存带宽限制。

## 1. 三种资源上限

GPU 推理同时受到三种约束：

1. **计算吞吐**：Tensor Core 每秒能完成多少 FLOPs；
2. **显存带宽**：HBM 每秒能搬多少字节；
3. **显存容量**：能否放下权重、KV Cache、临时张量和通信 buffer。

运行时间的粗略下界为：

```math
T\ge \max\left(\frac{F}{P_{peak}},\frac{M}{BW_{HBM}}\right),
```

其中 $`F`$ 是浮点运算量，$`M`$ 是必须从 HBM 搬运的字节数。

## 2. Arithmetic Intensity 与 Roofline

算术强度定义为：

```math
I=\frac{F}{M}\quad(\text{FLOPs/byte}).
```

硬件的转折点是：

```math
I^*=\frac{P_{peak}}{BW_{HBM}}.
```

- 若 $`I<I^*`$，搬数据先到上限，属于 memory-bound；
- 若 $`I>I^*`$，计算单元先到上限，属于 compute-bound。

## 3. 为什么 Prefill 通常 compute-bound

对线性层 $`Y=XW`$：

- $`X\in\mathbb R^{T\times d}`$；
- $`W\in\mathbb R^{d\times m}`$；
- 运算量约 $`2Tdm`$ FLOPs。

权重 $`W`$ 只需读取一次，却可被 $`T`$ 个 token 复用。随着 prompt 长度或 batch 增大，计算量按 $`T`$ 增长，而单位 token 分摊的权重读取成本下降，因此算术强度提高。

Prefill 还会计算近似 $`T\times T`$ 的 attention，但 FlashAttention 通过分块避免把完整注意力矩阵写回 HBM。

## 4. 为什么 Decode 通常 memory-bound

Decode 每一步只有一个新 token。线性层退化为矩阵向量乘：

```math
y=xW.
```

同一份权重几乎只服务一个 token，就要在下一层继续读取另一份权重。以参数量 $`N_p`$、权重每元素 $`b_w`$ 字节估算，单步至少需要读取：

```math
M_{weight}\approx N_pb_w.
```

理想化单 token 速度上限：

```math
\text{TPS}_{max}\lesssim\frac{BW_{HBM}}{N_pb_w+M_{KV/token}}.
```

这解释了为什么权重量化不仅省容量，也可能提高 decode 速度：每一步搬运的权重字节更少。

## 5. Batch 如何改变 Decode

若同时解码 $`B`$ 个请求，同一权重可服务 $`B`$ 个 token：

```math
Y=XW,\quad X\in\mathbb R^{B\times d}.
```

Batch 增大会提高算术强度和聚合吞吐，但每个请求可能因等待凑 batch 而增加延迟。Continuous Batching 的目标是在不长期等待的前提下动态维持足够大的有效 batch。

## 6. Attention 在 Decode 中的成本

第 $`t`$ 步需要当前 query 读取长度为 $`t`$ 的 K/V：

```math
F_{attn}=O(BHtd_h),\qquad M_{KV}=O(BHtd_hb_{kv}).
```

两者都随上下文线性增长，但算术强度仍低。长上下文、大 batch 时，KV Cache 带宽可能从次要成本变成主导成本。

## 7. 参数、KV 与临时内存预算

```math
M_{total}=M_{weights}+M_{KV}+M_{workspace}+M_{runtime}.
```

其中：

```math
M_{KV}=2BLT H_{kv}d_hb_{kv}.
```

不能把所有剩余显存都分给 KV：CUDA Graph、NCCL、采样、FlashAttention workspace 和显存碎片也需要空间。

## 8. 优化方法与瓶颈的对应关系

| 瓶颈 | 优先优化 |
|---|---|
| 权重带宽 | 权重量化、增大有效 batch、算子融合 |
| KV 带宽/容量 | GQA/MLA、KV 量化、PagedAttention、窗口注意力 |
| Prefill 算力 | FlashAttention、低精度 GEMM、Tensor Parallel |
| 调度干扰 | Chunked Prefill、Continuous Batching、PD 分离 |
| Kernel launch | CUDA Graph、算子融合、持久化 kernel |

## 9. 常见误区

1. **GPU utilization 高就代表快**：它可能只是有很多无效等待或小 kernel。
2. **参数量减半就一定快一倍**：还取决于量化 kernel、反量化和瓶颈位置。
3. **Prefill 永远 compute-bound**：极短 prompt、小 batch 或低效实现也可能带宽受限。
4. **Decode 只读权重**：长上下文下 KV Cache 读取同样关键。

## 10. 面试回答

> Prefill 对大量 token 做矩阵矩阵乘，权重可被重复使用，算术强度高，通常 compute-bound；Decode 每步只有一个 token，矩阵乘退化为 GEMV，必须反复搬运权重和历史 KV，通常 memory-bound。Batch 会提高权重复用和吞吐，但会影响排队延迟。

## 参考资料

- [Roofline: An Insightful Visual Performance Model](https://dl.acm.org/doi/10.1145/1498765.1498785)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [NVIDIA: Mastering LLM Techniques — Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
