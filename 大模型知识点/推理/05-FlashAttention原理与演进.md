---
tags: [大模型, 推理, FlashAttention, IO-Aware, CUDA]
created: 2026-07-13
updated: 2026-07-13
---

# FlashAttention 原理与演进

> FlashAttention 没有近似 Attention，也没有改变模型输出；它通过 IO-aware 分块减少 HBM 读写。

## 1. 标准 Attention 的内存问题

```math
O=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt d}+M\right)V.
```

朴素实现通常分成多个 kernel：

1. 计算 $`S=QK^\top`$；
2. 写出 $`S\in\mathbb R^{N\times N}`$；
3. 读取 $`S`$ 做 mask 与 softmax；
4. 再写出概率矩阵 $`P`$；
5. 读取 $`P,V`$ 计算输出。

当 $`N`$ 很大时，$`N^2`$ 中间矩阵的 HBM 读写比计算本身更昂贵。

## 2. 分块思想

FlashAttention 把 Q、K、V 切成能放入 SRAM/共享内存的小块。每次加载一个 Q 块与若干 K/V 块，在片上完成局部矩阵乘、softmax 更新和输出累加，不把完整 $`S`$、$`P`$ 写回 HBM。

难点是 softmax 分母依赖整行，不能简单独立计算各块。

## 3. Online Softmax 推导

对已经处理的元素维护最大值 $`m`$、指数和 $`l`$：

```math
m=\max_j s_j,\qquad l=\sum_j e^{s_j-m}.
```

加入新块，其最大值和指数和为 $`m_b,l_b`$。合并后：

```math
m'=\max(m,m_b),
```

```math
l'=e^{m-m'}l+e^{m_b-m'}l_b.
```

输出累加量也按相同系数重新缩放，因此扫描完所有 K/V 块后得到与完整 softmax 完全相同的结果。

## 4. IO 复杂度

朴素 attention 会物化两个 $`N^2`$ 矩阵；FlashAttention 主要读写 Q/K/V/O，显著降低 HBM traffic。它的优势来自减少数据移动，而不是减少理论 FLOPs。

这也是为什么短序列、小 head dimension 或旧硬件上，加速比可能不明显。

## 5. 反向传播为什么不保存概率矩阵

训练时可保存每行 softmax 的归一化统计量，在 backward 中重新计算局部 $`S`$、$`P`$。这是用额外 FLOPs 换取更少的显存读写和激活保存，整体反而更快。

## 6. 演进路线

### FlashAttention-1

- 提出 IO-aware exact attention；
- 分块与 online softmax；
- 大幅降低 $`N^2`$ 中间张量访问。

### FlashAttention-2

- 改进线程块与 warp 的工作划分；
- 减少非矩阵乘 FLOPs；
- 提升并行度和 GPU 占用率。

### FlashAttention-3

- 面向 Hopper；
- 利用 WGMMA、TMA 与 warp specialization；
- 重叠数据搬运、matmul 和 softmax；
- 引入更稳健的 FP8 路径。

### FlashAttention-4

- 面向更新一代硬件继续做算法—流水线协同设计；
- 重点不只是单个 tile，而是跨阶段 pipeline、异步执行与硬件专用数据路径。

## 7. Prefill 与 Decode 的区别

- Prefill 的 Q 长度和 KV 长度都大，FlashAttention 的二维分块收益明显。
- Decode 的 Q 长度通常为 1，瓶颈更偏 KV 读取，需要专门的 paged decode attention、FlashInfer 或 FlashMLA kernel。

所以“用了 FlashAttention”不能自动说明 decode 已最优。

## 8. 与 PagedAttention 的区别

| 技术 | 解决的问题 |
|---|---|
| FlashAttention | 一个 attention 算子内部如何少访问 HBM |
| PagedAttention | 多请求 KV Cache 如何分页分配与寻址 |

服务框架通常同时使用两者：调度器用分页管理 KV，kernel 按 block table 读取 KV 并完成 attention。

## 9. 常见误区

1. FlashAttention 不是稀疏注意力，也不是近似算法。
2. 它不会把 attention 理论复杂度从 $`O(N^2)`$ 变成 $`O(N)`$。
3. 显存峰值下降不代表 KV Cache 消失。
4. FA3/FA4 与特定硬件能力紧密相关，不是所有 GPU 都能得到同样收益。

## 10. 面试回答

> 朴素 attention 会把 $`N\times N`$ 的 score 和 probability 写入 HBM。FlashAttention 将 Q/K/V 分块放入片上存储，通过 online softmax 维护每行最大值和归一化因子，边扫描 K/V 边累加输出，因此结果精确不变，但显著减少 HBM IO。FA2 优化并行划分，FA3/FA4 进一步利用新 GPU 的异步流水线和低精度能力。

## 参考资料

- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3](https://arxiv.org/abs/2407.08608)
- [FlashAttention-4](https://arxiv.org/abs/2603.05451)
