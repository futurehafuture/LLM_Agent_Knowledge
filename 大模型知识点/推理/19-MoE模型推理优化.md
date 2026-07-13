---
tags: [大模型, 推理, MoE, Expert-Parallel, DeepEP]
created: 2026-07-13
updated: 2026-07-13
---

# MoE 模型推理优化

> MoE 每个 token 只激活少数 experts，计算量接近较小 dense 模型，但全部 expert 权重仍要存储并在设备间路由。

## 1. Router 与 Top-k

对 token 表示 $`h`$，router 产生 expert 分数：

```math
g=\mathrm{softmax}(W_rh).
```

选择 top-k experts：

```math
y=\sum_{e\in\mathrm{TopK}(g,k)}g_eE_e(h).
```

总参数量由所有 experts 决定，单 token 激活参数量只由 $`k`$ 个 experts 决定。

## 2. Expert Parallel

Experts 分散到不同 GPU。执行过程：

1. 本地 router 计算目的 expert；
2. 按目的 rank 对 token 排序；
3. All-to-All dispatch；
4. 每张卡执行本地 grouped GEMM；
5. All-to-All combine；
6. 恢复 token 原顺序。

通信和重排可能吞掉稀疏计算带来的收益。

## 3. 负载不均衡

若 batch 有 $`N`$ 个 token、$`E`$ 个 experts，理想每 expert 接收约：

```math
N_{ideal}=\frac{kN}{E}.
```

实际 router 分布不均，最慢 expert 决定 iteration 完成时间。小 batch decode 中 token 数少，统计波动更加严重。

## 4. Capacity 与 Token Drop

训练系统常设置 expert capacity；推理通常不能随意丢 token，否则改变输出。过载时需要动态 buffer、冗余 expert、路由约束或延迟部分 token。

## 5. Grouped GEMM

每个 expert 收到的 token 数不同，单独启动大量小 GEMM 效率很低。Grouped GEMM 将多个 expert 的不同尺寸矩阵乘在一次调度中执行，提高利用率。

## 6. 通信优化

- 节点内/节点间分层 All-to-All；
- dispatch 与上一层计算重叠；
- 低精度传输；
- fused permute + communication；
- topology-aware expert placement；
- DeepEP 等专用通信库。

## 7. Expert Placement 与复制

热门 experts 可以复制到多个 GPU，让 router 选择就近副本；冷 experts 可集中或 offload。代价是：

- 额外显存；
- 副本一致性和加载；
- 路由逻辑更复杂；
- 热点会随任务变化。

## 8. Prefill 与 Decode

Prefill token 多，expert batch 较大，grouped GEMM 更高效；Decode token 少，容易产生许多 tiny GEMM 和负载波动。高并发 continuous batching 对 MoE decode 尤其重要。

## 9. 与 TP/DP 组合

- EP 分布 experts；
- TP 可继续切每个大 expert；
- DP 复制整个 EP group；
- attention 层和 expert 层可能采用不同并行域。

复杂系统必须让 rank mapping 匹配 NVLink/IB 拓扑，避免每层跨慢链路 All-to-All。

## 10. 评测

除 TTFT/TPOT 外，还要观察：

- 每 expert token histogram；
- max/mean load ratio；
- All-to-All 时间；
- grouped GEMM 有效尺寸；
- token reorder 开销；
- 热点 expert 与跨节点流量。

## 11. 面试回答

> MoE 只激活 top-k experts，降低单 token FLOPs，但所有 expert 权重仍占容量。Expert Parallel 要通过 All-to-All 把 token 发给对应 experts，难点是路由负载不均、tiny GEMM 和通信。优化通常包括 grouped GEMM、拓扑感知 expert placement、通信计算重叠、热门 expert 复制和更大的有效 decode batch。

## 参考资料

- [DeepSpeed-MoE](https://arxiv.org/abs/2201.05596)
- [MegaBlocks](https://arxiv.org/abs/2211.15841)
- [DeepEP](https://github.com/deepseek-ai/DeepEP)
- [MegaScale-Infer](https://arxiv.org/abs/2504.02263)
