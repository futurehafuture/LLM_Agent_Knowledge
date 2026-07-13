---
tags: [大模型, 推理, CUDA-Graph, Operator-Fusion, Triton, CUTLASS]
created: 2026-07-13
updated: 2026-07-13
---

# CUDA Graph 与算子融合

> Decode 经常由大量短 kernel 组成。即使每个 kernel 计算很快，CPU launch、同步和中间 HBM 读写也可能成为显著开销。

## 1. Kernel Launch 开销

普通执行由 CPU 逐个提交 kernel：

```text
CPU launch → GPU kernel → CPU launch → GPU kernel → ...
```

单次 launch 只有微秒级，但一个 token 要经过几十层、每层多个 kernel。小 batch decode 中，累计空洞会明显。

## 2. CUDA Graph

CUDA Graph 先捕获一组固定 GPU 操作与依赖，后续一次 replay：

```text
capture once → replay graph many times
```

收益：

- 减少 CPU launch；
- 减少框架调度开销；
- 执行依赖更稳定；
- 降低 token 间抖动。

## 3. 动态请求如何进入固定图

图要求地址和大部分 shape 稳定。推理框架通常：

- 预分配 input/output/KV metadata buffer；
- 把本轮数据复制到固定地址；
- 为若干 batch size 分别 capture；
- 用 padding 或 piecewise graph 覆盖动态 shape。

捕获更多 shape 会增加显存和预热时间。

## 4. 算子融合为什么有效

两个算子分开执行：

```text
Kernel A: HBM → compute → HBM
Kernel B: HBM → compute → HBM
```

融合后中间值可留在寄存器/共享内存：

```text
Fused: HBM → A → B → HBM
```

融合减少 launch 和中间读写，尤其适合 memory-bound 的 elementwise 链。

## 5. LLM 常见融合

- RMSNorm + quantization；
- QKV projection + bias/reshape；
- RoPE；
- attention decode；
- residual + norm；
- gate/up projection + SiLU + multiply；
- logits processing + sampling；
- communication + GEMM。

大 GEMM 本身通常由高度优化库执行，不应为了“融合”而牺牲 Tensor Core 利用率。

## 6. Triton、CUTLASS 与库 kernel

- **cuBLAS/cuBLASLt**：成熟 GEMM 与 heuristic；
- **CUTLASS/CuTe**：构建硬件专用矩阵 kernel；
- **Triton**：用 Python-like DSL 写可调 tile kernel；
- **FlashInfer/FlashAttention**：专用 attention 与采样 kernel；
- **torch.compile/Inductor**：图级捕获、融合和代码生成。

选择原则不是“手写一定更快”，而是现有库是否覆盖当前 shape、dtype 与 layout。

## 7. Layout 与预打包

低精度权重常需要特定 tile/layout。离线预打包可以把重排成本从每次推理移到加载阶段。若框架在运行时反复 transpose/contiguous，可能抵消量化收益。

## 8. Persistent Kernel

Persistent kernel 让线程块长期驻留，内部循环处理多层或多个 token，减少 launch 和权重/元数据重复加载。代价是实现复杂、寄存器压力大、对模型结构和 shape 更专用。

## 9. 如何定位

使用 Nsight Systems 看：

- CPU 与 GPU 时间线空洞；
- kernel launch 间隔；
- NCCL 与计算是否重叠；
- H2D/D2H 拷贝。

使用 Nsight Compute 看：

- Tensor Core 利用；
- HBM throughput；
- occupancy、register spill；
- warp stall 原因。

## 10. 面试回答

> CUDA Graph 把固定的一组 GPU 操作一次捕获、多次 replay，主要减少 decode 中大量小 kernel 的 CPU launch 与调度开销。算子融合则把中间张量留在片上，减少 launch 和 HBM 往返。动态图通过预分配 buffer、多个 graph bucket 或 piecewise capture 适配，但 graph 覆盖越多，显存和预热成本越高。

## 参考资料

- [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [PyTorch CUDA Graphs](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)
- [OpenAI Triton](https://triton-lang.org/)
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)
