---
tags: [大模型, 推理, vLLM, Scheduler, PagedAttention]
created: 2026-07-13
updated: 2026-07-13
---

# vLLM 内部架构详解

> vLLM 不只是一个 OpenAI-compatible server。它把调度、Paged KV、模型执行、采样和分布式 worker 组合成高吞吐推理引擎。

## 1. 请求路径

```text
API Server
  → Input Processor / Tokenizer
  → Engine Core
  → Scheduler
  → KV Cache Manager
  → Model Executor / Workers
  → Attention & GEMM Kernels
  → Sampler
  → Streaming Output
```

控制面决定“谁在本轮运行”，数据面负责 GPU tensor 与 collective。

## 2. Engine 与 Scheduler

Scheduler 为 waiting/running 请求分配每轮 token 数，并同时检查：

- token budget；
- 可用 KV blocks；
- priority/FCFS；
- prefix cache 命中；
- speculative token；
- encoder input；
- 抢占与请求完成状态。

V1 统一把 prompt token、output token 和 speculative token 看成要计算的 token，不再要求 Prefill/Decode 使用完全不同的调度抽象。

## 3. KV Cache Manager

职责包括：

- 初始化 GPU KV block pool；
- 请求到逻辑/物理 block 的映射；
- 分配、释放与引用计数；
- prefix block hash；
- 检查公共前缀；
- 多 KV cache group 的协调。

PagedAttention 是 kernel 侧概念，KV Cache Manager 是系统侧资源管理。

## 4. Model Runner

Model Runner 把调度结果转成 GPU 输入：

- input ids 与 positions；
- slot mapping / block table；
- attention metadata；
- multimodal/encoder features；
- LoRA 映射；
- sampling metadata。

它还负责 CUDA Graph capture/replay、buffer 复用和输出拷贝。

## 5. Worker 与分布式 Executor

单卡时可直接执行；多卡时 executor 在 TP/PP/DP ranks 上启动 workers。每个 worker 加载自身权重 shard，并在每层执行 collective。

常见后端包括单进程、多进程和 Ray。后端选择影响启动、故障隔离和多节点编排，但不改变模型数学。

## 6. Attention Backend

vLLM 会根据设备、dtype、head size、block layout 和特性选择 FlashAttention、FlashInfer、FlashMLA、Triton 或其他 backend。Prefill、decode、prefix cache 和 sliding window 可能走不同 kernel。

因此“vLLM 版本相同”仍不足以复现实验，还应记录 attention backend。

## 7. CUDA Graph

Decode 的 kernel 很小，CPU launch 开销明显。vLLM 对常见 batch shape 捕获 CUDA Graph，之后 replay。动态请求通过预分配 buffer 与 metadata 映射进固定图。

完全动态 shape 无法直接复用同一张图，因此要权衡 graph 覆盖范围与显存占用。

## 8. Chunked Prefill 与 Prefix Cache

- Prefix cache 先返回已计算的 KV blocks；
- Scheduler 只为缺失 token 分配计算预算；
- 长 prompt 可以分多个 iteration；
- Decode token 与 prefill chunk 可在同轮组成 mixed batch。

这些能力都依赖统一调度和 paged KV，而不是彼此孤立的开关。

## 9. Speculative Decoding

Scheduler 还要管理 draft token 的验证、接受 token 的 KV 提交以及拒绝后 block 回收。不同方法可能使用 n-gram、draft model、EAGLE 或其他 proposer。

## 10. 调试与性能定位

建议按层次检查：

1. API 排队和 tokenizer；
2. scheduler 决策与 batch token 数；
3. prefix cache hit rate、KV usage；
4. model execution 与 collective；
5. attention/GEMM kernel；
6. sampler 与 streaming。

不要看到 GPU 利用率低就直接归因于 kernel；也可能是调度、CPU 或网络供给不足。

## 11. 面试回答

> vLLM 的核心是 scheduler 与 paged KV manager：scheduler 每轮按 token/KV 预算选择请求，KV manager 用物理 blocks 保存和复用缓存，model runner 将 block table、positions 和请求状态转成 GPU 输入，workers 执行模型与 collective。Continuous batching、chunked prefill、prefix caching 和 speculative decoding 都建立在这一统一运行时之上。

## 参考资料

- [vLLM Documentation](https://docs.vllm.ai/en/latest/)
- [vLLM V1 Architecture](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)
- [vLLM Paper](https://arxiv.org/abs/2309.06180)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
