---
tags: [大模型, 推理, Benchmark, AIPerf, Goodput, SLO]
created: 2026-07-13
updated: 2026-07-13
---

# 推理性能评测与 Benchmark

> 一个可信 benchmark 必须固定模型质量、硬件、精度和工作负载，并同时报告吞吐与延迟分布。单个 “tokens/s” 数字通常不足以做部署决策。

## 1. 先写清实验配置

### 模型

- checkpoint、revision；
- tokenizer/chat template；
- dtype/量化 recipe；
- max context；
- speculative draft；
- LoRA/多模态配置。

### 硬件

- GPU 型号、数量、显存；
- NVLink/NVSwitch/PCIe；
- 节点间网络；
- CPU、RAM；
- 驱动、CUDA、框架版本。

### 引擎

- vLLM/SGLang/TRT-LLM 版本；
- attention backend；
- TP/PP/DP/EP；
- max batched tokens/sequences；
- chunked prefill、prefix cache、CUDA Graph。

## 2. Workload 才是 benchmark 的一半

至少定义：

- 输入长度分布；
- 输出长度分布；
- 到达过程；
- 并发/请求率；
- prefix 共享率；
- stop/取消比例；
- 多轮与多模态比例。

固定 128-in/128-out 的结果不能代表 32K RAG 或长 CoT。

## 3. Offline 与 Online

### Offline throughput

预先准备所有请求，让引擎尽快完成。适合测最大吞吐，不代表交互延迟。

### Online serving

请求按时间到达，测排队和尾延迟。推荐用开环泊松到达：

```math
\Delta t\sim\mathrm{Exponential}(\lambda).
```

这样可扫描系统从低负载到过载的变化。

## 4. 核心指标

- TTFT P50/P95/P99；
- TPOT/ITL P50/P95/P99；
- E2E latency；
- input/output TPS；
- RPS；
- Goodput；
- 错误、超时、拒绝、取消；
- GPU memory、KV usage、功耗。

## 5. Goodput 曲线

设 SLO 为 $`S_{TTFT},S_{TPOT}`$。对不同请求率 $`\lambda`$ 运行压测，计算：

```math
G(\lambda)=\frac{N_{SLO\ satisfied}}{T}.
```

随着到达率增加，吞吐先上升；接近饱和后排队和尾延迟恶化，Goodput 可能下降。应比较最大可持续 Goodput，而不是只比较峰值 TPS。

## 6. Warm-up 与稳态

预热应覆盖：

- 权重加载与 page fault；
- CUDA context；
- kernel autotune/JIT；
- CUDA Graph capture；
- prefix/adapter cache 的冷暖状态。

要明确测冷启动还是稳态，二者不能混在一起平均。

## 7. 正确性与性能必须一起测

量化、投机解码、约束输出和不同 kernel 可能影响数值。至少检查：

- greedy 输出或 logits 差异；
- benchmark 任务准确率；
- perplexity/长上下文能力；
- JSON 合法率；
- speculative 分布一致性；
- 错误和 NaN。

错误输出很快没有意义。

## 8. 常见作弊或误导

1. 一个系统测 output TPS，另一个测 total TPS。
2. 不同输入/输出长度。
3. 一边量化、一边 BF16，却不报告质量。
4. 只报告平均延迟，不报告 P99。
5. 忽略 OOM/拒绝请求。
6. 用闭环压测掩盖排队崩溃。
7. 把 prefix cache 热命中结果与冷随机 prompt 比较。
8. 客户端成为瓶颈，却误判服务器吞吐。

## 9. Benchmark 工具

- **AIPerf**：生成式 AI 服务的综合压测与报告；
- **vLLM bench**：serve/throughput/latency；
- **SGLang benchmark**：框架与 serving 测试；
- **MLPerf Inference**：标准化 datacenter/edge 场景；
- **Nsight Systems/Compute**：不是负载工具，而是 kernel/时间线诊断工具。

工具只是请求发生器与统计器，实验设计仍由使用者负责。

## 10. 推荐报告模板

```text
模型/精度：
硬件/互联：
框架/commit：
并行与关键参数：
输入/输出长度分布：
到达率/并发：
缓存冷暖：
TTFT P50/P95/P99：
TPOT P50/P95/P99：
Output TPS / RPS：
Goodput 与 SLO：
错误率：
质量验证：
```

## 11. 面试回答

> 推理 benchmark 必须固定模型、精度、硬件、引擎和输入输出长度分布。Offline 测最大吞吐，Online 应按请求率测试排队与尾延迟。至少报告 TTFT、TPOT/ITL、E2E、output TPS、RPS、P95/P99 和 Goodput，并把 OOM、拒绝和质量退化算进去；否则 tokens/s 很容易产生误导。

## 参考资料

- [AIPerf](https://github.com/ai-dynamo/aiperf)
- [vLLM Benchmarking](https://docs.vllm.ai/en/latest/cli/bench/)
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
