---
tags: [大模型, 推理, 性能指标, TTFT, TPOT, Goodput]
created: 2026-07-13
updated: 2026-07-13
---

# LLM 推理完整流程与性能指标

> 目标：从一个 API 请求进入服务开始，追踪它如何变成 token，并建立一套不会被“tokens/s”误导的性能指标体系。

## 1. 一次请求经历什么

```text
HTTP/gRPC 请求
  → 排队与准入控制
  → Tokenization
  → Prefix Cache 查询
  → Prefill
  → 首 token 采样
  → 多轮 Decode
  → Detokenization 与流式返回
```

### 1.1 Tokenization

字符串被分词器转换为 token id。它通常在 CPU 上执行，短请求中可能占据可见比例；高并发时还要考虑线程池、内存分配和超长输入校验。

### 1.2 Prefill

设输入长度为 $`N_{in}`$。Prefill 一次并行处理整个 prompt，建立每层 KV Cache，并产生最后位置的 logits。矩阵乘法规模大、并行度高，通常偏 **compute-bound**。

### 1.3 Decode

每一步只输入上一步生成的一个 token，读取模型权重和全部历史 KV Cache，生成一个新 token。单步计算规模小但读写量大，通常偏 **memory-bound**。

### 1.4 流式返回

服务端将 token 增量解码成文本并发送给客户端。网络缓冲、代理服务器和客户端渲染都会影响用户观察到的延迟，所以“GPU 延迟”和“端到端延迟”不能混为一谈。

## 2. 单请求延迟指标

### 2.1 TTFT：Time To First Token

从请求发出到收到第一个输出 token：

```math
T_{TTFT}=T_{queue}+T_{tokenize}+T_{prefix}+T_{prefill}+T_{sample}+T_{network}.
```

TTFT 对聊天、搜索和 Agent 交互最敏感。长 prompt、排队、冷启动都会把它拉高。

### 2.2 TPOT 与 ITL

- **TPOT**：生成阶段平均每个输出 token 的时间。
- **ITL**：相邻两个流式 token 到达客户端的间隔。

若输出 $`N_{out}`$ 个 token，可近似写成：

```math
T_{E2E}\approx T_{TTFT}+(N_{out}-1)T_{TPOT}.
```

平均值会掩盖抖动，因此还应报告 ITL 的 P50、P95、P99。

### 2.3 端到端延迟

E2E latency 包括排队、CPU、GPU、网络和客户端可见时间。离线 benchmark 若只测模型 forward，会低估真实系统延迟。

## 3. 系统吞吐指标

### 3.1 Token throughput

```math
\text{Output TPS}=\frac{\sum_i N_{out}^{(i)}}{T_{wall}}.
```

必须说明它是：

- 单请求 TPS 还是服务器聚合 TPS；
- 仅输出 token，还是输入加输出 token；
- 在什么并发、输入长度和输出长度下测得。

### 3.2 Request throughput

```math
\text{RPS}=\frac{N_{completed}}{T_{wall}}.
```

RPS 会受输出长度强烈影响。两个系统即使 TPS 相同，短请求较多的一方 RPS 也会更高。

### 3.3 Goodput

Throughput 统计所有完成请求；Goodput 只统计满足服务等级目标的请求：

```math
\text{Goodput}(SLO)=\frac{\#\{r:T_{TTFT}^{(r)}\le S_{TTFT},\ T_{TPOT}^{(r)}\le S_{TPOT}\}}{T_{wall}}.
```

生产系统追求的通常不是“GPU 尽量忙”，而是在延迟约束下完成尽可能多的请求。

## 4. 为什么只报平均 tokens/s 不够

假设系统 A 把大量请求塞进一个 batch，聚合吞吐很高，但新请求要等待很久；系统 B 吞吐略低，却能持续接纳请求。若只看 TPS 会认为 A 更好，用户却可能明显感觉 B 更快。

完整报告至少包含：

| 类别 | 建议指标 |
|---|---|
| 延迟 | TTFT、TPOT、ITL、E2E 的 P50/P95/P99 |
| 吞吐 | input TPS、output TPS、RPS |
| 负载 | 并发数、到达率、输入/输出长度分布 |
| 资源 | GPU 利用率、HBM 占用、KV Cache 使用率、功耗 |
| 质量 | SLO 达标率、Goodput、错误率 |

## 5. 开环与闭环压测

- **闭环**：固定并发用户；一个请求完成后才发下一个。服务变慢时到达率也会自动下降，容易掩盖过载。
- **开环**：按固定或泊松到达率发送请求，不等待上一个完成。更适合观察排队崩溃点。

排队系统接近饱和时，即使吞吐只增加一点，尾延迟也可能急剧上升。压测必须扫多个到达率，而不是只选一个并发数。

## 6. 实践检查表

1. 固定模型、精度、硬件和框架版本。
2. 预热模型和 CUDA Graph。
3. 使用真实或明确给出的长度分布。
4. 分开统计 TTFT 与 decode 延迟。
5. 同时报告吞吐和尾延迟。
6. 记录 OOM、拒绝和超时请求，不能只统计成功样本。

## 7. 面试回答

> TTFT 主要覆盖排队和 prefill，TPOT 主要描述 decode。吞吐量衡量系统完成了多少工作，但可能通过牺牲延迟获得；Goodput 只统计满足 TTFT/TPOT SLO 的请求，更适合生产推理服务。比较系统时必须固定输入输出长度、并发与精度，并报告 P95/P99。

## 参考资料

- [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving](https://arxiv.org/abs/2401.09670)
- [vLLM Benchmarking CLI](https://docs.vllm.ai/en/latest/cli/bench/)
- [AIPerf](https://github.com/ai-dynamo/aiperf)
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/)
