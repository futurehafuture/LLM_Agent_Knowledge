---
tags: [大模型, 推理, Continuous-Batching, Scheduler, vLLM]
created: 2026-07-13
updated: 2026-07-13
---

# Continuous Batching 与推理调度

> 大模型请求的输入、输出长度不同。若必须等整个 batch 全部结束，GPU 会被最慢的请求拖住；Continuous Batching 允许请求在 token 边界动态加入和离开。

## 1. 静态 Batching 的问题

设一个 batch 有三条请求，分别生成 5、20、100 个 token。静态 batch 要运行 100 步：

```text
A: █████ 完成后空闲
B: ████████████████████ 完成后空闲
C: ██████████████████████████████████████████████████...
```

A、B 完成后留下的槽位不能服务新请求，造成 padding 与 GPU 利用率浪费。

## 2. Continuous / In-flight Batching

每完成一个 decode iteration，调度器都可以：

1. 移除已结束或取消的请求；
2. 接纳新请求做 prefill；
3. 为活跃请求各生成一个或多个 token；
4. 根据显存和 token budget 调整运行集合。

因此 batch 不再是固定请求集合，而是随时间变化的序列集合。

## 3. 为什么按 token 预算而不是请求数

不同请求本轮工作量不同：一个 8K prompt 的 prefill 远大于一个 decode token。调度器常限制：

```math
\sum_{r\in\mathcal B} n_r^{scheduled}\le N_{token\_budget}.
```

在统一 token budget 下，Prefill、chunked prefill 和 decode 可以由同一个调度循环管理。

## 4. 调度目标之间的冲突

| 目标 | 倾向 |
|---|---|
| 最大吞吐 | 大 batch、延迟更多请求 |
| 低 TTFT | 尽快插入新 prefill |
| 低 TPOT | 优先保持 decode 连续运行 |
| 公平性 | 防止长 prompt 或低优先级请求饥饿 |
| 高缓存命中 | 优先调度共享 prefix 的请求 |

不存在对所有负载都最优的固定策略。

## 5. 常见策略

### FCFS

按到达顺序服务，简单且可解释，但长 prompt 可能阻塞后续短请求。

### Decode-first

先调度已进入 decode 的请求，稳定 TPOT；新请求的 TTFT 可能上升。

### Prefill-first

新请求快速获得首 token，但大 prefill 会让正在聊天的用户出现 token 卡顿。

### SLO-aware / Priority

按剩余延迟预算、业务优先级或预计长度调度。需要防止低优先级请求永久饥饿。

## 6. KV Cache 与准入控制

一个请求不仅占本轮计算预算，还会持续增长 KV Cache。准入时要估算：

```math
M_{future}=M_{current}+N_{max\_new}m_{KV/token}.
```

若仅按当前已生成长度接纳请求，后续可能集体触发 OOM。

## 7. 抢占策略

显存不足时可选择：

- **Recompute**：释放请求 KV，之后从 prompt 重新 prefill；
- **Swap**：把 KV 换到 CPU/NVMe，恢复时再传回；
- **Reject/Queue**：不接纳新请求；
- **Abort**：只用于超时、取消或策略允许的场景。

重算消耗算力，swap 消耗 PCIe/NVLink 带宽。哪种更好取决于 prompt 长度与系统负载。

## 8. Head-of-Line Blocking

队首超长 prompt 会阻塞后面大量短请求。Chunked Prefill 将其拆成多个片段，使调度器可以在片段之间插入 decode 或短 prefill。

## 9. 调度循环伪代码

```python
while True:
    release_finished_blocks()
    running = select_decode_requests(token_budget, kv_budget)
    prefills = admit_or_chunk_waiting_requests(remaining_budget)
    batch = build_model_input(running, prefills)
    logits = execute_model(batch)
    sample_and_stream(logits)
```

实际系统还要处理多卡同步、CUDA Graph batch shape、请求取消和 prefix cache。

## 10. 面试回答

> Continuous Batching 在每个 token iteration 动态加入新请求、移除已完成请求，避免静态 batch 被最长序列拖住。现代调度器通常按 token 和 KV 显存预算调度，而不是只按请求数。吞吐、TTFT、TPOT、公平性和缓存命中相互冲突，因此需要 chunked prefill、抢占和 SLO-aware 策略。

## 参考资料

- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- [vLLM V1 Scheduler](https://docs.vllm.ai/en/latest/api/vllm/v1/core/sched/scheduler.html)
- [TensorRT-LLM In-flight Batching](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html)
