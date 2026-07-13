---
tags: [大模型, 推理, PD分离, DistServe, KV传输, SLO]
created: 2026-07-13
updated: 2026-07-13
---

# Prefill-Decode 分离部署

> PD 分离把 Prefill 和 Decode 放到不同 GPU 池，分别针对 TTFT 与 TPOT 优化；代价是必须把所有层的 KV Cache 从 Prefill 节点传给 Decode 节点。

## 1. 为什么考虑分离

共置系统把两种特性差异很大的工作混在一起：

| 阶段 | 常见瓶颈 | 主要 SLO |
|---|---|---|
| Prefill | 计算吞吐 | TTFT |
| Decode | 权重/KV 带宽 | TPOT/ITL |

长 prefill 会阻塞 decode；而为了 decode 配置的小 batch 又可能浪费 prefill 算力。

## 2. 基本架构

```text
          ┌────────────┐     KV transfer     ┌────────────┐
Request → │ Prefill GPU│ ──────────────────→ │ Decode GPU │ → Tokens
          └────────────┘                     └────────────┘
```

Router 需要选择 P 实例、D 实例，并记录请求状态。Prefill 完成后，Decode 必须拿到每层完整 KV 才能继续。

## 3. KV 传输成本

传输大小近似：

```math
S_{KV}=2LTH_{kv}d_hb_{kv}.
```

传输下界：

```math
T_{transfer}\ge\frac{S_{KV}}{BW_{network}}+T_{setup}.
```

长 prompt、MHA、大精度 KV 会显著放大成本；GQA、MLA、KV 量化能直接减轻 PD 网络压力。

## 4. TTFT 的新组成

```math
T_{TTFT}=T_{queue,P}+T_{prefill}+T_{transfer}+T_{queue,D}+T_{first\ decode}.
```

分离消除了 P/D GPU 干扰，却增加传输与第二次排队。若网络慢或负载轻，可能比共置更差。

## 5. 资源独立配置

P 池可以使用较大的 TP、较大 batch 和适合大 GEMM 的 GPU；D 池可以配置更多 replicas、更小 TP、低精度权重和更高 HBM 带宽。

还可以独立扩容：

- 输入长度或请求率上升：扩 P；
- 输出长度或并发会话上升：扩 D。

## 6. KV 传输路径

- GPU Direct RDMA；
- NVLink/NVSwitch 节点内传输；
- InfiniBand/RoCE 节点间传输；
- CPU staging（实现容易但通常更慢）；
- 分布式 KV store，允许其他 D 节点恢复或复用。

实现还要传 block table、位置、模型版本、采样状态等元数据。

## 7. Streaming Transfer

不一定等完整 prefill 全部结束再传。可以按层或按 KV block 流式发送，并将计算与网络传输重叠：

```math
T\approx\max(T_{prefill},T_{transfer})
```

这是理想上界；实际还受依赖、拥塞和接收端分配影响。

## 8. 路由与负载均衡

Router 不能只选最空闲 P/D，还要考虑：

- P 到 D 的网络拓扑；
- D 的可用 KV blocks；
- prefix cache 是否已在某节点；
- adapter/模型版本；
- 请求预计输入输出长度；
- SLO 剩余预算。

## 9. PD 分离并非总是更好

适合：

- 高负载且 P/D 干扰明显；
- 输入、输出长度差异大；
- 有高速互联；
- 需要独立 TTFT/TPOT 资源规划。

不适合：

- 单机或低并发；
- prompt 很短；
- KV 很大但网络较慢；
- 共置的 chunked prefill 已满足 SLO。

更新的系统研究开始在“聚合”和“分离”之间动态选择，而不是把 PD 分离视为绝对答案。

## 10. 故障处理

如果 Decode 节点在请求中途失败，仅有 Prefill KV 还不够恢复已生成的后缀。可选择：

- 持久化/复制增量 KV；
- 在新节点重放 prompt 与已生成 token；
- 让客户端重试；
- 对高价值长会话做检查点。

## 11. 面试回答

> PD 分离将 compute-bound 的 Prefill 和 memory-bound 的 Decode 放到不同 GPU 池，消除干扰并允许独立扩容、独立并行配置。代价是要传输所有层的 KV，TTFT 中新增传输和 Decode 排队。是否收益取决于 KV 大小、网络带宽、负载与 SLO，因此低负载或短 prompt 下共置往往更简单。

## 参考资料

- [DistServe](https://arxiv.org/abs/2401.09670)
- [Mooncake](https://arxiv.org/abs/2407.00079)
- [Splitwise](https://arxiv.org/abs/2311.18677)
- [TaiChi: Unifying PD Aggregation and Disaggregation](https://arxiv.org/abs/2508.01989)
- [NVIDIA Dynamo Disaggregated Serving](https://docs.nvidia.com/dynamo/latest/architecture/disaggregated_serving.html)
