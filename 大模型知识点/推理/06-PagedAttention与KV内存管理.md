---
tags: [大模型, 推理, PagedAttention, KV-Cache, vLLM]
created: 2026-07-13
updated: 2026-07-13
---

# PagedAttention 与 KV 内存管理

> PagedAttention 的核心不是修改注意力公式，而是借鉴虚拟内存分页，让请求的 KV Cache 不必物理连续。

## 1. 连续分配为什么浪费

请求输出长度事先未知。若按最大长度连续预留：

```text
请求 A：[已用 KV][未用但已预留................]
请求 B：[已用 KV][未用但已预留........]
```

会产生：

- **内部碎片**：预留块内部未使用；
- **外部碎片**：空闲区分散，难以找到大连续空间；
- **过度预留**：为了避免扩容，只能按最大长度保守分配。

## 2. 逻辑块与物理块

将每个请求的 token 序列按固定 block size 切分：

```text
逻辑块：0  1  2  3
物理块：7  2  9  4
页表：  0→7, 1→2, 2→9, 3→4
```

逻辑上连续的 KV 可以散落在不同物理位置。请求只在需要新 token 时申请新块。

## 3. 地址映射

若每块容纳 $`B_s`$ 个 token，逻辑 token 位置 $`t`$ 对应：

```math
b=\left\lfloor\frac{t}{B_s}\right\rfloor,\qquad o=t\bmod B_s.
```

从 block table 找到物理块 $`p=\text{table}[b]`$，再读取块内偏移 $`o`$。

Paged attention kernel 必须完成这种间接寻址，因此 block size 太小会增加页表和不连续访存开销，太大又增加内部碎片。

## 4. Block 生命周期

1. Prefill 根据 prompt 长度申请若干块；
2. Decode 写满当前块后再申请一个；
3. 请求结束或被取消时递减引用计数；
4. 引用计数归零后回收到 free list；
5. 显存不足时，调度器可以抢占、重算或换出部分请求。

## 5. Copy-on-Write

Beam Search、parallel sampling 或共享前缀会让多条序列拥有相同历史 KV。可以让它们引用相同物理块：

```text
公共前缀块 P0 P1
       ↙       ↘
   分支 A2     分支 B2
```

只有当某分支要修改共享块时才复制，即 Copy-on-Write。由于历史 KV 通常只追加、不原地修改，共享尤其自然。

## 6. Prefix Caching

PagedAttention 提供“块”这一存储单位，Prefix Caching 再用 prompt token、模型配置和缓存盐值等计算 block hash。相同前缀可以直接复用已有物理块，跳过对应 prefill。

两者关系：

- PagedAttention：解决怎么存；
- Prefix Caching：解决哪些请求可以复用。

## 7. 与操作系统分页的异同

| 操作系统 | LLM KV 管理 |
|---|---|
| 虚拟页 | 请求逻辑 KV block |
| 物理页 | GPU KV block |
| 页表 | block table |
| 页错误 | 申请、换入或重算 KV |
| Copy-on-Write | Beam/并行采样共享前缀 |

不同点是 LLM 调度器知道请求长度、生成状态和 SLO，可以做比通用 OS 更有语义的决策。

## 8. PagedAttention 不等于零碎片

最后一个块仍可能只用一部分。若 block size 为 $`B_s`$，单请求最多浪费接近 $`B_s-1`$ 个 token 槽位，但不再按最大序列长度浪费。

此外还存在 block table、hash 元数据、对齐和 allocator 的成本。

## 9. 与 FlashAttention 配合

FlashAttention 假设高效分块计算；PagedAttention 允许 KV 物理不连续。服务 kernel 需要：

1. 读取 block table；
2. gather 对应 K/V block；
3. 对这些块执行 online softmax；
4. 写出当前 token 的 attention 结果。

因此框架常有独立的 prefill attention 与 paged decode attention kernel。

## 10. 面试回答

> 传统 KV Cache 为每个请求连续预留最大长度，产生严重碎片。PagedAttention 把 KV 切成固定 block，用 block table 将逻辑块映射到任意物理块，按需分配；请求结束即可回收，Beam 和共享前缀还能通过引用计数与 Copy-on-Write 复用块。它改善的是 KV 内存管理，不改变 attention 数学定义。

## 参考资料

- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [vLLM Paged Attention Design](https://docs.vllm.ai/en/latest/design/paged_attention/)
- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
