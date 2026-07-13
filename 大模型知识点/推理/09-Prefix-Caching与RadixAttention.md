---
tags: [大模型, 推理, Prefix-Caching, RadixAttention, SGLang]
created: 2026-07-13
updated: 2026-07-13
---

# Prefix Caching 与 RadixAttention

> 多个请求只要拥有完全相同的 token 前缀，就可以复用该前缀在每一层产生的 KV Cache，跳过重复 prefill。

## 1. 常见重复前缀

- 相同 system prompt；
- few-shot 示例；
- 多轮对话的历史消息；
- RAG 中相同长文档、不同问题；
- Tree-of-Thought、多候选采样共享根节点。

## 2. 正确性条件

对 causal Transformer，前缀 token 的 hidden state 只依赖其左侧 token。若两个请求的以下条件一致，则前缀 KV 相同：

- token ids 与顺序；
- 模型权重和 adapter；
- RoPE/位置配置；
- dtype/量化与可能影响 KV 的模型参数；
- 多模态输入的内容与处理结果。

不能仅按原始字符串缓存，因为不同 tokenizer 配置可能得到不同 token。

## 3. 块哈希 Prefix Cache

将 token 按 KV block 切分，第 $`i`$ 块哈希可以链式计算：

```math
h_i=H(h_{i-1},\ tokens_i,\ model\_id,\ adapter\_id,\ extras).
```

只有完整块通常进入可复用缓存；请求最后一个未填满块仍由当前请求私有持有。

命中时，调度器把对应物理 KV blocks 映射进新请求，并只 prefill 未命中的后缀。

## 4. RadixAttention

SGLang 用 radix tree 组织 token prefix：

```text
          [system prompt]
            /        \
       [question A] [question B]
          /    \
     [branch1] [branch2]
```

树的公共路径对应共享 KV。Radix tree 能自然表示任意长度的公共前缀，不局限于“整个 prompt 完全相同”。

## 5. 缓存收益

若 prompt 长 $`N`$、命中前缀 $`H`$：

```math
\text{saved prefill tokens}=H,\qquad \text{new work}=N-H.
```

它主要降低 TTFT 与 prefill 算力，对之后每个 decode token 的模型 forward 成本没有直接减少。

## 6. 淘汰策略

KV Cache 容量有限，常用 LRU 或成本感知策略。简单 LRU 未考虑：

- 前缀长度越长，重算成本越高；
- 热门根节点被更多后代共享；
- 某些条目即将被正在排队的请求使用；
- 删除父节点可能影响大量子节点。

更先进的策略可综合访问频率、重算成本和引用数。

## 7. 安全与隔离

跨租户缓存可能泄露命中时间、提示长度或敏感前缀。实践中应：

- 按租户或信任域隔离；
- 哈希中加入 cache salt；
- 不把原始 prompt 写入可观察日志；
- 对模型版本与 LoRA adapter 做严格命名空间隔离。

## 8. 哪些情况收益有限

- prompt 大多随机且无共享前缀；
- 输入很短；
- cache 过小导致频繁抖动；
- tokenizer 模板中加入随机或动态字段；
- 共享内容不在开头而在中间。

KV Cache 只能复用前缀，不能直接复用任意中间子串。

## 9. 与 Prompt Cache API 的区别

商业 API 的 prompt caching 可能采用自动、显式或持久化策略；底层不一定公开。系统设计时要区分：

- 进程内 GPU KV cache；
- 跨实例的分布式 KV store；
- 只缓存 tokenization/embedding；
- 真正缓存所有层的 K/V。

## 10. 面试回答

> Prefix Caching 利用 causal 模型“相同前缀产生相同 KV”的性质，复用已有 KV blocks，只计算未命中的后缀。vLLM 常按 block hash 查找，SGLang 的 RadixAttention 用 radix tree 组织任意共享前缀。收益主要是降低重复 prefill 和 TTFT；难点在淘汰、显存容量、租户隔离与 adapter/模型版本一致性。

## 参考资料

- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)
- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
- [SGLang RadixAttention](https://docs.sglang.ai/advanced_features/radix_attention.html)
