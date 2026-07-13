---
tags: [大模型, 推理, TensorRT-LLM, Triton, NVIDIA]
created: 2026-07-13
updated: 2026-07-13
---

# TensorRT-LLM 推理优化

> TensorRT-LLM 面向 NVIDIA GPU，把模型图、低精度 kernel、KV 管理、调度和多卡通信编译/组合成专用推理运行时。

## 1. 与通用 PyTorch 推理的差异

通用框架强调动态图和开发便利；部署引擎知道模型只做 inference，可以：

- 常量折叠与权重预处理；
- 选择特定 shape/dtype 的 kernel；
- 融合算子；
- 规划 workspace 与 buffer；
- 捕获 CUDA Graph；
- 绑定 TP/PP/EP 通信；
- 使用低精度和硬件专用指令。

## 2. 两类使用路径

### Engine 路径

将 checkpoint 转换并构建 TensorRT engine。构建时间较长，但部署时图和 kernel 选择更固定。

### PyTorch/LLM API 路径

更接近 Hugging Face 使用方式，由运行时做优化，模型接入灵活。二者的功能覆盖和最佳性能可能不同，不能只说“用了 TensorRT-LLM”。

## 3. In-flight Batching

请求可以在 decode iteration 动态加入和退出，与 Continuous Batching 同一思想。Executor 还要处理：

- 最大 batch/token 数；
- 排队策略；
- KV block 分配；
- 请求取消；
- streaming response；
- 多 profile/shape。

## 4. GPT Attention Plugin

专用 attention plugin 会融合 QKV 相关处理，并支持：

- paged KV cache；
- RoPE/ALiBi；
- MHA/MQA/GQA；
- context/prefill 与 generation/decode；
- KV cache quantization；
- beam search 与 speculative decoding 的缓存布局。

不同模型结构与硬件会选择不同 kernel 路径。

## 5. 量化

TensorRT-LLM 支持多类 recipe，例如 FP8、INT8、INT4 weight-only、SmoothQuant、GPTQ/AWQ、FP4 等。需要同时确认：

1. 权重文件格式；
2. calibration/scale；
3. GEMM plugin 是否原生低精度；
4. accumulation dtype；
5. KV Cache dtype；
6. 当前 GPU 是否有对应 Tensor Core。

## 6. 多 GPU 与通信

- TP：节点内常通过 NCCL/NVLink；
- PP：按层切 stage；
- EP：MoE experts 分布式执行；
- Leader/Orchestrator：协调多实例和多节点执行。

Engine 构建配置必须与部署 world size、rank mapping 和硬件拓扑一致。

## 7. Triton Inference Server Backend

Triton 提供模型仓库、HTTP/gRPC、动态配置、metrics 和 ensemble。典型流水线：

```text
preprocessing → tensorrt_llm → postprocessing
```

也可使用 inflight batcher 模型把请求管理、流式输出和后端 executor 连接起来。

## 8. Speculative Decoding

支持 Draft Model、Medusa、Lookahead、ReDrafter、EAGLE 等路径。生产性能依赖 tree attention、候选验证、KV 提交和多 rank 同步是否融合，而不只是算法接受率。

## 9. 调优步骤

1. 先确认精度基线；
2. 固定输入/输出长度与并发；
3. 选择量化与 TP/PP；
4. 调整 max batch、max tokens、KV fraction；
5. 检查 attention/GEMM plugin 选择；
6. 测 TTFT、TPOT、Goodput；
7. 用 Nsight Systems/Compute 定位 kernel、通信和空洞。

## 10. 常见误区

1. Engine 构建成功不代表数值质量正确。
2. 最大 batch 越大不一定越好，可能恶化排队和 KV 压力。
3. TensorRT-LLM 与 Triton Server 不是同一个层次：前者是 LLM runtime，后者是服务平台。
4. 不能把不同版本、不同 plugin 的 benchmark 直接比较。

## 11. 面试回答

> TensorRT-LLM 将 LLM 图优化、专用 attention/GEMM plugin、低精度、paged KV、in-flight batching 和多卡通信组合成 NVIDIA GPU 推理运行时；Triton Backend 再提供网络服务、模型仓库和监控。性能取决于 engine/LLM API 路径、量化 recipe、plugin 选择、并行配置和真实负载。

## 参考资料

- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)
- [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)
- [Triton TensorRT-LLM Backend](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/README.html)
