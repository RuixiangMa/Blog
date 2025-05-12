---
title: A Survey on Inference Engines for Large Language Models
categories: [Paper]
tags: Survey Inference  LLM 
---

随着大型语言模型（Large Language Models, LLMs）的参数规模持续增长，其推理（inference）阶段的计算和内存需求也急剧上升。为了在有限的硬件资源下实现高效、低延迟、高吞吐量的部署，LLM推理优化 成为关键研究方向。本文将从多个维度详细解析当前主流的LLM推理优化技术。

一、LLM推理流程简述

在深入优化技术前，先回顾一下LLM的推理流程：

1. Prefill阶段

- 输入提示（prompt）被编码为token序列。
- 模型一次性处理所有输入token，生成初始上下文（context）。
- 计算并缓存注意力机制中的Key 和 Value 向量（KV Cache），用于后续解码阶段。

2. Decode阶段

- 逐个生成输出token。
- 每一步都依赖于之前生成的所有token，因此需要不断更新KV Cache。
- 随着生成长度增加，KV Cache占用内存迅速增长，影响性能。

二、LLM推理优化的核心挑战

- 内存瓶颈 ：KV Cache随序列长度呈线性增长，导致显存不足。
- 计算效率低 ：注意力机制（Attention）复杂度为O(n²)，长序列带来巨大计算压力。
- 吞吐量与延迟矛盾 ：批处理可提高吞吐，但会牺牲首token延迟（TTFT）。
- 多设备协同困难 ：分布式推理涉及通信开销和负载均衡问题。
- 硬件兼容性差 ：不同平台（GPU/TPU/Edge）支持不一致。

三、LLM推理优化技术详解

1. 批处理优化（Batch Optimization）

(1) 动态批处理（Dynamic Batching）

- 将多个请求合并成一个批次进行推理，提升GPU利用率。
- 支持引擎：vLLM、TensorRT-LLM、Sarathi-Serve等。

(2) 连续批处理（Continuous Batching）

- 不等待整个批次填满，而是实时将新请求加入正在处理的批次中。
- 减少空闲时间，进一步提升吞吐。
- 支持引擎：DeepSpeed-FastGen、Together Inference。

(3) 微批处理（Nano-batching）

- 将每个请求拆分为更小的微批次，便于调度和负载均衡。
- 支持引擎：NanoFlow。

(4) 分块预填充（Chunked-prefill）

- 将长输入分块处理，避免单次加载全部输入导致的显存溢出。
- 支持引擎：Sarathi-Serve、SGLang。

2. 并行化策略（Parallelism）

(1) 数据并行（Data Parallelism）

- 多个设备处理不同的请求或批次，适用于高并发场景。
- 支持引擎：HuggingFace Transformers、DeepSpeed。

(2) 张量并行（Tensor Parallelism）

- 将模型层切分到多个GPU上，适合超大规模模型。
- 支持引擎：DeepSpeed-FastGen、TensorRT-LLM。

(3) 流水线并行（Pipeline Parallelism）

- 将模型拆分为多个阶段，依次分布在多个设备上，重叠计算与通信。
- 支持引擎：Sarathi-Serve、DeepSpeed。

3. 缓存管理（KV Cache Management）

(1) PagedAttention

- 受操作系统内存分页启发，将KV缓存划分为固定大小的“页”，按需分配。
- 显著减少内存碎片，提高内存利用率。
- 支持引擎：vLLM、LMDeploy。

(2) KV Cache量化

- 对KV缓存中的K/V向量进行低精度量化（如FP8、INT8），降低显存占用。
- 支持引擎：TensorRT-LLM、PowerInfer。

(3) KV Cache复用与淘汰

- 利用LRU（最近最少使用）等策略淘汰旧缓存。
- 支持引擎：TensorRT-LLM。

(4) KV Cache抢占（Preemption）

- 在内存不足时，临时中断某些请求并清空其KV缓存，稍后重新恢复。
- 虽然会增加延迟，但能提升整体稳定性。
- 支持引擎：vLLM。

4. 注意力机制优化（Attention Optimization）

(1) FlashAttention
- 利用GPU内存层次结构优化注意力计算，减少内存访问次数。
- 时间复杂度仍为O(n²)，但常数项显著下降。
- 支持引擎：FlashAttention、TensorRT-LLM。

(2) FlashAttention-3

- 更高效的注意力计算方式，进一步压缩延迟。
- 支持引擎：FlashAttention-3。

(3) 稀疏注意力（Sparse Attention）

- 仅关注部分重要的token，减少冗余计算。
- 支持引擎：Switch Transformers、Triton。

5. 压缩与量化（Compression & Quantization）

(1) 权重量化（Weight Quantization）

- 将模型权重从FP32/FP16压缩到INT8、INT4等低精度格式。
- 支持方法：GPTQ、AWQ、GGUF。
- 支持引擎：llama.cpp、TensorRT-LLM、vLLM。

(2) LoRA（Low-Rank Adaptation）

- 在预训练模型基础上引入低秩适配矩阵，大幅减少参数量。
- 支持引擎：Friendli Inference、vLLM。

(3) 结构化剪枝（Structured Pruning）

- 移除冗余神经元或通道，保持结构紧凑。
- 支持引擎：LLM-Pruner、SparseGPT。

(4) 知识蒸馏（Knowledge Distillation, KD）

- 使用大模型指导小模型训练，压缩模型尺寸。
- 支持引擎：DistilBERT、TinyBERT。

6. 内核优化（Kernel Optimization）

- CUDA Kernel优化 ：定制化CUDA内核提升运算效率。
- Tensor Core利用 ：NVIDIA GPU上的专用矩阵计算单元，加速FP16/INT8运算。
- 编译器优化 ：如TensorRT编译器自动融合操作、选择最优算法。
- 支持引擎：TensorRT-LLM、Triton、PyTorch Inductor。

7. 采样与输出优化（Sampling & Output Optimization）

- Top-k / Top-p Sampling ：控制生成多样性。
- 束搜索（Beam Search）优化 ：减少重复计算。
- 异步生成（Async Generation） ：分离prefill和decode阶段，提高并发性。
- 支持引擎：vLLM、TensorRT-LLM、HuggingFace Transformers。

8. 边缘与异构设备支持

- CPU推理优化 ：如llama.cpp、NanoFlow支持纯CPU部署。
- 移动端支持 ：MLC LLM、Ollama支持Android/iOS。
- 边缘设备支持 ：TensorRT-LLM支持Jetson系列；PowerInfer支持骁龙芯片。
- 跨平台编译 ：如TVM支持多种目标架构。

9. 分布式推理与弹性伸缩

- Ray集成 ：vLLM、DistServe、Sarathi-Serve均基于Ray实现分布式推理。
- Kubernetes集成 ：BentoML、OpenLLM支持K8s部署。
- Serverless推理 ：Fireworks AI、Together Inference提供无服务器推理服务。

四、典型优化引擎对比

| 特性               | vLLM          | DeepSpeed-LLM       | TensorRT-LLM     | llama.cpp      | Friendli Inference |
|--------------------|---------------|---------------------|------------------|----------------|--------------------|
| 易用性             | 高            | 中                 | 高              | 低             | 中                |
| 吞吐优化           | 高            | 高                 | 高              | 低             | 中                |
| 延迟优化           | 高            | 高                 | 中              | 低             | 高                |
| KV Cache优化       | ✅ PagedAttention | ✅ 量化+循环缓存   | ✅ KV复用        | ❌              | ✅ TCache         |
| 量化支持           | FP16/INT8/INT4 | FP16/INT8/INT4     | FP16/INT8       | FP16/INT4      | INT4             |
| 并行支持           | 数据/流水线    | 数据/张量          | 张量/流水线     | 单线程         | 数据              |
| 商业支持           | ❌             | ✅ NVIDIA           | ✅ Microsoft     | ❌              | ✅                |

五、未来发展方向
1. 统一推理框架 ：支持多种模型架构（Transformer、Mamba、RetNet等）。
2. 自适应优化策略 ：根据输入长度、设备能力自动选择最优策略。
3. 端侧智能调度 ：在边缘设备上动态调整推理策略以节省能耗。
4. 安全增强 ：防止对抗攻击、数据泄露等安全问题。
5. 自动化调参工具 ：类似AutoML的系统，自动选择最佳优化组合。


六、总结

LLM推理优化是一个高度工程化和系统化的领域，涵盖了从底层硬件到上层服务架构的全栈优化。目前已有大量开源和商业引擎提供丰富的优化技术，开发者可根据具体应用场景（如实时交互、批量处理、边缘部署）灵活选择。未来，随着模型架构的演进和硬件的发展，LLM推理引擎将继续朝着高性能、低成本、易用性强、安全性好 的方向发展。
