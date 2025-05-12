---
title: A Survey on Efficient Inference for Large Language Models
categories: [Paper]
tags: Survey Inference  LLM 
---

本文系统性地总结了**大型语言模型（LLMs）高效推理**的各种优化策略，旨在解决 LLMs 在部署和推理过程中面临的计算资源消耗大、延迟高、成本高等问题。文章从三个维度对现有研究进行了分类与分析：

1. 数据级优化（Data-Level Optimization）
2. 模型级优化（Model-Level Optimization）
3. 系统级优化（System-Level Optimization）

🧩 一、背景介绍

📌 什么是 LLM？
- LLM（Large Language Model）是指参数规模达到数十亿甚至数千亿级别的语言模型。
- 它们在多个 NLP 任务上表现出色，如问答、摘要、翻译、代码生成等。
- 典型代表：GPT、BERT、LLaMA、ChatGLM、PaLM、Bloom 等。

🚨 高效推理的挑战
- 长序列处理困难：标准注意力机制复杂度为 O(n²)，难以处理长文本。
- 资源消耗大：需要大量显存和计算资源，限制了在边缘设备上的应用。
- 服务成本高：推理延迟高、吞吐量低，影响用户体验和商业落地。
🔍 二、高效推理优化的三大层级

🧮 1. 数据级优化（Data-Level Optimization）

🎯 目标：
减少输入长度或输出复杂度，提升推理效率，通常不修改模型本身。

📦 主要方法：

✅ 输入压缩（Input Compression）

- 目的：缩短输入提示（prompt），降低计算负载。
- 方法举例：
  - RECOMP：通过语义相似度剪枝冗余文档，适用于检索增强模型（RALM）。
  - LLMLingua / LongLLMLingua：粗到细的剪枝策略，包括句子级别 + token 级别剪枝。
  - Prompt Pruning：如 DYNAICL、Selective Context、PCRL 等方法动态选择上下文信息。

✅ 输出组织（Output Organization）
- 目的：优化解码过程，提高硬件利用率。
- 方法举例：
  - 批处理（Batching）：同时处理多个请求，提高 GPU 利用率。
  - 并行解码（Parallel Decoding）：如 Skeleton-of-Thought 提出的多路径并行生成策略。

✅ 软提示压缩（Soft Prompt-based Compression）
- 使用可学习前缀（prefix-tuning）代替完整 prompt。
- 如 Gisting、Auto-Compressors、ICAEM 等。

🤖 2. 模型级优化（Model-Level Optimization）

🎯 目标：
设计或压缩模型结构，在保证性能的前提下降低推理成本。

📦 主要方法：

🧱 A. 高效结构设计（Efficient Structure Design）

⚙️ Transformer Alternates（Transformer 替代架构）
- RWKV：融合 RNN 的递归特性和 Transformer 的并行能力，实现线性推理复杂度 O(d²)。
- Mamba：基于状态空间模型（SSM），适合长序列建模。
- RetNet：多尺度表示 + 并行递归结构。
- Hyena：使用长卷积替代注意力机制。

⚡ Attention 设计优化
- Low-Rank Attention：Linformer、LRT、FLuRKA 等，将注意力矩阵低秩近似。
- Kernel-based Attention：Linear Transformer、Performers、RFA 等，用核函数替代 softmax。
- Group/Query Multi-head Attention：减少 head 数量或共享 key/value 向量。

🔁 FFN 层优化
- Switch Transformers：引入 MoE（Mixture of Experts）机制，只激活部分专家。
- Sparse Upcycling、**StableMoE**、**Mixtral 8x7B**：稀疏 MoE 架构，节省计算资源。

🧨 B. 模型压缩（Model Compression）
✂️ 权重剪枝（Weight Pruning）
- SparseGPT：基于 OBS 方法的一次性剪枝。
- RIA：考虑权重与激活值重要性，转换为 N:M 结构稀疏。
- Pruner-Zero：自动识别最优剪枝指标。
- ZipLM：结构化剪枝 + 推理感知优化。

🔢 量化（Quantization）
- Post-Training Quantization (PTQ)：如 Kivi、KVQuant。
- Quantization-Aware Training (QAT)：训练时模拟量化误差，保持精度。
- AffineQuant、**QuIP#**：新型量化方法，支持更低比特（如 4-bit）。

📉 知识蒸馏（Knowledge Distillation, KD）
- Black-box KD：仅使用教师模型输出作为监督信号。
- White-box KD：利用中间层知识迁移。
- Distilling Step-by-Step：逐步蒸馏，提升小模型推理能力。

📐 结构优化
- NAS（神经网络搜索）：如 NAS-BERT、Structural Pruning via NAS。
- Low Rank Factorization (LRF)：对权重矩阵进行低秩分解。

🛠️ 3. 系统级优化（System-Level Optimization）

🎯 目标：
优化推理引擎和服务系统，提升吞吐量、降低延迟、充分利用硬件资源。

📦 主要方法：

🧰 A. 推理引擎优化（Inference Engine Optimization）

⚡ 注意力与算子优化
- FlashAttention：IO-aware attention 实现，减少内存占用。
- FlashDecoding++：专为解码阶段设计，加速推理流程。
- TensorRT-LLM：英伟达推出的 LLM 推理引擎，优化主流算子。

🔀 Speculative Decoding（推测解码）
- Tree Attention：加快验证预测结果的过程。
- RadixAttention：优化 prefix cache，提高缓存命中率。
- Draft & Verify：自推测解码，实现 lossless 加速。

🏗️ B. 服务系统优化（Serving System Optimization）

📦 内存管理
- PagedAttention：类比操作系统分页机制，灵活管理 key-value 缓存。
- Outlier Suppression+：抑制异常值以支持更精确的量化。

🧮 分布式调度
- 支持多卡或多节点推理，提高并发能力。
📈 批处理优化
- 动态 batch size 控制，平衡延迟与吞吐。

📊 三、关键技术对比（来自 Table 2）

| 方法                 | 训练复杂度     | Prefill 复杂度 | Decode 复杂度  | 显存占用   |
|----------------------|----------------|----------------|----------------|------------|
| Transformer          | O(n²d)         | O(n²d)         | O(n²d)         | O(nd)      |
| RWKV                 | O(nd²)         | O(nd²)         | O(d²)          | O(d²)      |
| Linformer            | O(nkd)         | O(nkd)         | O(nkd)         | O(kd)      |
| FlashAttention       | O(n²d)         | O(n²d)         | O(n²d)         | ↓ 显存占用 |
| Speculative Decoding | —              | —              | O(αn)          | —          |

🧭 四、未来研究方向
根据论文讨论，以下是几个值得探索的研究方向：
1. 跨层级联合优化：
  - 探索数据、模型和系统级优化的协同效应。
  - 例如：压缩输入 + 剪枝模型 + 优化引擎 = 更高效的端到端推理。
2. 自适应优化策略：
  - 构建能够根据不同任务、上下文长度、硬件条件动态调整的推理系统。
3. 安全性与效率权衡：
  - 在提升效率的同时，保障模型的安全性（如隐私保护、对抗攻击防御）。
4. 轻量化模型结构创新：
  - 继续探索非 Transformer 架构（如 RWKV、Mamba、SSM）的潜力。
5. 边缘设备适配：
  - 针对移动端、IoT 等资源受限场景设计轻量模型与推理系统。
6. 统一接口与工具链建设：
  - 开发标准化 API 和推理工具（如 sglang、LMDeploy、TensorRT-LLM），提升工程落地效率

📚 五、相关开源项目与工具推荐

| 工具名             | 类型           | 描述                                       |
|--------------------|----------------|--------------------------------------------|
| FlashAttention      | 算子优化       | 高效注意力实现，显著减少显存               |
| TensorRT-LLM        | 推理引擎       | NVIDIA 推出的 LLM 推理优化库               |
| sglang              | 推理系统       | 快速编程接口，支持高性能 LLM 服务          |
| PagedAttention      | 显存管理       | 优化 KV Cache，支持长序列                  |
| Llama.cpp           | CPU 推理       | 支持 GGUF 格式，可在 CPU 上运行 LLaMA      |
| LMDeploy            | 模型部署工具   | 支持多种后端（CUDA、TensorRT、OpenVINO）  |
| DeepSpeed-FastGen   | 分布式推理     | 微软 DeepSpeed 支持的大模型生成加速方案    |

✅ 总结

| 层级   | 技术方向     | 关键方法                                      | 效果                         |
|--------|--------------|-----------------------------------------------|------------------------------|
| 数据级 | 输入压缩     | RECOMP、LLMLingua、Prompt Pruning             | 减少输入长度，降低计算       |
|        | 输出组织     | Parallel Decoding、Batching                   | 提高吞吐，降低延迟           |
| 模型级 | 结构设计     | RWKV、Mamba、Linformer                        | 线性复杂度、低显存           |
|        | 压缩方法     | 剪枝（RIA、SparseGPT）、量化（PTQ/QAT）、蒸馏（KD） | 模型更小、更快、部署友好     |
| 系统级 | 引擎优化     | FlashAttention、FlashDecoding++               | 加快关键算子                 |
|        | 推理系统     | TensorRT-LLM、sglang、PagedAttention          | 提高整体吞吐与并发           |