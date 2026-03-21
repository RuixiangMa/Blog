---
title: "MSA: Memory Sparse Attention for Efficient End-to-End Memory Model Scaling to 100M Tokens"
categories: [Paper]
tags: Long Context Memory Attention Mechanism LLM KV Cache
---

## 论文信息

- **论文标题**: MSA: Memory Sparse Attention for Efficient End-to-End Memory Model Scaling to 100M Tokens
- **作者**: Yu Chen, Runkai Chen, Sheng Yi 等 (EverMind, Shanda Group, Peking University)
- **GitHub**: https://github.com/EverMind-AI/MSA

## 摘要

长期记忆是人类智能的基石。使AI能够处理终身规模的信息，达到数亿token的处理能力，一直是该领域的长期追求。由于全注意力架构的约束，大语言模型（LLM）的有效上下文长度通常限制在1M token以内。

本文提出**Memory Sparse Attention (MSA)**，一种端到端可训练、高效且可大规模扩展的记忆模型框架。通过可扩展稀疏注意力架构和文档级RoPE等核心创新，MSA在训练和推理中均实现线性复杂度，同时保持卓越的精度稳定性——从16K扩展到100M tokens时，性能衰减小于9%。

此外，结合KV缓存压缩和Memory Parallel技术，MSA仅需2×A800 GPU即可完成100M tokens的推理。通过Memory Interleaving机制，MSA还能有效支持跨分散记忆段的多跳推理。

## 核心挑战

### 现有方法的局限性

| 方法类型 | 代表方案 | 优势 | 劣势 |
|----------|---------|------|------|
| 参数化记忆 | LoRA, Continual Pre-training | 语义整合度高 | 容量受限，遗忘严重 |
| 外部存储 | RAG, MemAgent | 容量可扩展 | 非端到端，精度受限 |
| 潜在状态 | Sparse Attention, Linear Attention | 高精度 | O(L²)复杂度或容量受限 |

现有方法面临两大根本限制：
1. **高精度记忆的有限可扩展性**：高精度方法受限于固定上下文或状态容量
2. **缺乏端到端可训练性**：无法同时保持架构兼容性、高精度和对灾难性遗忘的鲁棒性

### 人类记忆规模估算

认知科学研究估计人类记忆的功能信息容量约为10⁹ bits。按每token 3-5 bits的有效语义密度计算，这相当于约200-300 million tokens的终身容量。要真正弥合与人类规模记忆的差距，模型必须能够处理扩展到数亿token的上下文。

## 方法：Memory Sparse Attention

### 稀疏注意力机制

MSA将标准密集自注意力替换为**基于文档的检索稀疏注意力机制**。核心设计：

1. **Router K Projector**：为每个文档生成专门的路由键矩阵
2. **分块压缩**：将每个文档分割为多个固定长度块，通过mean pooling压缩状态
3. **Top-k检索**：基于余弦相似度计算查询与记忆的相关性分数，选择Top-k最相关文档
4. **选择性应用**：仅在后半层应用MSA路由，保留层次表示对齐

### 文档级RoPE

为确保跨不同记忆规模的稳健泛化，MSA为每个文档独立应用RoPE。这解决了训练-推理上下文不匹配问题：模型通常在有限文档数量上训练（train-on-short），但推理时需操作大规模文档库（infer-on-long）。

通过为每个文档分配独立的起始位置ID（从0开始），MSA将位置语义与文档总数解耦，使模型能够有效外推到训练范围之外。

### 训练策略

**连续预训练阶段**（158.95B tokens）：
- 标准生成损失 $\mathcal{L}_{LLM}$ + 辅助路由损失 $\mathcal{L}_{aux}$
- $\mathcal{L}_{aux}$：监督对比损失，分离相关与无关文档块

**两阶段后训练**：
- 阶段一：8K上下文长度的SFT，建立基础指令遵循能力
- 阶段二：扩展到64K上下文长度，提高长度外推鲁棒性

### 三阶段推理流程

1. **全局记忆编码**（离线）：预计算所有文档的K、V、K^R，压缩并缓存
2. **路由与上下文组装**（在线）：计算查询路由向量，匹配Top-k文档
3. **稀疏生成**（在线）：在组装好的稀疏上下文上自回归生成

### Memory Interleaving

为处理需要多跳推理的复杂查询，MSA引入**Memory Interleaving机制**。推理过程在"生成检索"和"上下文扩展"之间交替进行：

1. 模型基于查询自回归生成文档ID序列
2. 系统获取对应原文并追加到原始查询
3. 重复直到模型判断累积文档足够

### Memory Parallel

为在标准单节点上支持极长上下文推理，MSA设计了专门的推理引擎：

**分层内存存储**：
- GPU驻留路由键（K̄^R）：分布在多卡VRAM中，确保低延迟检索
- CPU卸载内容KV（K̄, V̄）：存储在主机DRAM，按需异步获取

**分布式评分**：查询hidden states广播到所有GPU，每卡独立计算相似度分数。

## 实验结果

### QA任务

MSA在9个问答基准上显著优于基于相同backbone的RAG系统：

| 基准 | MSA vs 最佳RAG基线 | 相对提升 |
|------|-------------------|---------|
| 平均分数 | 3.760 | +16.0% |

对比SOTA RAG系统（KaLMv2 + Qwen3-235B），MSA在9个数据集中的4个取得最佳性能，平均提升7.2%。

### NIAH任务

在RULER "大海捞针"基准上，MSA展现出从32K到1M tokens的卓越稳定性：

| 模型 | 32K | 256K | 1M |
|------|-----|------|-----|
| Qwen3-4B-Instruct | 99% | 48% | 25% |
| Qwen3-Next-80B-A3B | 100% | 97% | 81% |
| RL-MemoryAgent-14B | 98% | 95% | 93% |
| **MSA (Ours)** | **99%** | **98%** | **95%** |

### 100M Token可扩展性

MSA在MS MARCO数据集上展示前所未有的可扩展性：从16K扩展到100M tokens，性能衰减小于9%。

## 结论

本文提出**Memory Sparse Attention (MSA)**，一种端到端可训练的稀疏注意力架构，实现了接近人类记忆规模的100M tokens处理能力：

1. **核心创新**：可扩展稀疏注意力 + 文档级RoPE + KV缓存压缩 + Memory Interleaving
2. **关键突破**：从16K到100M tokens仅<9%性能衰减
3. **实用部署**：仅需2×A800 GPU即可完成100M tokens推理
4. **性能领先**：在长上下文QA和NIAH基准上显著超越前沿LLM和SOTA RAG系统

MSA证明了通过将记忆容量与推理解耦，可以为通用模型提供内在的终身规模记忆，标志着向人类认知尺度记忆迈出的重要一步。
