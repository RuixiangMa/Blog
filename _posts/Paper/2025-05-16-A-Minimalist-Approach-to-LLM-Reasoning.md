---
title: 一种面向LLM推理的极简方法-从拒绝采样到Reinforce
categories: [Paper]
tags: Reinforce Rejection Sampling  LLM 
---

# A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce

## 摘要

强化学习（RL）已成为在复杂推理任务上微调大型语言模型（LLMs）的主要方法。在近期的方法中，GRPO 因其在训练如 DeepSeek-R1 等模型上的实证成功而脱颖而出，但其有效性来源仍不明确。在这项工作中，本文从类似 Reinforce 的算法角度重新审视 GRPO，并分析其核心组件。

作者发现一个简单的拒绝采样基线 RAFT，在仅使用正样本训练的情况下，其性能与 GRPO 和 PPO 相当甚至更优。本文的消融实验表明，GRPO 的主要优势来自于丢弃所有生成回答都错误的提示（prompt），而不是其奖励归一化机制。受此启发，本文提出了 **Reinforce-Rej**，这是策略梯度的一种最小扩展，它过滤掉全部正确或全部错误的样本。Reinforce-Rej 提高了 KL 效率和稳定性，成为比复杂 RL 算法更轻量级但有效的替代方案。本文建议将 RAFT 作为稳健且可解释的基线方法，并建议未来的研究应更加注重有原则地整合负样本，而非盲目依赖它们。本文的研究结果为基于奖励的 LLM 后训练提供了指导方向。

## 1 引言

在使用可验证奖励微调大型语言模型（LLMs）的背景下，研究强化学习（RL）算法的表现。本文的重点是数学推理任务，这些任务在 OpenAI 的 O1 模型和 DeepSeek-R1 发布后受到了广泛关注。

LLM 后训练的主流方法是近端策略优化（PPO）。然而，PPO 需要额外的 critic 网络，带来了计算开销和算法复杂性。同时，LLM 的确定性转移特性降低了方差，使得许多 PPO 的复杂组件可能不再必要。这一观察激发了人们对设计更简单但有效的 RL 算法的兴趣。

一些近期工作重新审视了 Reinforce 风格的方法，包括 ReMax、RLOO、GRPO 和 Reinforce++。与此同时，其他方法探索了除策略梯度之外的方向。例如 **RAFT（Reward-ranked fine-tuning）** 通过迭代生成多个响应、筛选错误答案来微调 LLM。

在这些方法中，GRPO 因其在数学推理任务上的卓越表现而脱颖而出，尤其是在训练 DeepSeek-R1 上的成功。然而，其算法细节尚未被充分公开，尚不清楚其优越性来源于内在优势还是延续过往方法的结果。相比之下，RAFT 已成为最简单且最可解释的基线之一，尽管设计极简，但在以往文献中表现出良好的实证性能。

在这项研究中，本文重新审视以下三种方法：

1. **RAFT**：也称为拒绝采样，在 LLM 文献中是最基本的 RL 算法；
2. **Vanilla Reinforce**：经典的策略梯度算法，是去除 critic 模型的简化版 PPO；
3. **GRPO**：一种 Reinforce 变体，每提示生成 n 个响应，并通过均值和标准差进行样本奖励归一化。

GRPO（Reinforce）与 RAFT 的关键区别在于如何处理负样本：GRPO 在训练中混合接受和拒绝的样本，而 RAFT 仅依赖正样本。

尽管普遍认为利用负反馈信号的 RL 方法显著优于仅使用正样本的 SFT 类算法，但本文在初步实验中发现性能差距出奇地小，甚至 RAFT 在早期训练阶段（如前 100-200 次迭代）表现出更快的收敛速度。进一步分析揭示，某些类型的负信号（如所有生成响应都完全错误的提示）实际上会显著损害模型性能，而诸如奖励归一化的技术影响较小。

为了更好地理解这些动态，本文使用 Qwen 和 LLaMA 模型进行了消融实验，隔离不同设计选择的影响。本文的结果突出了以下关键发现：

1. **RAFT 性能接近 GRPO**：尽管只使用正样本，RAFT 的性能与最先进的 RL 方法 GRPO 相当，差距出奇地小，并在早期训练阶段具有更快的收敛率。
2. **有害提示的过滤是 GRPO 的核心优势**：通过对不同 Reinforce 变体的控制实验，本文发现对于 on-policy 方法，训练那些所有采样响应都错误的提示会显著损害性能。GRPO 对这些有害提示的隐式过滤是其相对于标准 Reinforce 的主要优势来源。
3. **提出 Reinforce-Rej**：受 RAFT 和 Reinforce 的启发，本文研究了一种新的 Reinforce 变体——Reinforce-Rej，它选择性地过滤掉全部正确或全部错误的样本。该方法在最终性能上与 GRPO 相当，并展示了出色的 KL 效率。

这些洞察强调了在基于奖励的 LLM 后训练中，**样本选择比算法设计更重要**。该项目的代码可在 [GitHub](https://github.com/RLHFlow/Minimal-RL ) 获取。

## 2 方法

### 符号定义

给定一个提示（prompt）x，LLM 被表示为一个策略 π(a|x)，它可以将提示映射到响应 a 的分布。本文还定义 r(x, a) ∈ {−1, 1} 为一个二元奖励函数，用于评估提示-响应对的质量。本文记收集的提示-响应对数据集为 D。对于每个提示 x，本文可以生成 n 个候选响应 a₁, ..., aₙ，并获得相应的奖励 r₁, ..., rₙ。

设 aₜ 为响应 a 中的第 t 个 token，sₜ(θ) 表示 token t 的重要性采样比率：
$$
s_t(\theta) = \frac{\pi_\theta(a_t | x, a_{1:t-1})}{\pi_{\theta_{old}}(a_t | x, a_{1:t-1})}
$$
本文还定义奖励的基准为 mean(r₁, ..., rₙ)，标准差为 std(r₁, ..., rₙ)。

### RAFT

RAFT 算法也被称为拒绝采样微调。其步骤如下：

- **数据收集**：对一批提示 x₁,...,x_M，从参考模型中生成 n 个响应。
- **数据排序（拒绝采样）**：保留奖励最高的响应（通常是 r=1 的响应）。
- **模型微调**：最大化选中数据集上的对数似然：
$$
L_{\text{RAFT}}(\theta) = \sum_{(x,a) \in D} \log \pi_\theta(a|x)
$$

### Reinforce 与策略梯度

策略梯度的目标函数为：
$$
J(\theta) = \mathbb{E}_{x \sim d_0}\left[\mathbb{E}_{a \sim \pi_\theta(\cdot|x)} r(x, a)\right]
$$

更新方式为：
$$
\theta' \leftarrow \theta + \beta \nabla_\theta J(\theta)
$$

实际中，使用旧策略 π_old 收集轨迹，并用这些样本计算随机梯度。为了加速训练，通常采用重要性采样技巧修正分布差异。最终损失函数为：
$$
L_{\text{Reinforce}}(\theta) = \frac{1}{|D|} \sum_{x,a \in D} \min\left( s_t(\theta) \cdot r(x, a), \text{clip}(s_t(\theta), 1 - \epsilon, 1 + \epsilon) \cdot r(x, a) \right)
$$

### GRPO

GRPO 使用类似的损失函数，但用优势函数 At(x, a) 替代原始奖励 r(x, a)。具体来说，对于每个提示 x，GRPO 会采样 n > 1 个响应，并计算每个 token 的优势：
$$
A_t(x, a_i) = \frac{r_i - \text{mean}(r_1, ..., r_n)}{\text{std}(r_1, ..., r_n)}
$$

### DPO（Iterative）

DPO 基于成对比较数据 {(x, a⁺, a⁻)}，其中 a⁺ ≻ a⁻。优化对比损失：
$$
L_{\text{DPO}}(\theta) = -\log \sigma\left(\beta \log \frac{\pi_\theta(a^+|x)}{\pi_{\text{ref}}(a^+|x)} - \beta \log \frac{\pi_\theta(a^-|x)}{\pi_{\text{ref}}(a^-|x)}\right)
$$

### RAFT++

本文将 RAFT 扩展为 off-policy 版本，引入重要性采样和剪切技巧，得到：
$$
L_{\text{RAFT++}}(\theta) = \frac{1}{|D|} \sum_{x,a \in D} \frac{1}{|a|} \sum_{t=1}^{|a|} \min(s_t(\theta), \text{clip}(s_t(\theta), 1 - \epsilon, 1 + \epsilon)) \cdot I(r(x, a) = \max r(x, a_i))
$$

## 3 实验设置

本文专注于数学推理任务。使用的框架为 verl。训练数据来自 Numina-Math，包含约 86 万道数学题。实验模型为 Qwen2.5-Math-7B-base 和 LLaMA-3.2-3B-instruct。

超参数设置：

- 学习率：1e-6
- 每次迭代采样提示数：1024
- 每提示生成响应数：4
- 微批大小：512
- 最大生成长度：4096 tokens

评估任务包括 Math500、Minerva Math 和 Olympiad Bench，使用 average@16 进行评估。

## 4 主要结果

### RAFT 与 RAFT++ 接近深度 RL 方法

| Model Algorithm | Math500 | Minerva Math | Olympiad Bench | Average |
|----------------|---------|--------------|----------------|---------|
| Qwen2.5-Math-7B-base BaseRAFT | 41.3 | 11.0 | 18.6 | 23.6 |
| RAFT | 77.4 | 34.4 | 37.8 | 49.9 |
| RAFT++ | 80.5 | 35.8 | 41.2 | 52.5 |
| Iterative DPO | 75.7 | 30.5 | 38.3 | 48.2 |
| Reinforce | 80.6 | 36.1 | 42.1 | 52.9 |
| GRPO | 81.6 | 36.7 | 43.3 | 53.9 |
| PPO | 79.6 | 34.8 | 41.1 | 51.8 |

结果显示 RAFT 和 RAFT++ 的性能接近甚至超过复杂的 RL 方法。RAFT++ 通过引入重要性采样和剪切技巧，在保持训练稳定性的前提下提升了最终准确率。

### 消融实验

#### 正样本训练导致熵崩溃

RAFT++ 训练过程中策略熵迅速下降，导致探索能力减弱，最终被 GRPO 超过。负样本有助于维持探索，防止分布崩溃。

#### GRPO 成功的核心因素

GRPO 的优势主要来自于丢弃所有响应都错误的提示，而不是奖励归一化。本文提出的 **Reinforce-Rej** 仅保留部分样本，达到了与 GRPO 相当的性能。

## 5 结论

本文通过拒绝采样的视角重新审视了 LLM 后训练中的 RL 算法设计空间。本文的研究表明，RAFT 是一个出人意料的强大基线，其性能接近甚至超过 PPO 和迭代 DPO 等复杂方法。本文通过引入重要性采样和剪切技巧改进 RAFT，得到了 RAFT++，在保持训练流程简洁稳定的前提下，达到接近最先进水平的性能。

通过广泛的消融实验，本文识别出 GRPO 的主要优势不是来自奖励归一化，而是丢弃了所有响应都正确或错误的提示。在此基础上，本文提出了 Reinforce-Rej，这是一种最小化的策略梯度变体，过滤掉全部正确或错误的样本。Reinforce-Rej 提高了 KL 效率和熵稳定性，凸显了探索在基于奖励的微调中的作用。

本文的研究结果表明，基于 RL 的 LLM 训练中负样本的作用比之前假设的更为微妙。未来的方法不应盲目依赖原始负反馈，而应考虑更有选择性和原则性的方式来整合样本质量。本文推荐 RAFT 和 Reinforce-Rej 作为未来基于奖励的 LLM 后训练工作的轻量级、可解释且有效的基线。