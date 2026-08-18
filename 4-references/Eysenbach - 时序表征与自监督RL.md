---
tags:
- AI
summary: Eysenbach 组四篇论文构成"不学奖励学时序距离"纲领：对比学习拟合密度比即Q函数，深度解锁其规模化；本库七次实验界定了它的迁移边界。
---

# Eysenbach - 时序表征与自监督RL

**领域**：强化学习 / 自监督表征学习
**关联概念**：[[时序表征适用地图]] · [[harness直觉信号]] · [[带着技术找应用]]

## 核心内容

Benjamin Eysenbach（普林斯顿）组的研究纲领：goal-conditioned RL 不需要奖励工程——对比学习的最优分类器 logit 收敛到密度比 p(s_f|s,a)/p(s_f)，恰等于折扣占用度量（Successor Representation），即"到达型"任务的 Q 函数。**学时序表征本身就是一种 RL 算法**；本质是对环境转移算子做非线性低秩谱分解。

本目录四篇 PDF：

| 文件 | arXiv | 一句话 |
|---|---|---|
| 1000 Layer Networks for Self-Supervised RL.pdf | 2503.14858 | **NeurIPS 2025 最佳论文**。残差+LayerNorm+Swish 打破"RL 网络 2~5 层"禁忌，深至 1024 层，contrastive RL 提升 2×~50×；深度同时增强表达与探索（互锁），并解锁 batch size scaling 与部分 stitching |
| Is TD Learning the Gold Standard for Stitching in RL.pdf | 2510.21995 | MC 也能 stitching；TD 相对 MC 的优势小于大网络 vs 小网络的差距——RL 专属归纳偏置在 scale 时代收益递减 |
| Demystifying Emergent Exploration in Goal-conditioned RL.pdf | 2510.14129 | 涌现探索来自表征的**低秩性**而非网络逼近：压缩迫使未见状态被高估→乐观探索是免费副产品 |
| Temporal Representations for Exploration.pdf | 2603.02008 | 2026.03 B 站演讲（"追求奖励完全错误"标题党的原始出处）对应论文：时序对比表征直接驱动复杂探索行为，无外部奖励 |

## 与本知识库的关系

本库调研线（[[时序表征调研：从NeurIPS最佳论文到harness直觉]]）的理论源头。七次迁移实验的结论：该纲领在原生领域（连续状态+真轨迹+有控制者）成立，但迁移到可枚举实体数据（git、工具序列）时全程败给共现计数与 C@C 矩阵乘——神经网络是"表的替代品而非升级品"，见 [[时序表征适用地图]]。调研副产品 [[harness直觉信号]] 反而不依赖此纲领。

论文中"深度解锁质变行为"（U4 迷宫先背向目标绕行）与"评测低秩机制"的分析范式，与本库 [[适应度幻觉]]、meomory 的"评测方法论 > 算法设计"结论同构。
