---
tags: [AI, 工程]
---

# DeepSeek Engram — 条件记忆

**领域**：大模型架构 / 稀疏性 / 记忆系统
**论文**：Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models
**作者**：DeepSeek-AI（2026年1月12日）
**链接**：[arXiv 2601.07372](https://arxiv.org/abs/2601.07372) / [GitHub](https://github.com/deepseek-ai/Engram)
**PDF**：本目录 `Engram - Conditional Memory via Scalable Lookup.pdf`
**关联概念**：[[条件记忆]]、[[关联记忆]]、[[多层有损压缩]]、[[前意识激活层]]、[[记忆假肢]]

## 核心思想

语言处理有两种截然不同的任务：**组合推理**（需要深层计算）和**知识检索**（静态的、局部的模式查找）。当前 Transformer 把两者混在一起，用昂贵的运行时计算重建本可以直接查表的东西。Engram 在 MoE 的"条件计算"旁边引入"条件记忆"作为**第二条稀疏性轴**，把静态知识存入 embedding 查找表，用 O(1) 哈希检索代替深层重建。

## 关键机制

### 1. N-gram 哈希检索

对输入 token 做 NFKC 归一化 + 小写化（词表压缩 23%），取后缀 N-gram（2-gram、3-gram），通过多头哈希映射到 embedding 表索引。检索是确定性的、O(1) 的，不依赖中间激活值。

### 2. 上下文门控

检索到的 embedding 是上下文无关的先验。通过门控机制用当前隐状态调制：

```
αₜ = σ(RMSNorm(hₜ)ᵀ RMSNorm(kₜ) / √d)
```

检索是静态的，使用是动态的——当前上下文决定哪些记忆被放大、哪些被压低。

### 3. 异步预取

因为记忆索引只依赖输入 token（不依赖中间激活值），可以在 GPU 计算前面几层时，从主存/SSD 异步预取后面需要的 embedding。100B 参数从主存加载，推理开销不到 3%。

### 4. 多级缓存

利用 N-gram 的 Zipfian 分布：高频 embedding 驻留 GPU HBM，中频在 CPU DRAM，长尾低频在 NVMe SSD。

## 关键数字

| 发现 | 数值 |
|------|------|
| 稀疏分配法则 | 20-25% 给记忆，75-80% 给计算 |
| 100B 参数卸载到主存的开销 | < 3% |
| Engram-27B vs MoE-27B（MMLU） | +3.4 |
| Engram-27B vs MoE-27B（BBH） | +5.0 |
| 长上下文 Multi-Query NIAH | 97.0 vs 84.2（+12.8） |
| 关闭 Engram 后事实知识保留 | 29-44%（灾难性坍缩） |
| 关闭 Engram 后阅读理解保留 | 81-93%（几乎不受影响） |

## 机制分析亮点

- **等效加深网络**：Engram 通过接管静态模式查找，释放了早期层的负担，CKA 分析显示 Engram-27B 的第 5 层对齐到 MoE baseline 的第 ~12 层
- **事实 vs 推理的分离**：关闭 Engram 后事实知识坍缩但阅读理解几乎不受影响，说明 Engram 是知识仓库，backbone 是推理引擎
- **门控可视化**：多 token 实体（"Alexander the Great"、"四大发明"）和固定短语（"By the way"）上门控值最高

## 与本知识库的关系

Engram 从工程路径独立验证了本知识库从认知科学方向推导出的多个结论：

1. **条件记忆 vs 条件计算** = 我们的"记忆模块 vs 主 LLM"分离
2. **异步预取** = [[前意识激活层]]：在 LLM 开始思考前记忆已经准备好
3. **多级缓存（HBM/DRAM/SSD）** = L0/L1/L2 分层
4. **上下文门控** = [[主动遗忘]] 在推理时的表现形式
5. **Zipfian 分布驱动的缓存策略** = 频率决定记忆在哪层驻留

Engram 和 DGD/CMS（[[Nested Learning 学习指南：从理解到复现Hope]]）是互补的：Engram 提供大容量静态存储 + 快速预取，DGD/CMS 提供在线学习 + 遗忘控制。一个管"存多少"，一个管"怎么学"。
