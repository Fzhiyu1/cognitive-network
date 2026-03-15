# Nested Learning 复现计划

**状态**：未开始
**来源**：[[Nested Learning 学习指南：从理解到复现Hope]]
**设备**：48GB MacBook Pro（Apple Silicon，MPS后端）

## 阶段

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | 前置工作：Titans论文、线性关联记忆实现 | ⬜ |
| Phase 2 | 核心组件：DGD更新规则、MemoryModule | ⬜ |
| Phase 3 | 自修改Titans + 分块并行训练 | ⬜ |
| Phase 4 | CMS（连续记忆系统） | ⬜ |
| Phase 5 | 组装Hope（或先做Hope-Attention） | ⬜ |
| Phase 6 | 扩展验证：语言建模、持续学习 | ⬜ |

## 实现优先级（基于消融实验）

v的自修改 > DGD > k的自修改 > CMS > 动量 > 权重衰减 > q的自修改

## 应用方向：Hope 记忆外挂（2026-03-15 新增）

复现 Hope 不只是学术练习——它有一个具体的应用场景：训练一个小 Hope 模型（50-100M）作为主 LLM 的记忆激活层。详见 [[Hope 记忆外挂]]。

这意味着复现优先级可以重新考虑：CMS + 基础 Titans 可能比完整自修改 Titans 更早有实用价值，因为记忆匹配是一个窄任务。

## 关联概念

- [[关联记忆]]
- [[多层有损压缩]]
- [[Hope 记忆外挂]]
- [[前意识激活层]]
