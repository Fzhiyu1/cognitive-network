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

## 关联概念

- [[关联记忆]]
- [[多层有损压缩]]
