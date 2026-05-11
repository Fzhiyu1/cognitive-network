---
tags:
- 工程
- AI
- 方法论
summary: Interactive Evolution 应用于研究探索的 Claude Code skill；用户作 fitness function，递归 Teams 架构，把"判断力"挖成不可跳过的工程槽位。
---

# Atlas

**状态**：开发中（v0.1 spec 已写完，dual-mode 演化中）
**路径**：`/Users/fangzhiyu/run/atlas/`
**来源**：[[研究核与投射]] 核 3（设计-代码自动化）在研究探索这个载体上的表达

## 目标

把 **Interactive Evolution**（用户作 fitness function 的进化搜索）应用到研究探索场景：

- AI 提出候选研究方向
- 用户用 V/D/N（Value / Depth / Novelty）三轴打分
- 分数直接进入下一轮 spawn 决策（不是事后评估）
- 产物是分支拓扑树而非综述报告

## 核心架构表态

详见 [[判断力位]]。Atlas 的架构精髓在三个设计选择：

1. **V/D/N 评分进入 spawn 决策**：把传统进化算法挖空的 fitness 槽位留给人填
2. **AI 分支是无状态探针**：`claude --print` + 禁止 `Skill("atlas")` 自递归 → AI 端被刻意设计成不可累积
3. **dual-mode**：默认 AI 评分，`--human` 切手动 → 载体可替换，位置不能消失

## CLI 入口

```bash
bun run atlas-cli init '<question>'   # 启动研究会话
bun run atlas-server                  # Hono 控制平面
bun run dev                           # Web UI（观察者视角）
```

会话状态持久化在 `~/.tree-explorer/sessions/`。

## 关键设计文档

- `docs/superpowers/specs/2026-04-26-atlas-design.md` —— v0.1 spec
- `docs/plans/2026-04-28-supervisor-ux-p0-design.md`
- `docs/plans/2026-04-29-permission-isolation-design.md`
- `docs/plans/2026-04-29-sibling-crossover-design.md`

## 心智模型

> "CLI 入口 + 编排器 + 分支运行时 + 持久化 + 观察者 UI" 五件套，不要把它当 Web 服务或单 CLI 看待。

## v0.2+ 路径

数字分身方向：自实现 KNN + Embedding（不用 SaaS），让 AI 代理打分时也带"你自己的判断风格"。

## 关联概念

- [[判断力位]] —— 项目的核心架构表态
- [[校准的不可外包性]] —— 项目的哲学锚点
- [[meomory记忆假肢原型]] —— 同构对偶（evaluator 是 LLM Judge vs 人）
- [[研究核与投射]] —— 项目在思想谱系里的位置
- [[wake-atlas-AB实验]] —— 一次用 atlas 演示"substrate 的双刃"的实验
