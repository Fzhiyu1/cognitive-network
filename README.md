# Cognitive Network

输出驱动的个人认知网络。

这个仓库同时承担两件事：

1. Obsidian 知识库本体
2. 基于 Quartz 4 的公开数字花园站点

## 知识库结构

```text
0-inbox/         GTD 收集箱（未分类临时记录）
1-concepts/      原子化概念卡片
2-explorations/  完整探索与长文整理
3-projects/      项目页与实验计划
4-references/    外部理论、人物、论文与参考资料
tools/           本地工具（kb-index 索引生成器）
```

核心收录标准：内容里必须包含真实推导、重述、提炼或结构化思考，而不是纯收藏。

每张卡的 frontmatter 含 `summary` 字段（一句话摘要）作为 AI 协作的 L1 索引输入。详见 [`CLAUDE.md`](./CLAUDE.md)。

## AI 协作索引

为让 LLM 协作时能看到知识库全貌，`tools/kb-index` 工具基于 [[wikilink]] 拓扑 + Leiden 聚类生成 4 个 markdown 索引（GraphRAG-inspired，30ms 全量构建）：

- [`TERRAIN.md`](./TERRAIN.md) — L3 全局地形概览（~2 KB，AI 入场首读）
- [`INDEX.md`](./INDEX.md) — L1 每卡一行（摘要 + 度数 + 簇 + 标签）
- [`CLUSTERS.md`](./CLUSTERS.md) — L2 Leiden 主题簇（核心节点 + 成员）
- [`GRAPH.md`](./GRAPH.md) — 拓扑分析（PageRank / 桥梁 / 孤儿 / 死链）

加上 `KB-GRAPH.html`——D3 力导向图交互可视化（节点大小=PageRank，颜色=簇）。

工具源码：`tools/kb-index/`，spec：`3-projects/2026-04-27-kb-index-system.md`

```bash
# 重新生成索引
tools/kb-index/.venv/bin/kb-index update

# 生成可视化
tools/kb-index/.venv/bin/kb-index visualize --open

# 诊断（孤儿 / 死链 / 拼写相似）
tools/kb-index/.venv/bin/kb-index doctor
```

## 公开站点

公开站点现在基于 Quartz 4 构建，使用仓库根目录作为项目根。

- 内容入口：`content/`
- 实际内容来源：通过符号链接映射到 `1-concepts/`、`2-explorations/`、`3-projects/`、`4-references/`
- 图谱、反向链接、搜索、popover 预览：由 Quartz 原生提供
- 部署：GitHub Pages

## 本地开发

```bash
npm ci
npx quartz build --serve
```

构建产物输出到 `public/`。

## 部署

部署工作流位于 `.github/workflows/deploy-public-site.yml`。

- 发布分支：`master`
- 预期地址：`https://fzhiyu1.github.io/cognitive-network/`

首次启用时，需要在 GitHub 仓库设置中将 Pages Source 切换为 `GitHub Actions`。

## 说明

旧的 Astro 原型仍保留在 `site/`，但不再作为主站构建入口。
