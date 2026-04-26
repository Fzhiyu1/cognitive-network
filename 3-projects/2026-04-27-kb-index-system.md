---
tags:
- 工程
- 方法论
- AI
summary: 用Leiden聚类把知识库渲染成4个markdown，让AI入场即获地形感，触发更频繁的[[当场抽取]]
---

# KB-Index 知识库索引系统

**日期**：2026-04-27 初稿
**状态**：设计中（v0.1）
**形态**：本地 Python 脚本（~200-300 行）+ 4 个产物 markdown 文件 + CLAUDE.md 指引集成
**核心范式**：借鉴 GraphRAG 的多层级摘要思想，用纯 markdown 文件让 AI 在每次对话中重建知识库的"地形感"

---

## 1. 问题陈述

### 1.1 真实痛点

当前知识库已有 80+ 张卡片，分布在 `1-concepts/` `2-explorations/` `3-projects/` `4-references/`，使用 Obsidian-style `[[wikilink]]` 互联。但 AI（Claude Code）每次进入对话时**看不到全貌**：

- 只能 `ls` 看文件名，无法感知卡片之间的连接结构
- 临时 `grep` 找几张相关卡片，**永远不知道漏掉了什么**
- 做"创建链接"这种工作时缺少全图视野，提议的关联可能只是基于刚扫到的 5-6 张卡
- 无法识别"重新发明轮子"——某个新概念可能已在某张卡片的某段里被讨论过
- 无法做 meta 分析：哪片密、哪片稀、哪些 cluster 该合并、哪些是孤儿

### 1.2 问题的本质

**这是 RAG 的根本短板**：检索是 query-time 的，但人类思考时是**全图常驻**的。用户脑中的知识库拓扑，AI 每次进来都是空白。

**任何记忆系统的全图都不可能完整加载**——这是物理限制。能加载的是"关于全图的高层摘要"。所以正确目标不是"让 AI 看到全图"，而是**让 AI 在每次对话中重建那种地形感**。

### 1.3 已排除的路径

- **传统 RAG**（LangChain / LlamaIndex / Mem0 / vector DB）：query-time 检索 + 语义相似 ≠ 概念结构关联，不解决全貌问题
- **agent 记忆系统**（Letta / MemGPT）：为 agent 自主累积记忆设计，与"用户手写、AI 协作"的动机相反
- **Obsidian CLI**（`obsidian-cli` npm 包）：GUI 自动化工具，不是 AI 索引工具
- **Obsidian 插件**（Smart Connections / Dataview / Graph Analysis）：产物在 GUI 进程内，CLI 拿不到
- **完整 GraphRAG 库**（微软）：你的知识库已有 `[[link]]` 不需 LLM 抽实体；产物是 SQLite/parquet 而非 markdown；与"自实现派"风格不符

**结论**：借 GraphRAG 的**思想**，自实现轻量版。

---

## 2. 设计目标

### 2.1 核心目标

让任意 AI 进入知识库时，读 ~200 行 markdown 即可获得：

1. **L3 地形感**：整个知识库长什么样、有哪些区域
2. **L2 区域感**：每个 cluster 在讨论什么、包含哪些卡
3. **L1 定位感**：每张卡片的标题 + 一句话摘要 + 度数
4. **拓扑感**：哪些是中心节点、哪些是孤岛、哪些是桥梁

按需下钻到 L0（卡片本身）。

### 2.2 非目标

- ❌ 不替代 Obsidian GUI（人继续用 Obsidian 看图、查询、写卡）
- ❌ 不做 query-time 检索（AI 用 grep / Read 即可）
- ❌ 不做向量检索（不引入 embedding 模型）
- ❌ 不做实时索引（手动触发或 git hook 即可）

### 2.3 设计原则

1. **产物是纯 markdown**——人类可读、AI 可读、Obsidian 可索引、Quartz 可发布
2. **自实现 + 最小依赖**——只用 Python + igraph + leidenalg + 可选 anthropic SDK
3. **增量更新**——只重算受影响的 cluster，不每次全重算
4. **机械化优先**——L1（卡摘）和拓扑全机械抽取；L2/L3 摘要可选 LLM 增强
5. **零对 Obsidian 工作流的侵入**——卡片格式不变，只新增根目录 4 个文件

---

## 3. 架构总览

### 3.1 数据流

```
knowledge-base/                         knowledge-base/
├── 0-inbox/*.md                        ├── INDEX.md      (L1)
├── 1-concepts/*.md         ┌─────┐    ├── CLUSTERS.md   (L2)
├── 2-explorations/*.md ──→ │ kb- │ ──→├── GRAPH.md      (拓扑)
├── 3-projects/*.md         │index│    ├── TERRAIN.md    (L3)
└── 4-references/*.md       └─────┘    └── .kb-index/cache.json
```

### 3.2 处理管线

```
扫描 (mtime 增量)
  ↓
解析 (frontmatter + headings + [[link]] + 第一段)
  ↓
图构建 (节点=卡片，边=wiki link)
  ↓
聚类 (Leiden algorithm)
  ↓
拓扑分析 (度数、PageRank、孤岛、桥梁)
  ↓
摘要生成 (机械抽取 + 可选 LLM)
  ↓
渲染 (4 个 markdown 文件)
  ↓
缓存写入 (.kb-index/cache.json)
```

---

## 4. 产物规格

四个产物文件全部位于知识库根目录，被 Obsidian 看到、被 Quartz 看到、被 AI 通过 CLAUDE.md 指引读取。

### 4.1 INDEX.md（L1：每卡一行）

**作用**：AI 入场时的"卡片名册"，快速定位。

**格式**：

```markdown
---
generated: 2026-04-27T15:32:18+08:00
total_cards: 87
total_links: 234
generator: kb-index v0.1
---

# 知识库索引（L1）

> 每张卡片一行，含一句摘要 + 度数（in/out）+ 标签 + 所属 cluster。
> AI 进入对话时优先读此文件。需要详细内容请直接读对应 .md。

## 1-concepts/ (61 张)

- [[AI镜像效应]] :: AI 产出本质上是用户自身想法的镜像反射 :: in=8 out=3 :: #AI #哲学 :: cluster=C2
- [[校准的不可外包性]] :: 品味通过"接受/拒绝/修改 AI 输出"积累，训练集只属于你 :: in=2 out=4 :: #AI #认知科学 :: cluster=C2
- [[多层有损压缩]] :: 大模型压缩的不是意识，而是意识经多层有损压缩后的产物 :: in=4 out=2 :: #AI #哲学 :: cluster=C1
...

## 2-explorations/ (13 篇)
...

## 3-projects/ (5 个)
...

## 4-references/ (10 项)
...
```

**摘要来源**（按优先级）：
1. frontmatter 里的 `summary` 字段（如有）
2. 第一个 `## 定义` 后的第一句话
3. 第一段非空文本，截断到 80 字
4. 文件名（兜底）

### 4.2 CLUSTERS.md（L2：簇摘要）

**作用**：每个概念簇的中心思想 + 成员。

**格式**：

```markdown
---
generated: 2026-04-27T15:32:18+08:00
algorithm: leiden
n_clusters: 9
modularity: 0.71
---

# 知识库聚类（L2）

> 基于 [[wikilink]] 共现模式的自动聚类。Leiden 算法。
> 每簇含：核心节点（按 PageRank 排序）+ 主题摘要 + 成员列表。

## C1：AI 认知机制簇（11 张）

**核心节点**：[[多层有损压缩]] [[描述即污染]] [[思考与生成不分]]

**主题**：围绕"LLM 的内部运作机制"——怎么压缩、怎么生成、怎么受输入影响。
（机械摘要：列出簇内节点的标题；LLM 摘要：一段自然语言总结）

**成员**：
- [[多层有损压缩]] (PR=0.18)
- [[描述即污染]] (PR=0.15)
- [[思考与生成不分]] (PR=0.12)
- [[知道与推测不分]] (PR=0.10)
- ...

## C2：AI 与人类协作簇（8 张）

**核心节点**：[[AI镜像效应]] [[校准的不可外包性]]

**主题**：人在 AI 时代的认知校准、品味形成、协作陷阱。

**成员**：
...
```

### 4.3 GRAPH.md（拓扑分析）

**作用**：识别中心节点、孤岛、桥梁，发现结构问题。

**格式**：

```markdown
---
generated: 2026-04-27T15:32:18+08:00
n_nodes: 87
n_edges: 234
density: 0.062
---

# 知识库拓扑

## 中心节点（PageRank Top 15）

| 卡片 | PR | 入度 | 出度 | 簇 |
|------|----|----|----|----|
| [[AI镜像效应]] | 0.0421 | 8 | 3 | C2 |
| [[多层有损压缩]] | 0.0378 | 4 | 2 | C1 |
...

## 桥梁卡片（连接多个簇）

> 这些卡片同时被多个 cluster 引用，往往是关键的概念枢纽。

- [[知识即结构]]：连接 C1（认知机制）+ C5（学习理论）
- [[自由能原理]]：连接 C1 + C3 + C7
...

## 孤岛卡片（无入边或无出边）

> 可能需要补充链接，或是真正孤立的边缘概念。

- [[XYZ]]（无 backlink，可能未被其他卡片引用）
- [[ABC]]（无 outlink，可能是终端概念）
...

## 死链（链接到不存在的卡片）

- [[Foo]] 被 [[Bar]] 引用但文件不存在
...

## 入度分布

```
0    ████ 12 张
1-2  ████████████ 32 张
3-5  ████████ 24 张
6-10 ████ 14 张
11+  ██ 5 张
```
```

### 4.4 TERRAIN.md（L3：全局地形）

**作用**：AI 进来后第一眼看到的全库总览。最高层级的"地形感"。

**格式**：

```markdown
---
generated: 2026-04-27T15:32:18+08:00
---

# 知识库地形图（L3）

## 一句话总览

87 张卡片，9 个主题簇，密度 0.062，模块度 0.71。
主流：AI 与认知（38%）、记忆系统（24%）、工程实践（18%）、其他（20%）。

## 簇地图

| 簇 | 主题 | 卡数 | 密度 | 状态 |
|----|------|------|------|------|
| C1 | AI 认知机制 | 11 | 高 | 成熟 |
| C2 | AI 与人类协作 | 8 | 中 | 活跃增长 |
| C3 | 进化与记忆系统 | 14 | 高 | 成熟 |
| C4 | 工程实践 | 9 | 低 | 稀疏，可补 |
| ... |

## 结构观察

- **密度过高的区域**：C1 内部链接密集，可能存在概念冗余，建议审视是否合并
- **密度过低的区域**：C4 大量孤儿卡，需补关联
- **跨簇桥梁**：[[知识即结构]] 和 [[自由能原理]] 是关键枢纽，删除会断链
- **新增热区**（最近 30 天）：C2 增长 5 张

## 标签分布

| 标签 | 卡数 |
|------|------|
| #AI | 52 |
| #认知科学 | 31 |
| #哲学 | 22 |
...

## AI 使用提示

进入对话时：
1. 读 TERRAIN.md（本文件）获取地形感
2. 按对话主题查 INDEX.md 定位相关卡片
3. 用 CLUSTERS.md 找出"主题相邻"的卡片
4. 用 GRAPH.md 识别"是否在动到中心节点"
5. 按需 Read 具体卡片
```

### 4.5 .kb-index/cache.json（增量缓存）

```json
{
  "version": 1,
  "last_run": "2026-04-27T15:32:18+08:00",
  "cards": {
    "1-concepts/AI镜像效应.md": {
      "mtime": 1745740338.0,
      "sha": "abc123...",
      "summary": "AI 产出本质上是用户自身想法的镜像反射",
      "links_out": ["描述即污染", "校准的不可外包性"],
      "tags": ["AI", "哲学"],
      "in_degree": 8,
      "out_degree": 3,
      "cluster": "C2",
      "pagerank": 0.0421
    },
    ...
  },
  "clusters": {
    "C1": {
      "members": [...],
      "summary_machine": "...",
      "summary_llm": "...",
      "summary_llm_hash": "..."
    }
  }
}
```

---

## 5. 算法选择

### 5.1 解析

- **Markdown 解析**：`markdown-it-py` 或正则（足够）
- **Frontmatter**：`python-frontmatter`
- **Wikilink 抽取**：正则 `\[\[([^\]]+?)\]\]`，处理 `[[name|alias]]` 和 `[[name#section]]`
- **链接解析**：精确匹配卡片名（支持 fuzzy 兜底，但默认精确）

### 5.2 图构建

- 节点：每个 .md 文件
- 边：`A → B` 当 A 中出现 `[[B]]`
- 有向图（保留方向，便于区分入度/出度）
- 簇分析时转无向图

### 5.3 聚类：Leiden algorithm

**为什么选 Leiden**：
- Louvain 经典但有"分辨率限制"和"badly connected community"问题
- Leiden（Traag 2019）是 Louvain 改进版，保证社区内连通性，更稳定
- 库：`leidenalg` + `python-igraph`，成熟、装即用
- 资源占用：80+ 节点的图毫秒级完成

**参数**：
- `resolution_parameter=1.0` 默认；密度高时降低到 0.7 防止过分裂
- 多次运行取众数，避免随机性

**备选**：连通分量（太粗）、Label Propagation（不稳定）、谱聚类（慢）

### 5.4 拓扑度量

- **PageRank**：识别中心节点（damping=0.85）
- **入度/出度**：直接统计
- **桥梁检测**：跨簇链接最多的节点（簇分配后再算）
- **孤岛**：度数为 0 或入度为 0 的节点

### 5.5 摘要生成

**两种模式：**

**机械模式（默认，零依赖）：**
- 卡摘要：第一段或 `## 定义` 后第一句
- 簇摘要：簇内 top-3 节点的标题 + 标签 + 卡数
- 全局摘要：簇统计 + 标签分布

**LLM 模式（可选，质量增强）：**
- 簇摘要：用 Claude Haiku 4.5（便宜快），输入是簇内所有卡片的 L1 摘要 + 标签，输出 1-2 句话主题
- 全局摘要：基于簇摘要，再调一次 LLM 写"地形观察"
- 缓存机制：簇成员未变 + 簇摘要 hash 未变 → 跳过 LLM 调用
- 成本估算：~10 个簇 × 1 次调用 × ~500 tokens ≈ ¥0.001/次全量
- 配置：`KB_INDEX_LLM=on` 或 `--with-llm` 启用

### 5.6 增量更新

**判断变更**：对每个 .md 比较 `mtime` 和 `sha256`。
**重算范围**：
- 解析：仅变更文件
- 图：受影响节点的入边/出边重建
- 聚类：变更节点 > 5% 时全图重跑（Leiden 很快，不必复杂增量）
- LLM 摘要：只对成员变化的簇重调用

---

## 6. 使用方式

### 6.1 CLI

```bash
# 全量构建
kb-index build

# 增量更新（默认）
kb-index update

# 启用 LLM 增强摘要
kb-index update --with-llm

# 只验证不写入（dry run）
kb-index check

# 输出诊断（找孤岛、死链、过密区）
kb-index doctor
```

### 6.2 与 Obsidian / Quartz 共存

- 4 个产物 `.md` 在根目录，**会被 Obsidian 索引**——可以正常用图谱查看、用 Dataview 反向引用
- **会被 Quartz 发布**——成为知识库网站的一部分（如不想发布，加 `draft: true` frontmatter）
- `.kb-index/` 隐藏目录加入 `.gitignore`（可选）和 `quartz.config.ts` 的 ignorePatterns

### 6.3 与 AI 协作的 CLAUDE.md 集成

在 `knowledge-base/CLAUDE.md` 末尾追加：

```markdown
## AI 协作指引：必读索引

进入本知识库的任何 AI 必须先读以下文件以获得"地形感"：

1. `TERRAIN.md`（L3 全局地形）
2. `INDEX.md`（L1 卡片名册，按需扫读）
3. `CLUSTERS.md`（L2 主题簇，建立链接时必读）
4. `GRAPH.md`（拓扑，发现孤儿/桥梁/死链时用）

**读取策略**：
- 闲聊/单卡片操作：只读 TERRAIN.md
- 创建新卡片：读 TERRAIN.md + 相关簇的 CLUSTERS.md 段落
- 建立关联：必读 CLUSTERS.md + GRAPH.md
- meta 分析（找空洞、合并冗余）：全读

**索引更新**：
- 新建/修改卡片后，索引可能过期
- 用户可手动跑 `kb-index update`
- 或配置 git pre-commit hook 自动更新
```

### 6.4 git hook 自动化（可选）

`.git/hooks/pre-commit`：

```bash
#!/bin/sh
# 自动更新索引
cd $(git rev-parse --show-toplevel)
kb-index update --quiet
git add INDEX.md CLUSTERS.md GRAPH.md TERRAIN.md
```

---

## 7. 目录结构

```
knowledge-base/
├── tools/
│   └── kb-index/
│       ├── pyproject.toml
│       ├── README.md
│       └── kb_index/
│           ├── __init__.py
│           ├── cli.py          # CLI 入口（typer / click）
│           ├── parse.py        # markdown / frontmatter / wikilink 解析
│           ├── graph.py        # 图构建 + 拓扑度量 + Leiden 聚类
│           ├── summary.py      # 机械摘要 + LLM 摘要
│           ├── render.py       # 4 个 .md 渲染
│           ├── cache.py        # cache.json 读写 + 增量判断
│           └── tests/          # pytest
├── .kb-index/
│   └── cache.json              # 自动生成
├── INDEX.md                    # 自动生成
├── CLUSTERS.md                 # 自动生成
├── GRAPH.md                    # 自动生成
├── TERRAIN.md                  # 自动生成
├── 0-inbox/
├── 1-concepts/
├── 2-explorations/
├── 3-projects/
├── 4-references/
├── CLAUDE.md                   # 追加 §6.3 协作指引
└── ...（quartz 配置等不变）
```

**依赖**（`pyproject.toml`）：

```toml
[project]
name = "kb-index"
version = "0.1.0"
dependencies = [
    "python-igraph>=0.11",
    "leidenalg>=0.10",
    "python-frontmatter>=1.1",
    "typer>=0.12",
]

[project.optional-dependencies]
llm = ["anthropic>=0.39"]
dev = ["pytest>=8", "ruff>=0.6"]

[project.scripts]
kb-index = "kb_index.cli:app"
```

---

## 8. 验证标准

### 8.1 功能验证

- [ ] 跑 `kb-index build` 能在当前知识库上产出 4 个 .md，无错误
- [ ] INDEX.md 包含所有 .md 文件（除 INDEX/CLUSTERS/GRAPH/TERRAIN 自身和 .kb-index/）
- [ ] CLUSTERS.md 至少 5 个簇，模块度 > 0.4
- [ ] GRAPH.md 检测出至少 1 个已知孤儿卡（用户人工验证）
- [ ] TERRAIN.md 总览数据与 INDEX.md 统计一致
- [ ] 死链检测能找出至少 1 个已知死链（人工构造）
- [ ] 增量模式：只改 1 张卡 → 只该簇的摘要被重生成

### 8.2 体感验证（最关键）

**测试方法**：让一个全新对话的 Claude 进入知识库，**只读 TERRAIN.md + INDEX.md**，让它回答以下问题：

1. "知识库里和 AI 镜像效应相关的卡片有哪些？"——应能正确列出 C2 簇成员
2. "我想加一张关于'品味的不可外包性'的卡，会和已有什么卡片冲突或重复？"——应能识别 [[校准的不可外包性]]（如已存在）
3. "知识库里哪片最稀疏，需要补充？"——应能从 TERRAIN.md §结构观察 答出
4. "[[多层有损压缩]] 删除会断哪些链？"——应能从 GRAPH.md 答出

**通过标准**：4 题中至少 3 题答对，且不需要再 grep / ls 才能答出。

### 8.3 性能验证

- 全量构建：80+ 卡 < 5 秒（无 LLM）/ < 60 秒（含 LLM）
- 增量构建：单卡变更 < 1 秒
- 4 个产物 .md 总大小 < 50 KB（保证能塞进 AI 上下文）

---

## 9. 实施阶段

### V0.1（最小可用，~1 天）

- [ ] 解析 + 图构建 + Leiden 聚类
- [ ] 4 个 .md 机械生成（无 LLM）
- [ ] CLI: `build` / `update`
- [ ] 全量模式（不做增量）
- [ ] 在当前知识库上验证产物

### V0.2（增量 + LLM，~1 天）

- [ ] cache.json + 增量更新
- [ ] LLM 摘要（簇 + 全局）
- [ ] CLI: `--with-llm` `doctor`
- [ ] CLAUDE.md 协作指引

### V0.3（自动化 + 体感打磨，~1 天）

- [ ] git pre-commit hook 模板
- [ ] §8.2 体感验证（用真实 Claude 测）
- [ ] 根据结果调整摘要格式 / 字段

### V1.0（稳定）

- [ ] 4 个产物格式冻结
- [ ] 文档完整
- [ ] 在 ≥ 2 个知识库（kb-consolidation）跑通

---

## 10. 风险与权衡

### 10.1 已识别风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| Leiden 聚类对密度变化敏感，重跑簇号可能洗牌 | AI 看到的 cluster=C2 这次和下次可能指代不同簇 | 簇号根据 PageRank 最高节点的卡片 ID 稳定命名（`C-AI镜像效应`），不用 C1/C2 |
| LLM 摘要漂移：同样输入两次输出不同 | TERRAIN.md 风格不一致 | 摘要 hash 比对，未变更不重算 |
| 4 个产物 .md 被 Quartz 发布到公网，可能含敏感信息 | 隐私泄漏 | TERRAIN.md frontmatter 加 `draft: true`，或加入 quartz ignorePatterns |
| 卡片量增长到 500+ 时 INDEX.md 超过上下文限制 | AI 读不全 | 按 cluster 拆分 INDEX，AI 按需读分片 |
| Wikilink 写法不一致（`[[A]]` vs `[[A.md]]` vs `[[A\|alias]]`） | 死链误报 | 解析器规范化，支持所有三种 |

### 10.2 设计权衡

- **机械摘要 vs LLM 摘要**：机械精度低但稳定零成本；LLM 质量高但有漂移和成本。**采用双模式**，默认机械、按需 LLM。
- **每次全量 vs 增量**：增量复杂度高但快；全量简单但慢。**80 卡量级全量 < 5 秒**，可以先全量，后续增量。
- **生成产物 vs 实时计算**：生成方便 AI 读，缺点是会过期；实时计算永远新鲜，缺点是要 AI 主动跑工具。**生成产物**优先，git hook 解决过期问题。
- **保留 Obsidian/Quartz 兼容 vs 独立工具**：兼容意味着产物在根目录、被 Obsidian/Quartz 看到。**保留兼容**，零侵入。

### 10.3 不在本期范围

- 向量检索 / embedding（不需要，结构关联够用）
- Web UI（CLI + markdown 已足够）
- 多语言支持（仅中文）
- 跨 vault 联合索引

---

## 11. 关联

- 反向呼应：[[AI镜像效应]]——本工具的目标之一就是降低镜像效应（让 AI 看到外部结构而非自我反射）
- 延伸：[[描述即污染]]——CLAUDE.md 协作指引本身是一种"描述"，会改变 AI 行为分布
- 上游：[[知识库元认知]]
- 实践对照：[[meomory记忆假肢原型]]（meomory 是动态记忆，本系统是静态索引；两者对应"工作记忆"和"语义网络"）

---

## 12. 决策记录

| 决策点 | 选择 | 备选 | 理由 |
|--------|------|------|------|
| 聚类算法 | Leiden | Louvain / Label Propagation | Leiden 是 Louvain 改进版，保证连通性 |
| 图库 | python-igraph | networkx | igraph 性能更好；leidenalg 是 igraph 生态 |
| 摘要 | 机械 + 可选 LLM | 仅机械 / 仅 LLM | 默认零依赖，按需增强 |
| 产物格式 | markdown | JSON / SQLite | 人类 + AI + Obsidian + Quartz 多端可读 |
| 产物位置 | 知识库根目录 | `.kb-index/` 隐藏 | 让 Obsidian/Quartz 也能消费 |
| 簇命名 | 锚定 PageRank 顶点 | C1/C2 序号 | 重跑稳定，可作为知识库内的稳定 ID |
| 触发方式 | 手动 + 可选 git hook | watch / 实时 | 简单可靠 |
| 实现语言 | Python | Node.js / Rust | 用户已有 Python 生态（meomory） |

---

**下一步**：用户确认 spec → 进入 V0.1 实施 → 在当前知识库上跑通 → §8.2 体感验证。
