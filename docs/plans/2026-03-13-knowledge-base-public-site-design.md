# Knowledge Base Public Site Design

**Date:** 2026-03-13
**Status:** Approved for planning

## Goal

为当前知识库建立一个公开可访问的静态站点，满足以下目标：

- 首页是一个对外展示知识库的总览页
- 每篇文档都有稳定 URL，可单独分享
- 提供可缩放、可拖拽、可点击跳转详情页的交互式图谱
- 知识库内容更新后，站点自动重新构建并发布到 GitHub Pages
- `3-projects` 保留直链访问，但不进入首页、公开索引和默认图谱视图

## Constraints

- 当前仓库以 Markdown 为唯一内容源，继续保留现有目录结构
- 继续兼容 `[[wiki-link]]` 写法
- 不依赖 Obsidian 运行时，不要求访问者安装任何客户端
- 尽量保持站点代码与知识内容分离，避免破坏现有 Obsidian 工作流

## Architecture

采用自建静态站点方案，使用 `Astro` 负责路由和静态页面生成，使用一层自定义内容管线负责解析知识库内容并产出派生数据。

### Content Source

内容仍然来自仓库根目录下的：

- `0-inbox/`
- `1-concepts/`
- `2-explorations/`
- `3-projects/`
- `4-references/`

### Build Pipeline

构建前执行自定义索引脚本，完成以下工作：

1. 扫描所有公开内容目录中的 Markdown 文件
2. 解析 frontmatter、正文、首段摘要、标题
3. 提取 `[[wiki-link]]`，建立文档间边
4. 为每篇文档生成：
   - `id`
   - `title`
   - `slug`
   - `section`
   - `tags`
   - `summary`
   - `updatedAt`
   - `outgoingLinks`
   - `incomingLinks`
   - `isIndexed`
   - `isVisibleInDefaultGraph`
5. 输出：
   - `src/generated/content-index.json`
   - `src/generated/graph.json`

### Site Layer

站点由 `Astro` 生成以下页面：

- `/` 总览页
- `/concepts/` 概念索引页
- `/explorations/` 探索索引页
- `/references/` 参考资料索引页
- `/inbox/` inbox 索引页
- `/graph/` 图谱页
- `/concepts/<slug>/`
- `/explorations/<slug>/`
- `/references/<slug>/`
- `/inbox/<slug>/`
- `/projects/<slug>/`

不生成 `/projects/` 索引页。

## Information Architecture

### Home Page

首页负责“第一次访问的理解”，不承担完整阅读。

建议包含：

- Hero 区：一句话解释知识库定位
- Knowledge Overview：概念数、探索数、参考数、最近更新时间
- Entry Cards：概念、探索、参考、图谱四个入口
- Recent Updates：最近更新的 6 到 10 篇内容
- Featured Threads：挑出 3 到 5 条代表性概念链路
- Graph Preview：小型图谱预览，按钮跳转全屏图谱

### Detail Pages

详情页负责“阅读 + 浏览关系”。

建议结构：

- 标题、元信息、标签
- Markdown 正文
- 文内 `[[wiki-link]]` 转可点击链接
- 侧边栏或文末展示：
  - 出链
  - 反向链接
  - 同标签相关文章
  - 同类型最近更新

### Graph Page

图谱页使用交互式网络视图，满足：

- 缩放
- 拖拽
- Hover 高亮
- 点击节点跳详情页
- 左侧或右侧信息面板显示节点摘要
- 提供类型过滤和标签过滤

默认数据集排除 `3-projects`。
后续如需要，可以加“显示项目节点”的显式切换开关。

## Content Visibility Rules

### Public Index

进入公开索引：

- `0-inbox`
- `1-concepts`
- `2-explorations`
- `4-references`

不进入公开索引：

- `3-projects`

### Detail Pages

所有目录中的 Markdown 文档都生成详情页，包括 `3-projects`。

### Graph

默认图谱排除 `3-projects` 节点与边，但项目页本身可以在详情页展示关联链接。

## Technical Choices

- Framework: `Astro`
- Language: `TypeScript`
- Content parsing: `gray-matter` + 自定义 wiki-link 解析
- Graph data: 构建阶段预生成 JSON
- Graph rendering: `sigma.js` + `graphology`
- Testing: `Vitest`
- Deployment: `GitHub Actions` + `GitHub Pages`

## Non-Goals for V1

- 不做在线编辑
- 不做评论系统
- 不做登录权限
- 不做全文搜索以外的复杂推荐系统
- 不做 Obsidian 完全等价还原

## Risks

- `[[wiki-link]]` 标题变更后可能导致断链，需要构建阶段校验
- 中文标题 slug 规则需要稳定，否则历史链接会漂移
- 图谱节点数量增长后，前端布局性能可能下降，需要预计算坐标或限制初始渲染范围
- `3-projects` 虽不入索引，但保留直链，需确保不会在其他列表里意外泄露

## Recommendation

第一版优先做“稳定发布的公开站”，不是一次性追求所有高级功能。

MVP 顺序建议是：

1. 打通 Markdown 到静态详情页
2. 做首页总览和内容索引
3. 做默认图谱
4. 接上 GitHub Pages 自动发布
5. 再打磨视觉和交互细节
