"""渲染 4 个产物 markdown：INDEX / CLUSTERS / GRAPH / TERRAIN。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .graph import CardNode, Cluster, GraphStats, find_bridges, find_orphans
from .parse import VAULT_SUBDIRS
from .summary import summarize_cluster_machine, terrain_overview

GENERATED_BY = "kb-index v0.1"


def _now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def _frontmatter(extra: dict) -> str:
    lines = ["---"]
    lines.append(f"generated: {_now()}")
    lines.append(f"generator: {GENERATED_BY}")
    for k, v in extra.items():
        lines.append(f"{k}: {v}")
    lines.append("draft: true")
    lines.append("---")
    return "\n".join(lines)


def render_index(nodes: dict[str, CardNode], stats: GraphStats) -> str:
    fm = _frontmatter({
        "total_cards": stats.n_nodes,
        "total_links": stats.n_edges,
    })

    out = [fm, "", "# 知识库索引（L1）", ""]
    out.append("> 每张卡片一行：摘要 · 入度/出度 · 标签 · 所属 cluster。")
    out.append("> AI 进入对话时优先读 [[TERRAIN]] 获取地形感，需要定位单卡再查此文件。")
    out.append("")

    # 按子目录分组
    by_subdir: dict[str, list[CardNode]] = {sub: [] for sub in VAULT_SUBDIRS}
    for node in nodes.values():
        by_subdir.setdefault(node.card.subdir, []).append(node)

    for sub in VAULT_SUBDIRS:
        cards_in = sorted(by_subdir.get(sub, []), key=lambda n: -n.pagerank)
        if not cards_in:
            continue
        out.append(f"## {sub}/ ({len(cards_in)} 张)")
        out.append("")
        for node in cards_in:
            tag_str = " ".join(f"#{t}" for t in node.card.tags) if node.card.tags else ""
            summary = node.card.summary or "（无摘要）"
            out.append(
                f"- [[{node.name}]] :: {summary} "
                f":: in={node.in_degree} out={node.out_degree} "
                f":: {tag_str} :: cluster={node.cluster or '—'}"
            )
        out.append("")
    return "\n".join(out)


def render_clusters(
    clusters: list[Cluster],
    nodes: dict[str, CardNode],
    stats: GraphStats,
) -> str:
    fm = _frontmatter({
        "algorithm": "leiden",
        "n_clusters": stats.n_clusters,
        "modularity": f"{stats.modularity:.3f}",
    })

    out = [fm, "", "# 知识库聚类（L2）", ""]
    out.append("> 基于 [[wikilink]] 共现模式的自动聚类（Leiden 算法）。")
    out.append("> 簇号锚定 PageRank 顶点的卡片名（`C-XXX`），重跑稳定。")
    out.append("> AI 在为新卡建立链接时必读此文件——用于发现「主题相邻」的已有卡片。")
    out.append("")

    for cluster in clusters:
        machine_summary = summarize_cluster_machine(cluster, nodes)
        out.append(f"## {cluster.cid}")
        out.append("")
        out.append(f"**摘要**：{machine_summary}")
        out.append("")
        out.append(f"**核心节点**（PageRank Top 3）：" + " · ".join(f"[[{n}]]" for n in cluster.top_nodes))
        out.append("")
        out.append("**成员**：")
        for name in cluster.members:
            node = nodes[name]
            out.append(f"- [[{name}]] (PR={node.pagerank:.4f}, in={node.in_degree}, out={node.out_degree})")
        out.append("")
    return "\n".join(out)


def render_graph(
    nodes: dict[str, CardNode],
    clusters: list[Cluster],
    stats: GraphStats,
) -> str:
    cards = [n.card for n in nodes.values()]
    bridges = find_bridges(nodes, clusters, cards, top_k=10)
    no_in, no_out = find_orphans(nodes)

    fm = _frontmatter({
        "n_nodes": stats.n_nodes,
        "n_edges": stats.n_edges,
        "density": f"{stats.density:.4f}",
    })

    out = [fm, "", "# 知识库拓扑", ""]
    out.append(f"**节点数**: {stats.n_nodes}  ·  **边数**: {stats.n_edges}  ·  ")
    out.append(f"**密度**: {stats.density:.4f}  ·  **模块度**: {stats.modularity:.3f}")
    out.append("")

    # PageRank Top
    out.append("## 中心节点（PageRank Top 15）")
    out.append("")
    out.append("| 卡片 | PageRank | 入度 | 出度 | 簇 |")
    out.append("|------|---------:|-----:|-----:|----|")
    top_pr = sorted(nodes.values(), key=lambda n: -n.pagerank)[:15]
    for node in top_pr:
        out.append(
            f"| [[{node.name}]] | {node.pagerank:.4f} | {node.in_degree} | {node.out_degree} | {node.cluster or '—'} |"
        )
    out.append("")

    # 桥梁
    out.append("## 桥梁卡片（连接多个簇）")
    out.append("")
    if bridges:
        out.append("> 这些卡片同时被多个 cluster 引用，往往是关键的概念枢纽。删除它们会断链。")
        out.append("")
        for name, cls in bridges:
            out.append(f"- [[{name}]] 连接 " + " + ".join(cls))
    else:
        out.append("（暂无跨簇桥梁卡——可能是图过于稀疏或聚类把相关卡片归在了同一簇。）")
    out.append("")

    # 孤岛 & 死链
    out.append("## 孤岛卡片")
    out.append("")
    out.append(f"### 无 backlink（{len(no_in)} 张，可能未被引用）")
    out.append("")
    if no_in:
        for name in no_in[:30]:
            out.append(f"- [[{name}]]")
        if len(no_in) > 30:
            out.append(f"- … 还有 {len(no_in) - 30} 张")
    else:
        out.append("（无）")
    out.append("")

    out.append(f"### 无 outlink（{len(no_out)} 张，终端概念或待补充）")
    out.append("")
    if no_out:
        for name in no_out[:30]:
            out.append(f"- [[{name}]]")
        if len(no_out) > 30:
            out.append(f"- … 还有 {len(no_out) - 30} 张")
    else:
        out.append("（无）")
    out.append("")

    out.append("## 死链（链接到不存在的卡片）")
    out.append("")
    if stats.dead_links:
        for src, tgt in stats.dead_links[:30]:
            out.append(f"- [[{src}]] → `[[{tgt}]]` ✕")
        if len(stats.dead_links) > 30:
            out.append(f"- … 还有 {len(stats.dead_links) - 30} 条")
    else:
        out.append("（无死链 ✓）")
    out.append("")

    # 度数分布
    bins = [0, 1, 3, 6, 11, 9999]
    bin_labels = ["0", "1-2", "3-5", "6-10", "11+"]
    in_dist = [0] * len(bin_labels)
    for node in nodes.values():
        for i, b in enumerate(bins[1:]):
            if node.in_degree < b:
                in_dist[i] += 1
                break

    out.append("## 入度分布")
    out.append("")
    out.append("```")
    for label, count in zip(bin_labels, in_dist):
        bar = "█" * count
        out.append(f"{label:>5} {bar} {count}")
    out.append("```")
    out.append("")

    return "\n".join(out)


def render_terrain(
    nodes: dict[str, CardNode],
    clusters: list[Cluster],
    stats: GraphStats,
) -> str:
    overview = terrain_overview(nodes, clusters, stats)

    fm = _frontmatter({})

    out = [fm, "", "# 知识库地形图（L3）", ""]
    out.append("> AI 进入对话时**优先读此文件**。最高层级的全库地形概览。")
    out.append("> 完整索引见 [[INDEX]]，主题簇见 [[CLUSTERS]]，拓扑见 [[GRAPH]]。")
    out.append("")

    # 一句话总览
    out.append("## 一句话总览")
    out.append("")
    main_subdirs = ", ".join(f"{sub} {n}" for sub, n in overview["subdirs"][:4])
    out.append(
        f"**{overview['total_cards']}** 张卡片 · "
        f"**{overview['total_links']}** 条链接 · "
        f"**{overview['n_clusters']}** 个主题簇 · "
        f"密度 {overview['density']:.3f} · 模块度 {overview['modularity']:.3f}"
    )
    out.append("")
    out.append(f"分布：{main_subdirs}")
    out.append("")

    # 簇地图
    out.append("## 簇地图")
    out.append("")
    out.append("| 簇 | 卡数 | 核心节点 | 主导标签 |")
    out.append("|----|----:|--------|--------|")
    for c in clusters:
        top_tags = sorted(c.tags.items(), key=lambda x: -x[1])[:2]
        tag_str = " ".join(f"#{t}" for t, _ in top_tags) if top_tags else "—"
        top_nodes_str = " · ".join(f"[[{n}]]" for n in c.top_nodes)
        out.append(f"| `{c.cid}` | {len(c.members)} | {top_nodes_str} | {tag_str} |")
    out.append("")

    # 结构观察
    out.append("## 结构观察")
    out.append("")
    if overview["dense_clusters"]:
        names = ", ".join(f"`{c.cid}`" for c in overview["dense_clusters"][:3])
        out.append(f"- **密度过高的区域**：{names} 内部链接密集，可能存在概念冗余，建议审视是否合并/拆分")
    if overview["sparse_clusters"]:
        names = ", ".join(f"`{c.cid}`" for c in overview["sparse_clusters"][:5])
        out.append(f"- **密度过低的区域**：{names} 卡数 ≤3，可能需补充关联或并入其他簇")
    if overview["singleton_clusters"]:
        names = ", ".join(f"[[{c.anchor}]]" for c in overview["singleton_clusters"][:5])
        out.append(f"- **孤立簇（卡数=1）**：{names}（{len(overview['singleton_clusters'])} 个，建议建立外联或合并）")
    if stats.dead_links:
        out.append(f"- **死链**：{len(stats.dead_links)} 条，详见 [[GRAPH]]")
    else:
        out.append("- **死链**：✓ 无")
    out.append("")

    # 标签分布
    out.append("## 标签分布")
    out.append("")
    out.append("| 标签 | 卡数 |")
    out.append("|------|----:|")
    for tag, count in overview["tags"]:
        out.append(f"| #{tag} | {count} |")
    out.append("")

    # AI 使用指引
    out.append("## AI 使用指引")
    out.append("")
    out.append("进入本知识库的任何 AI 应按下表读取索引：")
    out.append("")
    out.append("| 场景 | 必读 | 按需 |")
    out.append("|------|------|------|")
    out.append("| 闲聊 / 单卡操作 | [[TERRAIN]] | — |")
    out.append("| 创建新卡 | [[TERRAIN]] | [[CLUSTERS]] 相关簇 |")
    out.append("| 建立关联 | [[TERRAIN]] [[CLUSTERS]] | [[GRAPH]] |")
    out.append("| meta 分析（找空洞 / 合并冗余） | 全读 | [[INDEX]] |")
    out.append("")
    out.append("索引可能过期，更新方式：在 vault 根目录运行 `kb-index update`。")
    out.append("")

    return "\n".join(out)


def write_all(
    vault_root: Path,
    nodes: dict[str, CardNode],
    clusters: list[Cluster],
    stats: GraphStats,
) -> dict[str, Path]:
    """渲染并写入 4 个产物文件。返回 {name: path}。"""
    products = {
        "INDEX": render_index(nodes, stats),
        "CLUSTERS": render_clusters(clusters, nodes, stats),
        "GRAPH": render_graph(nodes, clusters, stats),
        "TERRAIN": render_terrain(nodes, clusters, stats),
    }
    paths = {}
    for name, content in products.items():
        path = vault_root / f"{name}.md"
        path.write_text(content, encoding="utf-8")
        paths[name] = path
    return paths
