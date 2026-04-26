"""机械摘要生成（无 LLM 依赖）。"""

from __future__ import annotations

from collections import Counter

from .graph import CardNode, Cluster, GraphStats


def summarize_cluster_machine(cluster: Cluster, nodes: dict[str, CardNode]) -> str:
    """簇主题机械摘要：核心节点 + 主导标签 + 卡数。"""
    top_tags = sorted(cluster.tags.items(), key=lambda x: -x[1])[:3]
    tag_str = " ".join(f"#{t}({c})" for t, c in top_tags) if top_tags else "（无标签）"
    top3 = " · ".join(f"[[{n}]]" for n in cluster.top_nodes)
    return f"{len(cluster.members)} 张 · 核心: {top3} · 主导标签: {tag_str}"


def terrain_overview(
    nodes: dict[str, CardNode],
    clusters: list[Cluster],
    stats: GraphStats,
) -> dict:
    """L3 地形图所需的全局统计。"""
    # 标签分布
    tag_counter: Counter[str] = Counter()
    for node in nodes.values():
        for t in node.card.tags:
            tag_counter[t] += 1

    # 子目录分布
    subdir_counter: Counter[str] = Counter()
    for node in nodes.values():
        subdir_counter[node.card.subdir] += 1

    # 簇密度评估（>3 张算"高密度"）
    dense = [c for c in clusters if len(c.members) >= 8]
    sparse = [c for c in clusters if 1 < len(c.members) <= 3]
    singletons = [c for c in clusters if len(c.members) == 1]

    return {
        "total_cards": len(nodes),
        "total_links": stats.n_edges,
        "n_clusters": stats.n_clusters,
        "density": stats.density,
        "modularity": stats.modularity,
        "tags": tag_counter.most_common(),
        "subdirs": subdir_counter.most_common(),
        "dense_clusters": dense,
        "sparse_clusters": sparse,
        "singleton_clusters": singletons,
    }
