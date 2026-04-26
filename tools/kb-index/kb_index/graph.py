"""图构建 + Leiden 聚类 + 拓扑度量。"""

from __future__ import annotations

from dataclasses import dataclass, field

import igraph as ig
import leidenalg as la

from .parse import Card


@dataclass
class CardNode:
    name: str
    card: Card
    in_degree: int = 0
    out_degree: int = 0
    pagerank: float = 0.0
    cluster: str = ""  # cluster id, e.g. "C-AI镜像效应"


@dataclass
class Cluster:
    cid: str  # 锚定 PageRank 顶点的卡片名: "C-{anchor_name}"
    members: list[str] = field(default_factory=list)
    anchor: str = ""
    top_nodes: list[str] = field(default_factory=list)  # PageRank top 3
    tags: dict[str, int] = field(default_factory=dict)


@dataclass
class GraphStats:
    n_nodes: int
    n_edges: int
    density: float
    modularity: float
    n_clusters: int
    dead_links: list[tuple[str, str]] = field(default_factory=list)  # (source_card, missing_target)


def build_graph(cards: list[Card]) -> tuple[ig.Graph, dict[str, int], list[tuple[str, str]]]:
    """构建有向图。返回 (graph, name->index, dead_links)。"""
    name_to_idx = {card.name: i for i, card in enumerate(cards)}

    edges: list[tuple[int, int]] = []
    dead_links: list[tuple[str, str]] = []

    for card in cards:
        src = name_to_idx[card.name]
        for target in card.links_out:
            if target in name_to_idx:
                edges.append((src, name_to_idx[target]))
            else:
                dead_links.append((card.name, target))

    g = ig.Graph(n=len(cards), edges=edges, directed=True)
    g.vs["name"] = [c.name for c in cards]
    return g, name_to_idx, dead_links


def cluster_and_rank(
    cards: list[Card],
    resolution: float = 1.0,
    seed: int = 42,
) -> tuple[dict[str, CardNode], list[Cluster], GraphStats]:
    """跑 Leiden 聚类 + PageRank。返回节点字典、簇列表、图统计。"""
    g, name_to_idx, dead_links = build_graph(cards)

    # 节点字典
    nodes: dict[str, CardNode] = {
        card.name: CardNode(name=card.name, card=card) for card in cards
    }

    # 入度/出度（来自有向图）
    for v in g.vs:
        node = nodes[v["name"]]
        node.in_degree = g.degree(v.index, mode="in")
        node.out_degree = g.degree(v.index, mode="out")

    # PageRank（用有向图）
    if g.ecount() > 0:
        prs = g.pagerank(damping=0.85, directed=True)
    else:
        prs = [1.0 / max(g.vcount(), 1)] * g.vcount()

    for i, v in enumerate(g.vs):
        nodes[v["name"]].pagerank = prs[i]

    # Leiden 聚类（用无向版本，社区检测对方向不敏感）
    g_und = g.as_undirected(mode="collapse")

    if g_und.ecount() == 0:
        # 没有边：每个节点自成一簇（或全归一簇）
        partition = [list(range(g.vcount()))] if g.vcount() > 0 else []
        modularity = 0.0
    else:
        part = la.find_partition(
            g_und,
            la.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            seed=seed,
        )
        partition = list(part)
        modularity = part.modularity

    # 给每个簇命名（锚定 PageRank 最高的节点），并填充节点的 cluster 字段
    clusters: list[Cluster] = []
    for member_indices in partition:
        if not member_indices:
            continue
        member_names = [g.vs[i]["name"] for i in member_indices]
        # PageRank 排序
        member_names.sort(key=lambda n: nodes[n].pagerank, reverse=True)
        anchor = member_names[0]
        cid = f"C-{anchor}"
        top_nodes = member_names[:3]

        # 标签直方图
        tag_counts: dict[str, int] = {}
        for n in member_names:
            for t in nodes[n].card.tags:
                tag_counts[t] = tag_counts.get(t, 0) + 1

        cl = Cluster(
            cid=cid,
            members=member_names,
            anchor=anchor,
            top_nodes=top_nodes,
            tags=tag_counts,
        )
        clusters.append(cl)
        for n in member_names:
            nodes[n].cluster = cid

    # 按簇大小降序
    clusters.sort(key=lambda c: len(c.members), reverse=True)

    n_nodes = g.vcount()
    n_edges = g.ecount()
    density = g.density() if n_nodes > 1 else 0.0

    stats = GraphStats(
        n_nodes=n_nodes,
        n_edges=n_edges,
        density=density,
        modularity=modularity,
        n_clusters=len(clusters),
        dead_links=dead_links,
    )

    return nodes, clusters, stats


def find_bridges(
    nodes: dict[str, CardNode],
    clusters: list[Cluster],
    cards: list[Card],
    top_k: int = 10,
) -> list[tuple[str, list[str]]]:
    """识别桥梁卡片：链接到多个不同 cluster 的卡。返回 [(card_name, [cluster_ids...])]"""
    name_to_card = {c.name: c for c in cards}
    bridges: list[tuple[str, list[str], int]] = []  # (name, clusters_touched, bridge_count)

    for name, node in nodes.items():
        card = name_to_card[name]
        own = node.cluster
        touched: dict[str, int] = {}
        for tgt in card.links_out:
            if tgt in nodes:
                tgt_cl = nodes[tgt].cluster
                if tgt_cl and tgt_cl != own:
                    touched[tgt_cl] = touched.get(tgt_cl, 0) + 1
        if len(touched) >= 2:
            bridges.append((name, sorted(touched.keys()), sum(touched.values())))

    bridges.sort(key=lambda b: (-len(b[1]), -b[2]))
    return [(name, cls) for name, cls, _ in bridges[:top_k]]


def find_orphans(nodes: dict[str, CardNode]) -> tuple[list[str], list[str]]:
    """孤儿卡：无入边的（被遗忘）和无出边的（终端）。"""
    no_inbound = sorted([n for n, node in nodes.items() if node.in_degree == 0])
    no_outbound = sorted([n for n, node in nodes.items() if node.out_degree == 0])
    return no_inbound, no_outbound
