"""kb-index CLI: build / update / doctor / check."""

from __future__ import annotations

import time
from pathlib import Path

import typer

from .graph import cluster_and_rank
from .parse import scan_vault
from .render import write_all

app = typer.Typer(help="Knowledge base index generator (4-layer markdown digest).")


def _resolve_vault(vault: Path | None) -> Path:
    if vault is None:
        # 默认：从当前目录向上找含 0-inbox / 1-concepts 的目录
        cur = Path.cwd().resolve()
        for parent in [cur, *cur.parents]:
            if (parent / "1-concepts").is_dir():
                return parent
        raise typer.BadParameter("未找到 vault 根目录（含 1-concepts/），请用 --vault 指定。")
    return vault.resolve()


@app.command()
def build(
    vault: Path = typer.Option(None, "--vault", "-v", help="vault 根目录"),
    resolution: float = typer.Option(1.0, "--resolution", "-r", help="Leiden resolution"),
    seed: int = typer.Option(42, "--seed", help="Leiden 随机种子"),
):
    """全量构建 4 个产物文件。"""
    t0 = time.perf_counter()
    root = _resolve_vault(vault)
    typer.echo(f"vault: {root}")

    cards = scan_vault(root)
    typer.echo(f"scanned: {len(cards)} cards")

    nodes, clusters, stats = cluster_and_rank(cards, resolution=resolution, seed=seed)
    typer.echo(
        f"graph: {stats.n_nodes} nodes, {stats.n_edges} edges, "
        f"density={stats.density:.4f}, modularity={stats.modularity:.3f}, "
        f"clusters={stats.n_clusters}"
    )
    if stats.dead_links:
        typer.echo(f"dead links: {len(stats.dead_links)}")

    paths = write_all(root, nodes, clusters, stats)
    elapsed = time.perf_counter() - t0
    typer.echo(f"wrote {len(paths)} files in {elapsed:.2f}s:")
    for name, path in paths.items():
        size_kb = path.stat().st_size / 1024
        typer.echo(f"  {name}.md  {size_kb:6.1f} KB")


@app.command()
def update(
    vault: Path = typer.Option(None, "--vault", "-v"),
    resolution: float = typer.Option(1.0, "--resolution", "-r"),
    seed: int = typer.Option(42, "--seed"),
    with_llm: bool = typer.Option(False, "--with-llm", help="对缺 summary 的卡调 Claude 补摘要并写回 frontmatter"),
    refill_all: bool = typer.Option(False, "--refill-all", help="搭配 --with-llm 使用：覆盖所有卡的 summary（不只缺的）"),
):
    """更新索引。--with-llm 触发 LLM 摘要补全。"""
    if with_llm:
        from .llm import fill_missing_summaries
        root = _resolve_vault(vault)
        cards = scan_vault(root)
        fill_missing_summaries(cards, only_missing=not refill_all)
        typer.echo("---")
    build(vault=vault, resolution=resolution, seed=seed)


@app.command()
def doctor(
    vault: Path = typer.Option(None, "--vault", "-v"),
):
    """诊断：列出孤儿、死链、过密簇等问题。"""
    root = _resolve_vault(vault)
    cards = scan_vault(root)
    nodes, clusters, stats = cluster_and_rank(cards)

    from .graph import find_orphans

    no_in, no_out = find_orphans(nodes)
    typer.echo(f"=== KB Doctor ===")
    typer.echo(f"vault: {root}")
    typer.echo(f"cards: {stats.n_nodes}  edges: {stats.n_edges}  clusters: {stats.n_clusters}")
    typer.echo("")
    typer.echo(f"孤儿（无 backlink）: {len(no_in)}")
    for n in no_in[:10]:
        typer.echo(f"  - {n}")
    if len(no_in) > 10:
        typer.echo(f"  ... 还有 {len(no_in) - 10}")
    typer.echo("")
    typer.echo(f"终端（无 outlink）: {len(no_out)}")
    for n in no_out[:10]:
        typer.echo(f"  - {n}")
    if len(no_out) > 10:
        typer.echo(f"  ... 还有 {len(no_out) - 10}")
    typer.echo("")
    typer.echo(f"死链: {len(stats.dead_links)}")
    for src, tgt in stats.dead_links[:10]:
        typer.echo(f"  - {src} → {tgt}")
    if len(stats.dead_links) > 10:
        typer.echo(f"  ... 还有 {len(stats.dead_links) - 10}")
    typer.echo("")
    singletons = [c for c in clusters if len(c.members) == 1]
    typer.echo(f"单点簇: {len(singletons)}")
    for c in singletons[:10]:
        typer.echo(f"  - {c.anchor}")


@app.command()
def check(
    vault: Path = typer.Option(None, "--vault", "-v"),
):
    """干跑：解析 + 聚类，但不写文件。"""
    root = _resolve_vault(vault)
    cards = scan_vault(root)
    nodes, clusters, stats = cluster_and_rank(cards)
    typer.echo(f"OK: {stats.n_nodes} nodes, {stats.n_edges} edges, {stats.n_clusters} clusters")


@app.command()
def visualize(
    vault: Path = typer.Option(None, "--vault", "-v"),
    output: Path = typer.Option(None, "--output", "-o", help="输出 HTML 路径，默认 vault/KB-GRAPH.html"),
    resolution: float = typer.Option(1.0, "--resolution", "-r"),
    seed: int = typer.Option(42, "--seed"),
    open_browser: bool = typer.Option(False, "--open", help="生成后用浏览器打开"),
):
    """生成交互式力导向图（D3.js）。"""
    from .visualize import build_graph_data, render_html

    root = _resolve_vault(vault)
    cards = scan_vault(root)
    nodes, clusters, stats = cluster_and_rank(cards, resolution=resolution, seed=seed)
    data = build_graph_data(nodes, clusters, stats)
    html = render_html(data)

    out = output or (root / "KB-GRAPH.html")
    out.write_text(html, encoding="utf-8")
    size_kb = out.stat().st_size / 1024
    typer.echo(f"wrote {out.name}  ({size_kb:.1f} KB)  · {stats.n_nodes} nodes, {stats.n_edges} edges, {stats.n_clusters} clusters")
    typer.echo(f"open: file://{out.resolve()}")

    if open_browser:
        import webbrowser
        webbrowser.open(f"file://{out.resolve()}")


if __name__ == "__main__":
    app()
