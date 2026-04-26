"""生成知识库的交互式力导向图（D3.js v7）。"""

from __future__ import annotations

import json
from datetime import datetime

from .graph import CardNode, Cluster, GraphStats


# 簇配色（按出现顺序循环）—— 蓝图色调 + 高对比
CLUSTER_COLORS = [
    "#5fd3c5",  # cyan
    "#f0b45c",  # amber
    "#d77b6c",  # rust
    "#b690e8",  # violet
    "#7ec795",  # green
    "#e878a8",  # pink
    "#88aef0",  # sky
    "#f7d56a",  # gold
    "#9bcfd1",  # pale cyan
    "#c9986b",  # tan
]


def build_graph_data(
    nodes: dict[str, CardNode],
    clusters: list[Cluster],
    stats: GraphStats,
) -> dict:
    cluster_index = {c.cid: i for i, c in enumerate(clusters)}
    cluster_color = {c.cid: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, c in enumerate(clusters)}

    d3_nodes = []
    for name, node in nodes.items():
        d3_nodes.append({
            "id": name,
            "subdir": node.card.subdir,
            "tags": node.card.tags,
            "summary": node.card.summary or "",
            "in_degree": node.in_degree,
            "out_degree": node.out_degree,
            "pagerank": node.pagerank,
            "cluster": node.cluster,
            "cluster_idx": cluster_index.get(node.cluster, -1),
            "color": cluster_color.get(node.cluster, "#7d96b3"),
        })

    d3_links = []
    for name, node in nodes.items():
        for tgt in node.card.links_out:
            if tgt in nodes:
                d3_links.append({"source": name, "target": tgt})

    d3_clusters = []
    for c in clusters:
        d3_clusters.append({
            "cid": c.cid,
            "anchor": c.anchor,
            "size": len(c.members),
            "top_nodes": c.top_nodes,
            "color": cluster_color[c.cid],
        })

    return {
        "nodes": d3_nodes,
        "links": d3_links,
        "clusters": d3_clusters,
        "stats": {
            "n_nodes": stats.n_nodes,
            "n_edges": stats.n_edges,
            "density": stats.density,
            "modularity": stats.modularity,
            "n_clusters": stats.n_clusters,
        },
        "generated": datetime.now().astimezone().isoformat(),
    }


def render_html(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return _HTML_TEMPLATE.replace("__GRAPH_DATA__", payload)


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KB Graph · 知识库拓扑可视化</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
:root {
  --ink-0: #060d18;
  --ink-1: #0a1628;
  --ink-2: #102236;
  --ink-3: #18304a;
  --line: #2a4565;
  --paper: #e6ecf2;
  --paper-2: #b9c8db;
  --paper-3: #7d96b3;
  --paper-4: #5a738f;
  --cyan: #5fd3c5;
  --amber: #f0b45c;
  --rust: #d77b6c;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

html, body {
  height: 100%;
  width: 100%;
  font-family: 'PingFang SC', 'Hiragino Sans GB', -apple-system, sans-serif;
  background: var(--ink-1);
  color: var(--paper);
  overflow: hidden;
}

body {
  background-image:
    linear-gradient(to right, rgba(95,211,197,0.04) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(95,211,197,0.04) 1px, transparent 1px),
    radial-gradient(ellipse 60% 60% at 50% 50%, rgba(95,211,197,0.05), transparent 70%);
  background-size: 24px 24px, 24px 24px, 100% 100%;
}

.mono {
  font-family: 'SF Mono', Menlo, Monaco, monospace;
}

#app {
  position: fixed;
  inset: 0;
  display: grid;
  grid-template-areas:
    "head head"
    "graph side";
  grid-template-rows: 56px 1fr;
  grid-template-columns: 1fr 320px;
}

header {
  grid-area: head;
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 24px;
  align-items: center;
  padding: 0 24px;
  border-bottom: 1px solid var(--line);
  background: rgba(10, 22, 40, 0.85);
  backdrop-filter: blur(8px);
  z-index: 10;
}
header .title {
  font-family: 'SF Mono', Menlo, monospace;
  font-size: 14px;
  letter-spacing: 0.15em;
  color: var(--cyan);
}
header .stats {
  display: flex;
  gap: 24px;
  font-family: 'SF Mono', Menlo, monospace;
  font-size: 12px;
  color: var(--paper-3);
}
header .stats b { color: var(--paper); margin-right: 4px; font-weight: 500; }
header input.search {
  width: 240px;
  background: var(--ink-2);
  border: 1px solid var(--line);
  color: var(--paper);
  padding: 8px 12px;
  font-family: 'SF Mono', Menlo, monospace;
  font-size: 12px;
  outline: none;
}
header input.search:focus { border-color: var(--cyan); }

#graph {
  grid-area: graph;
  position: relative;
  overflow: hidden;
}
#graph svg {
  display: block;
  width: 100%;
  height: 100%;
}

aside {
  grid-area: side;
  border-left: 1px solid var(--line);
  background: rgba(10, 22, 40, 0.7);
  backdrop-filter: blur(8px);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.panel-section {
  padding: 16px 20px;
  border-bottom: 1px dashed var(--line);
}
.panel-section h3 {
  font-family: 'SF Mono', Menlo, monospace;
  font-size: 10px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--paper-4);
  margin-bottom: 10px;
  font-weight: 500;
}

#legend { padding: 14px 20px; flex-shrink: 0; }
#legend ul { list-style: none; }
#legend li {
  display: grid;
  grid-template-columns: 12px 1fr auto;
  gap: 10px;
  align-items: center;
  padding: 4px 0;
  cursor: pointer;
  user-select: none;
  font-family: 'SF Mono', Menlo, monospace;
  font-size: 11px;
  color: var(--paper-2);
  transition: color 0.2s;
}
#legend li:hover { color: var(--paper); }
#legend li.muted { opacity: 0.3; }
#legend .dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
}
#legend .anchor { color: var(--paper); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
#legend .count { color: var(--paper-4); font-size: 10px; }

#detail {
  flex: 1;
  overflow-y: auto;
  padding: 16px 20px;
  font-size: 13px;
}
#detail .name {
  font-size: 16px;
  font-weight: 500;
  margin-bottom: 8px;
  color: var(--paper);
}
#detail .summary {
  color: var(--paper-2);
  line-height: 1.7;
  margin-bottom: 16px;
}
#detail .field {
  display: grid;
  grid-template-columns: 80px 1fr;
  gap: 8px;
  font-family: 'SF Mono', Menlo, monospace;
  font-size: 11px;
  padding: 4px 0;
}
#detail .field .key { color: var(--paper-4); }
#detail .field .val { color: var(--paper); }
#detail .field .val .tag { color: var(--cyan); }
#detail .field .val .clu { color: var(--amber); }
#detail .empty {
  color: var(--paper-4);
  font-style: italic;
  text-align: center;
  padding: 32px 0;
  font-size: 12px;
}
#detail h4 {
  font-family: 'SF Mono', Menlo, monospace;
  font-size: 10px;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--paper-4);
  margin: 16px 0 6px;
  font-weight: 500;
}
#detail .neighbors {
  list-style: none;
  font-family: 'SF Mono', Menlo, monospace;
  font-size: 11px;
}
#detail .neighbors li {
  padding: 3px 6px;
  cursor: pointer;
  color: var(--paper-2);
  border-left: 2px solid transparent;
  transition: all 0.15s;
}
#detail .neighbors li:hover {
  color: var(--paper);
  border-left-color: var(--cyan);
  background: rgba(95, 211, 197, 0.05);
}

.tooltip {
  position: absolute;
  pointer-events: none;
  background: var(--ink-0);
  border: 1px solid var(--cyan);
  padding: 8px 12px;
  font-family: 'SF Mono', Menlo, monospace;
  font-size: 11px;
  color: var(--paper);
  z-index: 100;
  max-width: 320px;
  opacity: 0;
  transition: opacity 0.15s;
}
.tooltip.show { opacity: 1; }
.tooltip .t-name { color: var(--cyan); margin-bottom: 4px; font-weight: 500; }
.tooltip .t-meta { color: var(--paper-3); font-size: 10px; }

/* SVG element styles */
.link {
  stroke: rgba(125, 150, 179, 0.18);
  stroke-width: 0.6;
  fill: none;
  pointer-events: none;
}
.link.hl-out {
  stroke: var(--cyan);
  stroke-width: 1.5;
}
.link.hl-in {
  stroke: var(--amber);
  stroke-width: 1.5;
}
.link.dimmed { stroke: rgba(125, 150, 179, 0.04); }

.node {
  cursor: pointer;
  stroke: var(--ink-1);
  stroke-width: 1.2;
  transition: stroke-width 0.15s;
}
.node:hover, .node.selected {
  stroke: var(--paper);
  stroke-width: 2;
}
.node.dimmed { opacity: 0.18; }

.node-label {
  font-family: 'PingFang SC', sans-serif;
  font-size: 9px;
  fill: var(--paper-2);
  pointer-events: none;
  text-shadow:
    -1px -1px 0 var(--ink-1),
    1px -1px 0 var(--ink-1),
    -1px 1px 0 var(--ink-1),
    1px 1px 0 var(--ink-1);
}
.node-label.dimmed { opacity: 0.15; }
.node-label.hl { font-size: 11px; fill: var(--paper); font-weight: 500; }

footer {
  position: fixed;
  bottom: 8px;
  left: 16px;
  font-family: 'SF Mono', Menlo, monospace;
  font-size: 10px;
  color: var(--paper-4);
  pointer-events: none;
  letter-spacing: 0.1em;
  z-index: 5;
}
</style>
</head>
<body>

<div id="app">
  <header>
    <div class="title">KB · GRAPH</div>
    <div class="stats">
      <span><b id="s-nodes">—</b>nodes</span>
      <span><b id="s-edges">—</b>edges</span>
      <span><b id="s-clusters">—</b>clusters</span>
      <span><b id="s-mod">—</b>modularity</span>
      <span><b id="s-density">—</b>density</span>
    </div>
    <input class="search mono" id="search" placeholder="搜索卡片…  (/)" />
  </header>

  <div id="graph">
    <svg id="svg"></svg>
    <div class="tooltip" id="tooltip"></div>
  </div>

  <aside>
    <div class="panel-section" id="legend-section">
      <h3>Clusters · 点击切换显隐</h3>
      <ul id="legend"></ul>
    </div>
    <div id="detail" class="panel-section">
      <div class="empty">悬停或点击节点查看详情<br><br>拖拽 = 移动节点<br>滚轮 = 缩放<br>右键空白处 = 重置</div>
    </div>
  </aside>
</div>

<footer>KB-INDEX · INTERACTIVE TOPOLOGY · D3.js</footer>

<script>
const DATA = __GRAPH_DATA__;

// ---------- Stats header ----------
document.getElementById('s-nodes').textContent = DATA.stats.n_nodes;
document.getElementById('s-edges').textContent = DATA.stats.n_edges;
document.getElementById('s-clusters').textContent = DATA.stats.n_clusters;
document.getElementById('s-mod').textContent = DATA.stats.modularity.toFixed(3);
document.getElementById('s-density').textContent = DATA.stats.density.toFixed(3);

// ---------- Setup SVG ----------
const svgEl = document.getElementById('svg');
const graphEl = document.getElementById('graph');
const tooltipEl = document.getElementById('tooltip');

let width = graphEl.clientWidth;
let height = graphEl.clientHeight;

const svg = d3.select('#svg')
  .attr('viewBox', [0, 0, width, height]);

const g = svg.append('g');

// Zoom behavior
const zoom = d3.zoom()
  .scaleExtent([0.2, 4])
  .on('zoom', (event) => g.attr('transform', event.transform));
svg.call(zoom);

// Right-click resets view
svg.on('contextmenu', (event) => {
  event.preventDefault();
  svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
});

// ---------- Build adjacency ----------
const inEdges = new Map();
const outEdges = new Map();
DATA.nodes.forEach(n => { inEdges.set(n.id, []); outEdges.set(n.id, []); });
DATA.links.forEach(l => {
  outEdges.get(l.source).push(l.target);
  inEdges.get(l.target).push(l.source);
});

// ---------- Force simulation ----------
const prMax = d3.max(DATA.nodes, d => d.pagerank);
const radiusOf = d => 4 + Math.sqrt(d.pagerank / prMax) * 14;

const sim = d3.forceSimulation(DATA.nodes)
  .force('link', d3.forceLink(DATA.links).id(d => d.id).distance(40).strength(0.3))
  .force('charge', d3.forceManyBody().strength(-180))
  .force('center', d3.forceCenter(width / 2, height / 2))
  .force('collide', d3.forceCollide().radius(d => radiusOf(d) + 4))
  .force('cluster', clusterForce(0.04));

function clusterForce(strength) {
  // 将同簇节点拉向簇质心
  const centroids = new Map();
  return alpha => {
    centroids.clear();
    DATA.nodes.forEach(n => {
      if (!centroids.has(n.cluster)) centroids.set(n.cluster, { x: 0, y: 0, n: 0 });
      const c = centroids.get(n.cluster);
      c.x += n.x; c.y += n.y; c.n += 1;
    });
    centroids.forEach(c => { c.x /= c.n; c.y /= c.n; });
    DATA.nodes.forEach(n => {
      const c = centroids.get(n.cluster);
      if (!c) return;
      n.vx += (c.x - n.x) * strength * alpha;
      n.vy += (c.y - n.y) * strength * alpha;
    });
  };
}

// ---------- Draw links & nodes ----------
const link = g.append('g').attr('class', 'links').selectAll('line')
  .data(DATA.links)
  .join('line')
  .attr('class', 'link');

const node = g.append('g').attr('class', 'nodes').selectAll('circle')
  .data(DATA.nodes)
  .join('circle')
  .attr('class', 'node')
  .attr('r', radiusOf)
  .attr('fill', d => d.color)
  .call(drag(sim))
  .on('mouseenter', onHover)
  .on('mouseleave', onLeave)
  .on('click', onClick);

const label = g.append('g').attr('class', 'labels').selectAll('text')
  .data(DATA.nodes)
  .join('text')
  .attr('class', 'node-label')
  .attr('dy', d => -radiusOf(d) - 3)
  .attr('text-anchor', 'middle')
  .text(d => d.id.length > 12 ? d.id.slice(0, 11) + '…' : d.id);

sim.on('tick', () => {
  link
    .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
    .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
  node.attr('cx', d => d.x).attr('cy', d => d.y);
  label.attr('x', d => d.x).attr('y', d => d.y);
});

function drag(simulation) {
  return d3.drag()
    .on('start', (event, d) => {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x; d.fy = d.y;
    })
    .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
    .on('end', (event, d) => {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null; d.fy = null;
    });
}

// ---------- Hover / Tooltip ----------
function onHover(event, d) {
  highlightNeighborhood(d);
  showTooltip(event, d);
  showDetail(d);
}
function onLeave() {
  hideTooltip();
  clearHighlight();
}
function onClick(event, d) {
  event.stopPropagation();
  showDetail(d, true);  // pin to detail panel
}

function showTooltip(event, d) {
  tooltipEl.innerHTML = `
    <div class="t-name">${d.id}</div>
    <div class="t-meta">${d.cluster} · PR=${d.pagerank.toFixed(4)} · in=${d.in_degree} out=${d.out_degree}</div>
    ${d.summary ? `<div style="margin-top:6px;color:#b9c8db;font-size:11px;line-height:1.4;">${escapeHtml(d.summary)}</div>` : ''}
  `;
  const rect = graphEl.getBoundingClientRect();
  const x = event.clientX - rect.left + 12;
  const y = event.clientY - rect.top + 12;
  tooltipEl.style.left = x + 'px';
  tooltipEl.style.top = y + 'px';
  tooltipEl.classList.add('show');
}
function hideTooltip() { tooltipEl.classList.remove('show'); }

function highlightNeighborhood(d) {
  const outs = new Set(outEdges.get(d.id));
  const ins = new Set(inEdges.get(d.id));
  const nbrs = new Set([...outs, ...ins, d.id]);

  node.classed('dimmed', n => !nbrs.has(n.id));
  label.classed('dimmed', n => !nbrs.has(n.id))
       .classed('hl', n => n.id === d.id);
  link
    .classed('dimmed', l => !(l.source.id === d.id || l.target.id === d.id))
    .classed('hl-out', l => l.source.id === d.id)
    .classed('hl-in', l => l.target.id === d.id);
}
function clearHighlight() {
  node.classed('dimmed', false).classed('selected', false);
  label.classed('dimmed', false).classed('hl', false);
  link.classed('dimmed', false).classed('hl-out', false).classed('hl-in', false);
}

function showDetail(d, pinned) {
  const outs = outEdges.get(d.id);
  const ins = inEdges.get(d.id);
  document.getElementById('detail').innerHTML = `
    <div class="name">${escapeHtml(d.id)}</div>
    ${d.summary ? `<div class="summary">${escapeHtml(d.summary)}</div>` : ''}
    <div class="field"><span class="key">cluster</span><span class="val"><span class="clu">${d.cluster}</span></span></div>
    <div class="field"><span class="key">subdir</span><span class="val">${d.subdir}/</span></div>
    <div class="field"><span class="key">PageRank</span><span class="val">${d.pagerank.toFixed(4)}</span></div>
    <div class="field"><span class="key">in / out</span><span class="val">${d.in_degree} / ${d.out_degree}</span></div>
    <div class="field"><span class="key">tags</span><span class="val">${(d.tags||[]).map(t => '<span class="tag">#'+escapeHtml(t)+'</span>').join(' ') || '—'}</span></div>
    <h4>OUTGOING (${outs.length})</h4>
    <ul class="neighbors">${outs.map(t => `<li data-id="${escapeAttr(t)}">→ ${escapeHtml(t)}</li>`).join('') || '<li style="color:#5a738f;cursor:default;">（无）</li>'}</ul>
    <h4>BACKLINKS (${ins.length})</h4>
    <ul class="neighbors">${ins.map(s => `<li data-id="${escapeAttr(s)}">← ${escapeHtml(s)}</li>`).join('') || '<li style="color:#5a738f;cursor:default;">（无）</li>'}</ul>
  `;
  document.querySelectorAll('#detail .neighbors li[data-id]').forEach(li => {
    li.addEventListener('click', () => {
      const target = DATA.nodes.find(n => n.id === li.dataset.id);
      if (target) {
        highlightNeighborhood(target);
        showDetail(target, true);
      }
    });
  });
}

// ---------- Legend ----------
const hiddenClusters = new Set();
const legend = d3.select('#legend');
legend.selectAll('li')
  .data(DATA.clusters)
  .join('li')
  .attr('data-cid', d => d.cid)
  .html(d => `
    <span class="dot" style="background:${d.color}"></span>
    <span class="anchor">${escapeHtml(d.anchor)}</span>
    <span class="count">${d.size}</span>
  `)
  .on('click', function(event, d) {
    if (hiddenClusters.has(d.cid)) hiddenClusters.delete(d.cid);
    else hiddenClusters.add(d.cid);
    d3.select(this).classed('muted', hiddenClusters.has(d.cid));
    applyClusterFilter();
  });

function applyClusterFilter() {
  node.style('display', d => hiddenClusters.has(d.cluster) ? 'none' : null);
  label.style('display', d => hiddenClusters.has(d.cluster) ? 'none' : null);
  link.style('display', l =>
    (hiddenClusters.has(l.source.cluster) || hiddenClusters.has(l.target.cluster)) ? 'none' : null
  );
}

// ---------- Search ----------
const searchEl = document.getElementById('search');
document.addEventListener('keydown', (e) => {
  if (e.key === '/' && document.activeElement !== searchEl) {
    e.preventDefault();
    searchEl.focus();
  }
  if (e.key === 'Escape') {
    searchEl.value = '';
    searchEl.blur();
    clearHighlight();
  }
});
searchEl.addEventListener('input', () => {
  const q = searchEl.value.trim().toLowerCase();
  if (!q) { clearHighlight(); return; }
  const matches = DATA.nodes.filter(n => n.id.toLowerCase().includes(q));
  if (matches.length === 0) { clearHighlight(); return; }
  const ids = new Set(matches.map(m => m.id));
  node.classed('dimmed', n => !ids.has(n.id));
  label.classed('dimmed', n => !ids.has(n.id)).classed('hl', n => ids.has(n.id));
  link.classed('dimmed', true);
  if (matches.length === 1) showDetail(matches[0], true);
});

// ---------- Helpers ----------
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}
function escapeAttr(s) { return escapeHtml(s).replace(/"/g, '&quot;'); }

// ---------- Resize ----------
window.addEventListener('resize', () => {
  width = graphEl.clientWidth;
  height = graphEl.clientHeight;
  svg.attr('viewBox', [0, 0, width, height]);
  sim.force('center', d3.forceCenter(width / 2, height / 2));
  sim.alpha(0.3).restart();
});
</script>
</body>
</html>
"""
