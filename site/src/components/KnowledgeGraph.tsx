import type { CSSProperties } from "react";
import { useEffect, useRef, useState } from "react";

import { buildGraphRelationIndex, createGraphHoverState, reduceEdgeForHover, reduceNodeForHover } from "../lib/graph-focus";
import { getGraphDisplayConfig, getGraphHoverTheme, getGraphInteractionConfig, getGraphSectionTheme, type GraphSurface } from "../lib/graph-theme";
import { withBasePath } from "../lib/url";
import type { SectionKey } from "../lib/sections";

interface GraphNode {
  id: string;
  title: string;
  sectionKey: SectionKey;
  url: string;
  tags: string[];
}

interface GraphEdge {
  source: string;
  target: string;
}

interface GraphPayload {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

interface SectionOption {
  key: SectionKey;
  label: string;
}

interface KnowledgeGraphProps {
  fullGraph: GraphPayload;
  defaultGraph: GraphPayload;
  sectionOptions: SectionOption[];
  showFilters?: boolean;
  allowProjectToggle?: boolean;
  initialIncludeProjects?: boolean;
  height?: number;
  surface?: GraphSurface;
}

function hashToFloat(value: string): number {
  let hash = 0;

  for (const character of value) {
    hash = (hash * 31 + character.charCodeAt(0)) >>> 0;
  }

  return (hash % 1000) / 1000;
}

function createLayout(nodes: GraphNode[], spread: number) {
  const groups = new Map<SectionKey, GraphNode[]>();
  const sectionCenters: Record<SectionKey, { x: number; y: number }> = {
    inbox: { x: -7.5, y: -4.8 },
    concepts: { x: -6.2, y: 1.2 },
    explorations: { x: 0.8, y: 6.2 },
    projects: { x: 7, y: -4.6 },
    references: { x: 7.4, y: 1.6 }
  };
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));

  for (const node of nodes) {
    const group = groups.get(node.sectionKey) ?? [];
    group.push(node);
    groups.set(node.sectionKey, group);
  }

  const positions = new Map<string, { x: number; y: number }>();

  Object.entries(sectionCenters).forEach(([sectionKey, center]) => {
    const group = groups.get(sectionKey as SectionKey) ?? [];

    group
      .slice()
      .sort((left, right) => left.title.localeCompare(right.title, "zh-Hans-CN"))
      .forEach((node, nodeIndex) => {
        const angle = nodeIndex * goldenAngle;
        const radius = 1.2 + Math.sqrt(nodeIndex + 1) * spread;
        const wobble = (hashToFloat(node.id) - 0.5) * 0.55;
        const x = center.x + Math.cos(angle) * radius + wobble;
        const y = center.y + Math.sin(angle) * radius + wobble;
        positions.set(node.id, { x, y });
      });
  });

  return positions;
}

function drawRoundedRect(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number
) {
  context.beginPath();
  context.moveTo(x + radius, y);
  context.lineTo(x + width - radius, y);
  context.quadraticCurveTo(x + width, y, x + width, y + radius);
  context.lineTo(x + width, y + height - radius);
  context.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  context.lineTo(x + radius, y + height);
  context.quadraticCurveTo(x, y + height, x, y + height - radius);
  context.lineTo(x, y + radius);
  context.quadraticCurveTo(x, y, x + radius, y);
  context.closePath();
}

function drawNeonNodeHover(
  context: CanvasRenderingContext2D,
  data: {
    color: string;
    label: string | null;
    sectionKey?: SectionKey;
    size: number;
    x: number;
    y: number;
  },
  settings: {
    labelFont: string;
    labelSize: number;
    labelWeight: string;
  }
) {
  if (!data.label) {
    return;
  }

  const hoverTheme = getGraphHoverTheme(data.sectionKey ?? "concepts");
  const fontSize = settings.labelSize;
  const paddingX = 14;
  const boxHeight = fontSize + 16;

  context.save();
  context.font = `${settings.labelWeight} ${fontSize}px ${settings.labelFont}`;

  const textWidth = context.measureText(data.label).width;
  const dotSize = 8;
  const boxWidth = textWidth + paddingX * 2 + dotSize + 10;
  const canvasWidth = context.canvas.width / (window.devicePixelRatio || 1);
  const offset = data.size + 14;
  const placeOnLeft = data.x + offset + boxWidth > canvasWidth - 16;
  const boxX = placeOnLeft ? data.x - offset - boxWidth : data.x + offset;
  const boxY = data.y - boxHeight / 2;
  const dotX = boxX + paddingX + dotSize / 2;
  const textX = dotX + dotSize / 2 + 10;

  context.shadowColor = hoverTheme.shadow;
  context.shadowBlur = 18;
  context.fillStyle = hoverTheme.panelFill;
  drawRoundedRect(context, boxX, boxY, boxWidth, boxHeight, 12);
  context.fill();

  context.shadowBlur = 0;
  context.strokeStyle = hoverTheme.panelBorder;
  context.lineWidth = 1.1;
  drawRoundedRect(context, boxX, boxY, boxWidth, boxHeight, 12);
  context.stroke();

  context.beginPath();
  context.arc(data.x, data.y, data.size + 5, 0, Math.PI * 2);
  context.fillStyle = `${data.color}33`;
  context.fill();

  context.beginPath();
  context.arc(dotX, data.y, dotSize / 2, 0, Math.PI * 2);
  context.fillStyle = data.color;
  context.fill();

  context.fillStyle = hoverTheme.text;
  context.textBaseline = "middle";
  context.fillText(data.label, textX, data.y + 0.5);
  context.restore();
}

export default function KnowledgeGraph({
  fullGraph,
  defaultGraph,
  sectionOptions,
  showFilters = true,
  allowProjectToggle = true,
  initialIncludeProjects = false,
  height = 560,
  surface = "teaser"
}: KnowledgeGraphProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [includeProjects, setIncludeProjects] = useState(initialIncludeProjects);
  const [activeSections, setActiveSections] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(sectionOptions.map((section) => [section.key, true]))
  );
  const sourceGraph = includeProjects ? fullGraph : defaultGraph;
  const visibleNodes = sourceGraph.nodes.filter((node) => activeSections[node.sectionKey] ?? true);
  const visibleNodeIds = new Set(visibleNodes.map((node) => node.id));
  const visibleEdges = sourceGraph.edges.filter(
    (edge) => visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target)
  );

  useEffect(() => {
    const container = containerRef.current;

    if (!container) {
      return undefined;
    }

    if (visibleNodes.length === 0) {
      container.innerHTML = '<div class="graph-empty">当前过滤条件下没有节点。</div>';
      return undefined;
    }

    let renderer: { kill: () => void } | undefined;
    let cancelled = false;

    void (async () => {
      container.innerHTML = "";

      const [{ default: Graph }, { default: Sigma }] = await Promise.all([import("graphology"), import("sigma")]);

      if (cancelled) {
        return;
      }

      const graph = new Graph();
      const isCompact = window.innerWidth < 720;
      const displayConfig = getGraphDisplayConfig(isCompact, surface);
      const interactionConfig = getGraphInteractionConfig(surface);
      const layout = createLayout(visibleNodes, surface === "immersive" ? 1.75 : 1.45);
      const degreeByNode = new Map<string, number>();
      const nodeById = new Map(visibleNodes.map((node) => [node.id, node]));
      const graphEdges = visibleEdges.map((edge, index) => ({
        ...edge,
        key: `${edge.source}-${edge.target}-${index}`
      }));
      const relationIndex = buildGraphRelationIndex(
        graphEdges.map((edge) => ({
          edgeId: edge.key,
          source: edge.source,
          target: edge.target
        }))
      );
      let hoverState = createGraphHoverState(relationIndex, null);

      for (const edge of graphEdges) {
        degreeByNode.set(edge.source, (degreeByNode.get(edge.source) ?? 0) + 1);
        degreeByNode.set(edge.target, (degreeByNode.get(edge.target) ?? 0) + 1);
      }

      for (const node of visibleNodes) {
        const position = layout.get(node.id) ?? { x: 0, y: 0 };
        const degree = degreeByNode.get(node.id) ?? 0;
        const theme = getGraphSectionTheme(node.sectionKey);

        graph.addNode(node.id, {
          x: position.x,
          y: position.y,
          label: node.title,
          size: displayConfig.nodeSize + Math.min(degree, 10) * displayConfig.degreeStep,
          color: theme.fill,
          focusColor: theme.stroke,
          accentColor: theme.label,
          sectionKey: node.sectionKey,
          forceLabel: degree >= displayConfig.forceLabelDegree,
          url: node.url
        });
      }

      graphEdges.forEach((edge) => {
        const sourceNode = nodeById.get(edge.source);
        const edgeTheme = getGraphSectionTheme(sourceNode?.sectionKey ?? "concepts");

        graph.addEdgeWithKey(edge.key, edge.source, edge.target, {
          color: edgeTheme.edge,
          size: displayConfig.edgeSize,
          activeColor: edgeTheme.stroke
        });
      });

      renderer = new Sigma(graph, container, {
        renderEdgeLabels: false,
        renderLabels: true,
        labelDensity: displayConfig.labelDensity,
        labelGridCellSize: displayConfig.labelGridCellSize,
        labelRenderedSizeThreshold: displayConfig.labelThreshold,
        labelFont: "Trebuchet MS, Avenir Next, Segoe UI, sans-serif",
        labelSize: displayConfig.labelSize,
        labelWeight: surface === "immersive" ? "700" : "600",
        labelColor: { color: "#f3edff" },
        defaultEdgeColor: "rgba(179, 136, 255, 0.18)",
        defaultNodeColor: "#b388ff",
        defaultDrawNodeHover: drawNeonNodeHover,
        nodeReducer: (node, data) => reduceNodeForHover(node, data, hoverState, interactionConfig),
        edgeReducer: (edge, data) => reduceEdgeForHover(edge, data, hoverState, interactionConfig),
        allowInvalidContainer: true,
        hideLabelsOnMove: isCompact,
        stagePadding: 24,
        minCameraRatio: surface === "immersive" ? 0.14 : 0.2,
        maxCameraRatio: 5,
        zIndex: true
      });

      const setHoveredNode = (nodeId: string | null) => {
        if (hoverState.hoveredNodeId === nodeId) {
          return;
        }

        hoverState = createGraphHoverState(relationIndex, nodeId);
        renderer?.refresh();
      };

      renderer.on("enterNode", ({ node }: { node: string }) => {
        setHoveredNode(node);
      });

      renderer.on("leaveNode", () => {
        setHoveredNode(null);
      });

      renderer.on("leaveStage", () => {
        setHoveredNode(null);
      });

      renderer.on("clickNode", ({ node }: { node: string }) => {
        const url = graph.getNodeAttribute(node, "url");

        if (typeof url === "string") {
          window.location.assign(withBasePath(url));
        }
      });
    })();

    return () => {
      cancelled = true;
      renderer?.kill();
    };
  }, [activeSections, defaultGraph, fullGraph, includeProjects, surface, visibleEdges, visibleNodes]);

  return (
    <section className="graph-shell">
      {showFilters && (
        <div className="graph-toolbar">
          <div className="graph-filter-group">
            {sectionOptions.map((section) => (
              <button
                key={section.key}
                type="button"
                className={activeSections[section.key] ? "filter-pill is-active" : "filter-pill"}
                style={
                  {
                    "--section-fill": getGraphSectionTheme(section.key).fill,
                    "--section-stroke": getGraphSectionTheme(section.key).stroke,
                    "--section-glow": getGraphSectionTheme(section.key).glow
                  } as CSSProperties
                }
                onClick={() =>
                  setActiveSections((current) => ({
                    ...current,
                    [section.key]: !current[section.key]
                  }))
                }
              >
                <span className="filter-pill__dot" />
                <span>{section.label}</span>
              </button>
            ))}
          </div>

          <div className="graph-toolbar__aside">
            <div className="graph-status">
              <span>{visibleNodes.length} 节点</span>
              <span>{visibleEdges.length} 连边</span>
            </div>

            {allowProjectToggle && (
              <label className="graph-toggle">
                <input
                  type="checkbox"
                  checked={includeProjects}
                  onChange={(event) => setIncludeProjects(event.target.checked)}
                />
                <span>包含 projects</span>
              </label>
            )}
          </div>
        </div>
      )}

      <div className="graph-stage-wrap">
        <div className="graph-stage-hud">
          <span>knowledge field</span>
          <span>drag / zoom / click</span>
        </div>
        <div className="graph-stage" ref={containerRef} style={{ height: `${height}px` }} />
      </div>
    </section>
  );
}
