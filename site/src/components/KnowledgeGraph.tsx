import { useEffect, useRef, useState } from "react";

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
}

const SECTION_COLORS: Record<SectionKey, string> = {
  inbox: "#6f7d6d",
  concepts: "#b7572c",
  explorations: "#8d5537",
  projects: "#4e7096",
  references: "#54735d"
};

function hashToFloat(value: string): number {
  let hash = 0;

  for (const character of value) {
    hash = (hash * 31 + character.charCodeAt(0)) >>> 0;
  }

  return (hash % 1000) / 1000;
}

function createLayout(nodes: GraphNode[]) {
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
        const radius = 1.2 + Math.sqrt(nodeIndex + 1) * 1.45;
        const wobble = (hashToFloat(node.id) - 0.5) * 0.55;
        const x = center.x + Math.cos(angle) * radius + wobble;
        const y = center.y + Math.sin(angle) * radius + wobble;
        positions.set(node.id, { x, y });
      });
  });

  return positions;
}

export default function KnowledgeGraph({
  fullGraph,
  defaultGraph,
  sectionOptions,
  showFilters = true,
  allowProjectToggle = true,
  initialIncludeProjects = false,
  height = 560
}: KnowledgeGraphProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [includeProjects, setIncludeProjects] = useState(initialIncludeProjects);
  const [activeSections, setActiveSections] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(sectionOptions.map((section) => [section.key, true]))
  );

  useEffect(() => {
    const container = containerRef.current;

    if (!container) {
      return undefined;
    }

    const sourceGraph = includeProjects ? fullGraph : defaultGraph;
    const visibleNodes = sourceGraph.nodes.filter((node) => activeSections[node.sectionKey] ?? true);
    const visibleNodeIds = new Set(visibleNodes.map((node) => node.id));
    const visibleEdges = sourceGraph.edges.filter(
      (edge) => visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target)
    );

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
      const layout = createLayout(visibleNodes);
      const degreeByNode = new Map<string, number>();

      for (const edge of visibleEdges) {
        degreeByNode.set(edge.source, (degreeByNode.get(edge.source) ?? 0) + 1);
        degreeByNode.set(edge.target, (degreeByNode.get(edge.target) ?? 0) + 1);
      }

      for (const node of visibleNodes) {
        const position = layout.get(node.id) ?? { x: 0, y: 0 };
        const degree = degreeByNode.get(node.id) ?? 0;

        graph.addNode(node.id, {
          x: position.x,
          y: position.y,
          label: node.title,
          size: 4.8 + Math.min(degree, 10) * 0.75,
          color: SECTION_COLORS[node.sectionKey],
          url: node.url
        });
      }

      visibleEdges.forEach((edge, index) => {
        graph.addEdgeWithKey(`${edge.source}-${edge.target}-${index}`, edge.source, edge.target, {
          color: "rgba(68, 52, 38, 0.22)",
          size: 1
        });
      });

      const labelThreshold = window.innerWidth < 480 ? 18 : 12;

      renderer = new Sigma(graph, container, {
        renderEdgeLabels: false,
        labelDensity: 0.035,
        labelGridCellSize: 160,
        labelRenderedSizeThreshold: labelThreshold,
        defaultEdgeColor: "rgba(68, 52, 38, 0.22)",
        defaultNodeColor: "#b7572c",
        allowInvalidContainer: true,
        minCameraRatio: 0.2,
        maxCameraRatio: 5
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
  }, [activeSections, defaultGraph, fullGraph, includeProjects, sectionOptions]);

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
                onClick={() =>
                  setActiveSections((current) => ({
                    ...current,
                    [section.key]: !current[section.key]
                  }))
                }
              >
                {section.label}
              </button>
            ))}
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
      )}

      <div className="graph-stage" ref={containerRef} style={{ height: `${height}px` }} />
    </section>
  );
}
