import type { GraphInteractionConfig } from "./graph-theme";

interface GraphRelation {
  edgeId: string;
  source: string;
  target: string;
}

export interface GraphRelationIndex {
  activeEdgeIdsByNode: Map<string, Set<string>>;
  neighborNodeIdsByNode: Map<string, Set<string>>;
}

export interface GraphHoverState {
  hoveredNodeId: string | null;
  activeEdgeIds: Set<string>;
  neighborNodeIds: Set<string>;
  relatedNodeIds: Set<string>;
}

export interface GraphNodeFocusData {
  accentColor?: string;
  color: string;
  focusColor?: string;
  forceLabel?: boolean;
  hidden?: boolean;
  label: string | null;
  size: number;
  zIndex?: number;
}

export interface GraphEdgeFocusData {
  activeColor?: string;
  color: string;
  hidden?: boolean;
  size: number;
  zIndex?: number;
}

function getSetEntry(map: Map<string, Set<string>>, key: string): Set<string> {
  const entry = map.get(key);

  if (entry) {
    return entry;
  }

  const nextEntry = new Set<string>();
  map.set(key, nextEntry);
  return nextEntry;
}

function formatRgba(red: number, green: number, blue: number, alpha: number): string {
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

function withOpacity(color: string, alpha: number): string {
  const rgbaMatch = color.match(/^rgba?\(([^)]+)\)$/i);

  if (rgbaMatch) {
    const [red = "0", green = "0", blue = "0"] = rgbaMatch[1].split(",").map((channel) => channel.trim());
    return formatRgba(Number(red), Number(green), Number(blue), alpha);
  }

  const hexMatch = color.match(/^#([0-9a-f]{6}|[0-9a-f]{3})$/i);

  if (!hexMatch) {
    return color;
  }

  const hex = hexMatch[1].length === 3
    ? hexMatch[1]
        .split("")
        .map((character) => `${character}${character}`)
        .join("")
    : hexMatch[1];

  const red = Number.parseInt(hex.slice(0, 2), 16);
  const green = Number.parseInt(hex.slice(2, 4), 16);
  const blue = Number.parseInt(hex.slice(4, 6), 16);

  return formatRgba(red, green, blue, alpha);
}

export function buildGraphRelationIndex(relations: GraphRelation[]): GraphRelationIndex {
  const neighborNodeIdsByNode = new Map<string, Set<string>>();
  const activeEdgeIdsByNode = new Map<string, Set<string>>();

  relations.forEach(({ edgeId, source, target }) => {
    getSetEntry(neighborNodeIdsByNode, source).add(target);
    getSetEntry(neighborNodeIdsByNode, target).add(source);
    getSetEntry(activeEdgeIdsByNode, source).add(edgeId);
    getSetEntry(activeEdgeIdsByNode, target).add(edgeId);
  });

  return {
    activeEdgeIdsByNode,
    neighborNodeIdsByNode
  };
}

export function createGraphHoverState(relationIndex: GraphRelationIndex, hoveredNodeId: string | null): GraphHoverState {
  if (!hoveredNodeId) {
    return {
      hoveredNodeId: null,
      activeEdgeIds: new Set<string>(),
      neighborNodeIds: new Set<string>(),
      relatedNodeIds: new Set<string>()
    };
  }

  const neighborNodeIds = new Set(relationIndex.neighborNodeIdsByNode.get(hoveredNodeId) ?? []);
  const activeEdgeIds = new Set(relationIndex.activeEdgeIdsByNode.get(hoveredNodeId) ?? []);
  const relatedNodeIds = new Set<string>([hoveredNodeId, ...neighborNodeIds]);

  return {
    hoveredNodeId,
    activeEdgeIds,
    neighborNodeIds,
    relatedNodeIds
  };
}

export function reduceNodeForHover(
  nodeId: string,
  data: GraphNodeFocusData,
  hoverState: GraphHoverState,
  interactionConfig: GraphInteractionConfig
): GraphNodeFocusData {
  if (!hoverState.hoveredNodeId) {
    return { ...data };
  }

  if (nodeId === hoverState.hoveredNodeId) {
    return {
      ...data,
      color: data.focusColor ?? data.color,
      forceLabel: true,
      hidden: false,
      size: data.size * interactionConfig.focusNodeScale,
      zIndex: 3
    };
  }

  if (hoverState.neighborNodeIds.has(nodeId)) {
    return {
      ...data,
      color: data.accentColor ?? data.color,
      forceLabel: true,
      hidden: false,
      size: data.size * interactionConfig.neighborNodeScale,
      zIndex: 2
    };
  }

  return {
    ...data,
    color: withOpacity(data.color, interactionConfig.dimNodeOpacity),
    forceLabel: false,
    label: null,
    zIndex: 0
  };
}

export function reduceEdgeForHover(
  edgeId: string,
  data: GraphEdgeFocusData,
  hoverState: GraphHoverState,
  interactionConfig: GraphInteractionConfig
): GraphEdgeFocusData {
  if (!hoverState.hoveredNodeId) {
    return { ...data };
  }

  if (hoverState.activeEdgeIds.has(edgeId)) {
    return {
      ...data,
      color: data.activeColor ?? data.color,
      hidden: false,
      size: data.size * interactionConfig.activeEdgeScale,
      zIndex: 2
    };
  }

  return {
    ...data,
    color: withOpacity(data.color, interactionConfig.dimEdgeOpacity),
    hidden: true,
    zIndex: 0
  };
}
