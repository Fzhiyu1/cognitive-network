import type { SectionKey } from "./sections";

interface GraphSectionTheme {
  fill: string;
  stroke: string;
  glow: string;
  edge: string;
  label: string;
}

interface GraphDisplayConfig {
  labelThreshold: number;
  nodeSize: number;
  degreeStep: number;
  edgeSize: number;
  labelDensity: number;
  labelGridCellSize: number;
  forceLabelDegree: number;
  labelSize: number;
}

interface GraphHoverTheme {
  panelFill: string;
  panelBorder: string;
  text: string;
  shadow: string;
}

export type GraphSurface = "teaser" | "immersive";

export interface GraphInteractionConfig {
  focusNodeScale: number;
  neighborNodeScale: number;
  dimNodeOpacity: number;
  dimEdgeOpacity: number;
  activeEdgeScale: number;
}

const SECTION_THEMES: Record<SectionKey, GraphSectionTheme> = {
  inbox: {
    fill: "#63d9ff",
    stroke: "#e3fbff",
    glow: "rgba(99, 217, 255, 0.4)",
    edge: "rgba(99, 217, 255, 0.16)",
    label: "#dff9ff"
  },
  concepts: {
    fill: "#b388ff",
    stroke: "#f6ecff",
    glow: "rgba(179, 136, 255, 0.42)",
    edge: "rgba(179, 136, 255, 0.18)",
    label: "#f4ecff"
  },
  explorations: {
    fill: "#ff77e1",
    stroke: "#fff1fd",
    glow: "rgba(255, 119, 225, 0.44)",
    edge: "rgba(255, 119, 225, 0.22)",
    label: "#fff2fc"
  },
  projects: {
    fill: "#7f8dff",
    stroke: "#eff1ff",
    glow: "rgba(127, 141, 255, 0.4)",
    edge: "rgba(127, 141, 255, 0.18)",
    label: "#f1f3ff"
  },
  references: {
    fill: "#71f5b7",
    stroke: "#ebfff7",
    glow: "rgba(113, 245, 183, 0.34)",
    edge: "rgba(113, 245, 183, 0.18)",
    label: "#edfff8"
  }
};

const HOVER_THEMES: Record<SectionKey, GraphHoverTheme> = {
  inbox: {
    panelFill: "rgba(5, 10, 18, 0.96)",
    panelBorder: "rgba(99, 217, 255, 0.62)",
    text: "#eeffff",
    shadow: "rgba(99, 217, 255, 0.32)"
  },
  concepts: {
    panelFill: "rgba(10, 7, 24, 0.96)",
    panelBorder: "rgba(179, 136, 255, 0.62)",
    text: "#f7f2ff",
    shadow: "rgba(179, 136, 255, 0.34)"
  },
  explorations: {
    panelFill: "rgba(20, 7, 24, 0.96)",
    panelBorder: "rgba(255, 119, 225, 0.62)",
    text: "#fff2fc",
    shadow: "rgba(255, 119, 225, 0.34)"
  },
  projects: {
    panelFill: "rgba(7, 8, 24, 0.96)",
    panelBorder: "rgba(127, 141, 255, 0.58)",
    text: "#f3f4ff",
    shadow: "rgba(127, 141, 255, 0.3)"
  },
  references: {
    panelFill: "rgba(7, 10, 19, 0.96)",
    panelBorder: "rgba(113, 245, 183, 0.58)",
    text: "#effff7",
    shadow: "rgba(113, 245, 183, 0.28)"
  }
};

const DESKTOP_TEASER_DISPLAY_CONFIG: GraphDisplayConfig = {
  labelThreshold: 10,
  nodeSize: 5.6,
  degreeStep: 0.92,
  edgeSize: 1.1,
  labelDensity: 0.06,
  labelGridCellSize: 140,
  forceLabelDegree: 7,
  labelSize: 14
};

const COMPACT_TEASER_DISPLAY_CONFIG: GraphDisplayConfig = {
  labelThreshold: 18,
  nodeSize: 4.6,
  degreeStep: 0.76,
  edgeSize: 0.9,
  labelDensity: 0.05,
  labelGridCellSize: 150,
  forceLabelDegree: 8,
  labelSize: 13
};

const DESKTOP_IMMERSIVE_DISPLAY_CONFIG: GraphDisplayConfig = {
  labelThreshold: 7,
  nodeSize: 5.8,
  degreeStep: 1,
  edgeSize: 1.15,
  labelDensity: 0.18,
  labelGridCellSize: 104,
  forceLabelDegree: 3,
  labelSize: 16
};

const COMPACT_IMMERSIVE_DISPLAY_CONFIG: GraphDisplayConfig = {
  labelThreshold: 14,
  nodeSize: 4.8,
  degreeStep: 0.8,
  edgeSize: 0.95,
  labelDensity: 0.1,
  labelGridCellSize: 116,
  forceLabelDegree: 4,
  labelSize: 14
};

const INTERACTION_CONFIGS: Record<GraphSurface, GraphInteractionConfig> = {
  teaser: {
    focusNodeScale: 1.2,
    neighborNodeScale: 1.08,
    dimNodeOpacity: 0.26,
    dimEdgeOpacity: 0.1,
    activeEdgeScale: 2.4
  },
  immersive: {
    focusNodeScale: 1.24,
    neighborNodeScale: 1.1,
    dimNodeOpacity: 0.2,
    dimEdgeOpacity: 0.08,
    activeEdgeScale: 2.8
  }
};

export function getGraphSectionTheme(sectionKey: SectionKey): GraphSectionTheme {
  return SECTION_THEMES[sectionKey];
}

export function getGraphDisplayConfig(isCompact: boolean, surface: GraphSurface = "teaser"): GraphDisplayConfig {
  if (surface === "immersive") {
    return isCompact ? COMPACT_IMMERSIVE_DISPLAY_CONFIG : DESKTOP_IMMERSIVE_DISPLAY_CONFIG;
  }

  return isCompact ? COMPACT_TEASER_DISPLAY_CONFIG : DESKTOP_TEASER_DISPLAY_CONFIG;
}

export function getGraphHoverTheme(sectionKey: SectionKey): GraphHoverTheme {
  return HOVER_THEMES[sectionKey];
}

export function getGraphInteractionConfig(surface: GraphSurface): GraphInteractionConfig {
  return INTERACTION_CONFIGS[surface];
}
