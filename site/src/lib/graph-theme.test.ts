import { describe, expect, it } from "vitest";

import { getGraphDisplayConfig, getGraphHoverTheme, getGraphInteractionConfig, getGraphSectionTheme } from "./graph-theme";

describe("graph theme", () => {
  it("returns neon visual tokens for each section", () => {
    expect(getGraphSectionTheme("concepts")).toEqual({
      fill: "#b388ff",
      stroke: "#f6ecff",
      glow: "rgba(179, 136, 255, 0.42)",
      edge: "rgba(179, 136, 255, 0.18)",
      label: "#f4ecff"
    });

    expect(getGraphSectionTheme("explorations")).toEqual({
      fill: "#ff77e1",
      stroke: "#fff1fd",
      glow: "rgba(255, 119, 225, 0.44)",
      edge: "rgba(255, 119, 225, 0.22)",
      label: "#fff2fc"
    });
  });

  it("exposes teaser and immersive graph rendering presets", () => {
    expect(getGraphDisplayConfig(false, "teaser")).toEqual({
      labelThreshold: 10,
      nodeSize: 5.6,
      degreeStep: 0.92,
      edgeSize: 1.1,
      labelDensity: 0.06,
      labelGridCellSize: 140,
      forceLabelDegree: 7,
      labelSize: 14
    });

    expect(getGraphDisplayConfig(true, "teaser")).toEqual({
      labelThreshold: 18,
      nodeSize: 4.6,
      degreeStep: 0.76,
      edgeSize: 0.9,
      labelDensity: 0.05,
      labelGridCellSize: 150,
      forceLabelDegree: 8,
      labelSize: 13
    });

    expect(getGraphDisplayConfig(false, "immersive")).toEqual({
      labelThreshold: 7,
      nodeSize: 5.8,
      degreeStep: 1,
      edgeSize: 1.15,
      labelDensity: 0.18,
      labelGridCellSize: 104,
      forceLabelDegree: 3,
      labelSize: 16
    });

    expect(getGraphDisplayConfig(true, "immersive")).toEqual({
      labelThreshold: 14,
      nodeSize: 4.8,
      degreeStep: 0.8,
      edgeSize: 0.95,
      labelDensity: 0.1,
      labelGridCellSize: 116,
      forceLabelDegree: 4,
      labelSize: 14
    });
  });

  it("provides dark hover panel tokens that stay readable on neon graphs", () => {
    expect(getGraphHoverTheme("concepts")).toEqual({
      panelFill: "rgba(10, 7, 24, 0.96)",
      panelBorder: "rgba(179, 136, 255, 0.62)",
      text: "#f7f2ff",
      shadow: "rgba(179, 136, 255, 0.34)"
    });

    expect(getGraphHoverTheme("references")).toEqual({
      panelFill: "rgba(7, 10, 19, 0.96)",
      panelBorder: "rgba(113, 245, 183, 0.58)",
      text: "#effff7",
      shadow: "rgba(113, 245, 183, 0.28)"
    });
  });

  it("exposes interaction tuning for focus and dimming states", () => {
    expect(getGraphInteractionConfig("immersive")).toEqual({
      focusNodeScale: 1.24,
      neighborNodeScale: 1.1,
      dimNodeOpacity: 0.2,
      dimEdgeOpacity: 0.08,
      activeEdgeScale: 2.8
    });
  });
});
