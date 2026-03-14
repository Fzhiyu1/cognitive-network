import { describe, expect, it } from "vitest";

import { getGraphInteractionConfig } from "./graph-theme";
import { buildGraphRelationIndex, createGraphHoverState, reduceEdgeForHover, reduceNodeForHover } from "./graph-focus";

describe("graph focus", () => {
  it("builds adjacent nodes and active edges for a hovered node", () => {
    const relationIndex = buildGraphRelationIndex([
      { edgeId: "a-b", source: "a", target: "b" },
      { edgeId: "a-c", source: "a", target: "c" },
      { edgeId: "d-e", source: "d", target: "e" }
    ]);

    const hoverState = createGraphHoverState(relationIndex, "a");

    expect([...hoverState.neighborNodeIds].sort()).toEqual(["b", "c"]);
    expect([...hoverState.relatedNodeIds].sort()).toEqual(["a", "b", "c"]);
    expect([...hoverState.activeEdgeIds].sort()).toEqual(["a-b", "a-c"]);
  });

  it("amplifies the hovered node and dims unrelated nodes", () => {
    const hoverState = createGraphHoverState(
      buildGraphRelationIndex([{ edgeId: "a-b", source: "a", target: "b" }]),
      "a"
    );
    const interaction = getGraphInteractionConfig("immersive");

    expect(
      reduceNodeForHover(
        "a",
        {
          label: "Alpha",
          size: 10,
          color: "#b388ff",
          focusColor: "#f6ecff",
          accentColor: "#ead8ff",
          forceLabel: false,
          zIndex: 0
        },
        hoverState,
        interaction
      )
    ).toEqual({
      label: "Alpha",
      size: 12.4,
      color: "#f6ecff",
      focusColor: "#f6ecff",
      accentColor: "#ead8ff",
      forceLabel: true,
      hidden: false,
      zIndex: 3
    });

    expect(
      reduceNodeForHover(
        "b",
        {
          label: "Beta",
          size: 10,
          color: "#b388ff",
          focusColor: "#f6ecff",
          accentColor: "#ead8ff",
          forceLabel: false,
          zIndex: 0
        },
        hoverState,
        interaction
      )
    ).toEqual({
      label: "Beta",
      size: 11,
      color: "#ead8ff",
      focusColor: "#f6ecff",
      accentColor: "#ead8ff",
      forceLabel: true,
      hidden: false,
      zIndex: 2
    });

    expect(
      reduceNodeForHover(
        "z",
        {
          label: "Zeta",
          size: 10,
          color: "#b388ff",
          focusColor: "#f6ecff",
          accentColor: "#ead8ff",
          forceLabel: true,
          zIndex: 1
        },
        hoverState,
        interaction
      )
    ).toEqual({
      label: null,
      size: 10,
      color: "rgba(179, 136, 255, 0.2)",
      focusColor: "#f6ecff",
      accentColor: "#ead8ff",
      forceLabel: false,
      zIndex: 0
    });
  });

  it("thickens active edges and fades unrelated edges", () => {
    const hoverState = createGraphHoverState(
      buildGraphRelationIndex([
        { edgeId: "a-b", source: "a", target: "b" },
        { edgeId: "d-e", source: "d", target: "e" }
      ]),
      "a"
    );
    const interaction = getGraphInteractionConfig("immersive");

    expect(
      reduceEdgeForHover(
        "a-b",
        {
          size: 1.15,
          color: "rgba(179, 136, 255, 0.18)",
          activeColor: "#f6ecff",
          hidden: false,
          zIndex: 0
        },
        hoverState,
        interaction
      )
    ).toEqual({
      size: 3.2199999999999998,
      color: "#f6ecff",
      activeColor: "#f6ecff",
      hidden: false,
      zIndex: 2
    });

    expect(
      reduceEdgeForHover(
        "d-e",
        {
          size: 1.15,
          color: "rgba(179, 136, 255, 0.18)",
          activeColor: "#f6ecff",
          hidden: false,
          zIndex: 1
        },
        hoverState,
        interaction
      )
    ).toEqual({
      size: 1.15,
      color: "rgba(179, 136, 255, 0.08)",
      activeColor: "#f6ecff",
      hidden: true,
      zIndex: 0
    });
  });
});
