import { describe, expect, it } from "vitest";

import { parseWikiLinks } from "./wiki-links";

describe("parseWikiLinks", () => {
  it("parses wiki links, strips aliases and headings, and de-duplicates results", () => {
    const markdown = [
      "连接到 [[主动遗忘]]、[[主动遗忘|别名]]、[[主动遗忘#定义]]。",
      "还有 [[非单调可塑性假说|NMPH]]。"
    ].join("\n");

    expect(parseWikiLinks(markdown)).toEqual(["主动遗忘", "非单调可塑性假说"]);
  });
});
