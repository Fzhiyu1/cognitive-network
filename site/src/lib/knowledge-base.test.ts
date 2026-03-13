import { mkdtemp, rm, writeFile, mkdir, utimes } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { buildKnowledgeBase } from "./knowledge-base";
import { getGraphPayload } from "./graph-data";
import { getHomepageData, getSectionIndexData } from "./site-data";

const createdRoots: string[] = [];

async function createFixtureRoot(): Promise<string> {
  const root = await mkdtemp(path.join(os.tmpdir(), "kb-site-"));
  createdRoots.push(root);

  await Promise.all(
    ["0-inbox", "1-concepts", "2-explorations", "3-projects", "4-references"].map((directory) =>
      mkdir(path.join(root, directory), { recursive: true })
    )
  );

  await writeFile(
    path.join(root, "1-concepts", "主动遗忘.md"),
    [
      "---",
      "tags: [认知科学, AI]",
      "---",
      "",
      "# 主动遗忘",
      "",
      "主动遗忘是控制提取通道的过程。",
      "",
      "关联 [[人工脑干原型实验]] 与 [[非单调可塑性假说|NMPH]]。"
    ].join("\n"),
    "utf8"
  );

  await writeFile(
    path.join(root, "2-explorations", "记忆不是仓库.md"),
    [
      "---",
      "tags: [AI, 方法论]",
      "---",
      "",
      "# 记忆不是仓库",
      "",
      "总结 [[主动遗忘]]。"
    ].join("\n"),
    "utf8"
  );

  await writeFile(
    path.join(root, "3-projects", "人工脑干原型实验.md"),
    [
      "---",
      "tags: [工程]",
      "---",
      "",
      "# 人工脑干原型实验",
      "",
      "这是项目详情页。",
      "",
      "参见 [[主动遗忘]]。"
    ].join("\n"),
    "utf8"
  );

  await writeFile(
    path.join(root, "4-references", "非单调可塑性假说.md"),
    [
      "---",
      "tags: [认知科学]",
      "---",
      "",
      "# 非单调可塑性假说",
      "",
      "支持 [[主动遗忘]]。"
    ].join("\n"),
    "utf8"
  );

  await Promise.all([
    utimes(path.join(root, "1-concepts", "主动遗忘.md"), new Date("2026-03-10T00:00:00.000Z"), new Date("2026-03-10T00:00:00.000Z")),
    utimes(path.join(root, "2-explorations", "记忆不是仓库.md"), new Date("2026-03-12T00:00:00.000Z"), new Date("2026-03-12T00:00:00.000Z")),
    utimes(path.join(root, "3-projects", "人工脑干原型实验.md"), new Date("2026-03-13T00:00:00.000Z"), new Date("2026-03-13T00:00:00.000Z")),
    utimes(path.join(root, "4-references", "非单调可塑性假说.md"), new Date("2026-03-11T00:00:00.000Z"), new Date("2026-03-11T00:00:00.000Z"))
  ]);

  return root;
}

afterEach(async () => {
  await Promise.all(createdRoots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

describe("buildKnowledgeBase", () => {
  it("hides project notes from public indexes and the default graph while keeping them directly accessible", async () => {
    const root = await createFixtureRoot();

    const data = await buildKnowledgeBase({ rootDir: root });
    const project = data.noteByTitle.get("人工脑干原型实验");

    expect(project).toMatchObject({
      sectionKey: "projects",
      isIndexedPublicly: false,
      isInDefaultGraph: false,
      isDirectlyAccessible: true
    });

    expect(data.publicIndexNotes.map((note) => note.title)).not.toContain("人工脑干原型实验");
    expect(data.defaultGraph.nodes.map((node) => node.title)).not.toContain("人工脑干原型实验");
  });

  it("builds backlinks and can include projects in the full graph payload", async () => {
    const root = await createFixtureRoot();

    const data = await buildKnowledgeBase({ rootDir: root });
    const activeForgetting = data.noteByTitle.get("主动遗忘");

    expect(activeForgetting?.backlinks.map((note) => note.title)).toEqual([
      "非单调可塑性假说",
      "记忆不是仓库",
      "人工脑干原型实验"
    ]);

    expect(data.graph.nodes.map((node) => node.title)).toContain("人工脑干原型实验");
    expect(data.graph.edges).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ source: "主动遗忘", target: "人工脑干原型实验" }),
        expect.objectContaining({ source: "人工脑干原型实验", target: "主动遗忘" })
      ])
    );
  });

  it("returns homepage recent updates in descending modified order and excludes projects", async () => {
    const root = await createFixtureRoot();
    const data = await buildKnowledgeBase({ rootDir: root });

    expect(getHomepageData(data).recentUpdates.map((note) => note.title)).toEqual([
      "记忆不是仓库",
      "非单调可塑性假说",
      "主动遗忘"
    ]);
  });

  it("returns public section indexes and rejects the hidden projects index", async () => {
    const root = await createFixtureRoot();
    const data = await buildKnowledgeBase({ rootDir: root });

    expect(getSectionIndexData(data, "references").notes.map((note) => note.title)).toEqual(["非单调可塑性假说"]);
    expect(() => getSectionIndexData(data, "projects")).toThrow("projects is not publicly indexable");
  });

  it("serializes default and full graph payloads with the correct project visibility", async () => {
    const root = await createFixtureRoot();
    const data = await buildKnowledgeBase({ rootDir: root });

    expect(getGraphPayload(data).nodes.map((node) => node.title)).not.toContain("人工脑干原型实验");
    expect(getGraphPayload(data, { includeProjects: true }).nodes.map((node) => node.title)).toContain("人工脑干原型实验");
  });
});
