import { readdir, readFile, stat } from "node:fs/promises";
import path from "node:path";

import matter from "gray-matter";

import { SECTION_DEFINITIONS, SECTION_BY_KEY, type SectionDefinition, type SectionKey } from "./sections";
import { parseWikiLinks } from "./wiki-links";

export interface LinkedNote {
  title: string;
  url: string;
  sectionKey: SectionKey;
}

export interface Note {
  id: string;
  title: string;
  slug: string;
  tags: string[];
  summary: string;
  filePath: string;
  sectionKey: SectionKey;
  section: SectionDefinition;
  url: string;
  body: string;
  modifiedTime: string;
  links: string[];
  outgoingLinks: LinkedNote[];
  backlinks: LinkedNote[];
  relatedNotes: LinkedNote[];
  isIndexedPublicly: boolean;
  isInDefaultGraph: boolean;
  isDirectlyAccessible: boolean;
}

export interface GraphNode {
  id: string;
  title: string;
  sectionKey: SectionKey;
  url: string;
  tags: string[];
}

export interface GraphEdge {
  source: string;
  target: string;
}

export interface KnowledgeBaseData {
  notes: Note[];
  noteByTitle: Map<string, Note>;
  publicIndexNotes: Note[];
  graph: {
    nodes: GraphNode[];
    edges: GraphEdge[];
  };
  defaultGraph: {
    nodes: GraphNode[];
    edges: GraphEdge[];
  };
}

interface BuildKnowledgeBaseOptions {
  rootDir: string;
}

interface DraftNote {
  id: string;
  title: string;
  slug: string;
  tags: string[];
  summary: string;
  filePath: string;
  sectionKey: SectionKey;
  section: SectionDefinition;
  url: string;
  body: string;
  modifiedTime: string;
  links: string[];
  isIndexedPublicly: boolean;
  isInDefaultGraph: boolean;
  isDirectlyAccessible: boolean;
}

export async function buildKnowledgeBase(options: BuildKnowledgeBaseOptions): Promise<KnowledgeBaseData> {
  const notes = (await Promise.all(SECTION_DEFINITIONS.map((section) => readSectionNotes(options.rootDir, section)))).flat();
  const noteByTitle = new Map(notes.map((note) => [note.title, note]));
  const backlinksByTitle = new Map<string, Set<string>>();
  const graphEdges: GraphEdge[] = [];

  for (const note of notes) {
    for (const linkedTitle of note.links) {
      const linkedNote = noteByTitle.get(linkedTitle);

      if (!linkedNote) {
        continue;
      }

      graphEdges.push({ source: note.title, target: linkedNote.title });

      const backlinks = backlinksByTitle.get(linkedNote.title) ?? new Set<string>();
      backlinks.add(note.title);
      backlinksByTitle.set(linkedNote.title, backlinks);
    }
  }

  const finalizedNotes = notes.map<Note>((note) => {
    const outgoingLinks = note.links
      .map((linkedTitle) => noteByTitle.get(linkedTitle))
      .filter((linkedNote): linkedNote is DraftNote => Boolean(linkedNote))
      .map(toLinkedNote)
      .sort(compareLinkedNotes);

    const backlinks = [...(backlinksByTitle.get(note.title) ?? new Set<string>())]
      .map((linkedTitle) => noteByTitle.get(linkedTitle))
      .filter((linkedNote): linkedNote is DraftNote => Boolean(linkedNote))
      .map(toLinkedNote)
      .sort(compareLinkedNotes);

    return {
      ...note,
      outgoingLinks,
      backlinks,
      relatedNotes: buildRelatedNotes(note, noteByTitle, outgoingLinks, backlinks)
    };
  });

  const finalizedByTitle = new Map(finalizedNotes.map((note) => [note.title, note]));
  const graphNodes = finalizedNotes.map<GraphNode>((note) => ({
    id: note.title,
    title: note.title,
    sectionKey: note.sectionKey,
    url: note.url,
    tags: note.tags
  }));
  const defaultGraphNodeIds = new Set(finalizedNotes.filter((note) => note.isInDefaultGraph).map((note) => note.title));

  return {
    notes: finalizedNotes.sort(compareNotesByModifiedTime),
    noteByTitle: finalizedByTitle,
    publicIndexNotes: finalizedNotes.filter((note) => note.isIndexedPublicly).sort(compareNotesByModifiedTime),
    graph: {
      nodes: graphNodes,
      edges: graphEdges
    },
    defaultGraph: {
      nodes: graphNodes.filter((node) => defaultGraphNodeIds.has(node.id)),
      edges: graphEdges.filter((edge) => defaultGraphNodeIds.has(edge.source) && defaultGraphNodeIds.has(edge.target))
    }
  };
}

async function readSectionNotes(rootDir: string, section: SectionDefinition): Promise<DraftNote[]> {
  const sectionDir = path.join(rootDir, section.directory);
  let fileEntries;

  try {
    fileEntries = await readdir(sectionDir, { withFileTypes: true });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return [];
    }

    throw error;
  }

  const markdownEntries = fileEntries.filter((entry) => entry.isFile() && entry.name.endsWith(".md"));

  return Promise.all(
    markdownEntries.map(async (entry) => {
      const filePath = path.join(sectionDir, entry.name);
      const raw = await readFile(filePath, "utf8");
      const parsed = matter(raw);
      const fileStats = await stat(filePath);
      const title = extractTitle(parsed.content, entry.name);
      const slug = typeof parsed.data.slug === "string" && parsed.data.slug.trim() ? parsed.data.slug.trim() : title;

      return {
        id: title,
        title,
        slug,
        tags: normalizeTags(parsed.data.tags),
        summary: extractSummary(parsed.content),
        filePath,
        sectionKey: section.key,
        section,
        url: `/${section.routeSegment}/${encodeURIComponent(slug)}/`,
        body: parsed.content.trim(),
        modifiedTime: fileStats.mtime.toISOString(),
        links: parseWikiLinks(parsed.content),
        isIndexedPublicly: section.isPublicIndex,
        isInDefaultGraph: section.isDefaultGraph,
        isDirectlyAccessible: section.isDirectlyAccessible
      };
    })
  );
}

function extractTitle(content: string, fileName: string): string {
  const headingMatch = content.match(/^#\s+(.+)$/m);

  if (headingMatch?.[1]) {
    return headingMatch[1].trim();
  }

  return fileName.replace(/\.md$/u, "");
}

function normalizeTags(tags: unknown): string[] {
  if (!Array.isArray(tags)) {
    return [];
  }

  return tags.map((tag) => String(tag).trim()).filter(Boolean);
}

function extractSummary(content: string): string {
  const lines = content
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => Boolean(line) && !line.startsWith("#") && !line.startsWith("**"));

  const firstParagraph = lines.find((line) => !line.startsWith("-") && !line.startsWith(">") && !line.startsWith("|"));

  return sanitizeInlineMarkdown(firstParagraph ?? "");
}

function sanitizeInlineMarkdown(text: string): string {
  return text
    .replace(/\[\[([^|\]]+)(?:\|[^\]]+)?\]\]/g, "$1")
    .replace(/[*_`>#-]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function toLinkedNote(note: DraftNote): LinkedNote {
  return {
    title: note.title,
    url: note.url,
    sectionKey: note.sectionKey
  };
}

function buildRelatedNotes(
  note: DraftNote,
  noteByTitle: Map<string, DraftNote>,
  outgoingLinks: LinkedNote[],
  backlinks: LinkedNote[]
): LinkedNote[] {
  const relatedTitles = new Set<string>();

  for (const linked of [...outgoingLinks, ...backlinks]) {
    if (linked.title !== note.title) {
      relatedTitles.add(linked.title);
    }
  }

  for (const candidate of noteByTitle.values()) {
    if (candidate.title === note.title || relatedTitles.has(candidate.title)) {
      continue;
    }

    const sharedTagCount = candidate.tags.filter((tag) => note.tags.includes(tag)).length;

    if (sharedTagCount > 0 && candidate.sectionKey !== "projects") {
      relatedTitles.add(candidate.title);
    }
  }

  return [...relatedTitles]
    .map((title) => noteByTitle.get(title))
    .filter((linkedNote): linkedNote is DraftNote => Boolean(linkedNote))
    .map(toLinkedNote)
    .sort(compareLinkedNotes)
    .slice(0, 6);
}

function compareLinkedNotes(left: LinkedNote, right: LinkedNote): number {
  return left.title.localeCompare(right.title, "zh-Hans-CN");
}

function compareNotesByModifiedTime(left: Note | DraftNote, right: Note | DraftNote): number {
  return right.modifiedTime.localeCompare(left.modifiedTime);
}

export function getSectionLabel(sectionKey: SectionKey): string {
  return SECTION_BY_KEY.get(sectionKey)?.label ?? sectionKey;
}
