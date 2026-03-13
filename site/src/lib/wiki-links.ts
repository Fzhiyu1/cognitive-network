const WIKI_LINK_PATTERN = /\[\[([^[\]]+)\]\]/g;

export function parseWikiLinks(markdown: string): string[] {
  const titles = new Set<string>();

  for (const match of markdown.matchAll(WIKI_LINK_PATTERN)) {
    const rawTarget = match[1]?.trim();

    if (!rawTarget) {
      continue;
    }

    const title = rawTarget.split("|")[0]?.split("#")[0]?.trim();

    if (title) {
      titles.add(title);
    }
  }

  return [...titles];
}
