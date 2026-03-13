import MarkdownIt from "markdown-it";

import type { Note } from "./knowledge-base";
import { slugifyHeading, withBasePath } from "./url";

const markdown = new MarkdownIt({
  html: false,
  linkify: true,
  typographer: true
});

markdown.core.ruler.push("heading_ids", (state) => {
  for (let index = 0; index < state.tokens.length; index += 1) {
    const token = state.tokens[index];

    if (token.type !== "heading_open") {
      continue;
    }

    const inlineToken = state.tokens[index + 1];
    const text = inlineToken?.children?.map((child) => child.content).join("").trim() ?? "";

    if (text) {
      token.attrSet("id", slugifyHeading(text));
    }
  }
});

export function renderMarkdown(body: string, noteByTitle: Map<string, Note>): string {
  const withoutLeadingTitle = body.replace(/^#\s+.+\n+/u, "");
  const prepared = withoutLeadingTitle.replace(/\[\[([^[\]]+)\]\]/g, (_match, rawTarget: string) => {
    const [targetWithHeading, alias] = rawTarget.split("|");
    const [targetTitle, heading] = targetWithHeading.split("#");
    const title = targetTitle.trim();
    const label = (alias ?? heading ?? title).trim();
    const linkedNote = noteByTitle.get(title);

    if (!linkedNote) {
      return label;
    }

    const headingHash = heading ? `#${slugifyHeading(heading)}` : "";

    return `[${label}](${withBasePath(linkedNote.url)}${headingHash})`;
  });

  return markdown.render(prepared);
}
