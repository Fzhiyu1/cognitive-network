import type { APIRoute } from "astro";

import { getKnowledgeBaseData } from "../lib/site-store";

export const prerender = true;

export const GET: APIRoute = async () => {
  const data = await getKnowledgeBaseData();

  return Response.json({
    notes: data.notes.map((note) => ({
      title: note.title,
      sectionKey: note.sectionKey,
      url: note.url,
      tags: note.tags,
      summary: note.summary,
      modifiedTime: note.modifiedTime,
      isIndexedPublicly: note.isIndexedPublicly,
      isInDefaultGraph: note.isInDefaultGraph,
      isDirectlyAccessible: note.isDirectlyAccessible
    }))
  });
};
