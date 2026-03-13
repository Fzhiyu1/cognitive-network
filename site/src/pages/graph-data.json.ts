import type { APIRoute } from "astro";

import { getGraphPayload } from "../lib/graph-data";
import { getKnowledgeBaseData } from "../lib/site-store";

export const prerender = true;

export const GET: APIRoute = async () => {
  const data = await getKnowledgeBaseData();

  return Response.json({
    defaultGraph: getGraphPayload(data),
    fullGraph: getGraphPayload(data, { includeProjects: true })
  });
};
