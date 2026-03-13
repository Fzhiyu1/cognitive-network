import type { KnowledgeBaseData } from "./knowledge-base";

interface GraphPayloadOptions {
  includeProjects?: boolean;
}

export function getGraphPayload(data: KnowledgeBaseData, options: GraphPayloadOptions = {}) {
  return options.includeProjects ? data.graph : data.defaultGraph;
}
