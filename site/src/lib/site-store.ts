import path from "node:path";

import { buildKnowledgeBase, type KnowledgeBaseData } from "./knowledge-base";

const repositoryRoot = path.resolve(process.cwd(), "..");
let cache: Promise<KnowledgeBaseData> | undefined;

export function getRepositoryRoot(): string {
  return repositoryRoot;
}

export function getKnowledgeBaseData(): Promise<KnowledgeBaseData> {
  cache ??= buildKnowledgeBase({ rootDir: repositoryRoot });
  return cache;
}
