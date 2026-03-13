import { type KnowledgeBaseData } from "./knowledge-base";
import { SECTION_BY_KEY, type SectionKey } from "./sections";

export function getHomepageData(data: KnowledgeBaseData) {
  const publicNotes = data.publicIndexNotes;
  const featuredThreads = [...publicNotes]
    .sort((left, right) => {
      const leftScore = left.backlinks.length + left.outgoingLinks.length;
      const rightScore = right.backlinks.length + right.outgoingLinks.length;

      return rightScore - leftScore || right.modifiedTime.localeCompare(left.modifiedTime);
    })
    .slice(0, 4);

  const sectionEntries = [...SECTION_BY_KEY.values()]
    .filter((section) => section.isPublicIndex)
    .map((section) => ({
      ...section,
      count: publicNotes.filter((note) => note.sectionKey === section.key).length
    }));

  return {
    stats: {
      totalNotes: data.notes.length,
      publicNotes: publicNotes.length,
      concepts: data.notes.filter((note) => note.sectionKey === "concepts").length,
      explorations: data.notes.filter((note) => note.sectionKey === "explorations").length,
      references: data.notes.filter((note) => note.sectionKey === "references").length
    },
    recentUpdates: publicNotes.slice(0, 6),
    featuredThreads,
    sectionEntries
  };
}

export function getSectionIndexData(data: KnowledgeBaseData, sectionKey: SectionKey) {
  const section = SECTION_BY_KEY.get(sectionKey);

  if (!section?.isPublicIndex) {
    throw new Error(`${sectionKey} is not publicly indexable`);
  }

  return {
    section,
    notes: data.publicIndexNotes.filter((note) => note.sectionKey === sectionKey)
  };
}
