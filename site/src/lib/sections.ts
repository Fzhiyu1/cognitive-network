export type SectionKey = "inbox" | "concepts" | "explorations" | "projects" | "references";

export interface SectionDefinition {
  key: SectionKey;
  directory: string;
  routeSegment: string;
  label: string;
  isPublicIndex: boolean;
  isDefaultGraph: boolean;
  isDirectlyAccessible: boolean;
}

export const SECTION_DEFINITIONS: SectionDefinition[] = [
  {
    key: "inbox",
    directory: "0-inbox",
    routeSegment: "inbox",
    label: "Inbox",
    isPublicIndex: true,
    isDefaultGraph: true,
    isDirectlyAccessible: true
  },
  {
    key: "concepts",
    directory: "1-concepts",
    routeSegment: "concepts",
    label: "Concepts",
    isPublicIndex: true,
    isDefaultGraph: true,
    isDirectlyAccessible: true
  },
  {
    key: "explorations",
    directory: "2-explorations",
    routeSegment: "explorations",
    label: "Explorations",
    isPublicIndex: true,
    isDefaultGraph: true,
    isDirectlyAccessible: true
  },
  {
    key: "projects",
    directory: "3-projects",
    routeSegment: "projects",
    label: "Projects",
    isPublicIndex: false,
    isDefaultGraph: false,
    isDirectlyAccessible: true
  },
  {
    key: "references",
    directory: "4-references",
    routeSegment: "references",
    label: "References",
    isPublicIndex: true,
    isDefaultGraph: true,
    isDirectlyAccessible: true
  }
];

export const SECTION_BY_DIRECTORY = new Map(SECTION_DEFINITIONS.map((section) => [section.directory, section]));
export const SECTION_BY_KEY = new Map(SECTION_DEFINITIONS.map((section) => [section.key, section]));

export function getSectionByDirectory(directory: string): SectionDefinition | undefined {
  return SECTION_BY_DIRECTORY.get(directory);
}

export function isPublicIndexSection(sectionKey: SectionKey): boolean {
  return SECTION_BY_KEY.get(sectionKey)?.isPublicIndex ?? false;
}
