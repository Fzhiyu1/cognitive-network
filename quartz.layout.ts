import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"
import type { Options as ExplorerOptions } from "./quartz/components/Explorer"

const graph = Component.Graph({
  localGraph: {
    drag: true,
    zoom: true,
    depth: 1,
    scale: 1.15,
    repelForce: 0.7,
    centerForce: 0.35,
    linkDistance: 34,
    fontSize: 0.72,
    opacityScale: 1,
    showTags: true,
    removeTags: [],
    focusOnHover: true,
    enableRadial: false,
  },
  globalGraph: {
    drag: true,
    zoom: true,
    depth: -1,
    scale: 0.95,
    repelForce: 0.55,
    centerForce: 0.2,
    linkDistance: 38,
    fontSize: 0.72,
    opacityScale: 1,
    showTags: true,
    removeTags: [],
    focusOnHover: true,
    enableRadial: true,
  },
})

const explorerFilterFn: ExplorerOptions["filterFn"] = (node) => {
  return node.slugSegment !== "tags" && node.slugSegment !== "projects"
}

const explorerMapFn: ExplorerOptions["mapFn"] = (node) => {
  if (node.slugSegment === "concepts") {
    node.displayName = "概念"
  } else if (node.slugSegment === "explorations") {
    node.displayName = "探索"
  } else if (node.slugSegment === "references") {
    node.displayName = "引用"
  } else if (node.slugSegment === "index" && node.slug === "index") {
    node.displayName = "总览"
  }
}

const explorer = Component.Explorer({
  title: "导航",
  folderDefaultState: "open",
  folderClickBehavior: "collapse",
  useSavedState: true,
  filterFn: explorerFilterFn,
  mapFn: explorerMapFn,
})

const recentNotes = Component.RecentNotes({
  title: "最近更新",
  limit: 8,
  showTags: true,
  filter: (file) => file.slug !== "index" && !file.slug?.startsWith("projects/"),
})

export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [],
  afterBody: [
    Component.ConditionalRender({
      component: recentNotes,
      condition: ({ fileData }) => fileData.slug === "index",
    }),
  ],
  footer: Component.Footer({
    links: {
      GitHub: "https://github.com/Fzhiyu1/cognitive-network",
      Obsidian: "https://obsidian.md/",
    },
  }),
}

export const defaultContentPageLayout: PageLayout = {
  beforeBody: [
    Component.ConditionalRender({
      component: Component.Breadcrumbs({ rootName: "总览" }),
      condition: ({ fileData }) => fileData.slug !== "index",
    }),
    Component.ArticleTitle(),
    Component.ContentMeta(),
    Component.TagList(),
  ],
  left: [
    Component.PageTitle(),
    Component.MobileOnly(Component.Spacer()),
    Component.Flex({
      components: [
        {
          Component: Component.Search(),
          grow: true,
        },
        { Component: Component.Darkmode() },
        { Component: Component.ReaderMode() },
      ],
    }),
    explorer,
  ],
  right: [
    graph,
    Component.ConditionalRender({
      component: Component.DesktopOnly(Component.TableOfContents()),
      condition: ({ fileData }) => fileData.slug !== "index",
    }),
    Component.ConditionalRender({
      component: Component.Backlinks(),
      condition: ({ fileData }) => fileData.slug !== "index",
    }),
  ],
}

export const defaultListPageLayout: PageLayout = {
  beforeBody: [
    Component.Breadcrumbs({ rootName: "总览" }),
    Component.ArticleTitle(),
    Component.ContentMeta(),
  ],
  left: [
    Component.PageTitle(),
    Component.MobileOnly(Component.Spacer()),
    Component.Flex({
      components: [
        {
          Component: Component.Search(),
          grow: true,
        },
        { Component: Component.Darkmode() },
      ],
    }),
    explorer,
  ],
  right: [graph],
}
