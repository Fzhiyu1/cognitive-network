import assert from "node:assert/strict"
import fs from "node:fs"
import path from "node:path"
import test from "node:test"

import zhCN from "./i18n/locales/zh-CN"

const repoRoot = path.resolve(import.meta.dirname, "..")

test("zh-CN locale uses restrained public-facing terminology", () => {
  assert.equal(zhCN.components.graph.title, "关系图谱")
  assert.equal(zhCN.components.backlinks.title, "相关链接")
  assert.equal(zhCN.components.tableOfContents.title, "目录")
  assert.equal(zhCN.components.search.searchBarPlaceholder, "搜索主题、概念或线索")
})

test("layout exposes restrained navigation labels", () => {
  const layout = fs.readFileSync(path.join(repoRoot, "quartz.layout.ts"), "utf8")

  assert.match(layout, /title:\s*"导航"/)
  assert.match(layout, /title:\s*"最近更新"/)
  assert.match(layout, /rootName:\s*"首页"/)
})

test("homepage copy foregrounds public structure over persona", () => {
  const homepage = fs.readFileSync(path.join(repoRoot, "content/index.md"), "utf8")

  assert.match(homepage, /公开的是整理后的结构/)
  assert.match(homepage, /不是未经处理的自白/)
  assert.match(homepage, /三个入口/)
  assert.match(homepage, /推荐主题/)
  assert.match(homepage, /阅读方式/)
})

test("homepage exposes a research-terminal status rail", () => {
  const homepage = fs.readFileSync(path.join(repoRoot, "content/index.md"), "utf8")

  assert.match(homepage, /home-status-strip/)
  assert.match(homepage, /home-focus-band/)
  assert.match(homepage, /公开层已整理/)
  assert.match(homepage, /关系图谱在线/)
  assert.match(homepage, /检索优先/)
})

test("custom theme defines dedicated light-mode shell surfaces", () => {
  const styles = fs.readFileSync(path.join(repoRoot, "quartz/styles/custom.scss"), "utf8")

  assert.match(styles, /:root\s*\{[\s\S]*--shell-button-surface:/)
  assert.match(styles, /:root\[saved-theme="dark"\]\s*\{[\s\S]*--shell-button-surface:/)
  assert.match(styles, /\.darkmode,\s*\.readermode\s*\{[\s\S]*background:\s*var\(--shell-button-surface\)/)
  assert.match(styles, /\.graph\s*\{[\s\S]*>\s*\.graph-outer\s*\{[\s\S]*background:\s*var\(--shell-graph-surface\)/)
  assert.match(styles, /\.home-signal-item\s*\{[\s\S]*background:\s*var\(--shell-chip-surface\)/)
})

test("custom theme adds precision-terminal ambient hooks", () => {
  const styles = fs.readFileSync(path.join(repoRoot, "quartz/styles/custom.scss"), "utf8")

  assert.match(styles, /--shell-focus-line:/)
  assert.match(styles, /--shell-active-glow:/)
  assert.match(styles, /@keyframes shell-pulse/)
  assert.match(styles, /@keyframes shell-rise/)
  assert.match(styles, /\.home-status-strip/)
  assert.match(styles, /\.home-focus-band/)
  assert.match(styles, /\.graph-shell-meta/)
})

test("graph component exposes telemetry rails for local and global views", () => {
  const graphComponent = fs.readFileSync(path.join(repoRoot, "quartz/components/Graph.tsx"), "utf8")

  assert.match(graphComponent, /graph-shell-meta/)
  assert.match(graphComponent, /graph-shell-stats/)
  assert.match(graphComponent, /graph-container-shell/)
})

test("graph renderer strengthens hover emphasis and shell stats", () => {
  const graphScript = fs.readFileSync(
    path.join(repoRoot, "quartz/components/scripts/graph.inline.ts"),
    "utf8",
  )

  assert.match(graphScript, /activeLinkWidth/)
  assert.match(graphScript, /activeNodeStrokeWidth/)
  assert.match(graphScript, /updateGraphShellStats/)
})

test("custom theme defines graph telemetry and contrast controls", () => {
  const styles = fs.readFileSync(path.join(repoRoot, "quartz/styles/custom.scss"), "utf8")

  assert.match(styles, /--shell-graph-link-active:/)
  assert.match(styles, /--shell-graph-link-idle:/)
  assert.match(styles, /--shell-graph-node-ring:/)
  assert.match(styles, /\.graph-shell-meta/)
  assert.match(styles, /\.graph-shell-stat/)
  assert.match(styles, /\.graph-shell-legend/)
})

test("custom theme favors ambient background layers over scanline sweeps", () => {
  const styles = fs.readFileSync(path.join(repoRoot, "quartz/styles/custom.scss"), "utf8")

  assert.match(styles, /--shell-page-starfield:/)
  assert.match(styles, /--shell-page-orbit:/)
  assert.match(styles, /--shell-page-aurora:/)
  assert.match(styles, /--shell-page-nebula:/)
  assert.match(styles, /--shell-page-depth-glow:/)
  assert.match(styles, /--shell-page-prism:/)
  assert.match(styles, /--shell-page-dot-matrix:/)
  assert.match(styles, /--shell-page-dot-matrix-fine:/)
  assert.match(styles, /--shell-page-dot-bloom:/)
  assert.match(styles, /--shell-home-hero-aura:/)
  assert.match(styles, /--shell-home-hero-rim:/)
  assert.match(styles, /--shell-page-vignette:/)
  assert.match(styles, /body\s*\{[\s\S]*&::before\s*\{[\s\S]*var\(--shell-page-prism\)/)
  assert.match(styles, /body\s*\{[\s\S]*&::before\s*\{[\s\S]*var\(--shell-page-aurora\)/)
  assert.match(styles, /article\.home-page\s*\{[\s\S]*&::before,\s*&::after[\s\S]*&::before\s*\{[\s\S]*var\(--shell-home-hero-aura\)/)
  assert.doesNotMatch(styles, /article\.home-page\s*\{[\s\S]*&::before,\s*&::after[\s\S]*&::before\s*\{[\s\S]*var\(--shell-page-nebula\)/)
  assert.match(styles, /article\.home-page\s*\{[\s\S]*> section::before\s*\{[\s\S]*var\(--shell-page-depth-glow\)/)
  assert.match(styles, /\.page\s*\{[\s\S]*&::before\s*\{[\s\S]*var\(--shell-page-dot-matrix-fine\)/)
  assert.match(styles, /\.page\s*\{[\s\S]*&::before\s*\{[\s\S]*var\(--shell-page-dot-matrix\)/)
  assert.match(styles, /\.page\s*\{[\s\S]*&::before\s*\{[\s\S]*var\(--shell-page-dot-bloom\)/)
  assert.doesNotMatch(styles, /\.page\s*\{[\s\S]*&::before\s*\{[\s\S]*linear-gradient\(90deg,\s*transparent,\s*var\(--shell-page-grid-line\),\s*transparent\)/)
  assert.doesNotMatch(styles, /animation:\s*shell-sweep/)
  assert.doesNotMatch(styles, /--shell-page-scanline:/)
})

test("custom theme turns search into a dedicated shell rather than default quartz chrome", () => {
  const styles = fs.readFileSync(path.join(repoRoot, "quartz/styles/custom.scss"), "utf8")

  assert.match(styles, /--shell-search-shell:/)
  assert.match(styles, /--shell-search-pane:/)
  assert.match(styles, /\.search\s*\{[\s\S]*flex:\s*1 1 auto;/)
  assert.match(styles, /\.search\s*\{[\s\S]*min-width:\s*0;/)
  assert.match(styles, /\.search\s*\{[\s\S]*width:\s*100%;/)
  assert.match(styles, /\.search\s*\{[\s\S]*>\s*\.search-button\s*\{[\s\S]*justify-content:\s*flex-start;/)
  assert.match(styles, /\.search\s*\{[\s\S]*>\s*\.search-button\s*\{[\s\S]*gap:\s*0\.5rem;/)
  assert.match(styles, /\.search\s*\{[\s\S]*>\s*\.search-container\s*\{[\s\S]*>\s*\.search-space\s*\{[\s\S]*width:\s*min\(70rem,\s*calc\(100vw - 4rem\)\);/)
  assert.match(styles, /\.search\s*\{[\s\S]*>\s*\.search-container\s*\{[\s\S]*>\s*\.search-space\s*\{[\s\S]*>\s*\.search-layout\s*\{[\s\S]*overflow:\s*hidden;/)
  assert.match(styles, /\.search\s*\{[\s\S]*>\s*\.search-container\s*\{[\s\S]*>\s*\.search-space\s*\{[\s\S]*>\s*\.search-layout\s*\{[\s\S]*background:\s*var\(--shell-search-shell\);/)
  assert.match(styles, /\.search\s*\{[\s\S]*>\s*\.search-container\s*\{[\s\S]*>\s*\.search-space\s*\{[\s\S]*>\s*\.search-layout[\s\S]*\.preview-container\s*\{[\s\S]*background:\s*var\(--shell-search-pane\);/)
  assert.match(styles, /\.search\s*\{[\s\S]*>\s*\.search-container\s*\{[\s\S]*>\s*\.search-space\s*\{[\s\S]*>\s*\.search-layout[\s\S]*\.result-card:hover,[\s\S]*color:\s*var\(--dark\);/)
})

test("search trigger reads like an input shell instead of a plain button", () => {
  const searchComponent = fs.readFileSync(
    path.join(repoRoot, "quartz/components/Search.tsx"),
    "utf8",
  )
  const styles = fs.readFileSync(path.join(repoRoot, "quartz/styles/custom.scss"), "utf8")

  assert.match(searchComponent, /const searchButtonLabel =/)
  assert.match(searchComponent, /<p>\{searchButtonLabel\}<\/p>/)
  assert.match(searchComponent, /class="search-shortcut"/)
  assert.match(styles, /\.search-shortcut\s*\{/)
  assert.match(styles, /\.search\s*\{[\s\S]*>\s*\.search-button\s*\{[\s\S]*> \.search-shortcut\s*\{/)
  assert.match(styles, /\.left\.sidebar > \.flex-component\s*\{[\s\S]*display:\s*flex;/)
  assert.match(styles, /\.left\.sidebar > \.flex-component\s*\{[\s\S]*gap:\s*0\.75rem !important;/)
  assert.match(styles, /\.search\s*\{[\s\S]*min-width:\s*0;/)
  assert.match(styles, /\.search\s*\{[\s\S]*>\s*\.search-button\s*\{[\s\S]*min-width:\s*0;/)
  assert.match(styles, /\.search\s*\{[\s\S]*>\s*\.search-button\s*\{[\s\S]*> \.search-shortcut\s*\{[\s\S]*display:\s*none;/)
})

test("search preview is summarized into a dedicated shell rather than raw page chrome", () => {
  const searchScript = fs.readFileSync(
    path.join(repoRoot, "quartz/components/scripts/search.inline.ts"),
    "utf8",
  )
  const styles = fs.readFileSync(path.join(repoRoot, "quartz/styles/custom.scss"), "utf8")

  assert.match(searchScript, /const maxPreviewBlocks = \d+/)
  assert.match(searchScript, /function renderSearchIdleState\(/)
  assert.match(searchScript, /function buildPreviewShell\(/)
  assert.match(searchScript, /preview-shell-header/)
  assert.match(searchScript, /preview-shell-body/)
  assert.match(searchScript, /preview-shell-block/)
  assert.doesNotMatch(searchScript, /previewInner\.append\(\.\.\.innerDiv\)/)
  assert.match(styles, /\.search-layout\s*\{[\s\S]*&\.idle-state\s*\{/)
  assert.match(styles, /\.preview-shell\s*\{/)
  assert.match(styles, /\.preview-shell-body\s*\{/)
  assert.match(styles, /\.search-idle-panel\s*\{/)
  assert.doesNotMatch(styles, /body:has\(\.search-container\.active\)/)
})
