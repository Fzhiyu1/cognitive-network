# Knowledge Base Public Site Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a public static site for this knowledge base with an overview homepage, shareable detail pages, an interactive graph view, and automatic GitHub Pages deployment.

**Architecture:** Keep the current Markdown knowledge base as the single source of truth. Add a root-level Astro app plus a custom build-time content pipeline that parses frontmatter and `[[wiki-link]]`, generates structured JSON for pages and graph rendering, then emits static routes for all public content while excluding `3-projects` from public indexes and the default graph dataset.

**Tech Stack:** Astro, TypeScript, Node.js, gray-matter, graphology, sigma.js, Vitest, GitHub Actions, GitHub Pages

---

### Task 1: Bootstrap the Static Site Toolchain

**Files:**
- Create: `package.json`
- Create: `astro.config.mjs`
- Create: `tsconfig.json`
- Create: `src/env.d.ts`
- Create: `src/styles/global.css`
- Create: `src/layouts/BaseLayout.astro`
- Create: `src/pages/index.astro`
- Create: `tests/site-build.test.ts`

**Step 1: Write the failing test**

Create `tests/site-build.test.ts` with a smoke assertion that expects the generated homepage HTML to contain a known title string after build.

```ts
import { execSync } from "node:child_process"
import { existsSync, readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"

describe("site build", () => {
  it("generates a homepage", () => {
    execSync("npm run build", { stdio: "pipe" })
    expect(existsSync("dist/index.html")).toBe(true)
    expect(readFileSync("dist/index.html", "utf8")).toContain("Cognitive Network")
  })
})
```

**Step 2: Run test to verify it fails**

Run: `npm test -- --run tests/site-build.test.ts`

Expected: FAIL because the project has no Node toolchain or build script yet.

**Step 3: Write minimal implementation**

- Add Astro and Vitest dependencies in `package.json`
- Add `build`, `dev`, `test`, and `prebuild` scripts
- Create a minimal `BaseLayout.astro`
- Create a minimal `src/pages/index.astro`
- Add a minimal `global.css`

**Step 4: Run test to verify it passes**

Run: `npm install`
Run: `npm test -- --run tests/site-build.test.ts`

Expected: PASS and `dist/index.html` exists.

**Step 5: Commit**

```bash
git add package.json astro.config.mjs tsconfig.json src tests
git commit -m "feat: bootstrap knowledge base public site"
```

### Task 2: Build the Markdown Discovery and Metadata Pipeline

**Files:**
- Create: `scripts/build-knowledge-index.mjs`
- Create: `src/generated/.gitkeep`
- Create: `src/lib/content/types.ts`
- Create: `src/lib/content/sections.ts`
- Create: `tests/content-index.test.ts`

**Step 1: Write the failing test**

Create `tests/content-index.test.ts` to assert that the index builder:

- reads markdown from `0-inbox`, `1-concepts`, `2-explorations`, `3-projects`, `4-references`
- extracts title and section
- marks `3-projects` with `isIndexed: false`

```ts
import { execSync } from "node:child_process"
import { readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"

describe("content index builder", () => {
  it("marks projects as non-indexed", () => {
    execSync("node scripts/build-knowledge-index.mjs", { stdio: "pipe" })
    const content = JSON.parse(readFileSync("src/generated/content-index.json", "utf8"))
    const project = content.notes.find((note: { section: string }) => note.section === "projects")
    expect(project).toBeDefined()
    expect(project.isIndexed).toBe(false)
  })
})
```

**Step 2: Run test to verify it fails**

Run: `npm test -- --run tests/content-index.test.ts`

Expected: FAIL because the index builder does not exist.

**Step 3: Write minimal implementation**

- Implement directory scanning
- Parse frontmatter with `gray-matter`
- Infer section from top-level folder
- Emit `src/generated/content-index.json`
- Add `isIndexed` and `isVisibleInDefaultGraph` flags

**Step 4: Run test to verify it passes**

Run: `node scripts/build-knowledge-index.mjs`
Run: `npm test -- --run tests/content-index.test.ts`

Expected: PASS and `src/generated/content-index.json` contains notes from all content folders.

**Step 5: Commit**

```bash
git add scripts src/lib/content src/generated tests/content-index.test.ts
git commit -m "feat: add knowledge base content index pipeline"
```

### Task 3: Parse Wiki Links and Generate Graph Data

**Files:**
- Modify: `scripts/build-knowledge-index.mjs`
- Create: `src/lib/content/wiki-links.ts`
- Create: `tests/wiki-links.test.ts`

**Step 1: Write the failing test**

Create `tests/wiki-links.test.ts` to assert that:

- `[[标题]]` is extracted as an outgoing link
- reverse links are generated
- project nodes are excluded from the default graph dataset

```ts
import { execSync } from "node:child_process"
import { readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"

describe("graph builder", () => {
  it("builds graph edges from wiki links", () => {
    execSync("node scripts/build-knowledge-index.mjs", { stdio: "pipe" })
    const graph = JSON.parse(readFileSync("src/generated/graph.json", "utf8"))
    expect(Array.isArray(graph.nodes)).toBe(true)
    expect(Array.isArray(graph.edges)).toBe(true)
    expect(graph.meta.defaultSections).not.toContain("projects")
  })
})
```

**Step 2: Run test to verify it fails**

Run: `npm test -- --run tests/wiki-links.test.ts`

Expected: FAIL because graph generation is not implemented.

**Step 3: Write minimal implementation**

- Add a wiki-link parser
- Resolve note titles to note IDs/slugs
- Generate outgoing and incoming links
- Emit `src/generated/graph.json`
- Exclude `projects` from default graph nodes/edges

**Step 4: Run test to verify it passes**

Run: `node scripts/build-knowledge-index.mjs`
Run: `npm test -- --run tests/wiki-links.test.ts`

Expected: PASS and graph JSON includes nodes, edges, and filter metadata.

**Step 5: Commit**

```bash
git add scripts src/lib/content tests/wiki-links.test.ts src/generated
git commit -m "feat: generate knowledge graph data from wiki links"
```

### Task 4: Generate Shareable Detail Pages for Every Note

**Files:**
- Create: `src/lib/content/loaders.ts`
- Create: `src/pages/concepts/[slug].astro`
- Create: `src/pages/explorations/[slug].astro`
- Create: `src/pages/references/[slug].astro`
- Create: `src/pages/inbox/[slug].astro`
- Create: `src/pages/projects/[slug].astro`
- Create: `src/components/NoteMeta.astro`
- Create: `src/components/RelatedNotes.astro`
- Create: `tests/detail-routes.test.ts`

**Step 1: Write the failing test**

Create `tests/detail-routes.test.ts` to assert that:

- a concept page is generated
- a project page is generated
- project pages exist without a public index page

```ts
import { execSync } from "node:child_process"
import { existsSync } from "node:fs"
import { describe, expect, it } from "vitest"

describe("detail routes", () => {
  it("builds shareable note pages", () => {
    execSync("npm run build", { stdio: "pipe" })
    expect(existsSync("dist/concepts")).toBe(true)
    expect(existsSync("dist/projects")).toBe(true)
    expect(existsSync("dist/projects/index.html")).toBe(false)
  })
})
```

**Step 2: Run test to verify it fails**

Run: `npm test -- --run tests/detail-routes.test.ts`

Expected: FAIL because detail routes are not implemented.

**Step 3: Write minimal implementation**

- Build Astro dynamic routes for each section
- Render markdown content
- Render metadata, tags, outgoing links, incoming links
- Ensure project notes render under `/projects/<slug>/`
- Do not create `/projects/index.astro`

**Step 4: Run test to verify it passes**

Run: `npm test -- --run tests/detail-routes.test.ts`

Expected: PASS and detail pages exist in `dist/`.

**Step 5: Commit**

```bash
git add src/pages src/components src/lib/content tests/detail-routes.test.ts
git commit -m "feat: add shareable detail pages for knowledge notes"
```

### Task 5: Build Public Index Pages and Overview Homepage

**Files:**
- Modify: `src/pages/index.astro`
- Create: `src/pages/concepts/index.astro`
- Create: `src/pages/explorations/index.astro`
- Create: `src/pages/references/index.astro`
- Create: `src/pages/inbox/index.astro`
- Create: `src/components/HomeHero.astro`
- Create: `src/components/OverviewStats.astro`
- Create: `src/components/SectionEntryCard.astro`
- Create: `src/components/RecentUpdates.astro`
- Create: `tests/public-indexes.test.ts`

**Step 1: Write the failing test**

Create `tests/public-indexes.test.ts` to assert that:

- homepage contains overview sections
- concepts index exists
- projects index does not exist

```ts
import { execSync } from "node:child_process"
import { existsSync, readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"

describe("public indexes", () => {
  it("renders public indexes without projects", () => {
    execSync("npm run build", { stdio: "pipe" })
    expect(existsSync("dist/concepts/index.html")).toBe(true)
    expect(existsSync("dist/projects/index.html")).toBe(false)
    expect(readFileSync("dist/index.html", "utf8")).toContain("知识库总览")
  })
})
```

**Step 2: Run test to verify it fails**

Run: `npm test -- --run tests/public-indexes.test.ts`

Expected: FAIL because the homepage and list pages are incomplete.

**Step 3: Write minimal implementation**

- Build the homepage sections
- Build list pages for inbox, concepts, explorations, references
- Exclude projects from homepage stats, entry cards, recent updates, and public section listings

**Step 4: Run test to verify it passes**

Run: `npm test -- --run tests/public-indexes.test.ts`

Expected: PASS and index routing matches visibility rules.

**Step 5: Commit**

```bash
git add src/pages src/components tests/public-indexes.test.ts
git commit -m "feat: add public homepage and indexed section pages"
```

### Task 6: Implement the Interactive Graph Page

**Files:**
- Create: `src/pages/graph/index.astro`
- Create: `src/components/KnowledgeGraph.tsx`
- Create: `src/components/GraphSidebar.tsx`
- Create: `src/components/GraphFilters.tsx`
- Create: `tests/graph-page.test.ts`

**Step 1: Write the failing test**

Create `tests/graph-page.test.ts` to assert that:

- `/graph/` builds
- the page includes a graph container
- the page exposes default filter metadata without `projects`

```ts
import { execSync } from "node:child_process"
import { readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"

describe("graph page", () => {
  it("renders the graph entrypoint", () => {
    execSync("npm run build", { stdio: "pipe" })
    const html = readFileSync("dist/graph/index.html", "utf8")
    expect(html).toContain("data-graph-root")
    expect(html).toContain("graph")
    expect(html).not.toContain("\"projects\"")
  })
})
```

**Step 2: Run test to verify it fails**

Run: `npm test -- --run tests/graph-page.test.ts`

Expected: FAIL because the graph page is not implemented.

**Step 3: Write minimal implementation**

- Add a graph page route
- Render Sigma graph from generated data
- Support zoom, drag, hover, click-to-navigate
- Add section and tag filters
- Keep projects out of default graph data

**Step 4: Run test to verify it passes**

Run: `npm test -- --run tests/graph-page.test.ts`

Expected: PASS and the graph page builds successfully.

**Step 5: Commit**

```bash
git add src/pages/graph src/components tests/graph-page.test.ts
git commit -m "feat: add interactive knowledge graph page"
```

### Task 7: Add Build Validation and Broken-Link Guardrails

**Files:**
- Modify: `scripts/build-knowledge-index.mjs`
- Create: `tests/link-validation.test.ts`

**Step 1: Write the failing test**

Create `tests/link-validation.test.ts` to assert that unresolved wiki links fail the build, except where explicitly allowed.

```ts
import { execSync } from "node:child_process"
import { describe, expect, it } from "vitest"

describe("wiki-link validation", () => {
  it("exits successfully for the current knowledge base", () => {
    expect(() => execSync("node scripts/build-knowledge-index.mjs", { stdio: "pipe" })).not.toThrow()
  })
})
```

**Step 2: Run test to verify it fails**

Run: `npm test -- --run tests/link-validation.test.ts`

Expected: FAIL if the current builder does not validate or if unresolved links exist without explicit handling.

**Step 3: Write minimal implementation**

- Add unresolved-link detection
- Print actionable errors with source file paths
- Allow an explicit ignore list only if absolutely required

**Step 4: Run test to verify it passes**

Run: `npm test -- --run tests/link-validation.test.ts`

Expected: PASS and current content validates cleanly.

**Step 5: Commit**

```bash
git add scripts tests/link-validation.test.ts
git commit -m "feat: validate wiki links during site build"
```

### Task 8: Add GitHub Pages Deployment Workflow

**Files:**
- Create: `.github/workflows/deploy-site.yml`
- Modify: `package.json`
- Modify: `README.md`
- Create: `tests/build-command.test.ts`

**Step 1: Write the failing test**

Create `tests/build-command.test.ts` to assert that the production build command exits successfully.

```ts
import { execSync } from "node:child_process"
import { describe, expect, it } from "vitest"

describe("production build", () => {
  it("builds without errors", () => {
    expect(() => execSync("npm run build", { stdio: "pipe" })).not.toThrow()
  })
})
```

**Step 2: Run test to verify it fails**

Run: `npm test -- --run tests/build-command.test.ts`

Expected: FAIL until the full pipeline and workflow assumptions are correct.

**Step 3: Write minimal implementation**

- Add a GitHub Actions workflow that:
  - installs dependencies
  - runs `npm test`
  - runs `npm run build`
  - uploads `dist/`
  - deploys to GitHub Pages
- Update `README.md` with local dev and deploy instructions

**Step 4: Run test to verify it passes**

Run: `npm test -- --run tests/build-command.test.ts`
Run: `npm run build`

Expected: PASS and `dist/` is ready for Pages deployment.

**Step 5: Commit**

```bash
git add .github/workflows/deploy-site.yml package.json README.md tests/build-command.test.ts
git commit -m "feat: deploy knowledge base public site to github pages"
```

### Task 9: Final Verification

**Files:**
- Verify: `dist/index.html`
- Verify: `dist/graph/index.html`
- Verify: `dist/concepts/`
- Verify: `dist/projects/`

**Step 1: Run the full verification suite**

Run: `npm test`

Expected: PASS with all test files green.

**Step 2: Run the production build**

Run: `npm run build`

Expected: PASS and fresh assets generated in `dist/`.

**Step 3: Spot-check output**

Run:

```bash
test -f dist/index.html
test -f dist/graph/index.html
find dist/projects -name 'index.html' | grep -qv '^dist/projects/index.html$'
```

Expected: all commands exit 0 and no public `dist/projects/index.html` exists.

**Step 4: Commit final integration**

```bash
git add .
git commit -m "feat: launch knowledge base public site"
```
