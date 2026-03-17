import FlexSearch, { DefaultDocumentSearchResults } from "flexsearch"
import { ContentDetails } from "../../plugins/emitters/contentIndex"
import { registerEscapeHandler, removeAllChildren } from "./util"
import { FullSlug, normalizeRelativeURLs, resolveRelative } from "../../util/path"

interface Item {
  id: number
  slug: FullSlug
  title: string
  content: string
  tags: string[]
  [key: string]: any
}

// Can be expanded with things like "term" in the future
type SearchType = "basic" | "tags"
let searchType: SearchType = "basic"
let currentSearchTerm: string = ""
const encoder = (str: string): string[] => {
  const tokens: string[] = []
  let bufferStart = -1
  let bufferEnd = -1
  const lower = str.toLowerCase()

  let i = 0
  for (const char of lower) {
    const code = char.codePointAt(0)!

    const isCJK =
      (code >= 0x3040 && code <= 0x309f) ||
      (code >= 0x30a0 && code <= 0x30ff) ||
      (code >= 0x4e00 && code <= 0x9fff) ||
      (code >= 0xac00 && code <= 0xd7af) ||
      (code >= 0x20000 && code <= 0x2a6df)

    const isWhitespace = code === 32 || code === 9 || code === 10 || code === 13

    if (isCJK) {
      if (bufferStart !== -1) {
        tokens.push(lower.slice(bufferStart, bufferEnd))
        bufferStart = -1
      }
      tokens.push(char)
    } else if (isWhitespace) {
      if (bufferStart !== -1) {
        tokens.push(lower.slice(bufferStart, bufferEnd))
        bufferStart = -1
      }
    } else {
      if (bufferStart === -1) bufferStart = i
      bufferEnd = i + char.length
    }

    i += char.length
  }

  if (bufferStart !== -1) {
    tokens.push(lower.slice(bufferStart))
  }

  return tokens
}

let index = new FlexSearch.Document<Item>({
  encode: encoder,
  document: {
    id: "id",
    tag: "tags",
    index: [
      {
        field: "title",
        tokenize: "forward",
      },
      {
        field: "content",
        tokenize: "forward",
      },
      {
        field: "tags",
        tokenize: "forward",
      },
    ],
  },
})

const p = new DOMParser()
const fetchContentCache: Map<FullSlug, Document> = new Map()
const contextWindowWords = 30
const numSearchResults = 8
const numTagResults = 5
const maxPreviewBlocks = 5
const maxPreviewTags = 5
const maxPreviewBreadcrumbs = 3

const tokenizeTerm = (term: string) => {
  const tokens = term.split(/\s+/).filter((t) => t.trim() !== "")
  const tokenLen = tokens.length
  if (tokenLen > 1) {
    for (let i = 1; i < tokenLen; i++) {
      tokens.push(tokens.slice(0, i + 1).join(" "))
    }
  }

  return tokens.sort((a, b) => b.length - a.length) // always highlight longest terms first
}

function highlight(searchTerm: string, text: string, trim?: boolean) {
  const tokenizedTerms = tokenizeTerm(searchTerm)
  let tokenizedText = text.split(/\s+/).filter((t) => t !== "")

  let startIndex = 0
  let endIndex = tokenizedText.length - 1
  if (trim) {
    const includesCheck = (tok: string) =>
      tokenizedTerms.some((term) => tok.toLowerCase().startsWith(term.toLowerCase()))
    const occurrencesIndices = tokenizedText.map(includesCheck)

    let bestSum = 0
    let bestIndex = 0
    for (let i = 0; i < Math.max(tokenizedText.length - contextWindowWords, 0); i++) {
      const window = occurrencesIndices.slice(i, i + contextWindowWords)
      const windowSum = window.reduce((total, cur) => total + (cur ? 1 : 0), 0)
      if (windowSum >= bestSum) {
        bestSum = windowSum
        bestIndex = i
      }
    }

    startIndex = Math.max(bestIndex - contextWindowWords, 0)
    endIndex = Math.min(startIndex + 2 * contextWindowWords, tokenizedText.length - 1)
    tokenizedText = tokenizedText.slice(startIndex, endIndex)
  }

  const slice = tokenizedText
    .map((tok) => {
      // see if this tok is prefixed by any search terms
      for (const searchTok of tokenizedTerms) {
        if (tok.toLowerCase().includes(searchTok.toLowerCase())) {
          const regex = new RegExp(searchTok.toLowerCase(), "gi")
          return tok.replace(regex, `<span class="highlight">$&</span>`)
        }
      }
      return tok
    })
    .join(" ")

  return `${startIndex === 0 ? "" : "..."}${slice}${
    endIndex === tokenizedText.length - 1 ? "" : "..."
  }`
}

function escapeHTML(text: string) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;")
}

function normalizeText(text: string | null | undefined) {
  return (text ?? "").replace(/\s+/g, " ").trim()
}

function renderHighlightedText(searchTerm: string, text: string, trim = false) {
  const normalized = normalizeText(text)
  if (!normalized) return ""
  const safeText = escapeHTML(normalized)
  if (searchTerm.trim() === "") return safeText
  return highlight(searchTerm, safeText, trim)
}

function collectPreviewTokens(searchTerm: string) {
  return tokenizeTerm(searchTerm)
    .map((term) => term.trim().toLowerCase())
    .filter((term) => term.length > 0)
}

function matchesPreviewTerm(searchTerm: string, text: string) {
  const tokens = collectPreviewTokens(searchTerm)
  if (tokens.length === 0) return false

  const lower = text.toLowerCase()
  return tokens.some((term) => lower.includes(term))
}

function appendPreviewPills(container: HTMLElement, className: string, texts: string[], prefix = "") {
  if (texts.length === 0) return

  const list = document.createElement("ul")
  list.className = className

  texts.forEach((text) => {
    const item = document.createElement("li")
    item.innerHTML = renderHighlightedText(currentSearchTerm, `${prefix}${text}`)
    list.appendChild(item)
  })

  container.appendChild(list)
}

type PreviewBlock = {
  kind: "heading" | "body" | "quote"
  text: string
  order: number
  matches: boolean
}

function collectPreviewBlocks(doc: Document, searchTerm: string): PreviewBlock[] {
  const blocks: PreviewBlock[] = []
  const seen = new Set<string>()
  let order = 0

  const pushBlock = (node: Element) => {
    if (node.closest("pre, table, nav, aside, footer, .breadcrumb-container, .tags, .content-meta")) {
      return
    }

    const text = normalizeText(node.textContent)
    if (!text) return

    const tagName = node.tagName.toLowerCase()
    const kind =
      tagName === "blockquote" ? "quote" : tagName === "h2" || tagName === "h3" ? "heading" : "body"
    const minLength = kind === "heading" ? 8 : tagName === "li" ? 18 : 28
    if (text.length < minLength) return

    const dedupeKey = `${kind}:${text}`
    if (seen.has(dedupeKey)) return
    seen.add(dedupeKey)

    blocks.push({
      kind,
      text,
      order: order++,
      matches: matchesPreviewTerm(searchTerm, text),
    })
  }

  const article = doc.querySelector("article.popover-hint, .popover-hint article, article")
  article?.querySelectorAll("h2, h3, p, blockquote, li").forEach(pushBlock)

  const listing = doc.querySelector(".page-listing")
  if (blocks.length < maxPreviewBlocks) {
    listing?.querySelectorAll("p, li").forEach(pushBlock)
  }

  const selected: PreviewBlock[] = []
  const selectedText = new Set<string>()
  const addBlock = (block: PreviewBlock | undefined) => {
    if (!block || selectedText.has(block.text) || selected.length >= maxPreviewBlocks) return
    selected.push(block)
    selectedText.add(block.text)
  }

  addBlock(blocks.find((block) => block.kind === "heading"))

  blocks.filter((block) => block.matches).forEach((block) => addBlock(block))
  blocks.forEach((block) => addBlock(block))

  return selected.sort((a, b) => a.order - b.order).slice(0, maxPreviewBlocks)
}

function buildPreviewShell(doc: Document, searchTerm: string) {
  const shell = document.createElement("section")
  shell.className = "preview-inner preview-shell"

  const header = document.createElement("header")
  header.className = "preview-shell-header"

  const kicker = document.createElement("p")
  kicker.className = "preview-shell-kicker"
  kicker.textContent = "PREVIEW"
  header.appendChild(kicker)

  const titleText =
    normalizeText(doc.querySelector(".article-title")?.textContent) ||
    normalizeText(doc.querySelector("article h1, .page-listing h1, h1")?.textContent) ||
    "未命名条目"

  const title = document.createElement("h2")
  title.className = "preview-shell-title"
  title.innerHTML = renderHighlightedText(searchTerm, titleText)
  header.appendChild(title)

  const metaText = normalizeText(doc.querySelector(".content-meta")?.textContent)
  if (metaText) {
    const meta = document.createElement("p")
    meta.className = "preview-shell-meta"
    meta.textContent = metaText
    header.appendChild(meta)
  }

  const breadcrumbs = [...doc.querySelectorAll(".breadcrumb-container .breadcrumb-element")]
    .map((node) => normalizeText(node.textContent))
    .filter(Boolean)
    .slice(-maxPreviewBreadcrumbs)
  appendPreviewPills(header, "preview-shell-breadcrumbs", breadcrumbs)

  const tags = [
    ...doc.querySelectorAll(".tags li p, .tags li, .tags .internal.tag-link, .tags .tag-link"),
  ]
    .map((node) => normalizeText(node.textContent))
    .filter(Boolean)
    .map((tag) => tag.replace(/^#/, ""))
    .slice(0, maxPreviewTags)
  appendPreviewPills(header, "preview-shell-tags", tags, "#")

  const body = document.createElement("div")
  body.className = "preview-shell-body"

  const previewBlocks = collectPreviewBlocks(doc, searchTerm)
  if (previewBlocks.length === 0) {
    const empty = document.createElement("p")
    empty.className = "preview-shell-empty"
    empty.textContent = "当前条目没有可展示的摘要片段。"
    body.appendChild(empty)
  } else {
    previewBlocks.forEach((block) => {
      const blockElement = document.createElement(block.kind === "heading" ? "h3" : "p")
      blockElement.className = "preview-shell-block"
      blockElement.dataset.kind = block.kind
      blockElement.innerHTML = renderHighlightedText(searchTerm, block.text, block.kind !== "heading")
      body.appendChild(blockElement)
    })
  }

  shell.append(header, body)
  return shell
}

function renderSearchIdleState(
  searchLayout: HTMLElement,
  results: HTMLDivElement,
  preview?: HTMLDivElement,
) {
  searchLayout.classList.add("display-results", "idle-state")
  results.classList.add("idle")
  results.innerHTML = `
    <section class="search-idle-panel">
      <p class="search-idle-kicker">SEARCH MODE</p>
      <h3>输入主题、概念、人物或引用</h3>
      <ul class="search-idle-list">
        <li><span>正文检索</span><code>记忆</code></li>
        <li><span>标签检索</span><code>#AI</code></li>
        <li><span>快捷唤起</span><code>⌘K</code></li>
      </ul>
    </section>
  `

  if (!preview) return

  preview.classList.add("idle")
  preview.innerHTML = `
    <section class="preview-inner preview-shell">
      <header class="preview-shell-header">
        <p class="preview-shell-kicker">PREVIEW</p>
        <h2 class="preview-shell-title">搜索会返回结构化摘要，而不是整页投影。</h2>
        <p class="preview-shell-meta">结果列表优先给出入口，右侧只保留标题、标签与关键片段。</p>
      </header>
      <div class="preview-shell-body">
        <p class="preview-shell-block" data-kind="body">输入关键词后，左侧定位条目，右侧只展示可读摘要，避免出现“网页套网页”的干扰。</p>
        <p class="preview-shell-block" data-kind="quote">支持标签模式、键盘导航与高亮匹配。</p>
      </div>
    </section>
  `
}

async function setupSearch(searchElement: Element, currentSlug: FullSlug, data: ContentIndex) {
  const container = searchElement.querySelector(".search-container") as HTMLElement
  if (!container) return

  const sidebar = container.closest(".sidebar") as HTMLElement | null

  const searchButton = searchElement.querySelector(".search-button") as HTMLButtonElement
  if (!searchButton) return

  const searchBar = searchElement.querySelector(".search-bar") as HTMLInputElement
  if (!searchBar) return

  const searchLayout = searchElement.querySelector(".search-layout") as HTMLElement
  if (!searchLayout) return

  const idDataMap = Object.keys(data) as FullSlug[]
  const appendLayout = (el: HTMLElement) => {
    searchLayout.appendChild(el)
  }

  const enablePreview = searchLayout.dataset.preview === "true"
  let preview: HTMLDivElement | undefined = undefined
  const results = document.createElement("div")
  results.className = "results-container"
  appendLayout(results)

  if (enablePreview) {
    preview = document.createElement("div")
    preview.className = "preview-container"
    appendLayout(preview)
  }

  function hideSearch() {
    container.classList.remove("active")
    searchBar.value = "" // clear the input when we dismiss the search
    if (sidebar) sidebar.style.zIndex = ""
    currentSearchTerm = ""
    removeAllChildren(results)
    if (preview) {
      removeAllChildren(preview)
      preview.classList.remove("idle")
    }
    results.classList.remove("idle")
    searchLayout.classList.remove("display-results", "idle-state")
    searchType = "basic" // reset search type after closing
    searchButton.focus()
  }

  function showSearch(searchTypeNew: SearchType) {
    searchType = searchTypeNew
    if (sidebar) sidebar.style.zIndex = "1000"
    container.classList.add("active")
    renderSearchIdleState(searchLayout, results, preview)
    searchBar.focus()
  }

  let currentHover: HTMLInputElement | null = null
  async function shortcutHandler(e: HTMLElementEventMap["keydown"]) {
    if (e.key === "k" && (e.ctrlKey || e.metaKey) && !e.shiftKey) {
      e.preventDefault()
      const searchBarOpen = container.classList.contains("active")
      searchBarOpen ? hideSearch() : showSearch("basic")
      return
    } else if (e.shiftKey && (e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
      // Hotkey to open tag search
      e.preventDefault()
      const searchBarOpen = container.classList.contains("active")
      searchBarOpen ? hideSearch() : showSearch("tags")

      // add "#" prefix for tag search
      searchBar.value = "#"
      return
    }

    if (currentHover) {
      currentHover.classList.remove("focus")
    }

    // If search is active, then we will render the first result and display accordingly
    if (!container.classList.contains("active")) return
    if (e.key === "Enter" && !e.isComposing) {
      // If result has focus, navigate to that one, otherwise pick first result
      if (results.contains(document.activeElement)) {
        const active = document.activeElement as HTMLInputElement
        if (active.classList.contains("no-match")) return
        await displayPreview(active)
        active.click()
      } else {
        const anchor = document.getElementsByClassName("result-card")[0] as HTMLInputElement | null
        if (!anchor || anchor.classList.contains("no-match")) return
        await displayPreview(anchor)
        anchor.click()
      }
    } else if (e.key === "ArrowUp" || (e.shiftKey && e.key === "Tab")) {
      e.preventDefault()
      if (results.contains(document.activeElement)) {
        // If an element in results-container already has focus, focus previous one
        const currentResult = currentHover
          ? currentHover
          : (document.activeElement as HTMLInputElement | null)
        const prevResult = currentResult?.previousElementSibling as HTMLInputElement | null
        currentResult?.classList.remove("focus")
        prevResult?.focus()
        if (prevResult) currentHover = prevResult
        await displayPreview(prevResult)
      }
    } else if (e.key === "ArrowDown" || e.key === "Tab") {
      e.preventDefault()
      // The results should already been focused, so we need to find the next one.
      // The activeElement is the search bar, so we need to find the first result and focus it.
      if (document.activeElement === searchBar || currentHover !== null) {
        const firstResult = currentHover
          ? currentHover
          : (document.getElementsByClassName("result-card")[0] as HTMLInputElement | null)
        const secondResult = firstResult?.nextElementSibling as HTMLInputElement | null
        firstResult?.classList.remove("focus")
        secondResult?.focus()
        if (secondResult) currentHover = secondResult
        await displayPreview(secondResult)
      }
    }
  }

  const formatForDisplay = (term: string, id: number) => {
    const slug = idDataMap[id]
    return {
      id,
      slug,
      title: searchType === "tags" ? data[slug].title : highlight(term, data[slug].title ?? ""),
      content: highlight(term, data[slug].content ?? "", true),
      tags: highlightTags(term.substring(1), data[slug].tags),
    }
  }

  function highlightTags(term: string, tags: string[]) {
    if (!tags || searchType !== "tags") {
      return []
    }

    return tags
      .map((tag) => {
        if (tag.toLowerCase().includes(term.toLowerCase())) {
          return `<li><p class="match-tag">#${tag}</p></li>`
        } else {
          return `<li><p>#${tag}</p></li>`
        }
      })
      .slice(0, numTagResults)
  }

  function resolveUrl(slug: FullSlug): URL {
    return new URL(resolveRelative(currentSlug, slug), location.toString())
  }

  const resultToHTML = ({ slug, title, content, tags }: Item) => {
    const htmlTags = tags.length > 0 ? `<ul class="tags">${tags.join("")}</ul>` : ``
    const itemTile = document.createElement("a")
    itemTile.classList.add("result-card")
    itemTile.id = slug
    itemTile.href = resolveUrl(slug).toString()
    itemTile.innerHTML = `
      <h3 class="card-title">${title}</h3>
      ${htmlTags}
      <p class="card-description">${content}</p>
    `
    itemTile.addEventListener("click", (event) => {
      if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return
      hideSearch()
    })

    const handler = (event: MouseEvent) => {
      if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return
      hideSearch()
    }

    async function onMouseEnter(ev: MouseEvent) {
      if (!ev.target) return
      const target = ev.target as HTMLInputElement
      await displayPreview(target)
    }

    itemTile.addEventListener("mouseenter", onMouseEnter)
    window.addCleanup(() => itemTile.removeEventListener("mouseenter", onMouseEnter))
    itemTile.addEventListener("click", handler)
    window.addCleanup(() => itemTile.removeEventListener("click", handler))

    return itemTile
  }

  async function displayResults(finalResults: Item[]) {
    results.classList.remove("idle")
    preview?.classList.remove("idle")
    removeAllChildren(results)
    if (finalResults.length === 0) {
      results.innerHTML = `<a class="result-card no-match">
          <h3>未找到结果</h3>
          <p>换个词，或者试试标签模式。</p>
      </a>`
    } else {
      results.append(...finalResults.map(resultToHTML))
    }

    if (finalResults.length === 0 && preview) {
      preview.innerHTML = `
        <section class="preview-inner preview-shell">
          <header class="preview-shell-header">
            <p class="preview-shell-kicker">PREVIEW</p>
            <h2 class="preview-shell-title">没有找到可展示的条目。</h2>
            <p class="preview-shell-meta">建议缩短关键词，或尝试使用 <span class="highlight">#标签</span> 检索。</p>
          </header>
        </section>
      `
    } else {
      // focus on first result, then also dispatch preview immediately
      const firstChild = results.firstElementChild as HTMLElement
      firstChild.classList.add("focus")
      currentHover = firstChild as HTMLInputElement
      await displayPreview(firstChild)
    }
  }

  async function fetchContent(slug: FullSlug): Promise<Document> {
    if (fetchContentCache.has(slug)) {
      return fetchContentCache.get(slug) as Document
    }

    const targetUrl = resolveUrl(slug).toString()
    const contents = await fetch(targetUrl)
      .then((res) => res.text())
      .then((contents) => {
        if (contents === undefined) {
          throw new Error(`Could not fetch ${targetUrl}`)
        }
        const html = p.parseFromString(contents ?? "", "text/html")
        normalizeRelativeURLs(html, targetUrl)
        return html
      })

    fetchContentCache.set(slug, contents)
    return contents
  }

  async function displayPreview(el: HTMLElement | null) {
    if (!searchLayout || !enablePreview || !el || !preview) return
    const slug = el.id as FullSlug
    const previewShell = await fetchContent(slug).then((doc) => buildPreviewShell(doc, currentSearchTerm))
    preview.classList.remove("idle")
    preview.replaceChildren(previewShell)

    // scroll to longest
    const highlights = [...preview.getElementsByClassName("highlight")].sort(
      (a, b) => b.innerHTML.length - a.innerHTML.length,
    )
    highlights[0]?.scrollIntoView({ block: "start" })
  }

  async function onType(e: HTMLElementEventMap["input"]) {
    if (!searchLayout || !index) return
    currentSearchTerm = (e.target as HTMLInputElement).value
    searchType = currentSearchTerm.startsWith("#") ? "tags" : "basic"

    if (currentSearchTerm.trim() === "" || currentSearchTerm.trim() === "#") {
      renderSearchIdleState(searchLayout, results, preview)
      return
    }

    searchLayout.classList.add("display-results")
    searchLayout.classList.remove("idle-state")

    let searchResults: DefaultDocumentSearchResults<Item>
    if (searchType === "tags") {
      currentSearchTerm = currentSearchTerm.substring(1).trim()
      const separatorIndex = currentSearchTerm.indexOf(" ")
      if (separatorIndex != -1) {
        // search by title and content index and then filter by tag (implemented in flexsearch)
        const tag = currentSearchTerm.substring(0, separatorIndex)
        const query = currentSearchTerm.substring(separatorIndex + 1).trim()
        searchResults = await index.searchAsync({
          query: query,
          // return at least 10000 documents, so it is enough to filter them by tag (implemented in flexsearch)
          limit: Math.max(numSearchResults, 10000),
          index: ["title", "content"],
          tag: { tags: tag },
        })
        for (let searchResult of searchResults) {
          searchResult.result = searchResult.result.slice(0, numSearchResults)
        }
        // set search type to basic and remove tag from term for proper highlightning and scroll
        searchType = "basic"
        currentSearchTerm = query
      } else {
        // default search by tags index
        searchResults = await index.searchAsync({
          query: currentSearchTerm,
          limit: numSearchResults,
          index: ["tags"],
        })
      }
    } else if (searchType === "basic") {
      searchResults = await index.searchAsync({
        query: currentSearchTerm,
        limit: numSearchResults,
        index: ["title", "content"],
      })
    }

    const getByField = (field: string): number[] => {
      const results = searchResults.filter((x) => x.field === field)
      return results.length === 0 ? [] : ([...results[0].result] as number[])
    }

    // order titles ahead of content
    const allIds: Set<number> = new Set([
      ...getByField("title"),
      ...getByField("content"),
      ...getByField("tags"),
    ])
    const finalResults = [...allIds].map((id) => formatForDisplay(currentSearchTerm, id))
    await displayResults(finalResults)
  }

  document.addEventListener("keydown", shortcutHandler)
  window.addCleanup(() => document.removeEventListener("keydown", shortcutHandler))
  searchButton.addEventListener("click", () => showSearch("basic"))
  window.addCleanup(() => searchButton.removeEventListener("click", () => showSearch("basic")))
  searchBar.addEventListener("input", onType)
  window.addCleanup(() => searchBar.removeEventListener("input", onType))

  registerEscapeHandler(container, hideSearch)
  await fillDocument(data)
}

/**
 * Fills flexsearch document with data
 * @param index index to fill
 * @param data data to fill index with
 */
let indexPopulated = false
async function fillDocument(data: ContentIndex) {
  if (indexPopulated) return
  let id = 0
  const promises: Array<Promise<unknown>> = []
  for (const [slug, fileData] of Object.entries<ContentDetails>(data)) {
    promises.push(
      index.addAsync(id++, {
        id,
        slug: slug as FullSlug,
        title: fileData.title,
        content: fileData.content,
        tags: fileData.tags,
      }),
    )
  }

  await Promise.all(promises)
  indexPopulated = true
}

document.addEventListener("nav", async (e: CustomEventMap["nav"]) => {
  const currentSlug = e.detail.url
  const data = await fetchData
  const searchElement = document.getElementsByClassName("search")
  for (const element of searchElement) {
    await setupSearch(element, currentSlug, data)
  }
})
