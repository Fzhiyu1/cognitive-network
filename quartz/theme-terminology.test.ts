import assert from "node:assert/strict"
import fs from "node:fs"
import path from "node:path"
import test from "node:test"

import zhCN from "./i18n/locales/zh-CN"

const repoRoot = path.resolve(import.meta.dirname, "..")

test("zh-CN locale uses oracle terminal terminology", () => {
  assert.equal(zhCN.components.graph.title, "回响星图")
  assert.equal(zhCN.components.backlinks.title, "回响链路")
  assert.equal(zhCN.components.tableOfContents.title, "仪轨目录")
  assert.equal(zhCN.components.search.searchBarPlaceholder, "输入真名、命题或线索")
})

test("layout exposes themed navigation labels", () => {
  const layout = fs.readFileSync(path.join(repoRoot, "quartz.layout.ts"), "utf8")

  assert.match(layout, /title:\s*"终端索引"/)
  assert.match(layout, /title:\s*"最新回响"/)
  assert.match(layout, /rootName:\s*"原点"/)
})

test("homepage copy adopts mixed-oracle narrative", () => {
  const homepage = fs.readFileSync(path.join(repoRoot, "content/index.md"), "utf8")

  assert.match(homepage, /把真实藏进术式/)
  assert.match(homepage, /三重投影/)
  assert.match(homepage, /高能回响/)
  assert.match(homepage, /接入协议/)
})
