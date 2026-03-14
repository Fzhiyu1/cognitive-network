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
