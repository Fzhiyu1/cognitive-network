import assert from "node:assert/strict"
import fs from "node:fs"
import path from "node:path"
import test from "node:test"
import vm from "node:vm"
import ts from "typescript"

const scriptPath = path.join(import.meta.dirname, "darkmode.inline.ts")
const darkmodeScript = ts.transpileModule(fs.readFileSync(scriptPath, "utf8"), {
  compilerOptions: {
    module: ts.ModuleKind.CommonJS,
    target: ts.ScriptTarget.ES2022,
  },
}).outputText

type HarnessOptions = {
  prefersLight?: boolean
  storedTheme?: string | null
}

function runDarkmodeScript({ prefersLight = false, storedTheme = null }: HarnessOptions = {}) {
  const attributes = new Map<string, string>()
  const storage = new Map<string, string>()

  if (storedTheme !== null) {
    storage.set("theme", storedTheme)
  }

  const context = {
    window: {
      matchMedia: (query: string) => ({
        matches: query.includes("light") ? prefersLight : !prefersLight,
        addEventListener: () => {},
        removeEventListener: () => {},
      }),
      addCleanup: () => {},
    },
    document: {
      documentElement: {
        setAttribute: (key: string, value: string) => attributes.set(key, value),
        getAttribute: (key: string) => attributes.get(key),
      },
      addEventListener: () => {},
      dispatchEvent: () => {},
      getElementsByClassName: () => [],
    },
    localStorage: {
      getItem: (key: string) => storage.get(key) ?? null,
      setItem: (key: string, value: string) => storage.set(key, value),
    },
    CustomEvent: class {
      detail: unknown

      constructor(_type: string, init?: { detail?: unknown }) {
        this.detail = init?.detail
      }
    },
  }

  vm.runInNewContext(darkmodeScript, context)

  return {
    savedTheme: attributes.get("saved-theme"),
  }
}

test("defaults to dark theme when there is no saved preference", () => {
  const result = runDarkmodeScript({ prefersLight: true })

  assert.equal(result.savedTheme, "dark")
})

test("respects a saved theme preference", () => {
  const result = runDarkmodeScript({ prefersLight: true, storedTheme: "light" })

  assert.equal(result.savedTheme, "light")
})
