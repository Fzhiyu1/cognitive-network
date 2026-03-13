import { defineConfig } from "astro/config";
import react from "@astrojs/react";

const repoBase = process.env.PUBLIC_BASE_PATH ?? (process.env.GITHUB_ACTIONS ? "/cognitive-network" : "/");

export default defineConfig({
  integrations: [react()],
  output: "static",
  site: "https://fzhiyu1.github.io",
  base: repoBase
});
