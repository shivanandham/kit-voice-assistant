import path from "node:path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// Built into ../frontend/app, served by the FastAPI app when UI_VARIANT=app.
// Assets live under /ui-assets/ so they never collide with the classic /static/ files.
export default defineConfig({
  base: "/ui-assets/",
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(import.meta.dirname, "./src") } },
  build: {
    outDir: "../frontend/app",
    emptyOutDir: true,
    assetsDir: "",
    // Never inline assets as data: URIs for scripts/workers; keep fonts and the worklet as files.
    assetsInlineLimit: 0,
    modulePreload: { polyfill: false },
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8770",
      "/ws": { target: "ws://127.0.0.1:8770", ws: true },
    },
  },
})
