import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The frontend always talks to the local Stable server (stable/server) via
// relative /api and /ws paths — see src/api/client.ts and src/api/ws.ts.
// In dev, Vite proxies both to the FastAPI process so there is never a
// cross-origin request; in production the same FastAPI process serves this
// build's dist/ directly (see stable/server/app.py's web_dist_dir()/mount), so
// the relative paths still resolve without any base-URL configuration.
const BACKEND = process.env.STABLE_BACKEND_URL ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: BACKEND,
        changeOrigin: true,
      },
      "/ws": {
        target: BACKEND,
        ws: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
