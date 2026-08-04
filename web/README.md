# web/ — React + TypeScript + Vite frontend

Built in **Phase 4**. Placeholder so the repo layout matches the Architecture
section of `PLAN.md`.

The frontend talks to the local FastAPI server (`stable serve`) over HTTP and a
WebSocket for job progress. Frames arrive as JPEG proxies over HTTP and are
drawn to a canvas overlay — no WebRTC.
