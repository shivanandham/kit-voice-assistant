# KIT UI (React)

Vite + React + TypeScript + Tailwind + shadcn/ui, with ElevenLabs UI (Orb, Conversation, Message; MIT)
and Vercel AI Elements (Sources, Suggestion; Apache-2.0) installed through the shadcn CLI.

- `npm ci && npm run build` writes the production build to `../frontend/app/` (commit it; the server has no Node).
- The server serves it when `UI_VARIANT=app` (pages `/`, `/login`, `/assistant`, `/history`; files under `/ui-assets/`).
- `src/voice/engine.ts` is the voice logic from `frontend/static/voice.js`, without DOM; `public/pcm-worklet.js` is a verbatim copy of `frontend/static/pcm-worklet.js`.
- `npm run dev` proxies `/api` and `/ws` to a backend on port 8770.
- Never render server or user text as HTML; `backend/tests/test_moderation_loop.py` scans `ui/` for HTML sinks.
