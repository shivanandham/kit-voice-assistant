# KIT Voice Assistant — Plan

A real-time voice assistant for Knotopian Institute of Technology (KIT), built for a hackathon.
One web app, one assistant, two modes: public (admissions) and logged-in (students and teachers),
plus a history page. It runs 24/7 on a droplet at `https://kit.bixops.in`.

Interfaces between modules are in [CONTRACT.md](CONTRACT.md).

## Scoring priorities

1. Must-haves (count double): voice conversation, correct answers, leave and on-duty requests, saved history.
2. Nice-to-haves: placement registration, lesson doubts, admission enquiries and applications, surviving an AI service failure.
3. Also scored: security and reliability, architecture and design, code quality and tests.

## Architecture

```
Browser ── HTTPS / WebSocket ──> Caddy ──> FastAPI app (one process, systemd)
 mic up, audio down                         ├ Deepgram: speech-to-text and text-to-speech
                                            ├ Claude Haiku 5.5 with tools
                                            ├ College API client
                                            ├ Gemini embeddings (RAG)
                                            └ Postgres + pgvector
```

One deployable: FastAPI serves the API, the voice WebSocket and the static frontend (same origin,
no build step). `backend/` and `frontend/` stay separate in the repo and talk only through the API.

| Part | Choice |
|---|---|
| Speech | Deepgram: Nova-3 streaming STT, Aura-2 TTS, relayed through our server |
| Brain | Claude Haiku 5.5 on the Anthropic SDK, streamed, plain tool loop (no LangGraph) |
| Documents | Whole documents in a cached prompt by default, RAG fallback, automatic switch |
| Storage | Postgres: sessions, conversations, messages, tool-call log, documents, chunks with vectors |
| Frontend | Plain HTML and JavaScript: home, login, assistant, history |

## Voice loop

- Mic audio (16 kHz PCM) streams to our server and on to Deepgram, which returns live text and
  detects when the speaker stops.
- Haiku's reply is streamed, cut into sentences, and each sentence is spoken as soon as it is ready.
- Barge-in: speaking over a reply stops playback and cancels the reply.
- College terms and course codes are passed to Deepgram as recognition hints.
- A text box sits beside the mic.

## Tools by role

| Role | Tools |
|---|---|
| Visitor | College info, courses, submit admission application |
| Student | Profile, academics, attendance, timetable, calendar, leave and on-duty (list, submit, cancel), placement drives and registration |
| Teacher | Classes, class attendance, pending requests, approve, reject |

The server picks the tool set from the login; no tool accepts a user id. Every submission becomes a
pending action that is read back, confirmed by voice in a later turn, then sent with an idempotency key.

## Documents

| Mode | Behaviour |
|---|---|
| `context` | Applicable documents go into the cached system prompt in full |
| `rag` | A `search_documents` tool over pgvector with Gemini embeddings |
| `auto` (default) | `context` while documents fit a 60k-token budget, `rag` above it |

- Visitors get the brochure only. Logged-in users get the current handbook, circulars and lesson notes.
- The handbook edition in force on the college's date is used; superseded editions are excluded.
- Circulars are always included in full in both modes, because they override handbook rules.
- The document list is re-checked every couple of minutes; new or changed documents are embedded in
  the background so the switch needs no re-ingest.

## Security

- All keys and the college login token stay on the server; the browser holds only an HttpOnly cookie.
- History is visible only to its owner.
- Document text is treated as data, not instructions.
- Keys live in a gitignored `.env`; the repo is scanned for secrets before the public push.

## Correctness traps

- The college's date (`/v1/today`) is a week ahead of the real clock. All date maths uses it.
- Some working days follow another weekday's timetable (`follows_weekday` in the calendar).
- Circular 12/2026 changes the on-duty proof rule for B.Tech only.
- Delegated teacher access has an expiry date.
- Zoho's placement drive has one seat left: never register in tests.

## Hosting

- Droplet `bixops-prod-1` (1 vCPU, 961 MB). Caddy stays the front door.
- Stays up: Caddy, the bixops static site, Springboard backend and both frontends.
- Stopped for now: the Solo Lite API container; its database only if memory is still short.
- Added: a pgvector Postgres container on port 5433, a `kit-assistant` systemd service on port 5210,
  and one Caddy block for `kit.bixops.in`.
- A restore script on the droplet brings Solo Lite back.

## Build approach

Build locally first, in parallel, one agent per module with fixed file ownership:

| Module | Files |
|---|---|
| Spine | `backend/app/{main,config,db,auth,history}.py`, `schema.sql` |
| College API and tools | `backend/app/kit/`, `backend/app/agent/{tools,confirm}.py` |
| Assistant loop | `backend/app/agent/{loop,prompts}.py` |
| Knowledge | `backend/app/knowledge/` |
| Voice | `backend/app/voice/` |
| Frontend | `frontend/` |
| Droplet | `deploy/` and the droplet itself |

Then: integrate and evaluate, test locally (with the user on a real microphone), deploy, submit.

## Testing

- Unit tests against a mocked college API: tool gating, dates, confirm-before-submit, document mode switch.
- Voice path: recorded audio streamed through the WebSocket as a fake microphone.
- Real microphone check by a person in a browser.
- No live writes to the college sandbox during the build.

## After submission

Backups for speech and the LLM, planned once Deepgram alone works.
