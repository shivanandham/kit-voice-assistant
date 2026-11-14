# KIT Voice Assistant

A real-time voice assistant for Knotopian Institute of Technology (KIT), a dummy college used in a
hackathon. One web app and one assistant with two modes. The public home page lets visitors ask
admission questions by voice or text and submit an application. After login the same assistant serves
students (attendance, timetable, rules, leave and on-duty requests, placements, lesson doubts) and
teachers (class requests, approvals). A history page lists past conversations.

Live: https://kit.bixops.in. Test logins are provided by the organisers; no credentials or keys are in
this repository.

Interface: a React + TypeScript + Tailwind + shadcn/ui app (source in `ui/`, production build committed
in `frontend/app`, served when `UI_VARIANT=app`). The classic no-build pages in `frontend/` remain the
default fallback. The composer has text and voice modes, a live waveform while listening, Send what I
said (sends the interim transcript now), Stop, a voice replies toggle, the sources behind each reply,
and a history page of past conversations.

Design documents: [PLAN.md](PLAN.md) (scope and choices), [CONTRACT.md](CONTRACT.md) (module
interfaces and the WebSocket protocol), [ACCEPTANCE.md](ACCEPTANCE.md) (the criteria the tests prove), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (components, one spoken turn, data model, trust boundaries), [SECURITY_AUDIT.md](SECURITY_AUDIT.md).

## Requirements

Criterion ids refer to ACCEPTANCE.md; tests are in `backend/tests/`.

| Requirement | Status | Where | How it is tested |
|---|---|---|---|
| Voice conversation (must-have) | Implemented | `backend/app/voice/session.py`, `deepgram.py`, `gate.py`; `ui/src/voice/engine.ts`, `ui/src/components/kit/voice-assistant.tsx` | `test_acc_voice_*.py` (VOX1-5: protocol, handshake, barge-in, turns, limits, failure), `test_voice_session.py`; `scripts/fake_mic.py` for manual runs |
| Correct answers (must-have) | Implemented | `backend/app/agent/loop.py`, `prompts.py`; `backend/app/knowledge/` | `test_acc_knowledge_*.py` (DOC, CIT), `test_acc_agent_citations.py`, `test_acc_tools_dates.py` (COR1, COR2) |
| Leave and on-duty requests (must-have) | Implemented | `backend/app/agent/tools.py`, `confirm.py` | `test_acc_tools_leave.py` (COR3), `test_acc_tools_confirm.py` (AC8, COR7), `test_acc_tools_teacher.py` (AC9) |
| Saved history (must-have) | Implemented | `backend/app/history.py`, `schema.sql`; `ui/src/pages/history.tsx` | `test_spine_history.py`, `test_spine_history_api.py` (HIS1-3), `test_e2e_isolation_history.py` (AC3), `test_security_history_isolation.py` |
| Placement registration | Implemented | `agent/tools.py` (`list_placement_drives`, `register_for_placement_drive`, eligibility with reasons) | `test_acc_tools_placements.py` (COR4) |
| Lesson doubts | Implemented | lesson notes selected in `knowledge/store.py` (students: own department); `search_documents` in RAG mode | `test_acc_knowledge_boundaries.py` (DOC4), `test_acc_knowledge_rag.py` |
| Admission enquiries and applications on the public page | Implemented | `agent/tools.py` (`submit_admission_application`); `ui/src/pages/home.tsx` | `test_acc_tools_admissions.py` (COR5), `test_e2e_isolation_visitor.py` (AC1) |
| Still working if one AI service fails | Implemented, one row per fallback below | | |
| - second model provider | Implemented | `backend/app/agent/fallback.py` (Gemini via google-genai; no tools; answers from documents and conversation); settings `LLM_FALLBACK_*` in `config.py` | `test_fallback_loop.py` (exercised with faked providers) |
| - browser speech synthesis when the speech service returns no audio | Implemented | `ui/src/voice/engine.ts` (`speechSynthesis`) | UI code; no automated test in `backend/tests/` |
| - full-text search when embeddings are unavailable | Implemented | `backend/app/knowledge/rag.py` | `test_acc_knowledge_rag.py`, `test_knowledge_rag.py` |
| - college API retries | Implemented | `backend/app/kit/client.py` (GETs retried; POSTs only with an idempotency key) | `test_acc_kit_client.py`, `test_kit_client.py` |
| - typed input when microphone or speech recognition fails | Implemented | `ui/src/components/kit/voice-assistant.tsx`; text message in `voice/session.py` | `test_acc_voice_failure.py`, `test_acc_voice_turns.py` |
| - safe error events that keep the socket usable | Implemented | `voice/session.py`, `agent/loop.py` | `test_acc_voice_failure.py`, `test_acc_tools_reliability.py` |
| Public mode cannot see student data | Implemented | role tool sets in `agent/tools.py`; brochures-only document selection in `knowledge/store.py`; `auth.py` | `test_acc_tools_gating.py`, `test_e2e_isolation_visitor.py` (AC1), `test_security_routes.py` |
| Speech-to-text through a speech service, not the browser's recognition | Implemented | `backend/app/voice/deepgram.py` (server-side Deepgram Nova-3); the browser only captures PCM audio | `test_acc_voice_deepgram.py` |
| Public home page, login, history page | Implemented | `ui/src/pages/home.tsx`, `login.tsx`, `history.tsx` (build in `frontend/app`); classic `frontend/index.html`, `login.html`, `history.html` | `test_security_routes.py`, `test_spine_login.py` |

Other behaviour: the fallback model has no tools, so during a Claude outage actions and live lookups
are unavailable. Sessions, history and pending actions are in Postgres and survive restarts; systemd
restarts the service (`Restart=always` in `deploy/kit-assistant.service`).

The brief states that fancy design is not scored, so the frontend is deliberately thin: it is assembled from an existing component library (source in `ui/`), and `frontend/app/` is its committed production build, marked as generated in `.gitattributes`.

## Security and reliability

Audit result: [SECURITY_AUDIT.md](SECURITY_AUDIT.md) (an audit using Cloudflare's security-audit skill
found no critical, high or medium issues).

| Control | Where | Tests |
|---|---|---|
| Role-based tool sets chosen server-side; no tool takes a user id | `agent/tools.py`, `auth.py` | `test_acc_tools_gating.py`, `test_tools_gating.py`, `test_acc_tools_identity.py` |
| Confirm-before-write: explicit approval as the latest message, 10-minute expiry, idempotency keys | `agent/confirm.py`, `agent/tools.py` | `test_acc_tools_confirm.py`, `test_tools_confirm.py` |
| Session re-validated while a voice socket is open (close code 4401) | `voice/session.py`, `auth.py` | `test_acc_voice_handshake.py`, `test_security_websocket.py` |
| Cross-user isolation with several users at once over the real app | `history.py`, `agent/`, `voice/` | `test_e2e_isolation_*.py` |
| Rate limits and input limits (visitors per IP, users per login id) | `limits.py` | `test_limits_*.py`, `test_e2e_isolation_limits.py` |
| Moderation of user input and replies | `moderation.py`, `agent/loop.py` | `test_moderation_rules.py`, `test_moderation_loop.py`, `test_acc_agent_moderation.py` |
| Output sanitising (plain text, secrets replaced) | `agent/loop.py`, `moderation.py` | `test_acc_tools_secrets.py`, `test_e2e_isolation_secrets.py` |
| CSP and security headers | `main.py` | `test_security_http.py` |
| Prompt-injection handling of college data and documents | `agent/prompts.py`, `knowledge/mode.py`, `agent/confirm.py` | `test_acc_agent_prompt.py`, `test_e2e_isolation_citations.py` |
| Secrets stay server-side (environment only; never in browser, prompts or tool results) | `config.py`, `kit/client.py` (token redaction) | `test_acc_tools_secrets.py`, `test_e2e_isolation_secrets.py` |

## Architecture and design

- One deployable FastAPI process with module boundaries: `voice/` (WebSocket, speech relay), `agent/`
  (model loop, tools, pending actions, fallback), `knowledge/` (sync, selection, prompt mode, search),
  `kit/` (college client), `auth.py`, `history.py`, `limits.py`, `moderation.py`, `db.py`/`schema.sql`.
- Provider seams are small Protocols in `backend/app/ports.py`, checked by `test_ports_conformance.py`.
- Documents go into a cached prompt while they fit a token budget and switch to RAG automatically above it.
- Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Module interfaces: [CONTRACT.md](CONTRACT.md).
  Specification: [ACCEPTANCE.md](ACCEPTANCE.md).

Design decisions and their trade-offs are recorded in [docs/DECISIONS.md](docs/DECISIONS.md).

## Code quality and tests

- One command: `make test` (whole suite with coverage). 5,390+ tests and 97% line and branch coverage
  at the last full run.
- Tests are derived from ACCEPTANCE.md; criterion ids appear in test names.
- Third parties are stubbed; a network guard fails any test that touches a real host
  (`test_spine_network_guard.py`).
- Lint: ruff rules `E, F, W, I, B` pinned in `pyproject.toml`; `make lint`. CI: `.github/workflows/ci.yml`.
- Public functions have type hints and docstrings.

## How it works

Spoken turn:

```
Browser (mic, 16 kHz PCM)
   |  WebSocket /ws/voice (session cookie, origin checked)
   v
FastAPI app (voice/session.py)
   |-- voice gate drops silence --> Deepgram speech-to-text (Nova-3, streaming)
   |        transcript + end of speech
   |-- moderation check on the user text
   |-- agent loop (agent/loop.py): Claude Haiku with the tools for the user's role
   |        |-- college API client (kit/client.py), always with the session's own token
   |        |-- pending actions in Postgres (confirm before write)
   |        '-- knowledge: documents in the cached prompt, or search_documents (pgvector)
   |-- output moderation, sentence splitter
   |-- Deepgram text-to-speech (Aura-2), one sentence at a time
   v
Browser plays audio; speaking over it sends an interrupt that cancels the reply
```

Everything runs in one FastAPI process that also serves the frontend from the same origin: the classic
no-build pages by default, or the built React app when `UI_VARIANT=app`. More detail in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Caddy terminates TLS in front of it.

Module layout (`backend/app/`):

| Path | Responsibility |
|---|---|
| `main.py`, `config.py`, `db.py`, `schema.sql` | App, security headers, settings, Postgres pool and schema |
| `auth.py`, `history.py` | Login, sessions, conversation storage and API |
| `limits.py`, `moderation.py` | Abuse limits; input and output moderation |
| `kit/client.py` | College API client (retries, idempotency keys) |
| `agent/` | `loop.py` model loop, `tools.py` role tool sets, `confirm.py` pending actions, `prompts.py` |
| `knowledge/` | Document sync, edition selection, mode switch, chunking, embeddings, search, citations |
| `voice/` | WebSocket protocol, Deepgram relay, voice gate, sentence splitter |

Design decisions:

- Speech runs on the server. The Deepgram key stays server-side and never reaches the browser, and
  the browser's built-in speech recognition is not used, so behaviour does not depend on the browser.
- The server chooses the tool set from the role in the college's login answer. No tool takes a user id,
  and a tool outside the role's set is refused even if the model asks for it.
- Every write is a pending action: the tool validates and stores it, the assistant reads it back, and
  it executes only after a later user message in the same session and conversation. Each write carries
  an idempotency key, so a retry or double confirmation creates one record.
- Documents go into a cached system prompt while they fit a token budget (default 60k) and switch to
  retrieval over pgvector above it (`DOC_MODE=auto`). The prompt text is byte-stable while documents are
  unchanged, so prompt caching works. Circulars are always included in full.
- Citations are validated after the model answers: references to documents the user may not read, or
  to sections that do not exist, are dropped. The machine-readable source line is never spoken.
- All date logic uses the college's own date (`/v1/today`), which is ahead of the real clock, and
  follows `follows_weekday` in the college calendar.
- A plain tool loop on the Anthropic SDK with a bounded number of model calls per turn, no agent
  framework.

## Security details

An audit using Cloudflare's security-audit skill found no critical, high or medium issues; see
[SECURITY_AUDIT.md](SECURITY_AUDIT.md). Main controls: role-based tool sets chosen server-side;
confirm-before-write (an explicit approval as the latest message, expiry, idempotency keys); the
session is re-validated on every voice turn; rate limits (visitors per IP, logged-in users per login
id); moderation on input and output; output sanitising; a strict CSP; and prompt-injection handling of
college data and documents. Details:

- Access: visitors reach only the admissions brochure and public college data. Students and teachers
  reach only their own data; the college token is held server-side and every college call uses the
  token of the calling session. A conversation id belonging to someone else returns 404 (REST,
  WebSocket resume and tools).
- Sessions: HttpOnly cookie, only a SHA-256 hash of the token is stored, the session id changes at
  login, logout ends the session on the server and closes open voice sockets (close code 4401), and an
  expired session or college token counts as no session. The WebSocket needs a valid session and an
  allowed Origin.
- Secrets: read from the environment only, never in the repo, responses, prompts, tool results, logs or
  stored messages. Assistant text containing a secret is replaced before it is sent.
- Prompt injection: document text and tool output are labelled as data in the system prompt. User
  profile fields are put into the prompt from a fixed list, so a planted instruction in a free-text
  field of the college profile never reaches the model; the tests plant one and check it does not appear
  in the prompt, tool results, replies or storage. Request text written by students is flagged as data
  in teacher tool results.
- Moderation: each user message is checked (built-in rules plus a Haiku classifier, which falls back to
  the rules if it fails) before any tool runs. Harmful requests, attempts to obtain the system prompt
  or secrets, and instruction overrides get a short refusal. Distress or self-harm gets a supportive
  reply that points to a person. Each decision is logged with its category.
- Output: replies are plain text with no HTML, markdown or control characters, and are moderated
  before being sent or spoken. The frontend inserts dynamic text as text. The Content-Security-Policy
  is `default-src 'self'` with no inline script, plus `frame-ancestors 'none'`, `nosniff` and a
  restrictive Permissions-Policy.
- Rate limits and abuse controls (`limits.py`, all configurable, defaults shown). IP limits apply to
  visitors only; logged-in users are limited by login id so people on a shared network do not use up
  each other's allowance:
  - Voice sockets: 3 per visitor IP, 2 per user, 40 overall (refused with close code 4429).
  - Turns: visitors 20 per day per IP, users 100 per day per login id, 200 per conversation, global
    daily budget 5000 (visitors stop first; users continue to 8000).
  - Audio: 30 s per utterance, 3 min idle timeout, socket lifetime 15 min (visitor) or 60 min (user),
    speech-service time 15 min per visitor IP and 60 min per user per day. The server-side voice gate
    forwards audio only while someone is speaking, so silence is never sent or billed.
  - HTTP: 120 requests per minute (per IP for visitors, per login id for users); 30 logins per minute
    per IP; 5 failed logins per login id per 15 minutes; 30 new anonymous sessions per IP per hour.
  - Writes: 3 pending actions per conversation; 2 admission applications per session and 5 per IP per
    day.
  - Input size: 1000 typed characters, 64 KB HTTP body, 8 KB WebSocket control message, 32 KB audio
    frame. There are no file uploads.
  - The client IP comes from the proxy's `X-Forwarded-For` entry, not from a header the client adds.

## Tests

```sh
make test        # whole suite with coverage
make test-fast   # same tests, no coverage, stop at first failure
make lint
```

- Tests are derived from ACCEPTANCE.md and CONTRACT.md, not from the code. Each test names the
  criterion it proves (for example `AC3`, `VOX2`). When the two disagree, the code is wrong.
- All third parties (college API, Anthropic, Deepgram, Gemini) are stubbed. A network guard in
  `backend/tests/conftest.py` fails any test that opens a socket or does a DNS lookup to a real
  host (REL2, checked by `test_spine_network_guard.py`). The suite needs only local Postgres with
  pgvector.
- Full run: 5,390 passed, 0 failed, 97% line and branch coverage across 81 test files (plus shared
  helpers and fakes). Many tests are parametrised over roles and endpoints. `make test` runs it all with
  one command. The `test_e2e_isolation_*` suite drives the real assembled app with several users at once.
- Suites by area (test file counts):

| Area | Files | Covers |
|---|---|---|
| `test_acc_agent_*` | 7 | model loop, cancellation, bounds, events, citations, moderation, prompt layout |
| `test_acc_tools_*` | 11 | role gating, identity, confirm-before-write, leave, admissions, placements, dates, secrets |
| `test_acc_knowledge_*` | 6 | editions, document visibility, mode switch, RAG, citations, sync |
| `test_acc_voice_*` | 9 | protocol, handshake, barge-in, turns, limits, failure, Deepgram relay, splitter |
| `test_acc_kit_client.py`, `test_kit_client.py` | 2 | college client, retries, idempotency |
| `test_e2e_isolation_*` | 7 | cross-user isolation over the real app: visitor, history, actions, citations, secrets, limits, concurrency |
| `test_security_*` | 4 | routes, HTTP headers, WebSocket, history isolation |
| `test_limits_*` | 5 | abuse limits ABU1-ABU11 |
| `test_moderation_*` | 2 | input and output rules, moderation inside the loop |
| `test_spine_*` | 12 | config, database, auth, login, history, lifespan, fake college, network guard |
| `test_agent_loop.py`, `test_knowledge_*` | 7 | module-level unit tests |

- Defects the acceptance suites found, fixed in the code rather than in the tests:
  - Logging out did not close a voice socket that was already open, so the old session could keep
    talking. Sockets now re-check their session periodically and close with 4401 (AC6, AC7).
  - The echo guard, which stops the assistant hearing its own voice, swallowed legitimate follow-up
    questions that reused words from the last reply. It now treats a transcript as echo only when a
    long contiguous run of the reply covers most of it.
  - The criterion id in each test name shows what it guards, so further findings can be traced to
    ACCEPTANCE.md.

## Running locally

Prerequisites: Python 3.12, [uv](https://docs.astral.sh/uv/), Postgres with the pgvector extension.

```sh
createdb kit_assistant
psql kit_assistant -c 'CREATE EXTENSION IF NOT EXISTS vector'
cp .env.example .env     # then fill in keys
uv sync
make run                 # http://127.0.0.1:8000
```

The schema is applied at startup. Environment variables (see `.env.example` for defaults):

- College: `KIT_BASE_URL`, `KIT_PARTICIPANT_KEY`
- Model: `ANTHROPIC_API_KEY`, `LLM_MODEL`, `LLM_EFFORT`, `LLM_KEEP_WARM`, `LLM_THINKING`,
  `HISTORY_WINDOW`, `MODERATION_LLM`
- Speech: `DEEPGRAM_API_KEY`, `DG_STT_MODEL`, `DG_TTS_MODEL`
- Model fallback: `LLM_FALLBACK_ENABLED`, `LLM_FALLBACK_MODEL`
- Embeddings: `GEMINI_API_KEY`, `EMBED_MODEL`, `EMBED_DIM`
- Storage and documents: `DATABASE_URL`, `DOC_MODE` (`context`, `rag` or `auto`),
  `DOC_CONTEXT_TOKEN_BUDGET`, `DOC_REFRESH_SECONDS`
- Web: `APP_HOST`, `APP_PORT`, `COOKIE_SECURE`, `PUBLIC_ORIGIN`
- Limits: `ANON_SESSIONS_PER_HOUR`, `HTTP_LIMIT_PER_MINUTE`, `LOGIN_LIMIT_PER_MINUTE`,
  `LOGIN_FAILURES_PER_ID`, `LOGIN_FAILURE_WINDOW_SECONDS`, `LIMIT_*` (sockets, turns, audio, writes,
  sizes, daily budgets), `VOICE_GATE_ENABLED`

Two helper scripts: `scripts/chat.py` (text client) and `scripts/fake_mic.py` (streams recorded audio
through the WebSocket as a microphone). With `--login` they read the shared test password from
`KIT_TEST_PASSWORD`. Web: also `UI_VARIANT` (`app` serves the React build).

## Run with Docker

`docker-compose.yml` starts Postgres with pgvector (`db`) and the app (`app`, built from the
`Dockerfile` with the locked dependencies from `uv.lock`). The schema is applied at startup.

```sh
cp .env.example .env     # then fill in keys; DATABASE_URL is set by compose
docker compose up --build -d
curl http://localhost:8000/healthz
```

## Deployment

Production is one DigitalOcean droplet. There is no pipeline.

- The repository is cloned on the droplet and run by systemd as the `deploy` user
  (`deploy/kit-assistant.service`) on 127.0.0.1:5210.
- Caddy proxies `kit.bixops.in` to it (`deploy/kit.bixops.in.caddy`) and handles TLS.
- Postgres with pgvector runs in a container on 127.0.0.1:5433.
- Settings and keys are in a mode-600 env file on the droplet, outside git.
- Update: `git pull && uv sync --no-dev` as `deploy`, `systemctl restart kit-assistant`, then check
  `/healthz`. Logs: `journalctl -u kit-assistant -f`.

Full steps, including the rollback script that restores the service stopped to free memory, are in
[deploy/DEPLOY.md](deploy/DEPLOY.md).

## Known limits

- Daily allowances (turns, voice seconds, admissions) are persisted to Postgres every 30 seconds
  and reloaded at startup; short per-minute windows are in memory and reset on restart.
- No second speech-recognition provider: if Deepgram is down, voice input fails (typed input works).
- Reply latency depends on the model's thinking time (roughly two to four seconds).
- Voice gate thresholds were tuned on synthetic audio, not a range of real microphones and rooms.
