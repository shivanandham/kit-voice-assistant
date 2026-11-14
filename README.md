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

## Requirements and evidence

Criterion ids refer to ACCEPTANCE.md. Test files are in `backend/tests/`.

| Requirement | Implemented in | Tested in |
|---|---|---|
| Voice conversation (must) | `backend/app/voice/session.py`, `deepgram.py`, `gate.py`; `ui/src/voice/engine.ts`, `ui/src/components/kit/voice-assistant.tsx` (classic: `frontend/static/voice.js`, `pcm-worklet.js`) | `test_acc_voice_*` (VOX1-5), `test_security_websocket.py`, `scripts/fake_mic.py` for manual runs |
| Correct answers (must) | `backend/app/agent/loop.py`, `prompts.py`; `backend/app/knowledge/` (documents, RAG, citations) | `test_acc_knowledge_*` (DOC, CIT), `test_acc_agent_citations.py`, `test_acc_tools_dates.py` (COR1, COR2) |
| Leave and on-duty requests (must) | `backend/app/agent/tools.py`, `confirm.py` | `test_acc_tools_leave.py` (COR3), `test_acc_tools_confirm.py` (AC8, COR7), `test_acc_tools_teacher.py` (AC9) |
| Saved history (must) | `backend/app/history.py`, `schema.sql`; `ui/src/pages/history.tsx` (classic: `frontend/history.html`) | `test_spine_history*.py` (HIS1-3), `test_e2e_isolation_history.py` (AC3), `test_security_history_isolation.py` |
| Placement registration (nice) | `agent/tools.py` (`list_placement_drives`, `register_for_placement_drive`, eligibility with reasons) | `test_acc_tools_placements.py` (COR4) |
| Lesson doubts (nice) | lesson notes served by `knowledge/store.py` (students: own department); `search_documents` in RAG mode | `test_acc_knowledge_boundaries.py` (DOC4), `test_acc_knowledge_rag.py` |
| Admission enquiries and applications (nice) | `agent/tools.py` (`submit_admission_application`); `ui/src/pages/home.tsx` (classic: `frontend/index.html`) | `test_acc_tools_admissions.py` (COR5), `test_e2e_isolation_visitor.py` (AC1) |
| Still working if one AI service fails (nice) | Full-text search when embeddings fail (`knowledge/rag.py`); college client retries (`kit/client.py`); safe error messages with a usable socket (`voice/session.py`); typed input beside voice (`ui/`); state in Postgres; `Restart=always` in `deploy/kit-assistant.service`. Gemini model fallback (`agent/fallback.py`) and browser speech synthesis (`ui/src/voice/engine.ts`) | `test_acc_voice_failure.py`, `test_acc_kit_client.py`, `test_kit_client.py`, `test_acc_knowledge_rag.py` |
| Public mode cannot see student data (must) | `agent/tools.py` role tool sets, `knowledge/store.py` `select_documents` (visitors get brochures only), `auth.py` | `test_acc_tools_gating.py`, `test_e2e_isolation_visitor.py` (AC1), `test_security_*` |
| Speech-to-text through a speech service, not the browser's recognition | `backend/app/voice/deepgram.py` (server-side Deepgram Nova-3); the browser only captures PCM audio | `test_acc_voice_deepgram.py`, `test_acc_voice_*` |
| Security and reliability | `auth.py`, `limits.py`, `moderation.py`, `main.py` | `test_security_*`, `test_e2e_isolation_*`, `test_limits_*`, `test_moderation_*`, `test_spine_network_guard.py` (REL2) |
| Architecture and design | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | `test_spine_*` for the shared foundations |
| Code quality and tests | `make lint` (ruff), type hints and docstrings on public functions | `make test` |

## Resilience

The assistant keeps working when one service fails:

- Embeddings unavailable: document search falls back to Postgres full-text ranking.
- College API: GETs are retried; POSTs are retried only when they carry an idempotency key.
- Every failure path (speech, model, college, moderation) returns a short safe message and leaves the
  WebSocket usable for the next turn.
- Microphone or speech recognition fails: typed input keeps working in the same conversation.
- Sessions, history and pending actions are in Postgres and survive restarts; systemd restarts the
  service automatically (`Restart=always`).
- Second model provider: when the Claude call fails (`anthropic.APIError`) before anything was spoken,
  `agent/fallback.py` makes one Gemini call (google-genai, `LLM_FALLBACK_MODEL`, default
  `gemini-2.5-flash`, 12 s timeout, `LLM_FALLBACK_ENABLED`). It has no tools, so it answers only from the
  documents and conversation and says that live lookups and actions are unavailable. The reply is
  logged as a `fallback` tool call.
- Browser speech synthesis: when the speech service returns no audio, `ui/src/voice/engine.ts` speaks
  the reply with the browser's `speechSynthesis`, and the interface notes the browser voice is in use.

## Architecture

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

## Security and privacy

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

## Not done yet

- No second speech-recognition provider: if Deepgram is down, voice input fails (typed input still
  works). The fallback model cannot call tools, so during a Claude outage actions and live lookups are unavailable.
- Rate-limit counters are in memory and reset when the service restarts.
- Reply latency is roughly two to four seconds, because the model thinks before answering.
- The voice gate thresholds were tuned on synthetic audio, not on a range of real microphones and
  rooms; in a noisy room it may pass noise or clip quiet speech.
