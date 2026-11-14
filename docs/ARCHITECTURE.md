# Architecture

One FastAPI process (`backend/app`) serves the web interface, the REST API and the voice WebSocket.
Postgres (with pgvector) holds state. Caddy terminates TLS in front of it.

## Components

```
Browser (ui/ React app, or classic frontend/ pages)
   |  HTTPS: /api/*, pages        WebSocket: /ws/voice (cookie, Origin checked)
   v
FastAPI (main.py: headers, CSP, routes)
   |-- auth.py ........ login, sessions        |-- limits.py ..... abuse limits (memory+db)
   |-- history.py ..... conversations API      |-- moderation.py . input/output checks
   |-- voice/ ......... session, gate, Deepgram relay, sentence splitter
   |-- agent/ ......... loop (model), tools (by role), confirm (pending writes), prompts
   |-- knowledge/ ..... store (sync), mode (prompt), rag (search), structure (refs)
   |-- kit/client.py .. college API client
   v
Postgres (sessions, conversations, messages, tool_calls, pending_actions, documents, doc_chunks)
External: college API, Anthropic (Claude), Deepgram (STT, TTS), Gemini (embeddings, and answers via agent/fallback.py if Claude fails)
```

Provider seams: `backend/app/ports.py` names every external service the app depends on as a small
`typing.Protocol` (`SpeechToText`, `TextToSpeech`, `ChatModel` for the fallback model, `Embedder`,
`CollegeSystem`), listing only the methods the app calls and which adapter implements each today
(Deepgram, Gemini, `KitClient`). That file is where a new provider plugs in: write an adapter with the
same methods and pass it at the injection point named in the docstring (for speech, the
`stt_factory`/`tts_factory` parameters of `voice.session.build_router`).
`backend/tests/test_ports_conformance.py` checks the current adapters still match, so the seams are
enforced by the test suite.

## Life of one spoken turn

1. Browser captures the microphone as 16 kHz PCM and streams frames over the WebSocket
   (`ui/src/voice/engine.ts`, `pcm-worklet.js`).
2. The socket is accepted only with a valid session and allowed Origin; the session is re-checked
   while it is open and closed with 4401 if it ended (`voice/session.py`, `auth.py`).
3. The voice gate forwards audio only while someone is speaking (`voice/gate.py`); audio time is
   counted against limits (`limits.py`).
4. Deepgram Nova-3 streams interim and final transcripts (`voice/deepgram.py`). The browser's own
   speech recognition is not used. Speech over a reply sends an interrupt that cancels it.
5. The final transcript becomes a turn (`voice/session.py` `start_turn`). Turn and conversation limits
   are checked (`limits.py`).
6. Input moderation runs before any tool (`moderation.py`, in `agent/loop.py`).
7. The loop builds the request: cached system prompt with documents, the role's tools, recent history
   (`agent/loop.py`, `prompts.py`, `knowledge/mode.py`).
8. Claude calls tools (`agent/tools.py`); each uses the session's own college token via
   `kit/client.py`. Writes only create a pending action (`agent/confirm.py`).
9. The reply streams out; citations are validated (`loop.py`) and output is moderated and sanitised.
10. `_Speech` in `agent/loop.py` and the splitter in `voice/` cut the text into sentences; Deepgram Aura-2
    turns each into audio (`voice/session.py` `_speak`). The browser plays it.
11. Messages and tool calls are saved (`history.py`, `agent/loop.py`) and shown in the history page.

Typed input skips steps 1 to 4 and joins at step 5.

## Data model (`schema.sql`)

| Table | Holds |
|---|---|
| `sessions` | Hashed session id, owner key (`visitor:...` or `user:<login>`), role, user profile, college token and its expiry, session expiry |
| `conversations` | Id, owner key, role, title, timestamps |
| `messages` | Conversation id, role, text |
| `tool_calls` | Conversation id, tool name, input, output, error flag |
| `pending_actions` | Session, conversation, tool, args, summary, status, created time (expires after 10 minutes) |
| `documents` | College documents: type, version, effective date, `applies_to`, text, content hash, active flag |
| `doc_chunks` | Section chunks with heading and a `vector(768)` embedding |

## Trust boundaries

| Boundary | What crosses | Controls |
|---|---|---|
| Browser to server | Cookie, typed text (max 1000 chars), PCM audio, control messages | HttpOnly cookie, Origin check, size limits, rate limits, strict CSP |
| Server to college API | Session's own bearer token, tool arguments, idempotency keys on writes | Token never leaves the server; role tool sets; GET retries, POST retries only with a key |
| Server to model (Anthropic) | System prompt with documents, history, tool results | Documents and tool output labelled as data; profile fields from a fixed list; secrets scrubbed |
| Server to speech service (Deepgram) | Gated audio out, transcripts and synthesis text | Key held server-side; silence never sent |
| Server to embeddings (Gemini) | Document chunks and search queries | Full-text fallback if it fails |

## Document pipeline

- Sync: `knowledge/store.py` lists the college's documents on a loop (`DOC_REFRESH_SECONDS`), downloads
  changed PDFs, cleans the text, hashes it and upserts `documents`; chunks are embedded in the
  background into `doc_chunks`.
- Selection by audience (`select_documents`): visitors get brochures only. Students and teachers get
  the handbook edition in force on the college date, every circular, lesson notes (students only for
  their department), brochures and unknown types.
- Prompt mode (`knowledge/mode.py`, `DOC_MODE`): whole documents in the cached prompt while under the
  token budget; above it, circulars in full plus a catalogue, and the model calls `search_documents`
  (pgvector, full-text fallback). The prompt text is byte-stable so caching works.
- Citations: sections carry `[ref D<id>:<section>]` lines; the model cites them, the server drops refs the
  user may not read or that do not exist, and the source line is never spoken.

## Key design decisions and trade-offs

- Speech on the server: the key stays private and behaviour is the same in every browser, at the cost
  of streaming audio through us and one speech provider as a dependency.
- Tools chosen by role on the server, no tool takes a user id: safer than prompt rules, but adding a
  capability means code, not prompt text.
- Confirm before write with expiry and idempotency keys: prevents accidental or repeated records, adds
  one conversational step.
- Documents in the cached prompt when they fit, RAG when not: best accuracy and lowest latency for the
  current corpus, with a route to scale; two code paths to test.
- Plain tool loop on the Anthropic SDK, bounded model calls per turn: easy to reason about, no
  framework features such as built-in retries or tracing.
- State in Postgres, rate limits checked in memory: daily allowances are persisted to Postgres every
  30 seconds and reloaded at startup; short per-minute windows are in memory and reset on restart.
  Counters are not shared across several processes.
- College date from `/v1/today`, not the system clock: the sandbox clock is ahead; every date rule depends
  on it, so one failing call affects date answers.
- Two interfaces: the React app is the main one and the classic pages stay as a no-build fallback, at the
  cost of maintaining both.
