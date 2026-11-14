# Module contract

Every module is built in parallel by a different person. Code against these signatures exactly, even
if the module you depend on does not exist yet. If you must deviate, say so in your final report.

## Ground rules

- Python 3.12, async everywhere. Package root is `backend/`; import as `app.*`
  (`uvicorn app.main:app --app-dir backend`). Tests: `uv run pytest backend/tests/test_<area>*.py`.
- Only touch the files you own. Dependencies are already installed (`pyproject.toml`); if you need
  another, `uv pip install` it and name it in your final report. Do not edit `pyproject.toml`.
- Secrets live in `.env` at the repo root (already created, gitignored). Never print, log or commit
  key values. Never put a key or the college token in anything sent to the browser or the LLM.
- College sandbox: live **reads and logins are fine**; **no live writes** (no POST to admissions,
  requests, cancel, placement register, approve, reject). Mock them in tests.
- Test logins (the shared password is provided by the organisers and read from the environment as `KIT_TEST_PASSWORD`): students `arjun priya kavya rahul karthik meena divya sneha`,
  teachers `lakshmi senthil`.
- Block docstrings on public functions, type hints, no dead code. Tests live in
  `backend/tests/test_<area>_*.py`; `conftest.py` belongs to the spine.

## Settings — `app/config.py` (spine)

`get_settings() -> Settings` (cached, pydantic-settings, reads `.env`). Fields (see `.env.example`):
`kit_base_url, kit_participant_key, anthropic_api_key, llm_model, llm_effort, llm_max_tokens=4096,
deepgram_api_key, dg_stt_model, dg_tts_model, gemini_api_key, embed_model="", embed_dim=768,
database_url, doc_mode, doc_context_token_budget, doc_refresh_seconds, app_host, app_port,
cookie_secure, public_origin, session_ttl_hours=8`.

## Database — `app/db.py`, `app/schema.sql` (spine)

```python
async def init_db() -> None      # create asyncpg pool, register pgvector codec, apply schema.sql (idempotent)
def pool() -> asyncpg.Pool
async def close_db() -> None
```

Tests use database `kit_assistant_test` (exists). `conftest.py` sets `DATABASE_URL` to it before
imports and provides an autouse fixture that calls `init_db()` once and truncates all tables per test.

```sql
sessions(id text primary key,            -- sha256 of the cookie token
         owner_key text not null,        -- 'user:<login_id>' or 'visitor:<random>'
         role text not null,             -- 'visitor' | 'student' | 'teacher'
         user_json jsonb, kit_token text, kit_token_expires_at timestamptz,
         created_at timestamptz default now(), expires_at timestamptz not null)
conversations(id uuid primary key, owner_key text not null, role text not null, title text,
              started_at timestamptz default now(), updated_at timestamptz default now())
messages(id bigserial primary key, conversation_id uuid references conversations on delete cascade,
         role text not null,             -- 'user' | 'assistant'
         content text not null, created_at timestamptz default now())
tool_calls(id bigserial primary key, conversation_id uuid, name text, input jsonb, output jsonb,
           is_error boolean, created_at timestamptz default now())
pending_actions(id uuid primary key, session_id text, conversation_id uuid, tool_name text,
                args jsonb, summary text, status text default 'pending',  -- pending|done|cancelled|failed
                created_at timestamptz default now())
documents(id int primary key,            -- college document id
          title text, type text, version text, effective_date date, applies_to jsonb,
          is_surprise boolean, text text, token_estimate int, content_hash text,
          active boolean default true, fetched_at timestamptz default now())
doc_chunks(id bigserial primary key, document_id int references documents on delete cascade,
           chunk_index int, heading text, content text, embedding vector(768))
```

## Session — `app/auth.py` (spine)

```python
@dataclass(frozen=True)
class SessionCtx:
    session_id: str
    owner_key: str
    role: Literal["visitor", "student", "teacher"]
    user: dict | None          # college UserSummary: id, login_id, full_name, role, department,
                               # class_id, class_code, programme, year_of_study
    kit_token: str | None      # college JWT; server-side only

async def session_from_request(request: Request, response: Response) -> SessionCtx   # creates a visitor session + cookie if none
async def session_from_websocket(ws: WebSocket) -> SessionCtx | None                 # cookie only; None if missing/expired
router: APIRouter   # POST /api/login {login_id,password} -> {user}; POST /api/logout; GET /api/me -> {role,user}
```

Cookie `kit_session`: HttpOnly, SameSite=Lax, Secure when `cookie_secure`. Login calls the college
`/v1/auth/login`, stores the token server-side, and issues a **new** session id.

## History — `app/history.py` (spine)

```python
async def create_conversation(ctx: SessionCtx) -> str                  # uuid
async def conversation_owned(ctx: SessionCtx, conversation_id: str) -> bool
async def add_message(conversation_id: str, role: str, content: str) -> None   # sets title from first user message, bumps updated_at
async def recent_messages(conversation_id: str, limit: int = 40) -> list[dict] # [{role, content}]
async def log_tool_call(conversation_id: str, name: str, input: dict, output: Any, is_error: bool) -> None
router: APIRouter   # GET /api/history -> [{id,title,started_at,updated_at,message_count}]; GET /api/history/{id} -> {id,title,messages:[{role,content,created_at}]}
```

History is always filtered by `ctx.owner_key`.

## App — `app/main.py` (spine)

Lifespan: `init_db()`, `app.knowledge.store.start_sync_loop()`; on shutdown stop both. Routers:
auth, history, `app.voice.session.router`. Also:

- `GET /healthz` -> `{status, db, doc_mode}`
- `GET /api/public/college`, `GET /api/public/courses` (from the college API, cached 5 min)
- Pages: `/` -> `frontend/index.html`, `/login` -> `login.html`, `/assistant` -> `assistant.html`
  (redirect to `/login` if not logged in), `/history` -> `history.html`; `/static/*` -> `frontend/static/`.
- Security headers, WebSocket origin check against `public_origin`, basic per-IP rate limit on `/api/login`.

## College API — `app/kit/client.py` (college and tools)

```python
class KitError(Exception):      # .status, .code, .message, .field   (error JSON: {"error": {code, message, field}})
class KitClient:
    def __init__(self, token: str | None = None)   # shared httpx.AsyncClient; X-Participant-Key on every call
    # one async method per endpoint, returning parsed JSON:
    health, today, college, courses, documents, document_file(id) -> bytes,
    submit_admission(payload, idempotency_key), login(login_id, password), me, academics, subjects,
    timetable, calendar(from_date=None, to_date=None), attendance,
    attendance_sessions(subject=None, from_date=None, to_date=None),
    my_requests, create_request(payload, idempotency_key), get_request(id), cancel_request(id),
    placement_drives, register_drive(id), my_placements,
    teacher_classes, teacher_requests(status=None), class_attendance(class_id),
    approve_request(id, comment=None), reject_request(id, reason)
async def college_today() -> dict     # /v1/today cached 60 s: {date, weekday, weekday_name, timezone}
```

GETs retry twice on timeouts and 5xx. POSTs retry only when they carry an idempotency key.
OpenAPI spec: `https://ht.knowvally.com/openapi.json`.

## Tools — `app/agent/tools.py`, `app/agent/confirm.py` (college and tools)

```python
@dataclass
class ToolResult:
    content: str        # JSON text for the model
    is_error: bool = False

def tool_specs(role: str, include_search: bool) -> list[dict]   # Anthropic tool definitions, stable order, strict schemas
async def run_tool(name: str, args: dict, ctx: SessionCtx, conversation_id: str) -> ToolResult
async def pending_summary(ctx: SessionCtx, conversation_id: str) -> str | None   # text describing actions awaiting confirmation
```

- `run_tool` rejects any tool not in the role's set, and logs every call via `history.log_tool_call`.
- Identity always comes from `ctx`; no tool has a user-id parameter.
- **Write tools** (`submit_admission_application`, `submit_leave_request`, `cancel_request`,
  `register_for_placement_drive`, `approve_request`, `reject_request`) never call the college API
  directly. They validate, run pre-checks (e.g. exam days or holidays in a leave range), store a
  `pending_actions` row and return `{"status": "needs_confirmation", "action_id", "summary", "warnings"}`.
- `confirm_pending_action(action_id)` executes it **only if** it belongs to this session and
  conversation, is still `pending`, and a user message exists in `messages` created after the
  action. Otherwise it returns an error telling the model to ask the user first.
  `cancel_pending_action(action_id)` drops it. College error messages are returned verbatim.
- `search_documents(query)` is included only when `include_search`; it calls
  `app.knowledge.rag.search(query, ctx)`.
- `get_timetable` resolves `follows_weekday` using the calendar and the college date.
- `list_placement_drives` annotates each drive with whether this student is eligible and why.

## Knowledge — `app/knowledge/{store,mode,rag}.py` (knowledge)

```python
# store.py
async def sync_documents() -> bool            # list, download new/changed, extract text (pypdf), upsert; index changed docs; True if anything changed
async def start_sync_loop() -> None; async def stop_sync_loop() -> None   # every doc_refresh_seconds; must not crash the app if the college API is down
async def applicable_documents(ctx: SessionCtx) -> list[dict]             # documents rows this user may use, deterministic order

# mode.py
@dataclass
class KnowledgeBlock:
    mode: Literal["context", "rag"]
    text: str            # goes into the cached system prompt: all applicable documents (context)
                         # or circulars only (rag); each prefixed with a metadata header
async def knowledge_block(ctx: SessionCtx) -> KnowledgeBlock   # byte-stable between calls unless documents change

# rag.py
async def index_document(document_id: int) -> None            # chunk by numbered rule/section, embed with Gemini, replace rows in doc_chunks
async def search(query: str, ctx: SessionCtx, k: int = 6) -> list[dict]   # [{document_title, version, heading, content, score}], limited to applicable_documents
```

Selection rules: visitors get `brochure` only. Logged-in users get the handbook edition with the
latest `effective_date` on or before the college date (others excluded), all circulars, lesson notes
(students: matching their department; teachers: all) and the brochure. `auto` picks `context` while
the applicable documents' total `token_estimate` is within `doc_context_token_budget`.

## Assistant — `app/agent/{loop,prompts}.py` (assistant loop)

```python
async def run_turn(ctx: SessionCtx, conversation_id: str, user_text: str) -> AsyncIterator[dict]
```

Saves the user message first (`history.add_message`), then streams events:
`{"type":"text_delta","text"}`, `{"type":"tool","name","state":"start"|"end","is_error"}`,
`{"type":"done","text"}` (full reply, already saved as an assistant message), `{"type":"error","message"}`.
Cancelling the consuming task must stop the model call and save the partial reply.

System prompt order (for caching): tools, then instructions + `KnowledgeBlock.text` (cache
breakpoint), then per-user facts: name, role, class, the college date, `pending_summary`. Replies are
for speech: short sentences, no markdown, numbers and dates said naturally, never invent facts.

## Voice — `app/voice/{deepgram,session}.py` (voice)

`router` exposes `WebSocket /ws/voice`. Session comes from `auth.session_from_websocket`; close with
code 4401 if there is none.

Client to server:

- binary frame: microphone audio, PCM signed 16-bit little-endian, mono, 16 kHz, 20–100 ms each
- `{"type":"start","conversation_id": "<uuid>" | null}` — first message
- `{"type":"text","text":"..."}` — typed input
- `{"type":"interrupt"}` — stop the current reply

Server to client:

- `{"type":"ready","conversation_id","role","user_name"}`
- `{"type":"transcript","final":bool,"text"}` — what the user said
- `{"type":"assistant_delta","text"}`, `{"type":"assistant_done","text"}`
- `{"type":"tool","name","state"}`
- `{"type":"audio_start","sample_rate":24000,"format":"pcm_s16le"}`, then binary PCM frames, then `{"type":"audio_end"}`
- `{"type":"interrupted"}` — client must drop queued audio immediately
- `{"type":"error","message"}`

A final transcript (or typed text) starts `agent.loop.run_turn`; text deltas are cut into sentences
and each sentence is synthesised and streamed in order. User speech during a reply cancels it.

## Citations (added)

**Chunks** — `doc_chunks` gains `ref text`, `section text`, `page_start int`, `page_end int`
(`ALTER TABLE ... ADD COLUMN IF NOT EXISTS` appended to `schema.sql` by the knowledge owner).
`ref` is `D<document_id>:<section>` (for example `D2:6.4.3`), or `D<document_id>:c<chunk_index>`
when a chunk has no section number; unique within a document and stable across re-indexing of
unchanged text.

**Knowledge**

- `rag.search` results also carry `ref, section, page_start, page_end, document_id, document_type, effective_date`.
- `KnowledgeBlock.text`: every section of every document is preceded by a line `[ref D2:6.4.3]`.
- `async def resolve_refs(refs: list[str], ctx: SessionCtx) -> list[dict]` in `rag.py`: returns
  `[{ref, document_title, document_type, version, effective_date, section, heading, page_start, page_end}]`
  for refs that exist in `applicable_documents(ctx)`; anything else is dropped. Order follows input, no duplicates.

**Assistant loop**

- The model ends a reply that used documents with a final line `SOURCES: D2:6.4.3, D5:c0`.
  The loop never emits that line as text; it resolves the refs and emits
  `{"type":"citations","sources":[...]}` just before `done`, and `done` carries the same list as `citations`.
- Each source is `{"kind":"document", ref, document_title, version, section, heading, page_start, page_end}`
  or `{"kind":"college_record","label":"College system: attendance"}` (one per distinct successful
  data tool used in the turn).
- `messages` gains `citations jsonb` (assistant owner appends the `ALTER TABLE` to `schema.sql`);
  `history.add_message(conversation_id, role, content, citations=None)`; `GET /api/history/{id}`
  returns `citations` on each message.

**Voice WebSocket** — forwards `{"type":"citations","sources":[...]}` to the client unchanged, before `assistant_done`.

**Frontend** — shows a compact "Sources" list under the assistant's reply and in history, for example
"Student Handbook 2026 (2026.1), rule 6.4.3, page 14" or "College system: attendance".

## Send now (added)

Client to server: `{"type":"finalize"}` — "send what you have heard so far". The server ends the
current utterance immediately: it asks the speech service to finalise, uses the final transcript if
one arrives within about half a second and otherwise the latest interim text, sends it as a final
`transcript` and starts the turn. The rest of that utterance, if the speech service finalises it
later, is discarded so the turn is not started twice. If nothing has been heard, the message is ignored.
