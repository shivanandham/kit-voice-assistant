# Security audit: kit-voice-assistant

- **Date:** 2026-11-14.
- **Method:** Cloudflare `security-audit` skill (github.com/cloudflare/security-audit-skill, commit c1c8a8c). It was installed only in `.claude/skills/security-audit`, which is gitignored. Run profile `quick`.
- **Source reviewed:** commit `2d2c5f8`. Line numbers refer to that commit. Other engineers are editing `backend/app/main.py`, `backend/app/config.py`, `.env.example` and tests, and have an untracked `frontend/app/` and `ui/` in progress. Those uncommitted changes were **not** reviewed, and lines in `main.py` and `config.py` may have moved since.
- **Run artefacts** (outside the repo): `/private/tmp/claude-501/-Users-itachiuchiha-personal/430689e0-9d4a-429a-9a8a-b01e450d88db/scratchpad/audit-run/run-1/`. This holds the architecture summary, the coverage ledger (validated with the skill's validator) and the hunter brief.

## How to read this report

The skill sorts results into three kinds:

- **Confirmed:** the full source path is established. Only confirmed findings get a severity.
- **Needs validation:** a lead grounded in specific code, where one decisive fact cannot be settled from source (model behaviour, runtime memory, the production env file). These carry **no** severity by the skill's rules. Each one names the exact missing fact.
- **Hardening:** a missing defence with no demonstrated boundary violation.

How each item was established:

- **Code reading.** No application code was run. The skill forbids running target code without an OS sandbox that enforces network, environment, filesystem and memory limits, and memory limits cannot be enforced on this Mac. So the test suite was not run either.
- **Read-only tools.** git, grep, and the Python standard library on `uv.lock` (`[git]`).
- **Passive checks against https://kit.bixops.in** (`[live]`). The owner authorised these: about 15 ordinary requests, no logins, nothing against other hostnames.

## Summary

The core boundaries hold.

- No route, tool or query reaches another user's data. Every history, pending-action and tool path is scoped by `owner_key` or `session_id`.
- Identity comes only from the college login.
- Sessions are random, stored hashed, rotated on login, deleted on logout, and re-checked by open sockets.
- The WebSocket rejects foreign origins and sessionless clients `[live]`.
- No secret reaches the browser, the model, logs or stored messages.
- Git history contains no real key `[git]`.
- The frontend has no HTML sink.

**Counts:**

| Kind | Count |
|---|---|
| Critical | 0 |
| High | 0 |
| Medium | 0 |
| Low | 3 |
| Needs validation | 6 |
| Hardening notes | ~15 |

The most important lead is **NV-1**: the confirm-before-write gate accepts *any* later user message as approval. Its real-world impact depends on model behaviour. If NV-1 proves exploitable, it would be **medium**: a teacher's approval executed without the teacher's intent. Treat it as the first thing to fix.

## Needs validation (ordered by likely impact; no severity by the skill's rules)

### NV-1. The confirm gate treats any later user message as approval, and student-written text sits in the teacher's model context

**Location:**
- `backend/app/agent/confirm.py:133-139` (the gate)
- `confirm.py:95-101` (reuse of an existing pending row)
- `confirm.py:22-23` (`REQUEST_FIELDS` keeps `reason` and `event_name`)
- `backend/app/agent/tools.py:606-609` (`list_class_requests`)
- `tools.py:619-633` (approve summary)
- `backend/app/agent/prompts.py:207-209` (pending summaries in the system block)
- `backend/app/agent/loop.py:154,658` (tool results cached across turns)

**What an attacker could do:**
1. A student puts instructions in the reason or event name of their own leave or on-duty request.
2. A teacher lists class requests, so that text enters the teacher's model context and stays there for later turns.
3. If the model is steered into calling `approve_request` for that request, it creates a pending action.
4. The teacher's next message of any content then satisfies the server gate, for example "next one", a misheard voice fragment, or even "no".
5. The model can then call `confirm_pending_action`. The approval runs under the teacher's college token, though the teacher never agreed to it.

There is a related problem. An identical re-proposal returns the old row with its old `created_at`. So a re-proposed action can be confirmed in the very reply that re-proposed it. Only a prompt instruction says not to (`prompts.py:165`).

**Evidence (code reading):**
- The only deterministic check is `SELECT 1 FROM messages WHERE conversation_id=$1 AND role='user' AND created_at > $2`. It looks at no content and has no binding to the action and no expiry.
- Moderation runs only on the user's own text (`loop.py:521`), never on tool results.
- Two independent hunters found the same root cause.

**Blocker:** Whether Claude Haiku will actually follow the injected text and confirm on an unrelated message is model behaviour. It was not tested.

**Recommended fix:** Bind confirmation to an explicit user act for that action id. The best option is a confirm or cancel button event over the WebSocket that carries the `action_id`. Failing that, accept only if the first user message after the read-back is a short affirmative, checked by the server. Also:
- refresh `created_at` when an identical row is re-proposed (or refuse to confirm in the same turn);
- give pending actions a short expiry;
- keep student-entered `reason` and `event_name` out of the system block, or quote them clearly as data.

**Regression test:** use a scripted fake model that proposes an action, then the user sends an unrelated message, then the model confirms. The test asserts that no college POST happens.

### NV-2. WebSocket size limits are enforced only after uvicorn has accepted and inflated messages of up to 16 MiB

**Location:** `deploy/kit-assistant.service:10` (no `--ws-max-size` and no `--ws-per-message-deflate false`); `backend/app/voice/session.py:652-660,460,610`.

**What an attacker could do:** Anyone gets a visitor cookie by loading a page, and can hold 3 sockets per IP and 40 in total. Each socket can send highly compressible frames: about 16 KB of deflate stream becomes 16 MiB, which the single uvicorn process inflates and holds. Only after that does the app reject it against its 8 KB and 32 KB limits. With several sockets and IPs this could exhaust memory on the 961 MB droplet. That would drop every session and reset the in-memory abuse counters on restart.

**Evidence:**
- Code reading.
- I checked the defaults in the locked uvicorn 0.54.0: `ws_max_size = 16*1024*1024` and `ws_per_message_deflate = True` (`.venv/.../uvicorn/config.py:207,211`).
- websockets 17.4 applies `max_size` to the decompressed size.

**Blockers:** Peak memory per socket was not measured. It is also unknown whether Caddy passes the permessage-deflate negotiation through to uvicorn.

**Fix:** Add `--ws-max-size 65536` (or just above 32 KiB) to `ExecStart`. Consider `--ws-per-message-deflate false`.

### NV-3. With the voice gate on, silent audio frames cost CPU but are never metered

**Location:** `backend/app/voice/session.py:468-473`; `backend/app/voice/gate.py:24-30,57-72`.

**What an attacker could do:** A visitor streams all-zero 32 KB frames continuously. Each frame runs a pure-Python RMS over 16,384 samples on the shared event loop. Silent frames are not charged to voice time, do not refresh idle time, and have no rate limit. Three sockets per IP, reconnecting every 180 s, could slow everyone else.

**Evidence:** Code reading.

**Blockers:**
- The CPU cost per frame was not measured.
- Whether `VOICE_GATE_ENABLED=true` in production could not be checked, because the env file was not read.

**Fix:** Drop or close sockets that send audio faster than real time plus a margin, and count received audio seconds per principal regardless of the gate.

### NV-4. The shared college-sandbox password and all test login ids are published in the public repo

**Location** (value not repeated here):
- `CONTRACT.md:16-17`
- `scripts/chat.py:22`
- `scripts/fake_mic.py:32`
- `backend/tests/fakes/college.py:67`
- git history from commit `b98b021`

The README says no credentials are in the repository, which contradicts this.

**What an attacker could do:** Any reader of the repository can log in at kit.bixops.in as either teacher, which gives the approve and reject tools, or as any of the eight students. That person could read those students' college data, file requests, and spend the operator's model and speech quota up to the per-user limits. They could also lock those accounts out (see L-1).

**Evidence:** Code reading and `[git]`.

**Blocker:** Whether the organisers already give this password to all participants, which would make the repo copy no new exposure. It was not confirmed by logging in.

**Fix:** Move it to a gitignored env variable (for example `KIT_TEST_PASSWORD`) used by the scripts. Remove it from `CONTRACT.md`. If it is not already public, ask the organisers to rotate it; removing it from history does not help once it has been public.

### NV-5. Arbitrary cookie values are cached in memory, and looked up in the database, before the HTTP rate limit applies

**Location:** `backend/app/limits.py:413-440`.

**What an attacker could do:** `http_allowed` runs `_login_key_for` on the raw `kit_session` cookie of every non-static request. That means a DB lookup and an insert into `_session_keys`, with the raw token as the key. All of it happens *before* `HTTP.take`, so requests that get a 429 still cost a query and a cache entry. The cache is capped at 50,000 entries but has no byte limit. Large random cookies could pin hundreds of MB on a 961 MB host.

**Evidence:** Code reading.

**Blocker:** The effective header-size limit in Caddy and httptools was not measured.

**Fix:**
- apply the IP limit first, then look up the session only for requests that pass;
- reject cookies that are not 43 characters (`token_urlsafe(32)`);
- key the cache on `sha256(token)`.

### NV-6. A few visitor IPs can hold all 40 voice sockets, which shuts out students and teachers

**Location:** `backend/app/limits.py:153,177`; `backend/app/voice/session.py:579-589,627-642`.

**What an attacker could do:** Opening a socket without sending `start` is free, and the cap is 3 sockets per visitor IP. About 14 IPv4 addresses that reconnect every 180 s can therefore keep every other user, including logged-in ones, getting close code 4429. Typed turns also go over this socket.

**Evidence:** Code reading.

**Blocker:** The production limit values are in the unread env file.

**Fix:**
- reserve slots for logged-in users;
- require `start` within a few seconds of connecting;
- throttle reconnects per IP.

## Confirmed findings

### L-1 (low). Anyone can lock a known account out of login for 15 minutes at a time

**Location:** `backend/app/auth.py:215` (`check_login_id` runs before the college is asked); `backend/app/limits.py:455-470`.

**What an attacker could do:** An anonymous client sends 5 wrong passwords for a victim's login id. The correct password is then refused with 429 for 900 s, and the attacker can repeat this indefinitely. One IP's allowance of 30 login attempts a minute is enough to keep about 90 accounts locked at once. The demo login ids are public (NV-4), so the accounts evaluators would use are easy targets. Sessions that are already logged in are not affected.

**Evidence:**
- Code reading. The counter key is only `login_id.strip().lower()`.
- The committed test `backend/tests/test_spine_abuse.py:162-168` asserts this exact behaviour. It was not run.
- ACCEPTANCE ABU5 specifies a lockout per login id, so this is a side effect of the design.
- The condition is that the college answers a wrong password with a 4xx (`auth.py:219-222`).

**Fix:** Key the lockout on (login id, client IP), or apply a progressive delay rather than a hard block. Count only the college's invalid-credentials error, not every 4xx.

### L-3 (low). Concurrent login attempts get past the per-account failure limit

**Location:** `backend/app/auth.py:215` (check), `auth.py:217` (`await` on the college login), `auth.py:221` (failure recorded); `backend/app/limits.py:459-470`.

**What an attacker could do:** The check only reads the counter, and the failure is recorded after an await. Attempts sent concurrently all pass, so an attacker gets about 30 guesses per IP per minute against one account instead of 5 per 15 minutes.

**Evidence:** Code reading; I re-read the ordering myself.

**Fix:** Reserve the attempt (`LOGIN_FAILURES.take`) before calling the college. Release the reservation on success or on a college outage.

### L-2 (low). A coverage database with the developer's local paths is tracked in git

**Location:** `.coverage`, added in commit `bb32a9a`. The worktree deletion is uncommitted. It is not in `.gitignore`.

**What is exposed:** Absolute paths from the developer's machine, including the local username. That is non-secret internal detail.

**Evidence:** `[git]`. `git show HEAD:.coverage` holds paths under the developer's home directory.

**Fix:** Commit the deletion. Add `.coverage`, `htmlcov/`, `prod.env` and `*.env` to `.gitignore`. `DEPLOY.md:25` tells the operator to keep a local `prod.env`, which `.env.*` does not match.

## Hardening notes (no demonstrated boundary violation)

**Deployment configuration**

- **Insecure defaults** (`backend/app/config.py:38,39,68`): `cookie_secure=False`, `public_origin=http://localhost:8000` and `voice_gate_enabled=False` are dev values. While `cookie_secure` is false, `origin_allowed` also accepts any loopback origin (`auth.py:168-169`). Production sets these only in the untracked env file. `[live]` shows production does send `Secure` and the right CSP origin. Make secure values the default, or refuse to start when `public_origin` is https and `cookie_secure` is false. Document the required production values in `DEPLOY.md`.
- **Unlocked install** (`deploy/DEPLOY.md:16,32`): deploys run `uv sync --no-dev` without `--locked`. A stale lock would be re-resolved from PyPI against the unpinned `pyproject.toml`. Use `uv sync --locked --no-dev`.
- **Service unit** (`deploy/kit-assistant.service`):
  - no systemd sandboxing (`NoNewPrivileges`, `ProtectSystem=strict`, `ProtectHome`, `PrivateTmp`, `MemoryMax`);
  - the service user owns its own code and `.venv`;
  - a `MemoryMax` would also contain NV-2.
- **Published infrastructure details** (`deploy/DEPLOY.md:15,24,31`, `restore-bixops.sh:17-18,40`): the docs publish the droplet IP, root SSH login and the key filename, plus another product's container name. Consider a private runbook and turning off root SSH.

**Sessions and login**

- **College token at rest** (`auth.py:128-136`): the college JWT is stored in plaintext in `sessions.kit_token`, while the cookie token is hashed. A database read would expose live college tokens. Consider encrypting it at rest.
- **Logout CSRF** (`auth.py:256-263`): a cross-site form POST can clear the victim's cookie through the response, forcing a sign-out. The server row survives because the SameSite=Lax cookie is not sent. Check `Origin` on POST login and logout.
- **Login error wording** (`auth.py:222`): the college's 401 message is returned verbatim. It is redacted of keys and tokens, but differences in wording could allow account enumeration. Return a fixed message.
- **Lockout memory** (`limits.py:27,183`): all abuse counters are in memory, bounded at 50k keys and reset on restart. Flooding with distinct ids can evict a target's failure history.

**WebSocket**

- **Origin check after accept** (`voice/session.py:630-633`): the socket is accepted before the Origin check and then closed with 4403. Checking before `accept()` is cleaner.
- **Logout window** (`voice/session.py:529-535,571-576`): `start` does not re-check the session, and a turn that is already running keeps going for up to 30 s after a logout in another tab. This is same-principal only; new turns are refused.
- **Admission caps race** (`confirm.py:144,159`): the caps are checked and then recorded only after the awaited college call, and tool calls in one model message run concurrently, so up to 3 applications can pass the per-session cap of 2.
- **No global voice cap** (`limits.py:294-300`): there is no overall limit on voice seconds, and audio keeps going to Deepgram after the global turn budget is spent.
- **Login field lengths** (`auth.py:203-207`): `LoginBody` has no length limits on `login_id` or `password`.

**Agent, tools and model context**

- **Class attendance pre-check** (`agent/tools.py:589-595`): `get_class_attendance` with only `class_id` skips its own "your classes, delegation not expired" pre-check. The college still authorises the call under the teacher's own token, so there is no privilege gain.
- **Shared HTTP client cookies** (`kit/client.py:52-56`): one shared `httpx.AsyncClient` keeps the default cookie jar. If the college ever set a cookie, it would be sent on every user's requests. Disable cookies on this client.
- **No string length caps** (`agent/tools.py:775-791`): `validate_args` has no maximum length on free-text fields.
- **Unredacted error code** (`kit/client.py:93`): `KitError.code` is not passed through `redact` the way the message is.
- **Unescaped document metadata** (`knowledge/mode.py:56,80-84`): document title, version and `applies_to` are not tag-escaped the way bodies are. The source is the college, so this crosses no principal boundary.
- **Classifier tag escape** (`moderation.py:142`): user text can close the `<message>` tag in the classifier prompt. This weakens a guardrail only.
- **Secret check per piece** (`moderation.py:233`): the output secret check runs per clause, and tool arguments are never checked. No secret is in model context today.
- **Stale docstring** (`confirm.py:27`): it says student notes are excluded, but `REQUEST_FIELDS` keeps them.
- **Cache owner default** (`loop.py:171`): `_load` treats a cache entry with no recorded owner as a match. It is safe today because `_remember` always records the owner and ownership is checked in the database first. Make it fail closed.

## Checked and found sound

**Authentication and sessions** (code reading; `[live]` for flags)
- Token is `secrets.token_urlsafe(32)` and only its SHA-256 is stored (`auth.py:43,123-124`).
- The session id rotates on login, and the old row is deleted (`auth.py:242-252`).
- Logout deletes the row; expiry is checked in SQL; a college token near expiry ends the session (`auth.py:74-85`).
- `/api/login` and `/api/me` return only the allow-listed profile.
- Live cookie is `HttpOnly; Secure; SameSite=Lax`, with an 8 h max age `[live]`.
- Login CSRF is blocked: FastAPI 0.162.7 accepts only `application/json` for this body, which needs a CORS preflight, and there is no CORS middleware.

**Cross-user isolation** (code reading)
- History list and detail filter by `owner_key`; a foreign id and a missing id both give the same 404 (`history.py:99-152`).
- Voice resume accepts a conversation id only if `conversation_owned` (`voice/session.py:529-535`).
- `run_tool` re-checks conversation ownership (`tools.py:817-838`).
- Pending actions are keyed by id, `session_id` and `conversation_id`, locked `FOR UPDATE`, and executed at most once (`confirm.py:111-160`).
- The in-memory conversation cache is owner-checked (`loop.py:148-185`).
- Tools are gated by role on the server, and no tool takes a user id.
- Every college call uses the calling session's own token.
- Teacher approve and reject pre-check that the request belongs to the teacher's classes.

**WebSocket** (code reading; `[live]`)
- Exact Origin match; a bad Origin closes 4403 and no session closes 4401 `[live]`.
- The session is re-checked before every turn and every 30 s.
- Logout, or login as another user, deletes the old row, so open sockets close with 4401.
- Socket slots are acquired atomically and released on every exit path.
- Deepgram keys go only in headers, and provider error text is never forwarded.

**Secrets** (code reading; `[git]`)
- No key or token in model context, tool results, stored messages, `tool_calls`, logs or responses.
- College errors are redacted (`kit/client.py:67-81`).
- Output cleaning replaces key-shaped text.
- All 20 commits were scanned: the only key-shaped strings are short fake fixtures in `backend/tests`.
- A hash-free comparison found none of the 10 real `.env` values in history (only non-secret config values matched).
- `.env` was never committed.

**Prompt injection** (code reading)
- Visitors get only the brochure in every path: context, RAG search and `resolve_refs`.
- Lesson notes are filtered by department.
- Document bodies are wrapped, and `<document` tags inside them are escaped.
- Profile fields come from a fixed allow-list, so free-text notes never reach the model.
- Documents are written only by the college, which already holds every capability they could steer. The one cross-principal channel is NV-1.

**SQL**
- Every query I read and that the hunters read uses asyncpg `$n` parameters. No SQL is built from strings.

**Headers and CSP**
- Production sends all of these `[live]`: CSP `default-src 'self'` (no inline script or eval), `frame-ancestors 'none'`, `X-Frame-Options DENY`, `nosniff`, `Referrer-Policy`, `Permissions-Policy`, and HSTS for 1 year with includeSubDomains.
- HTTP gets a 308 redirect to HTTPS `[live]`.
- `/docs`, `/openapi.json`, `/.env` and `/.git/config` return 404 `[live]`.
- In code, the headers also apply to 429, 413 and 500 responses.

**TLS** (`[live]`)
- TLS 1.2 and 1.3 with ECDHE-ECDSA AES-GCM; Let's Encrypt certificate for exactly kit.bixops.in, valid until 2027-02-12.
- TLS 1.0 and 1.1 could not be tested, because the local OpenSSL refuses them. Caddy's default minimum is TLS 1.2.

**Frontend** (code reading)
- No `innerHTML`, `insertAdjacentHTML`, `document.write`, `eval` or dynamic `href`/`src`. All dynamic text uses `textContent`.
- Redirects go to constant paths.
- sessionStorage holds only a conversation-id hint.

**Dependencies**
- OSV advisory batch query on all 58 locked name==version pairs: **no known vulnerabilities**. This sent only the public package list, which is already in the public `uv.lock`.
- All 58 packages come from pypi.org with sha256 hashes, and no alternate index is configured.
- `httpx2`, `httpcore2` and `httpx2-jsfetch` are legitimate dependencies of the official anthropic and google-genai SDKs. `httpx2-jsfetch` is emscripten-only and not installed.

**Deploy**
- uvicorn binds 127.0.0.1 only, behind Caddy.
- `X-Forwarded-For` is trusted only from loopback, and the right-most entry is used (`limits.py:75-84`).
- The service runs as a non-root user.
- No debug or reload flag.

## Not covered

**Skill steps skipped or shortened.** The owner's 14:00 IST deadline was the reason, along with the execution policy.
- `quick` profile: one hunter wave of 6 independent agents over 18 coverage units.
- **The post-wave coverage critic was not run.**
- **Phase 3 and 5 independent verifiers were not run** as separate agents. Instead, I re-read the decisive lines myself, and two leads (NV-1, and L-1's lines) were found independently by two hunters. Under the skill's rules, the low findings are therefore not fully verified.
- `findings.json`, `REPORT.md`, `FINDINGS-DETAIL.md` and `NEEDS-VALIDATION.md` were not produced. This file is the report.
- The reconnaissance in Phase 1 was done by me directly, not by four separate agents.
- All six hunters returned; the SQL, rate-limit and wildcard pass was the last.
- One lead was rejected by a live check: visitor limits keyed on the full IPv6 address could be rotated within a prefix, but `kit.bixops.in` has no AAAA record `[live]`, so it is not reachable today. Aggregate by /64 if IPv6 is ever enabled.

**No code was run.** Neither the test suite nor any harness: no OS sandbox meeting the skill's controls was available. Every "needs validation" item has a bounded local test plan in the run artefacts.

**Out of scope or not reviewed:**
- The untracked `ui/` and `frontend/app/`, and the uncommitted changes to `main.py` and `config.py` that serve them via `UI_VARIANT` and `/ui-assets`. These need their own review, covering DOM sinks, CSP compatibility and source maps, before they are deployed.
- The production env file (`COOKIE_SECURE`, `PUBLIC_ORIGIN`, `VOICE_GATE_ENABLED`). It was deliberately not read; the live headers suggest the first two are set correctly.
- College API behaviour, Caddy's runtime WebSocket extension handling, and the droplet's other services.
- No logins were made and no write actions were confirmed.

**Prior runs:** none existed, so coverage is first-pass only. The skill's authors note that one run finds roughly half of what repeated runs find.
