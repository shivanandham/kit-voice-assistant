# Acceptance criteria

What the system must do, independent of how it is written. Tests are derived from this file and from
[CONTRACT.md](CONTRACT.md), never from the implementation: when the code disagrees with a criterion,
the code is wrong. Each test names the criterion it proves (for example `AC3`).

## Access control and privacy (AC)

- **AC1** A visitor (no login) can never obtain any student or teacher data: not through a REST
  route, the voice WebSocket, a tool, conversation history, a pending action, or document search.
- **AC2** A logged-in user only ever gets their own data. No route or tool accepts a user id, login
  id or token from the client or from the model. Every college call for a session carries that
  session's token and nobody else's.
- **AC3** Conversation history is visible only to its owner. Another user's conversation id returns
  404 and no data, whether listed, read, resumed over the WebSocket or used in a tool call.
- **AC4** Role (visitor, student, teacher) comes only from the college's login answer. The server
  picks the tool set from the role; a tool outside the set is refused even if the model asks for it.
- **AC5** Secrets (participant key, college tokens, passwords, API keys) never appear in a response
  body, header, WebSocket message, model prompt, tool result, log line or stored message.
- **AC6** Sessions: HttpOnly cookie; only a hash of the token is stored; the id rotates on login;
  logout ends the session on the server; an expired session or expired college token is treated as
  no session; logging in as B in a browser that was A leaves nothing of A reachable.
- **AC7** The voice WebSocket needs a valid session and an allowed Origin. A conversation can be
  resumed only by its owner.
- **AC8** A write to the college system happens only through a pending action that the same session
  confirms in the same conversation after a later user message, and it executes at most once.
- **AC9** A teacher acts only on classes the college authorises for them; the app never widens that,
  and an expired delegation is reported, not bypassed.

## Documents (DOC)

- **DOC1** Visitors get the admissions brochure only, in both modes, including search results.
- **DOC2** The handbook edition in force on the college's date is used; superseded and not-yet-effective
  editions are excluded.
- **DOC3** Circulars are always given in full, in both modes.
- **DOC4** Lesson notes go to teachers and to students of a listed department; notes that list no
  department go to every student. A document of a type not named here (the college may add new
  kinds mid-event) goes to logged-in users only, never to visitors.
- **DOC9** Only the roles `student` and `teacher` count as logged in. Any other role value is
  treated as a visitor.
- **DOC5** `auto` uses `context` while applicable documents fit the token budget and `rag` above it;
  `context` and `rag` force the mode.
- **DOC6** New, changed and removed documents are picked up by the next sync; a failed sync or a
  failed embedding call never breaks the app.
- **DOC7** Search returns only chunks of documents the caller may use, and falls back to full-text
  ranking when embeddings are unavailable.
- **DOC8** The rendered knowledge text is byte-identical between calls while documents are unchanged.

## Correctness of answers and actions (COR)

- **COR1** Every date calculation uses the college's date (`/v1/today`), never the server clock.
- **COR2** Timetable for a date: a holiday or Sunday has no classes; a working day that follows
  another weekday uses that weekday's timetable.
- **COR3** Leave and on-duty: `to_date` before `from_date` and on-duty without an event name are
  rejected before anything is stored; exam days, holidays and past dates in the range are warnings
  shown to the user before confirmation.
- **COR4** Placement drives are marked eligible or not, with reasons: CGPA, standing arrears,
  department, year, deadline, seats left, already registered, already placed.
- **COR5** Admission applications are validated (name, phone, email, course code, percentage) and
  warn when the percentage is below the course minimum.
- **COR6** A refusal from the college system is relayed with the college's own message.
- **COR7** Every write carries an idempotency key, and a retry or double confirmation never creates
  a second record.

## Assistant loop (AGT)

- **AGT1** The user's message is saved before the model is called; the reply is saved when done.
- **AGT2** Every tool call the model makes receives exactly one result, including failed, refused
  and cancelled ones; the stored conversation is always valid for the next model call.
- **AGT3** Cancelling a turn stops the model call and saves what was already said.
- **AGT4** A turn stops after a bounded number of model calls.
- **AGT5** Nothing user-specific or time-dependent sits before the prompt-cache breakpoint.
- **AGT6** Model or college failures produce a short, safe error for the user, never a crash or a
  stack trace.

## Voice (VOX)

- **VOX1** The WebSocket follows the protocol in CONTRACT.md exactly.
- **VOX2** Speech or an `interrupt` during a reply cancels it and sends `interrupted`; the new
  utterance becomes the next turn.
- **VOX3** Reply audio arrives between `audio_start` and `audio_end`, sentences in order, with no
  audio after `interrupted`.
- **VOX4** Typed text works without a microphone and is not echoed back as a transcript.
- **VOX5** One turn at a time per connection; a failing speech service produces an `error` event and
  leaves the socket usable.

## History (HIS)

- **HIS1** Voice and typed turns are both saved, in order, with the conversation titled from the
  first user message.
- **HIS2** Visitors have history too, tied to their browser session.
- **HIS3** History survives a server restart.

## Reliability (REL)

- **REL1** With the college system down, public pages and the assistant answer with a clear message.
- **REL2** No test touches a real third-party service.
- **REL3** If the primary model call fails before any reply text has been emitted, a second model
  provider answers from the documents and conversation, without tools; if it also fails, the user gets
  the safe error. A failure after text has been emitted does not switch providers mid-reply.
- **REL4** If the speech service returns no audio for a reply, the reply is still shown and is read
  aloud by the browser's speech synthesis; speech recognition never uses the browser.

## Abuse and limits (ABU)

Rule of thumb: IP-based limits apply only to visitors. A logged-in user is always limited by login
id, so people sharing a network (a lab, a hostel, the judges' room) never use up each other's allowance.

All limits are settings with defaults; exceeding one gives a clear, polite message (HTTP 429 or an
`error` event with a spoken-style text), never a crash, and never affects other users.

- **ABU1** Voice connections: a visitor IP may hold at most 3 open sockets; a logged-in user at most
  2, counted per login id and never per IP, so any number of students behind one college network
  can talk at once. At most 40 sockets are open overall. Extra connections are refused with close
  code 4429.
- **ABU2** Turns (owner's numbers): a visitor gets at most 20 turns per day, counted per IP across
  all of that IP's sessions, so clearing cookies does not reset it. A logged-in user gets at most
  100 turns per day, counted per login id across all of that user's sessions and devices.
- **ABU3** Audio: one utterance is capped at 30 seconds; a socket with no speech or text for 3 minutes
  is closed; a visitor socket lives at most 15 minutes and a logged-in one at most 60, after which
  the client must reconnect.
- **ABU4** New anonymous sessions are limited to 30 per IP per hour, and a page load that already
  has a valid session never creates another.
- **ABU5** HTTP requests from visitors are limited to 120 per minute per IP (static files excluded);
  requests from logged-in sessions are limited to 120 per minute per login id instead, never per
  IP. Logins are limited to 30 attempts per minute per IP and 5 failures per login id per 15 minutes.
- **ABU6** Writes: at most 3 pending actions per conversation; a visitor session can submit at most
  2 admission applications per day and an IP at most 5.
- **ABU7** A conversation holds at most 200 turns; after that the user is asked to start a new one.
- **ABU8** A global daily turn budget (default 5000) protects the paid speech and model credits;
  when it is spent, visitors get a "busy, try later" message while logged-in users keep working
  until a second, higher budget (default 8000).
- **ABU9** The client IP is taken from the proxy's `X-Forwarded-For` entry and cannot be forged by a
  header the client adds itself.
- **ABU10** Input limits: typed text at most 1000 characters; one spoken utterance at most 30 seconds;
  an HTTP request body at most 64 KB; a WebSocket control message at most 8 KB and an audio frame at
  most 32 KB. Oversized input is refused with a clear message and is never passed to the model, the
  speech service or the college system. The app accepts no file uploads.
- **ABU11** Audio reaches the speech service only while someone is speaking, plus a short lead-in so
  the first word is not clipped and a short tail so the end of speech is detected. Silence is never
  forwarded or billed, however long the microphone stays open. The server enforces this (a client
  cannot bypass it), and the browser also stops uploading silence. Speech-service time is capped
  per day: 15 minutes per visitor IP and 60 minutes per logged-in user.

## Moderation and output safety (MOD, OUT)

- **MOD1** Every user message, spoken or typed, is checked before any tool runs and before any reply
  is shown or spoken. Abusive, hateful, sexual, violent or otherwise harmful requests, attempts to
  obtain secrets or the system prompt, and attempts to override the assistant's instructions get a
  short polite refusal: no tool call, no college call.
- **MOD2** A message that signals personal distress or self-harm gets a short supportive reply that
  points to a person who can help, never a bare refusal.
- **MOD3** Every piece of assistant text is checked before it is sent or spoken. Text containing a
  secret (API key, token, participant key), profanity or slurs is replaced by a safe fallback line.
- **MOD4** Moderation never blocks ordinary college questions (false positives are defects), adds no
  noticeable delay to a normal turn, and if the checking service fails the turn still completes
  using the built-in rules.
- **MOD5** Each moderation decision is logged with its category, without storing secrets.
- **OUT1** Assistant text reaching the client is plain text: no HTML tags, no markdown syntax, no
  control characters. The frontend inserts all dynamic text as text, never as HTML, and the page's
  Content-Security-Policy forbids inline script.
- **AGT7** At most the last 20 messages of a conversation are sent to the model as context; this is
  a setting.

## Citations (CIT)

- **CIT1** Every stored chunk carries what is needed to cite it: a stable reference, its document
  (id, title, type, edition, effective date), its section or rule number and heading, and its page range.
- **CIT2** An answer that uses a document carries citations in both document modes: structured ones
  (document, edition, section, page) sent to the client and saved with the message, and a short
  natural mention of the source in the spoken reply.
- **CIT3** A citation can only point at a document the user may use and a section that exists.
  Invented or foreign references are dropped; a visitor never receives a citation to anything but
  the brochure.
- **CIT4** An answer that uses live college records is labelled as coming from the college system.
- **CIT5** The machine-readable source line is never spoken and never shown as reply text.
- **CIT6** History shows each past reply with its citations.
