# Design decisions

Short records of the choices that shape the system, with what was given up. Details: [ARCHITECTURE.md](ARCHITECTURE.md).

| # | Decision | Why | Trade-off |
|---|---|---|---|
| 1 | One deployable: FastAPI serves the API, the voice WebSocket and the built interface | Same origin for cookie and socket; one process to run on a small server | Scaling out needs the shared state in decision 8 moved to a shared store |
| 2 | Speech-to-text and text-to-speech run on the server side of our WebSocket | No provider key reaches the browser; the browser's speech recognition is never used | Audio makes an extra hop through our server |
| 3 | A plain streamed tool loop on the model SDK instead of an agent framework | One agent, about twenty tools: less code, lower latency, easy to test with a fake client | No graph tooling if the flow grows into several agents |
| 4 | Tool sets are chosen by the server from the session role; no tool takes a user id | The model cannot ask for another user's data or a tool outside its role | Each new capability must be registered per role |
| 5 | Writes go through a pending action that needs an explicit approval in a later turn, with expiry and an idempotency key | A misheard sentence or injected text cannot submit, approve or cancel anything | One extra spoken turn before every write |
| 6 | Documents live in a cached prompt and switch to RAG over pgvector above a token budget | Whole documents are more accurate than retrieval at this size; RAG takes over when the college adds large documents | Two code paths to keep correct, both covered by tests |
| 7 | The live college system is the source of truth for dates, schedules and who may approve; documents supply rules | It is what the college enforces when a request is submitted | A document that disagrees is mentioned, not followed |
| 8 | Short-window limits and the conversation cache are in process memory; daily allowances are persisted to Postgres | No database call on the request path; conversations rebuild from Postgres on a cache miss | Per-minute windows restart with the process |
| 9 | External services sit behind the interfaces in `backend/app/ports.py`, with a conformance test | A second provider is an adapter, not a rewrite; the model fallback is one such adapter | Interfaces must be kept in step with what the app calls |
| 10 | Tests are derived from `ACCEPTANCE.md`, by someone other than the module's author where possible | The suites found defects the authors' own tests missed (see README) | More tests to maintain |
