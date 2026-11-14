# Tests

Run everything with `make test` (needs Postgres with pgvector; no third-party service is contacted).

| Files | What they are |
|---|---|
| `test_acc_*.py` | Acceptance suites written from `ACCEPTANCE.md` by someone other than the module's author. Each test name carries its criterion id (for example `test_ac8_...`). |
| `test_e2e_isolation_*.py` | End-to-end isolation: the real assembled app (REST, WebSocket, assistant loop, tools, knowledge) with several users at once; only the college, model, speech and embedding services are faked. |
| `test_spine_*.py`, `test_security_*.py` | Sessions, history, routes, headers, origin checks, abuse limits at the HTTP layer. |
| `test_tools_*.py`, `test_kit_*.py` | College client, role-based tools, the confirm-before-write gate. |
| `test_agent_*.py`, `test_moderation_*.py`, `test_fallback_*.py` | Assistant loop, prompts, moderation, the second-provider fallback. |
| `test_knowledge_*.py` | Document sync, selection by audience, cached-prompt and RAG modes, citations. |
| `test_voice_*.py`, `test_limits_*.py` | Voice WebSocket, barge-in, the voice gate, rate and input limits. |
| `test_ports_conformance.py` | The provider adapters satisfy the interfaces in `app/ports.py`. |

Shared foundation: `conftest.py` (test database from `TEST_DATABASE_URL`, per-test reset, a network guard
that fails any test reaching a real host, dummy API keys) and `fakes/college.py` (an in-memory college
system where each token maps to exactly one user and every request is recorded).
