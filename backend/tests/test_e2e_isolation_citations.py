# ruff: noqa: F811  (the e2e fixture is imported, then requested by name)
"""DOC1 / DOC4 / CIT3 end to end: citations and document text follow the reader's rights.

A visitor never receives a citation to anything but the brochure, even when the model names
handbook, lesson-note and circular refs on its SOURCES line; a student's citations resolve only
to documents that student may use. Checked in frames, stored messages and GET /api/history.
"""

import json

import pytest

from tests.e2e_helpers import (
    BROCHURE_ID,
    DOC_MARK,
    NON_BROCHURE_MARKS,
    TURN_MODES,
    A,
    C,
    e2e,  # noqa: F401
    events,
    hits,
    stored_messages,
    stored_tool_calls,
)

DOC_MODES = pytest.mark.parametrize("doc_mode", ["context", "rag"])


def all_refs(e2e) -> list[str]:
    refs = [r for doc in e2e.refs.values() for r in doc]
    return refs + ["D2:6.4.3", "D1:6.4.3", "D4:1.3.1", "D99:1.1", "D3:999"]


def doc_of(ref: str) -> int:
    return int(ref.split(":")[0][1:])


@pytest.fixture
def doc_mode_setting(monkeypatch):
    from app.config import get_settings
    from app.knowledge import mode

    def set_mode(value: str) -> None:
        monkeypatch.setattr(get_settings(), "doc_mode", value)
        mode._cache.clear()

    return set_mode


async def cited_turn(e2e, actor, how, text):
    s = await actor.socket()
    await s.start()
    sources = ", ".join(all_refs(e2e))
    e2e.model.plan(text, f"The rule is explained in the documents.\nSOURCES: {sources}")
    frames = await s.turn(text, how)
    assert events(frames, "assistant_done"), frames
    return s, frames


def document_refs(sources: list[dict]) -> list[str]:
    return [x["ref"] for x in sources if x.get("kind") == "document"]


async def all_citation_exits(actor, s, frames) -> list[list[dict]]:
    """Citations as the client got them (frames), as stored, and as history returns them."""
    out = [f["sources"] for f in events(frames, "citations")]
    out += [f.get("citations") or [] for f in events(frames, "assistant_done")]
    stored = [m for m in await stored_messages(s.conversation_id) if m["role"] == "assistant"]
    for m in stored:
        c = m["citations"]
        out.append(json.loads(c) if isinstance(c, str) else (c or []))
    hist = (await actor.get(f"/api/history/{s.conversation_id}")).json()
    out += [m.get("citations") or [] for m in hist["messages"] if m["role"] == "assistant"]
    return out


@TURN_MODES
@DOC_MODES
async def test_doc1_cit3_visitor_citations_are_brochure_only(e2e, how, doc_mode, doc_mode_setting):
    doc_mode_setting(doc_mode)
    v = await e2e.actor("V1")
    s, frames = await cited_turn(e2e, v, how, "what are the fees and the on-duty rules")
    exits = await all_citation_exits(v, s, frames)
    assert exits
    for sources in exits:
        assert all(doc_of(r) == BROCHURE_ID for r in document_refs(sources)), sources
        assert hits(json.dumps(sources), NON_BROCHURE_MARKS + ["Handbook", "Lesson", "Circular"]) \
            == []
    assert any(document_refs(x) for x in exits), "control: the brochure ref should resolve"
    # CIT5: the SOURCES line is never shown or spoken.
    assert "SOURCES" not in s.text()
    assert hits(s.text(), NON_BROCHURE_MARKS) == []


@TURN_MODES
async def test_doc1_visitor_search_results_are_brochure_only(e2e, how, doc_mode_setting):
    doc_mode_setting("rag")
    v = await e2e.actor("V1")
    s = await v.socket()
    await s.start()
    queries = ["on-duty proof working days", "foreign key primary key", "circular on-duty proof",
               "tuition fee", DOC_MARK[2], DOC_MARK[4], DOC_MARK[5]]
    e2e.model.plan("search everything", [("search_documents", {"query": q}) for q in queries],
                   "Okay.")
    await s.turn("search everything", how)
    rows = [r for r in await stored_tool_calls(s.conversation_id) if r["name"] == "search_documents"]
    assert len(rows) == len(queries)
    out = json.dumps([r["output"] for r in rows])
    assert hits(out, NON_BROCHURE_MARKS) == []
    assert DOC_MARK[BROCHURE_ID] in out  # control: the brochure is searchable
    for r in rows:
        for hit in (r["output"] or {}).get("results", []):
            assert "Brochure" in hit["document_title"], hit
    model_in = "\n".join(c.results_text() for c in e2e.model.by("V1"))  # minus its own queries
    assert hits(model_in, NON_BROCHURE_MARKS) == []


@TURN_MODES
@DOC_MODES
@pytest.mark.parametrize("login,allowed", [(A, {2, 3, 4, 5}), (C, {2, 3, 5})])
async def test_cit3_doc4_student_citations_only_to_documents_they_may_use(
        e2e, how, doc_mode, login, allowed, doc_mode_setting):
    doc_mode_setting(doc_mode)
    st = await e2e.actor("S", login)
    s, frames = await cited_turn(e2e, st, how, "what do the rules and notes say")
    exits = await all_citation_exits(st, s, frames)
    for sources in exits:
        docs = {doc_of(r) for r in document_refs(sources)}
        assert docs <= allowed, (login, sources)
        assert 1 not in docs  # superseded handbook (DOC2)
    assert any(2 in {doc_of(r) for r in document_refs(x)} for x in exits), "control"
    if login == C:
        model_in = "\n".join(c.text() for c in e2e.model.by("S"))
        assert DOC_MARK[4] not in model_in  # CSE lesson notes never reach an ECE student
