"""Acceptance: citations, the SOURCES line, and the history window sent to the model.

Criteria: CIT2, CIT3, CIT4, CIT5, CIT6, AGT7. Model, classifier and (mostly) the resolver are faked.
"""

import json
from datetime import date

import pytest
from app import history
from app.agent import loop
from app.agent.tools import ToolResult
from app.db import pool
from app.knowledge import rag, store
from tests.acc_agent_helpers import (  # noqa: F401
    ARJUN, PRIYA, VIS1, Reply, assert_valid, blocks, collect, db_messages, dump, env_factory,
    evict_memory, new_conv, text, tool_use,
)
from tests.test_acc_agent_events import check_event_shapes, deltas

DOC_KEYS = {"kind", "ref", "document_title", "version", "section", "heading", "page_start",
            "page_end"}
KNOWN = {
    "D2:6.4.3": {"ref": "D2:6.4.3", "document_title": "Student Handbook 2026", "version": "2026.1",
                 "section": "6.4.3", "heading": "On-duty proof", "page_start": 14, "page_end": 14,
                 "document_type": "handbook", "effective_date": "2026-07-01"},
    "D5:c0": {"ref": "D5:c0", "document_title": "Circular 12/2026", "version": "1",
              "section": None, "heading": "Circular", "page_start": 1, "page_end": 2,
              "document_type": "circular", "effective_date": "2026-09-01"},
}


@pytest.fixture
def resolver(monkeypatch):
    """A fake rag.resolve_refs that knows KNOWN and drops everything else."""
    seen = []

    async def resolve(refs, ctx):
        seen.append((list(refs), ctx.owner_key))
        out, done = [], set()
        for r in refs:
            if r in KNOWN and r not in done:
                done.add(r)
                out.append(dict(KNOWN[r]))
        return out
    monkeypatch.setattr(rag, "resolve_refs", resolve)
    return seen


def sources_of(events):
    c = [e for e in events if e["type"] == "citations"]
    return c[0]["sources"] if c else []


def docs_of(events):
    return [s for s in sources_of(events) if s["kind"] == "document"]


async def stored_citations(cid):
    rows = await pool().fetch(
        "SELECT citations FROM messages WHERE conversation_id=$1 AND role='assistant' ORDER BY id",
        __import__("uuid").UUID(cid))
    out = []
    for r in rows:
        v = r["citations"]
        out.append(json.loads(v) if isinstance(v, str) else v)
    return out


def assert_line_hidden(events, cid_messages):
    flat = " ".join(e["text"] for e in events if e["type"] in ("text_delta", "done")) \
        + dump(cid_messages)
    assert "SOURCES" not in flat.upper().replace("SOURCES OF", "")
    assert "D2:6.4.3" not in flat and "D5:c0" not in flat


# ------------------------------------------------------------------ CIT2 / CIT5

async def test_CIT2_sources_line_becomes_a_citations_event_and_done_carries_it(env_factory, resolver):
    env_factory([Reply(["Under rule six point four point three you must submit proof in five days.",
                        "\nSOURCES: D2:6.4.3, D5:c0"])])
    cid = await new_conv(ARJUN)
    events = await collect(ARJUN, cid, "on duty proof?")
    check_event_shapes(events)
    docs = docs_of(events)
    assert [d["ref"] for d in docs] == ["D2:6.4.3", "D5:c0"]
    for d in docs:
        assert set(d) == DOC_KEYS and d["kind"] == "document"
    assert docs[0] == {"kind": "document", "ref": "D2:6.4.3", "document_title": "Student Handbook 2026",
                       "version": "2026.1", "section": "6.4.3", "heading": "On-duty proof",
                       "page_start": 14, "page_end": 14}
    assert events[-2]["type"] == "citations" and events[-1]["citations"] == events[-2]["sources"]
    assert "submit proof" in events[-1]["text"]
    assert resolver[0][0] == ["D2:6.4.3", "D5:c0"] and resolver[0][1] == "user:arjun"
    assert_line_hidden(events, await db_messages(cid))


LINE_VARIANTS = [
    ["Answer here.\nSOURCES: D2:6.4.3"],
    ["Answer here. SOURCES: D2:6.4.3"],
    ["Answer here.\nsources: D2:6.4.3"],
    ["Answer here.\nSources: D2:6.4.3"],
    ["Answer here.\nSOURCES:D2:6.4.3"],
    ["Answer here.\nSOURCES : D2:6.4.3"],
    ["Answer here.\n  SOURCES:   D2:6.4.3  "],
    ["Answer here.\n", "SOU", "RCES", ": D2:", "6.4.3"],
    ["Answer here.\nS", "O", "U", "R", "C", "E", "S", ":", " ", "D", "2", ":", "6", ".", "4.3"],
    ["Answer here.", "\nSOURCES: D2", ":6.4.3"],
    ["Answer here.\nSOURCES: D2:6.4.3\n"],
    ["Answer here.\nSOURCES: D2:6.4.3."],
]


@pytest.mark.parametrize("chunks", LINE_VARIANTS, ids=range(len(LINE_VARIANTS)))
async def test_CIT5_sources_line_never_appears_in_text_and_is_resolved(env_factory, resolver, chunks):
    env_factory([Reply(chunks)])
    cid = await new_conv(ARJUN)
    events = await collect(ARJUN, cid, "q")
    check_event_shapes(events)
    assert [d["ref"] for d in docs_of(events)] == ["D2:6.4.3"]
    assert events[-1]["text"].startswith("Answer here")
    assert_line_hidden(events, await db_messages(cid))
    for t in await stored_citations(cid):
        pass


async def test_CIT5_line_after_a_tool_call_in_the_second_model_call(env_factory, resolver):
    env_factory([Reply(["Checking. "], [text("Checking. "), tool_use("t", "get_my_profile")],
                       "tool_use"),
                 Reply(["Rule six applies.\nSOURCES: D2:6.4.3"])])
    cid = await new_conv(ARJUN)
    events = await collect(ARJUN, cid, "q")
    refs = [d["ref"] for d in docs_of(events)]
    assert refs == ["D2:6.4.3"]
    assert_line_hidden(events, await db_messages(cid))
    assert any(s["kind"] == "college_record" for s in sources_of(events))


async def test_CIT5_line_in_the_middle_of_the_reply_is_still_never_spoken(env_factory, resolver):
    env_factory([Reply(["First part. ", "SOURCES: D2:6.4.3\n", "Second part."])])
    cid = await new_conv(ARJUN)
    events = await collect(ARJUN, cid, "q")
    assert_line_hidden(events, await db_messages(cid))
    assert "First part" in events[-1]["text"]


async def test_CIT5_word_sources_in_ordinary_sentence_is_kept(env_factory, resolver):
    env_factory([Reply(["Your sources of income do not matter here. Fees are fixed."])])
    cid = await new_conv(ARJUN)
    events = await collect(ARJUN, cid, "q")
    assert "sources of income" in events[-1]["text"]
    assert sources_of(events) == []


# ------------------------------------------------------------------ CIT3

async def test_CIT3_refs_the_resolver_rejects_are_dropped(env_factory, resolver):
    env_factory([Reply(["Answer.\nSOURCES: D99:1.1, D2:6.4.3, D2:99.9, garbage, D5:c0"])])
    cid = await new_conv(ARJUN)
    events = await collect(ARJUN, cid, "q")
    assert [d["ref"] for d in docs_of(events)] == ["D2:6.4.3", "D5:c0"]
    flat = dump(events) + dump(await stored_citations(cid))
    assert "D99" not in flat and "garbage" not in flat and "99.9" not in flat


async def test_CIT3_all_refs_rejected_means_no_citation_at_all(env_factory, resolver):
    env_factory([Reply(["Answer.\nSOURCES: D99:1, D98:2"])])
    cid = await new_conv(ARJUN)
    events = await collect(ARJUN, cid, "q")
    check_event_shapes(events)
    assert sources_of(events) == []
    assert not events[-1].get("citations")
    assert (await stored_citations(cid)) == [None] or (await stored_citations(cid)) == [[]]
    assert_line_hidden(events, await db_messages(cid))


async def test_CIT3_duplicate_refs_are_cited_once(env_factory, resolver):
    env_factory([Reply(["Answer.\nSOURCES: D2:6.4.3, D2:6.4.3, D2:6.4.3"])])
    cid = await new_conv(ARJUN)
    events = await collect(ARJUN, cid, "q")
    assert [d["ref"] for d in docs_of(events)] == ["D2:6.4.3"]


async def test_CIT3_resolver_failure_never_breaks_the_turn(env_factory, monkeypatch):
    async def boom(refs, ctx):
        raise RuntimeError("db down")
    monkeypatch.setattr(rag, "resolve_refs", boom)
    env_factory([Reply(["Answer.\nSOURCES: D2:6.4.3"])])
    cid = await new_conv(ARJUN)
    events = await collect(ARJUN, cid, "q")
    check_event_shapes(events)
    assert events[-1]["type"] == "done" and "db down" not in dump(events)
    assert_line_hidden(events, await db_messages(cid))


async def test_CIT3_resolution_uses_the_callers_own_context(env_factory, resolver):
    env_factory([Reply(["A.\nSOURCES: D2:6.4.3"]), Reply(["B.\nSOURCES: D2:6.4.3"])])
    ca, cp = await new_conv(ARJUN), await new_conv(PRIYA)
    await collect(ARJUN, ca, "q")
    await collect(PRIYA, cp, "q")
    assert [owner for _r, owner in resolver] == ["user:arjun", "user:priya"]


async def test_CIT3_visitor_citing_handbook_gets_no_document_citation_real_resolver(
        env_factory, monkeypatch):
    from tests.test_knowledge_fakes import college_date, insert_documents
    await insert_documents()
    monkeypatch.setattr(store, "_college_date", college_date(date(2026, 11, 21)))
    env_factory([Reply(["The rule says five days.\nSOURCES: D2:6.4.3, D1:6.4.3, D4:1"])])
    cid = await new_conv(VIS1)
    events = await collect(VIS1, cid, "on duty proof?")
    check_event_shapes(events)
    assert docs_of(events) == []
    assert "Student Handbook" not in dump(events) + dump(await stored_citations(cid))
    assert_line_hidden(events, await db_messages(cid))


async def test_CIT3_visitor_may_cite_the_brochure_real_resolver(env_factory, monkeypatch):
    from tests.test_knowledge_fakes import college_date, insert_documents
    await insert_documents()
    monkeypatch.setattr(store, "_college_date", college_date(date(2026, 11, 21)))
    docs = await store.applicable_documents(VIS1)
    assert [d["id"] for d in docs] == [3]
    refs = list(rag._structure(docs[0]).refs)
    assert refs, "the brochure has sections that can be cited"
    ref = refs[0]
    env_factory([Reply([f"Fees are listed.\nSOURCES: {ref}"])])
    cid = await new_conv(VIS1)
    events = await collect(VIS1, cid, "fees?")
    assert [d["ref"] for d in docs_of(events)] == [ref]


async def test_CIT3_student_real_resolver_cites_real_section_and_drops_invented(
        env_factory, monkeypatch):
    from tests.test_knowledge_fakes import college_date, insert_documents
    await insert_documents()
    monkeypatch.setattr(store, "_college_date", college_date(date(2026, 11, 21)))
    from tests.test_knowledge_fakes import CSE_STUDENT
    env_factory([Reply(["Five working days.\nSOURCES: D2:6.4.3, D2:9.9.9, D77:1"])])
    cid = await new_conv(CSE_STUDENT)
    events = await collect(CSE_STUDENT, cid, "on duty proof?")
    docs = docs_of(events)
    assert [d["ref"] for d in docs] == ["D2:6.4.3"]
    assert docs[0]["section"] == "6.4.3" and docs[0]["document_title"]
    assert set(docs[0]) == DOC_KEYS


# ------------------------------------------------------------------ CIT4

async def test_CIT4_a_successful_read_tool_adds_a_college_record_source(env_factory):
    env_factory([Reply([], [tool_use("t", "get_attendance_summary")], "tool_use"),
                 Reply(["You have eighty percent."])])
    cid = await new_conv(ARJUN)
    events = await collect(ARJUN, cid, "attendance?")
    check_event_shapes(events)
    assert sources_of(events) == [{"kind": "college_record", "label": "College system: attendance"}]
    assert events[-1]["citations"] == sources_of(events)


async def test_CIT4_failed_tool_adds_no_college_record_source(env_factory):
    env_factory([Reply([], [tool_use("t", "get_attendance_summary")], "tool_use"), Reply(["Sorry."])],
                behaviour={"get_attendance_summary": (0, ToolResult('{"error": "down"}', True))})
    cid = await new_conv(ARJUN)
    events = await collect(ARJUN, cid, "attendance?")
    assert sources_of(events) == []
    assert not events[-1].get("citations")


async def test_CIT4_raising_tool_adds_no_college_record_source(env_factory):
    env_factory([Reply([], [tool_use("t", "get_attendance_summary")], "tool_use"), Reply(["Sorry."])],
                behaviour={"get_attendance_summary": (0, RuntimeError("x"))})
    cid = await new_conv(ARJUN)
    events = await collect(ARJUN, cid, "attendance?")
    assert sources_of(events) == []


async def test_CIT4_refused_tool_adds_no_college_record_source(env_factory):
    env_factory([Reply([], [tool_use("t", "approve_request")], "tool_use"), Reply(["No."])])
    cid = await new_conv(ARJUN)
    events = await collect(ARJUN, cid, "x")
    assert sources_of(events) == []


async def test_CIT4_one_source_per_distinct_tool_and_mixed_success(env_factory):
    env_factory([Reply([], [tool_use("a", "get_attendance_summary"),
                            tool_use("b", "get_attendance_summary"),
                            tool_use("c", "get_my_profile"), tool_use("d", "list_courses")],
                       "tool_use"), Reply(["Done."])],
                behaviour={"list_courses": (0, ToolResult('{"error": "x"}', True))})
    cid = await new_conv(ARJUN)
    events = await collect(ARJUN, cid, "x")
    labels = [s["label"] for s in sources_of(events) if s["kind"] == "college_record"]
    assert len(labels) == len(set(labels)) == 2
    assert all(l.startswith("College system") for l in labels)
    assert "College system: attendance" in labels


async def test_CIT4_no_tool_no_college_record(env_factory):
    env_factory([Reply(["Hello."])])
    cid = await new_conv(ARJUN)
    events = await collect(ARJUN, cid, "hi")
    assert sources_of(events) == [] and events[-1] == {"type": "done", "text": "Hello."}


async def test_CIT4_documents_and_records_together(env_factory, resolver):
    env_factory([Reply([], [tool_use("t", "get_attendance_summary")], "tool_use"),
                 Reply(["Both.\nSOURCES: D2:6.4.3"])])
    cid = await new_conv(ARJUN)
    events = await collect(ARJUN, cid, "x")
    kinds = [s["kind"] for s in sources_of(events)]
    assert sorted(kinds) == ["college_record", "document"]


async def test_CIT4_citations_do_not_leak_between_turns(env_factory):
    env_factory([Reply([], [tool_use("t", "get_attendance_summary")], "tool_use"),
                 Reply(["Eighty."]), Reply(["Just chatting."])])
    cid = await new_conv(ARJUN)
    await collect(ARJUN, cid, "attendance?")
    events = await collect(ARJUN, cid, "thanks")
    assert sources_of(events) == [] and "citations" not in events[-1]


# ------------------------------------------------------------------ CIT6

async def test_CIT6_stored_assistant_message_has_its_citations(env_factory, resolver):
    env_factory([Reply([], [tool_use("t", "get_attendance_summary")], "tool_use"),
                 Reply(["Done.\nSOURCES: D2:6.4.3"]), Reply(["Plain."])])
    cid = await new_conv(ARJUN)
    events = await collect(ARJUN, cid, "x")
    await collect(ARJUN, cid, "y")
    stored = await stored_citations(cid)
    assert stored[0] == sources_of(events)
    assert not stored[1]
    rows = await pool().fetch(
        "SELECT role, citations FROM messages WHERE conversation_id=$1 AND role='user'",
        __import__("uuid").UUID(cid))
    assert all(r["citations"] is None for r in rows)


async def test_CIT6_history_api_function_returns_citations_with_each_message(
        env_factory, resolver):
    env_factory([Reply(["Done.\nSOURCES: D2:6.4.3"])])
    cid = await new_conv(ARJUN)
    await collect(ARJUN, cid, "x")
    import uuid
    row = await pool().fetchrow(
        "SELECT citations FROM messages WHERE conversation_id=$1 AND role='assistant'",
        uuid.UUID(cid))
    v = row["citations"]
    v = json.loads(v) if isinstance(v, str) else v
    assert v[0]["ref"] == "D2:6.4.3"


async def test_CIT6_citations_survive_a_rebuild_without_reaching_the_model_as_text(
        env_factory, resolver):
    env = env_factory([Reply(["Done.\nSOURCES: D2:6.4.3"]), Reply(["Next."])])
    cid = await new_conv(ARJUN)
    await collect(ARJUN, cid, "x")
    evict_memory()
    await collect(ARJUN, cid, "y")
    assert "SOURCES" not in dump(env.client.calls[1]["messages"]).upper()
    assert_valid(env.client.calls[1]["messages"])


# ------------------------------------------------------------------ AGT7

def sent_text(call):
    return dump(call["messages"])


@pytest.mark.parametrize("turns", [9, 10, 11, 50])
async def test_AGT7_only_the_last_twenty_messages_reach_the_model_from_memory(env_factory, turns):
    env = env_factory(lambda kw, n: Reply([f"Reply{n:03d}."]))
    cid = await new_conv(VIS1)
    for i in range(turns):
        await collect(VIS1, cid, f"Question{i:03d}")
    await collect(VIS1, cid, "Final question")
    msgs = env.client.calls[-1]["messages"]
    assert 19 <= len(msgs) <= 20 or len(msgs) == 2 * turns + 1 and len(msgs) <= 20
    assert len(msgs) <= 20
    assert_valid(msgs)
    assert msgs[0]["role"] == "user"
    assert "Final question" in sent_text(env.client.calls[-1])
    if turns >= 11:
        assert "Question000" not in sent_text(env.client.calls[-1])
    assert f"Reply{turns - 1:03d}" in sent_text(env.client.calls[-1])


@pytest.mark.parametrize("stored", [19, 20, 21, 100])
async def test_AGT7_only_the_last_twenty_messages_reach_the_model_after_a_rebuild(
        env_factory, stored):
    env = env_factory([Reply(["Fresh."])])
    cid = await new_conv(VIS1)
    for i in range(stored):
        role = "user" if i % 2 == 0 else "assistant"
        await history.add_message(cid, role, f"{role}-msg-{i:03d}")
    evict_memory()
    await collect(VIS1, cid, "Final question")
    msgs = env.client.calls[0]["messages"]
    assert len(msgs) <= 20
    assert_valid(msgs)
    assert msgs[0]["role"] == "user"
    flat = sent_text(env.client.calls[0])
    assert "Final question" in flat
    assert f"-msg-{stored - 1:03d}" in flat, "the newest stored message must be kept"
    if stored >= 21:
        assert "-msg-000" not in flat
    if stored <= 19:
        assert "-msg-000" in flat, "nothing may be dropped below the limit"
    # nothing lost from storage
    assert len(await db_messages(cid)) == stored + 2


async def test_AGT7_window_with_tool_turns_keeps_tool_pairs_together(env_factory):
    def script(kw, n):
        if "tool_result" not in dump(kw["messages"][-1]):
            return Reply([], [tool_use(f"t{n}", "get_my_profile")], "tool_use")
        return Reply([f"Answer{n:03d}."])
    env = env_factory(script)
    cid = await new_conv(ARJUN)
    for i in range(25):
        await collect(ARJUN, cid, f"Q{i:03d}")
    for call in env.client.calls:
        assert len(call["messages"]) <= 20 + 3
        assert_valid(call["messages"])
    last = env.client.calls[-1]["messages"]
    assert last[0]["role"] == "user" and "tool_result" not in dump(last[0])


# DEFECT: the 20-message window is not a setting (Settings has no history_window), so it cannot be configured.
async def test_AGT7_window_is_a_setting(env_factory, monkeypatch):
    from app.config import get_settings
    s = get_settings()
    assert hasattr(s, "history_window"), "the history window must be a setting"
    monkeypatch.setattr(s, "history_window", 6)
    env = env_factory(lambda kw, n: Reply([f"R{n}."]))
    cid = await new_conv(VIS1)
    for i in range(8):
        await collect(VIS1, cid, f"Q{i}")
    assert len(env.client.calls[-1]["messages"]) <= 6
    assert_valid(env.client.calls[-1]["messages"])


async def test_AGT7_default_window_is_twenty():
    from app.config import get_settings
    assert getattr(get_settings(), "history_window", 20) == 20
