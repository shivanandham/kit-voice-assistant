"""Handbook edition and college date (DOC2, COR1), circulars (DOC3), sync (DOC6, REL1)."""

import asyncio
import hashlib
from datetime import date

import pytest
from app import db
from app.knowledge import rag, store

from tests.acc_knowledge_helpers import (
    BRO, CIR_ALL, CIR_UG, CSE, ECE, HB25, HB26, LN_CSE, MARK, TEACHER_CSE, TITLE, UNKNOWN, VISITOR,
    base_items, base_texts, blank_pdf, env, ids, make_pdf, pdf_env, seed, set_server_clock,  # noqa: F401
)

ORIG_LIST_REMOTE = store._list_remote
STUDENTS_AND_TEACHER = [pytest.param(CSE, id="student"), pytest.param(TEACHER_CSE, id="teacher")]


def handbooks(docs):
    return [d["id"] for d in docs if d["type"] == "handbook"]


def hb_item(doc_id, eff, title=None):
    return {"id": doc_id, "title": title or f"Handbook {doc_id}", "type": "handbook", "version": str(doc_id),
            "effective_date": eff, "applies_to": {}, "is_surprise": False}


async def put(items):
    for it in items:
        await store._upsert(it["id"], store._meta(it), f"Chapter 1 · Rules\n1.1 Scope\n1.1.1 Text of {it['id']} here.", f"h{it['id']}")


# ---------------------------------------------------------------- DOC2 / COR1: edition in force

EDITIONS = [hb_item(10, "2024-07-01"), hb_item(11, "2025-07-01"), hb_item(12, "2026-07-01")]


@pytest.mark.parametrize("n_editions", [1, 2, 3])
@pytest.mark.parametrize("today,expected_index", [
    ("2024-06-30", None), ("2024-07-01", 0), ("2024-07-02", 0), ("2025-06-30", 0), ("2025-07-01", 1),
    ("2025-12-31", 1), ("2026-06-30", 1), ("2026-07-01", 2), ("2026-07-02", 2), ("2031-01-01", 2),
])
@pytest.mark.parametrize("c", STUDENTS_AND_TEACHER)
async def test_DOC2_COR1_edition_in_force_is_latest_on_or_before_college_date(env, c, n_editions, today, expected_index):
    items = EDITIONS[:n_editions]
    await put(items)
    env.today = today
    want = None
    if expected_index is not None:
        # the latest edition among those supplied that is effective on or before the date
        eligible = [i for i in items if i["effective_date"] <= today]
        want = max(eligible, key=lambda i: i["effective_date"])["id"] if eligible else None
    assert handbooks(await store.applicable_documents(c)) == ([want] if want else [])


@pytest.mark.parametrize("server_day", [date(2000, 1, 1), date(2099, 12, 31), date(2025, 8, 1)])
@pytest.mark.parametrize("college_day,expected", [("2024-12-31", 10), ("2025-09-01", 11), ("2026-08-01", 12)])
async def test_COR1_server_clock_is_never_used_for_the_edition(env, monkeypatch, server_day, college_day, expected):
    await put(EDITIONS)
    set_server_clock(monkeypatch, server_day)
    env.today = college_day
    assert handbooks(await store.applicable_documents(CSE)) == [expected]


async def test_COR1_future_dated_edition_is_excluded_even_when_server_is_ahead(env, monkeypatch):
    await put([hb_item(10, "2025-07-01"), hb_item(11, "2027-07-01")])
    set_server_clock(monkeypatch, date(2030, 1, 1))
    env.today = "2026-01-01"
    assert handbooks(await store.applicable_documents(CSE)) == [10]


async def test_DOC2_only_future_edition_means_no_handbook(env):
    await put([hb_item(11, "2027-07-01")])
    assert handbooks(await store.applicable_documents(CSE)) == []


async def test_DOC2_superseded_edition_excluded_once_new_one_is_effective(env):
    await put(EDITIONS[1:])
    env.today = "2026-07-01"
    got = ids(await store.applicable_documents(CSE))
    assert 11 not in got and 12 in got


async def test_COR1_edition_changes_when_college_date_crosses_effective_date(env):
    await put(EDITIONS[1:])
    env.today = "2026-06-30"
    assert handbooks(await store.applicable_documents(CSE)) == [11]
    env.today = "2026-07-01"
    assert handbooks(await store.applicable_documents(CSE)) == [12]


@pytest.mark.parametrize("c", STUDENTS_AND_TEACHER)
async def test_COR1_date_endpoint_failing_keeps_last_known_college_date(env, monkeypatch, c):
    await put(EDITIONS[1:])
    env.today = "2026-06-30"
    assert handbooks(await store.applicable_documents(c)) == [11]
    env.today_fail = True
    set_server_clock(monkeypatch, date(2099, 1, 1))
    assert handbooks(await store.applicable_documents(c)) == [11]


@pytest.mark.parametrize("bad", ["garbage", "", None, "2026-13-45", "21/11/2026"])
async def test_COR1_unusable_date_answer_is_treated_as_failure_not_crash(env, monkeypatch, bad):
    await put(EDITIONS[1:])
    env.today = "2026-06-30"
    await store.applicable_documents(CSE)
    env.today = bad
    set_server_clock(monkeypatch, date(2099, 1, 1))
    assert handbooks(await store.applicable_documents(CSE)) == [11]


# DEFECT: with no college date ever seen, _college_date falls back to date.today() (server clock).
async def test_COR1_date_endpoint_failing_from_the_start_does_not_use_server_clock(env, monkeypatch):
    await put([hb_item(11, "2025-07-01"), hb_item(12, "2029-01-01")])
    env.today_fail = True
    set_server_clock(monkeypatch, date(2030, 1, 1))
    assert 12 not in ids(await store.applicable_documents(CSE))


async def test_COR1_date_endpoint_failing_from_the_start_never_raises(env):
    await put(EDITIONS)
    env.today_fail = True
    await store.applicable_documents(CSE)


async def test_COR1_visitor_result_does_not_depend_on_any_date(env, monkeypatch):
    await seed()
    env.today_fail = True
    set_server_clock(monkeypatch, date(1999, 1, 1))
    assert ids(await store.applicable_documents(VISITOR)) == [BRO]


async def test_DOC2_handbook_without_effective_date_is_not_in_force(env):
    await put([hb_item(11, "2025-07-01"), hb_item(12, None)])
    assert handbooks(await store.applicable_documents(CSE)) == [11]


async def test_DOC2_two_editions_same_date_choice_is_deterministic(env):
    await put([hb_item(11, "2025-07-01"), hb_item(12, "2025-07-01")])
    first = handbooks(await store.applicable_documents(CSE))
    assert len(first) == 1
    for _ in range(5):
        assert handbooks(await store.applicable_documents(CSE)) == first


async def test_DOC2_exactly_one_handbook_ever_delivered(env):
    await put(EDITIONS + [hb_item(13, "2025-01-01"), hb_item(14, "2023-01-01")])
    env.today = "2026-12-01"
    assert len(handbooks(await store.applicable_documents(CSE))) == 1


# ---------------------------------------------------------------- DOC3: circulars

def circ(doc_id, eff, applies=None):
    return {"id": doc_id, "title": f"Circular {doc_id}", "type": "circular", "version": f"{doc_id}/2026",
            "effective_date": eff, "applies_to": applies or {}, "is_surprise": False}


async def put_circ(items):
    for it in items:
        await store._upsert(it["id"], store._meta(it), f"Circular {it['id']}\nSubject: x\nBody {it['id']} full text.", f"c{it['id']}")


@pytest.mark.parametrize("c", STUDENTS_AND_TEACHER + [pytest.param(ECE, id="ece")])
@pytest.mark.parametrize("n", [1, 2, 5])
async def test_DOC3_every_circular_reaches_logged_in_users(env, c, n):
    items = [circ(30 + i, f"2026-0{i + 1}-01") for i in range(n)]
    await put_circ(items)
    assert {30 + i for i in range(n)} <= set(ids(await store.applicable_documents(c)))


@pytest.mark.parametrize("n", [1, 3])
async def test_AC1_DOC3_circulars_never_reach_visitors(env, n):
    await put_circ([circ(30 + i, "2026-01-01") for i in range(n)])
    assert await store.applicable_documents(VISITOR) == []


@pytest.mark.parametrize("applies", [{}, {"programmes": ["PG"]}, {"departments": ["MECH"]}, {"programmes": ["UG"]}])
async def test_DOC3_circular_is_given_whatever_its_applies_to(env, applies):
    await put_circ([circ(30, "2026-01-01", applies)])
    assert 30 in ids(await store.applicable_documents(CSE))


@pytest.mark.parametrize("c", STUDENTS_AND_TEACHER)
async def test_DOC3_circular_with_future_effective_date_is_still_given(env, c):
    # DOC3: circulars are always given in full; the model is told the effective date in the header.
    await put_circ([circ(30, "2027-05-01")])
    assert 30 in ids(await store.applicable_documents(c))


async def test_DOC3_circular_without_effective_date_is_given(env):
    await put_circ([circ(30, None)])
    assert 30 in ids(await store.applicable_documents(CSE))


async def test_DOC3_circular_of_inactive_status_is_not_given(env):
    await put_circ([circ(30, "2026-01-01")])
    await db.pool().execute("UPDATE documents SET active=false WHERE id=30")
    assert 30 not in ids(await store.applicable_documents(CSE))


# ---------------------------------------------------------------- selection ordering / determinism

@pytest.mark.parametrize("c", [CSE, TEACHER_CSE, VISITOR], ids=["student", "teacher", "visitor"])
async def test_DOC8_applicable_documents_order_is_deterministic_and_independent_of_insertion(env, c):
    await seed()
    first = ids(await store.applicable_documents(c))
    await db.pool().execute("TRUNCATE documents CASCADE")
    items = base_items()
    items.reverse()
    await seed(items)
    assert ids(await store.applicable_documents(c)) == first
    await db.pool().execute("UPDATE documents SET fetched_at = now() - (id || ' hours')::interval")
    assert ids(await store.applicable_documents(c)) == first


async def test_DOC1_applicable_documents_for_nothing_stored_is_empty(env):
    for c in (VISITOR, CSE, TEACHER_CSE):
        assert await store.applicable_documents(c) == []


async def test_DOC4_row_shape_has_the_columns_callers_need(env):
    await seed()
    for d in await store.applicable_documents(CSE):
        for key in ("id", "title", "type", "version", "effective_date", "applies_to", "text", "token_estimate",
                    "content_hash"):
            assert key in d
        assert isinstance(d["applies_to"], dict)


# ---------------------------------------------------------------- DOC6 / REL1: sync

async def sync_all():
    changed = await store.sync_documents()
    await store.wait_for_indexing()
    return changed


async def rows():
    return {r["id"]: r for r in await db.pool().fetch("SELECT * FROM documents")}


async def test_DOC6_first_sync_stores_everything_and_reports_change(env):
    assert await sync_all() is True
    r = await rows()
    assert set(r) == set(range(1, 10))
    assert all(x["active"] and x["text"] and x["token_estimate"] > 0 for x in r.values())
    assert r[HB26]["text"] == base_texts()[HB26]
    assert r[HB26]["content_hash"] == hashlib.sha256(env.college.files[HB26]).hexdigest() or r[HB26]["content_hash"]


async def test_DOC6_first_sync_indexes_every_document_with_embeddings(env):
    await sync_all()
    for doc_id in range(1, 10):
        n = await db.pool().fetchval("SELECT count(*) FROM doc_chunks WHERE document_id=$1", doc_id)
        e = await db.pool().fetchval("SELECT count(*) FROM doc_chunks WHERE document_id=$1 AND embedding IS NOT NULL", doc_id)
        assert n > 0 and e == n


async def chunk_snapshot():
    return [tuple(r) for r in await db.pool().fetch(
        "SELECT id, document_id, chunk_index, content, embedding::text FROM doc_chunks ORDER BY id")]


async def test_DOC6_second_sync_without_changes_reports_no_change_and_leaves_index_untouched(env):
    await sync_all()
    before = await chunk_snapshot()
    embed_calls = env.emb.calls
    assert await sync_all() is False
    assert await chunk_snapshot() == before          # same rows, same ids, same embeddings
    assert env.emb.calls == embed_calls              # nothing re-embedded


async def test_DOC6_repeated_unchanged_syncs_are_idempotent(env):
    await sync_all()
    before = {k: (v["text"], v["version"], v["content_hash"]) for k, v in (await rows()).items()}
    chunks = await db.pool().fetchval("SELECT count(*) FROM doc_chunks")
    for _ in range(3):
        await sync_all()
    assert {k: (v["text"], v["version"], v["content_hash"]) for k, v in (await rows()).items()} == before
    assert await db.pool().fetchval("SELECT count(*) FROM doc_chunks") == chunks


async def test_DOC6_new_document_is_picked_up_by_next_sync(env):
    await sync_all()
    before = await chunk_snapshot()
    env.college.items.append({"id": 50, "title": "Circular new", "type": "circular", "version": "50",
                              "effective_date": "2026-11-20", "applies_to": {}, "is_surprise": False})
    env.college.files[50] = b"Circular 50\nSubject: fresh newcirc text here."
    assert await sync_all() is True
    assert 50 in env.college.downloads
    after = await chunk_snapshot()
    assert all(r in after for r in before)       # existing documents' chunks and embeddings untouched
    assert 50 in ids(await store.applicable_documents(CSE))
    assert 50 not in ids(await store.applicable_documents(VISITOR))
    assert any("newcirc" in r["content"] for r in await rag.search("newcirc", CSE))


async def test_DOC6_changed_version_is_picked_up_and_old_text_and_chunks_replaced(env):
    await sync_all()
    env.college.set_item(CIR_ALL, version="13/2026-rev2")
    env.college.files[CIR_ALL] = b"Circular 13\nSubject: revised\nThe revisedtext circular now says two days."
    assert await sync_all() is True
    row = (await rows())[CIR_ALL]
    assert row["version"] == "13/2026-rev2" and "revisedtext" in row["text"]
    assert "circallmark" not in row["text"]
    old = await db.pool().fetchval("SELECT count(*) FROM doc_chunks WHERE document_id=$1 AND content ILIKE '%three working days%'", CIR_ALL)
    assert old == 0
    assert (await rag.search("revisedtext circular now says two days", CSE))[0]["version"] == "13/2026-rev2"
    assert all(r["document_title"] != TITLE[CIR_ALL] or "three working days" not in r["content"]
               for r in await rag.search("three working days proof", CSE, k=20))


@pytest.mark.parametrize("field,value", [("title", "Renamed doc"), ("effective_date", "2026-12-12"),
                                         ("applies_to", {"programmes": ["PG"]}), ("is_surprise", True)])
async def test_DOC6_changed_metadata_only_is_picked_up(env, field, value):
    await sync_all()
    env.college.set_item(CIR_UG, **{field: value})
    assert await sync_all() is True
    row = (await rows())[CIR_UG]
    key = {"effective_date": lambda v: date.fromisoformat(v)}.get(field, lambda v: v)
    got = row[field]
    if field == "applies_to" and isinstance(got, str):
        import json
        got = json.loads(got)
    assert got == key(value)


async def test_DOC6_changed_lesson_notes_department_changes_who_gets_them_at_next_sync(env):
    await sync_all()
    assert LN_CSE in ids(await store.applicable_documents(CSE))
    env.college.set_item(LN_CSE, applies_to={"departments": ["ECE"]})
    await sync_all()
    assert LN_CSE not in ids(await store.applicable_documents(CSE))
    assert LN_CSE in ids(await store.applicable_documents(ECE))


# DEFECT: same-metadata file changes are only detected on every 5th sync (CONTENT_RECHECK_EVERY).
async def test_DOC6_changed_content_under_same_metadata_is_picked_up_by_next_sync(env):
    await sync_all()
    env.college.files[BRO] = b"2. Programmes and fees\nThe brand new fee is 999999 per year for everybody."
    assert await sync_all() is True
    assert "999999" in (await rows())[BRO]["text"]
    assert "999999" in (await store.applicable_documents(VISITOR))[0]["text"]


async def test_DOC6_content_change_found_eventually_within_five_syncs(env):
    await sync_all()
    env.college.files[BRO] = b"2. Programmes and fees\nThe brand new fee is 999999 per year for everybody."
    for _ in range(5):
        await sync_all()
    assert "999999" in (await rows())[BRO]["text"]


async def test_DOC6_removed_document_becomes_inactive_and_disappears_for_every_audience(env):
    await sync_all()
    for doc in (BRO, CIR_UG, LN_CSE, HB26):
        env.college.remove(doc)
    assert await sync_all() is True
    r = await rows()
    assert all(r[d]["active"] is False for d in (BRO, CIR_UG, LN_CSE, HB26))
    for c in (VISITOR, CSE, ECE, TEACHER_CSE):
        got = ids(await store.applicable_documents(c))
        assert not {BRO, CIR_UG, LN_CSE, HB26} & set(got)
        titles = {x["document_title"] for x in await rag.search("fee rule circular lesson identity", c, k=20)}
        assert not titles & {TITLE[d] for d in (BRO, CIR_UG, LN_CSE, HB26)}


async def test_DOC6_removed_handbook_does_not_resurrect_superseded_edition_silently(env):
    await sync_all()
    env.college.remove(HB26)
    await sync_all()
    # the older edition is in force again on the college date; it must be the one delivered, not removed 2026
    assert handbooks(await store.applicable_documents(CSE)) == [HB25]


async def test_DOC6_removed_then_relisted_document_is_active_again(env):
    await sync_all()
    item = next(i for i in env.college.items if i["id"] == CIR_UG)
    env.college.remove(CIR_UG)
    await sync_all()
    assert CIR_UG not in ids(await store.applicable_documents(CSE))
    env.college.items.append(item)
    assert await sync_all() is True
    assert CIR_UG in ids(await store.applicable_documents(CSE))
    assert any(r["document_title"] == TITLE[CIR_UG] for r in await rag.search("circugmark", CSE))


async def test_DOC6_everything_removed_leaves_nothing_for_anyone(env):
    await sync_all()
    env.college.items = []
    await sync_all()
    for c in (VISITOR, CSE, TEACHER_CSE):
        assert await store.applicable_documents(c) == []
        assert await rag.search("fee", c) == []


@pytest.mark.parametrize("doc,typ,audiences", [
    (BRO, "brochure", {"visitor", "student"}), (CIR_ALL, "circular", {"student"}),
    (LN_CSE, "lesson_notes", {"student"}), (HB26, "handbook", {"student"}),
])
async def test_DOC6_surprise_document_is_routed_by_type_like_any_other(env, doc, typ, audiences):
    env.college.set_item(doc, is_surprise=True)
    await sync_all()
    assert (await rows())[doc]["is_surprise"] is True
    assert (doc in ids(await store.applicable_documents(VISITOR))) == ("visitor" in audiences)
    assert (doc in ids(await store.applicable_documents(CSE))) == ("student" in audiences)


async def test_DOC6_new_surprise_document_arriving_later_is_picked_up_and_never_given_to_visitors(env):
    await sync_all()
    env.college.items.append({"id": 60, "title": "Surprise circular", "type": "circular", "version": "s1",
                              "effective_date": "2026-11-21", "applies_to": {}, "is_surprise": True})
    env.college.files[60] = b"Circular 60\nSubject: surprise exam moved to tomorrow morning."
    await sync_all()
    assert 60 in ids(await store.applicable_documents(CSE))
    assert 60 not in ids(await store.applicable_documents(VISITOR))
    assert await rag.search("surprise exam", VISITOR) == [] or all("surprise" not in r["content"].lower() for r in await rag.search("surprise exam", VISITOR))


@pytest.mark.parametrize("failing", [[BRO], [HB26], [BRO, HB26, CIR_UG], list(range(1, 9))])
async def test_REL1_DOC6_download_failure_for_some_documents_does_not_lose_the_others(env, failing):
    env.college.download_fail = set(failing)
    await sync_all()
    r = await rows()
    for doc in range(1, 10):
        assert (doc in r) == (doc not in failing)
    env.college.download_fail = set()
    assert await sync_all() is True
    assert set(await rows()) == set(range(1, 10))


async def test_REL1_download_failure_of_changed_document_keeps_the_old_text_active(env):
    await sync_all()
    env.college.set_item(CIR_ALL, version="v2")
    env.college.files[CIR_ALL] = b"Circular 13 changed text body here."
    env.college.download_fail = {CIR_ALL}
    await sync_all()
    row = (await rows())[CIR_ALL]
    assert row["active"] and "circallmark" in row["text"]
    assert CIR_ALL in ids(await store.applicable_documents(CSE))
    env.college.download_fail = set()
    await sync_all()
    assert "changed text body" in (await rows())[CIR_ALL]["text"]


async def test_REL1_list_endpoint_failure_keeps_existing_documents_serving(env):
    await sync_all()
    before = {c.owner_key: ids(await store.applicable_documents(c)) for c in (VISITOR, CSE, TEACHER_CSE)}
    env.college.list_fail = True
    try:
        result = await store.sync_documents()
    except Exception:
        result = False
    assert result is False
    for c in (VISITOR, CSE, TEACHER_CSE):
        assert ids(await store.applicable_documents(c)) == before[c.owner_key]
    assert all(r["active"] for r in (await rows()).values())
    env.college.list_fail = False
    assert await sync_all() is False


async def test_REL1_list_failure_on_empty_database_leaves_assistant_working(env):
    env.college.list_fail = True
    try:
        await store.sync_documents()
    except Exception:
        pass
    assert await store.applicable_documents(CSE) == []
    assert await rag.search("anything", CSE) == []


# DEFECT: sync_documents raises when the listing fails (only the loop catches it).
async def test_REL1_DOC6_sync_documents_itself_does_not_raise_when_listing_fails(env):
    env.college.list_fail = True
    assert await store.sync_documents() is False


async def test_REL1_list_failure_does_not_deactivate_documents(env):
    await sync_all()
    env.college.list_fail = True
    try:
        await store.sync_documents()
    except Exception:
        pass
    assert all(r["active"] for r in (await rows()).values())


# DEFECT: one listing entry with a bad effective_date makes _meta raise and aborts the whole sync.
async def test_REL1_listing_with_one_malformed_entry_does_not_lose_the_others(env):
    env.college.items.insert(0, {"id": 70, "title": "Bad date", "type": "circular", "version": "1",
                                 "effective_date": "not-a-date", "applies_to": {}, "is_surprise": False})
    env.college.files[70] = b"Circular 70\nSubject: broken date entry."
    try:
        await store.sync_documents()
    except Exception:
        pass
    await store.wait_for_indexing()
    assert set(range(1, 10)) <= set(await rows())


@pytest.mark.parametrize("payload", [b"this is not a pdf at all", b"%PDF-1.4\ngarbage garbage", b"", b"\x00" * 64])
async def test_REL1_DOC6_corrupt_pdf_does_not_break_sync_or_lose_other_documents(pdf_env, payload):
    for i in pdf_env.college.items:
        pdf_env.college.files[i["id"]] = make_pdf(base_texts()[i["id"]].splitlines())
    pdf_env.college.files[CIR_UG] = payload
    await sync_all()
    r = await rows()
    assert set(range(1, 10)) - {CIR_UG} <= set(r)
    got = ids(await store.applicable_documents(CSE))
    assert CIR_UG not in got
    pdf_env.college.files[CIR_UG] = make_pdf(base_texts()[CIR_UG].splitlines())
    assert await sync_all() is True
    assert CIR_UG in ids(await store.applicable_documents(CSE))


async def test_DOC6_real_pdf_text_is_extracted_and_served(pdf_env):
    for i in pdf_env.college.items:
        pdf_env.college.files[i["id"]] = make_pdf(base_texts()[i["id"]].splitlines())
    await sync_all()
    row = (await rows())[BRO]
    assert "brochuremark tuition fee" in row["text"]
    assert [r["document_title"] for r in await rag.search("brochuremark tuition fee", VISITOR)][:1] == [TITLE[BRO]]


# DEFECT: a PDF with no text is stored as an empty-text document and served to users.
async def test_REL1_empty_pdf_neither_breaks_sync_nor_is_served_as_a_document(pdf_env):
    for i in pdf_env.college.items:
        pdf_env.college.files[i["id"]] = make_pdf(base_texts()[i["id"]].splitlines())
    pdf_env.college.files[CIR_UG] = blank_pdf()
    await sync_all()
    assert set(range(1, 10)) - {CIR_UG} <= set(await rows())
    served = {d["id"]: d for d in await store.applicable_documents(CSE)}
    assert CIR_UG not in served or served[CIR_UG]["text"].strip()


# DEFECT: a document whose text is blank is stored and served as an empty document.
async def test_REL1_whitespace_only_text_never_served_as_empty_document(env):
    env.college.files[CIR_UG] = b"   \n  \n"
    await sync_all()
    for d in await store.applicable_documents(CSE):
        assert d["text"].strip()


async def test_REL1_very_large_document_syncs_and_is_searchable(env):
    big = "Chapter 1 · Huge rules\n1.1 Everything\n" + "\n".join(
        f"1.1.{i} Rule number {i} says bigword{i % 50} must be followed by everyone concerned." for i in range(1, 6001))
    env.college.files[HB26] = big.encode()
    await sync_all()
    row = (await rows())[HB26]
    assert len(row["text"]) == len(big) and row["token_estimate"] > 20000
    n = await db.pool().fetchval("SELECT count(*) FROM doc_chunks WHERE document_id=$1", HB26)
    assert n > 50
    assert await rag.search("bigword7 rule", CSE)


async def test_DOC6_token_estimate_grows_with_text_and_is_positive(env):
    await sync_all()
    r = await rows()
    assert all(x["token_estimate"] > 0 for x in r.values())
    longest = max(r.values(), key=lambda x: len(x["text"]))
    shortest = min(r.values(), key=lambda x: len(x["text"]))
    assert longest["token_estimate"] >= shortest["token_estimate"]


@pytest.mark.parametrize("fail_for", ["all-syncs", "first-sync-only"])
async def test_REL1_DOC6_embedding_failures_do_not_fail_the_sync(env, fail_for):
    env.emb.fail = True
    assert await store.sync_documents() is True
    await store.wait_for_indexing()
    assert set(await rows()) == set(range(1, 10))
    n = await db.pool().fetchval("SELECT count(*) FROM doc_chunks")
    assert n > 0
    assert (await rag.search("tuition fee", VISITOR))[0]["document_title"] == TITLE[BRO]
    if fail_for == "first-sync-only":
        env.emb.fail = False
        monkey_reset_cooldown()
        for _ in range(6):
            await sync_all()
        e = await db.pool().fetchval("SELECT count(*) FROM doc_chunks WHERE embedding IS NULL")
        assert e == 0


def monkey_reset_cooldown():
    rag._embed_down_until = 0.0


async def test_REL1_embedding_returning_wrong_shape_does_not_fail_the_sync(env):
    async def bad(texts, task_type):
        return [[0.1, 0.2]]

    rag._embed = bad
    assert await store.sync_documents() is True
    await store.wait_for_indexing()
    assert set(await rows()) == set(range(1, 10))


# DEFECT: overlapping syncs can index the same document twice at once, leaving duplicate chunks (intermittent).
async def test_DOC6_concurrent_syncs_do_not_duplicate_or_crash(env):
    results = await asyncio.gather(store.sync_documents(), store.sync_documents(), store.sync_documents())
    await store.wait_for_indexing()
    assert sorted(results) == [False, False, True] or True in results
    assert await db.pool().fetchval("SELECT count(*) FROM documents") == 9
    per = await db.pool().fetch("SELECT document_id, chunk_index, count(*) c FROM doc_chunks GROUP BY 1,2 HAVING count(*) > 1")
    assert per == []


@pytest.mark.parametrize("wrapper", ["documents", "items"])
async def test_DOC6_listing_wrapped_in_object_is_accepted(monkeypatch, wrapper):
    import app.kit.client as client

    class K:
        async def documents(self):
            return {wrapper: base_items()}

    monkeypatch.setattr(client, "KitClient", lambda *a, **k: K())
    assert len(await ORIG_LIST_REMOTE()) == 9


# ---------------------------------------------------------------- the loop (REL1)

@pytest.fixture
def fast_loop(monkeypatch):
    real_sleep = asyncio.sleep

    async def quick(_s, *a, **k):
        await real_sleep(0.001)

    monkeypatch.setattr(asyncio, "sleep", quick)
    yield real_sleep


async def test_REL1_loop_survives_repeated_failures_and_recovers(env, fast_loop):
    env.college.list_fail = True
    await store.start_sync_loop()
    for _ in range(200):
        await fast_loop(0.005)
        if env.college.list_calls >= 4:
            break
    assert env.college.list_calls >= 4
    assert store._loop_task is not None and not store._loop_task.done()
    assert await store.applicable_documents(CSE) == []
    env.college.list_fail = False
    for _ in range(400):
        await fast_loop(0.005)
        if await db.pool().fetchval("SELECT count(*) FROM documents") == 9:
            break
    await store.stop_sync_loop()
    assert await db.pool().fetchval("SELECT count(*) FROM documents") == 9


async def test_REL1_loop_survives_download_and_embedding_and_extraction_errors(env, fast_loop, monkeypatch):
    env.emb.fail = True
    env.college.download_fail = {BRO}

    def boom(b):
        raise ValueError("bad pdf")

    monkeypatch.setattr(store, "extract_text", boom)
    await store.start_sync_loop()
    for _ in range(40):
        await fast_loop(0.005)
    assert not store._loop_task.done()
    await store.stop_sync_loop()


async def test_REL1_start_twice_runs_one_loop_and_stop_ends_it(env, fast_loop):
    await store.start_sync_loop()
    first = store._loop_task
    await store.start_sync_loop()
    assert store._loop_task is first
    await store.stop_sync_loop()
    assert first.done() and store._loop_task is None
    await store.stop_sync_loop()  # stopping again is harmless


async def test_REL1_loop_runs_first_sync_immediately(env, fast_loop):
    await store.start_sync_loop()
    for _ in range(200):
        await fast_loop(0.005)
        if await db.pool().fetchval("SELECT count(*) FROM documents") == 9:
            break
    await store.stop_sync_loop()
    assert await db.pool().fetchval("SELECT count(*) FROM documents") == 9


async def test_REL1_loop_can_be_restarted_after_stop(env, fast_loop):
    await store.start_sync_loop()
    await store.stop_sync_loop()
    await store.start_sync_loop()
    assert store._loop_task is not None and not store._loop_task.done()
    await store.stop_sync_loop()


# ---------------------------------------------------------------- REL2

async def test_REL2_no_real_network_during_a_full_sync_and_search(env):
    await sync_all()
    await rag.search("fee", CSE)
    # the autouse network guard fails the test on any non-local connection; reaching here proves none
