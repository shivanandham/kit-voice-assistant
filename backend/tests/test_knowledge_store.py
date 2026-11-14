"""Document sync, text cleanup and selection rules."""

import asyncio
import copy
from datetime import date

import pytest
from app import db
from app.knowledge import rag, store

from tests.test_knowledge_fakes import (
    CSE_STUDENT,
    ECE_STUDENT,
    ITEMS,
    TEACHER,
    TEXTS,
    VISITOR,
    college_date,
    insert_documents,
)

TODAY = date(2026, 11, 21)


# ---------------------------------------------------------------- text cleanup

def test_clean_page_drops_running_header_and_joins_split_headings():
    raw = ("KIT Student Handbook 2025 · Edition 2025.1 · Superseded · page 3 of 11\n"
           "Chapter\n6\n·\nLeave\nrules\n6.4\nOn-duty\n(OD)\n"
           "6.4.3 Within 5 working days after the event, the student must submit proof of\n"
           "participation.\n")
    assert store.clean_page(raw) == [
        "Chapter 6 · Leave rules",
        "6.4 On-duty (OD)",
        "6.4.3 Within 5 working days after the event, the student must submit proof of",
        "participation.",
    ]


def test_clean_page_keeps_rule_references_inline():
    assert store.clean_page("Subject:\nChange\nto\n(Rule\n6.4.3)\nBody text here.") == [
        "Subject: Change to (Rule 6.4.3)", "Body text here."]


# ---------------------------------------------------------------- selection rules

async def _docs():
    await insert_documents()
    return await store.active_documents()


def _ids(docs):
    return [d["id"] for d in docs]


async def test_visitor_gets_brochure_only():
    docs = await _docs()
    assert _ids(store.select_documents(docs, "visitor", None, TODAY)) == [3]


async def test_btech_student_gets_current_handbook_circulars_own_lesson_notes_and_brochure():
    docs = await _docs()
    assert _ids(store.select_documents(docs, "student", CSE_STUDENT.user, TODAY)) == [2, 5, 4, 3]
    assert _ids(store.select_documents(docs, "student", ECE_STUDENT.user, TODAY)) == [2, 5, 3]


async def test_teacher_gets_all_lesson_notes():
    docs = await _docs()
    teacher = {"department": "ECE"}
    assert _ids(store.select_documents(docs, "teacher", teacher, TODAY)) == [2, 5, 4, 3]


async def test_handbook_edition_follows_college_date():
    docs = await _docs()
    before = store.select_documents(docs, "student", CSE_STUDENT.user, date(2026, 6, 30))
    on = store.select_documents(docs, "student", CSE_STUDENT.user, date(2026, 7, 1))
    assert [d["id"] for d in before if d["type"] == "handbook"] == [1]
    assert [d["id"] for d in on if d["type"] == "handbook"] == [2]


async def test_superseded_and_inactive_documents_excluded(monkeypatch):
    await insert_documents()
    await db.pool().execute("UPDATE documents SET active=false WHERE id=5")
    monkeypatch.setattr(store, "_college_date", college_date(TODAY))
    ids = _ids(await store.applicable_documents(CSE_STUDENT))
    assert 1 not in ids and 5 not in ids and ids == [2, 4, 3]


async def test_applicable_documents_visitor_never_sees_handbook(monkeypatch):
    await insert_documents()
    monkeypatch.setattr(store, "_college_date", college_date(TODAY))
    assert _ids(await store.applicable_documents(VISITOR)) == [3]
    assert _ids(await store.applicable_documents(TEACHER)) == [2, 5, 4, 3]


# ---------------------------------------------------------------- sync and change detection

@pytest.fixture
def remote(monkeypatch):
    """A fake college listing; records downloads and indexing calls."""
    state = {"items": copy.deepcopy(ITEMS), "files": {i: TEXTS[i].encode() for i in TEXTS},
             "downloads": [], "indexed": [], "down": False}

    async def list_remote():
        if state["down"]:
            raise ConnectionError("college API down")
        return copy.deepcopy(state["items"])

    async def download(doc_id):
        state["downloads"].append(doc_id)
        return state["files"][doc_id]

    async def index_document(doc_id):
        state["indexed"].append(doc_id)

    async def embed_missing(document_id=None, *, skip_if_busy=False):
        return None

    monkeypatch.setattr(store, "_list_remote", list_remote)
    monkeypatch.setattr(store, "_download", download)
    monkeypatch.setattr(store, "extract_text", lambda pdf: pdf.decode())
    monkeypatch.setattr(rag, "index_document", index_document)
    monkeypatch.setattr(rag, "embed_missing", embed_missing)
    return state


async def test_first_sync_stores_and_indexes_everything(remote):
    assert await store.sync_documents() is True
    await store.wait_for_indexing()
    rows = await db.pool().fetch("SELECT id, token_estimate, active FROM documents ORDER BY id")
    assert [r["id"] for r in rows] == [1, 2, 3, 4, 5]
    assert all(r["active"] and r["token_estimate"] > 0 for r in rows)
    assert sorted(remote["indexed"]) == [1, 2, 3, 4, 5]


async def test_unchanged_documents_are_not_stored_or_indexed_again(remote):
    await store.sync_documents()
    await store.wait_for_indexing()
    await db.pool().execute(
        "INSERT INTO doc_chunks (document_id, chunk_index, heading, content, ref) "
        "SELECT id, 0, 'h', 'c', 'D' || id || ':c0' FROM documents")
    remote["downloads"].clear()
    remote["indexed"].clear()
    assert await store.sync_documents() is False
    await store.wait_for_indexing()
    assert sorted(remote["downloads"]) == [1, 2, 3, 4, 5]     # content is compared every sync
    assert remote["indexed"] == []


async def test_new_and_changed_documents_are_stored_and_indexed(remote):
    await store.sync_documents()
    await store.wait_for_indexing()
    await db.pool().execute(
        "INSERT INTO doc_chunks (document_id, chunk_index, heading, content, ref) "
        "SELECT id, 0, 'h', 'c', 'D' || id || ':c0' FROM documents")
    remote["downloads"].clear()
    remote["indexed"].clear()
    remote["items"][4]["version"] = "12/2026-rev2"
    remote["files"][5] = b"Circular revised: proof within 2 working days."
    remote["items"].append({"id": 6, "title": "Circular 13/2026", "type": "circular",
                            "version": "13/2026", "effective_date": "2026-11-20",
                            "applies_to": {}, "is_surprise": True})
    remote["files"][6] = b"New circular text."
    assert await store.sync_documents() is True
    await store.wait_for_indexing()
    assert sorted(remote["indexed"]) == [5, 6]
    row = await db.pool().fetchrow("SELECT version, text FROM documents WHERE id=5")
    assert row["version"] == "12/2026-rev2" and "2 working days" in row["text"]


async def test_content_change_under_same_metadata_detected_by_next_sync(remote):
    await store.sync_documents()
    await store.wait_for_indexing()
    await db.pool().execute(
        "INSERT INTO doc_chunks (document_id, chunk_index, heading, content, ref) "
        "SELECT id, 0, 'h', 'c', 'D' || id || ':c0' FROM documents")
    remote["files"][3] = b"Brochure with new fees."
    remote["indexed"].clear()
    assert await store.sync_documents() is True
    await store.wait_for_indexing()
    assert remote["indexed"] == [3]
    assert await db.pool().fetchval("SELECT text FROM documents WHERE id=3") == "Brochure with new fees."


async def test_vanished_document_marked_inactive(remote):
    await store.sync_documents()
    remote["items"] = [i for i in remote["items"] if i["id"] != 4]
    assert await store.sync_documents() is True
    assert await db.pool().fetchval("SELECT active FROM documents WHERE id=4") is False


async def test_failing_indexing_never_breaks_sync(remote, monkeypatch):
    async def broken(doc_id):
        raise RuntimeError("embedding service down")

    monkeypatch.setattr(rag, "index_document", broken)
    assert await store.sync_documents() is True
    await store.wait_for_indexing()
    assert await db.pool().fetchval("SELECT count(*) FROM documents WHERE active") == 5


async def test_sync_loop_survives_college_api_down(remote, monkeypatch):
    remote["down"] = True
    assert await store.sync_documents() is False
    await store.start_sync_loop()
    await asyncio.sleep(0.05)
    assert store._loop_task is not None and not store._loop_task.done()
    await store.stop_sync_loop()
    assert await db.pool().fetchval("SELECT count(*) FROM documents") == 0


async def test_listing_failure_keeps_stored_documents(remote):
    await store.sync_documents()
    remote["down"] = True
    assert await store.sync_documents() is False
    assert await db.pool().fetchval("SELECT count(*) FROM documents WHERE active") == 5


async def test_malformed_entry_is_skipped_and_the_rest_kept(remote):
    remote["items"].insert(0, {"id": 70, "title": "Bad", "type": "circular", "version": "1",
                               "effective_date": "not-a-date", "applies_to": {}})
    remote["items"].insert(1, {"title": "no id"})
    assert await store.sync_documents() is True
    ids = [r["id"] for r in await db.pool().fetch("SELECT id FROM documents ORDER BY id")]
    assert ids == [1, 2, 3, 4, 5]


async def test_blank_document_is_neither_stored_nor_served(remote, monkeypatch):
    remote["files"][5] = b"  \n \n"
    await store.sync_documents()
    assert await db.pool().fetchval("SELECT count(*) FROM documents WHERE id=5") == 0
    await db.pool().execute("UPDATE documents SET text='   ' WHERE id=2")
    monkeypatch.setattr(store, "_college_date", college_date(TODAY))
    assert _ids(await store.applicable_documents(CSE_STUDENT)) == [4, 3]


@pytest.mark.parametrize("role", ["admin", "guest", "", None, "Student", "TEACHER"])
async def test_unrecognised_roles_are_treated_as_visitors(role):
    docs = await _docs()
    assert _ids(store.select_documents(docs, role, {"department": "CSE"}, TODAY)) == [3]


async def test_unknown_college_date_never_uses_the_server_clock():
    docs = await _docs()
    chosen = store.select_documents(docs, "student", CSE_STUDENT.user, None)
    assert [d["id"] for d in chosen if d["type"] == "handbook"] == [2]   # bounded by the circular
    handbooks_only = [d for d in docs if d["type"] == "handbook"]
    assert store.select_documents(handbooks_only, "student", CSE_STUDENT.user, None) == []
