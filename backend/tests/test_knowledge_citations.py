"""Citations: chunk metadata (CIT1), ref tags in the prompt text, and ref resolution (CIT3)."""

import re
from datetime import date

import pytest
from app import db
from app.knowledge import mode, rag, store, structure

from tests.test_knowledge_fakes import (
    CSE_STUDENT,
    ECE_STUDENT,
    ITEMS,
    VISITOR,
    college_date,
    fake_vector,
)

PAGED_HANDBOOK = "\n".join([
    "\f1", "Handbook front page text that introduces the rules.",
    "\f2", "Chapter 6 · Leave and on-duty rules",
    "6.4 On-duty (OD)",
    "6.4.1 On-duty may be approved for activities on behalf of the college.",
    "\f3", "6.4.3 Within 5 working days after the event, the student must submit proof",
    "of participation to the Class Advisor.",
    "6.4.4 If the proof is not submitted in time, the request is rejected.",
    "6.5 Who approves requests",
    "6.5.1 Requests are approved by the Class Advisor of the student's class.",
])
CIRCULAR_TEXT = "\n".join([
    "\f1", "Circular 12/2026",
    "Subject: Change to the time allowed for submitting on-duty proof (Rule 6.4.3)",
    "1. A B.Tech student must submit on-duty proof within 3 working days after the event.",
])


@pytest.fixture
async def indexed(monkeypatch):
    async def embed(texts, task_type):
        return [fake_vector(t) for t in texts]

    monkeypatch.setattr(rag, "_embed", embed)
    monkeypatch.setattr(rag, "_embed_down_until", 0.0)
    monkeypatch.setattr(store, "_college_date", college_date(date(2026, 11, 21)))
    texts = {1: "6.4 On-duty\n6.4.3 Seven working days (old edition).", 2: PAGED_HANDBOOK,
             3: "2. Programmes and fees\nHostel fee per year 60000 for every programme.",
             4: "1.3 Keys\nA foreign key refers to the primary key of another table.",
             5: CIRCULAR_TEXT}
    for item in ITEMS:
        await store._upsert(item["id"], store._meta(item), texts[item["id"]], f"h{item['id']}")
        await rag.index_document(item["id"])
    mode._cache.clear()


async def _chunk_rows(doc_id: int) -> list:
    return await db.pool().fetch(
        "SELECT * FROM doc_chunks WHERE document_id=$1 ORDER BY chunk_index", doc_id)


async def test_cit1_every_chunk_has_ref_section_heading_and_pages(indexed):
    rows = await db.pool().fetch("SELECT * FROM doc_chunks")
    assert rows and all(r["ref"] and r["heading"] and r["page_start"] >= 1
                        and r["page_end"] >= r["page_start"] for r in rows)
    on_duty = next(r for r in await _chunk_rows(2) if r["section"] == "6.4")
    assert on_duty["ref"] == "D2:6.4" and on_duty["heading"].endswith("6.4 On-duty (OD)")
    assert (on_duty["page_start"], on_duty["page_end"]) == (2, 3)


async def test_cit1_refs_unique_within_a_document_and_stable_across_reindexing(indexed):
    before = [(r["ref"], r["content"]) for r in await _chunk_rows(2)]
    assert len({ref for ref, _ in before}) == len(before)
    await rag.index_document(2)
    assert [(r["ref"], r["content"]) for r in await _chunk_rows(2)] == before
    assert await db.pool().fetchval("SELECT count(*) FROM doc_chunks WHERE embedding IS NULL") == 0


async def test_cit1_unnumbered_chunk_uses_chunk_index_and_circular_uses_its_number(indexed):
    front = (await _chunk_rows(2))[0]
    assert front["ref"] == "D2:c0" and front["section"] is None and front["page_start"] == 1
    circular = await _chunk_rows(5)
    assert [r["ref"] for r in circular] == ["D5:12/2026"] and circular[0]["section"] == "12/2026"


def test_cit1_long_section_chunks_get_unique_suffixed_refs():
    body = "\n".join(f"6.4.{i} " + "word " * 60 for i in range(1, 15))
    s = structure.analyse(9, "T", "1", "handbook", "6.4 On-duty\n" + body)
    refs = [c.ref for c in s.chunks]
    assert refs[0] == "D9:6.4" and refs[1] == "D9:6.4#2" and len(set(refs)) == len(refs)
    assert all(c.section == "6.4" for c in s.chunks)


def test_cit1_page_marks_from_extraction_are_not_in_chunk_content():
    s = structure.analyse(2, "H", "1", "handbook", PAGED_HANDBOOK)
    assert all("\f" not in c.content for c in s.chunks)
    assert s.refs["D2:6.4.3"] == {"section": "6.4.3", "heading": "Chapter 6 · Leave and on-duty rules"
                                  " › 6.4 On-duty (OD)", "page_start": 3, "page_end": 3}


async def test_cit1_search_results_carry_citation_fields(indexed):
    results = await rag.search("submit on-duty proof working days", CSE_STUDENT)
    top = {r["ref"]: r for r in results}
    assert "D5:12/2026" in top and "D2:6.4" in top
    r = top["D2:6.4"]
    assert (r["document_id"], r["document_type"], r["section"], r["effective_date"],
            r["page_start"], r["page_end"]) == (2, "handbook", "6.4", "2026-07-01", 2, 3)


async def test_cit1_rendered_text_tags_every_section_and_rule_and_is_byte_stable(indexed):
    text = (await mode.knowledge_block(CSE_STUDENT)).text
    for ref in ("D2:c0", "D2:6.4", "D2:6.4.1", "D2:6.4.3", "D2:6.4.4", "D2:6.5", "D2:6.5.1",
                "D5:12/2026", "D4:1.3", "D3:2"):
        assert f"[ref {ref}]\n" in text
    assert text.index("[ref D2:6.4.3]\n6.4.3 Within 5") > text.index("[ref D2:6.4]\n6.4 On-duty")
    assert "\f" not in text
    mode._cache.clear()
    assert (await mode.knowledge_block(CSE_STUDENT)).text == text


async def test_cit1_unchunked_document_still_renders_in_full(indexed):
    await db.pool().execute("DELETE FROM doc_chunks")
    mode._cache.clear()
    text = (await mode.knowledge_block(CSE_STUDENT)).text
    assert "of participation to the Class Advisor." in text and "[ref D2:6.4.3]" in text


async def test_cit3_resolve_refs_returns_details_in_input_order_without_duplicates(indexed):
    got = await rag.resolve_refs(["D5:12/2026", "D2:6.4.3", "D2:6.4.3", "D2:6.4"], CSE_STUDENT)
    assert [g["ref"] for g in got] == ["D5:12/2026", "D2:6.4.3", "D2:6.4"]
    assert got[1] == {"ref": "D2:6.4.3", "document_title": "KIT Student Handbook 2026",
                      "document_type": "handbook", "version": "2026.1",
                      "effective_date": "2026-07-01", "section": "6.4.3",
                      "heading": "Chapter 6 · Leave and on-duty rules › 6.4 On-duty (OD)",
                      "page_start": 3, "page_end": 3}


@pytest.mark.parametrize("ref", ["D2:9.9.9", "D99:c0", "nonsense", "", None, "D2:", "d2:6.4",
                                 "D2:6.4.3 extra", "D1:6.4.3"])
async def test_cit3_malformed_unknown_and_superseded_refs_are_dropped(indexed, ref):
    assert await rag.resolve_refs([ref], CSE_STUDENT) == []


async def test_cit3_visitor_only_resolves_brochure_refs(indexed):
    got = await rag.resolve_refs(["D2:6.4.3", "D5:12/2026", "D4:1.3", "D3:2"], VISITOR)
    assert [g["ref"] for g in got] == ["D3:2"]


async def test_cit3_student_cannot_cite_another_departments_lesson_notes(indexed):
    assert await rag.resolve_refs(["D4:1.3"], ECE_STUDENT) == []
    assert [g["ref"] for g in await rag.resolve_refs(["D4:1.3"], CSE_STUDENT)] == ["D4:1.3"]


async def test_cit3_every_tag_in_the_rendered_text_resolves(indexed):
    for ctx in (CSE_STUDENT, VISITOR):
        refs = re.findall(r"^\[ref (\S+)\]$", (await mode.knowledge_block(ctx)).text, re.MULTILINE)
        assert refs and [g["ref"] for g in await rag.resolve_refs(refs, ctx)] == refs
