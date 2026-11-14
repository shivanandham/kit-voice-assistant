"""Chunking, indexing and search (embedding calls mocked)."""

from datetime import date
from types import SimpleNamespace

import pytest
from app import db
from app.config import get_settings
from app.knowledge import rag, store

from tests.test_knowledge_fakes import (
    CSE_STUDENT,
    ECE_STUDENT,
    VISITOR,
    college_date,
    fake_vector,
    insert_documents,
)


@pytest.fixture
def embedder(monkeypatch):
    """Deterministic embeddings; flip state['fail'] to simulate a Gemini outage."""
    state = {"fail": False, "calls": 0}

    async def embed(texts, task_type):
        state["calls"] += 1
        if state["fail"]:
            raise RuntimeError("gemini down")
        return [fake_vector(t) for t in texts]

    monkeypatch.setattr(rag, "_embed", embed)
    monkeypatch.setattr(rag, "_embed_down_until", 0.0)
    monkeypatch.setattr(store, "_college_date", college_date(date(2026, 11, 21)))
    return state


async def _index_all():
    await insert_documents()
    for doc_id in range(1, 6):
        await rag.index_document(doc_id)


def test_chunks_follow_sections_and_carry_title_and_heading():
    text = ("Chapter 6 · Leave rules\n6.1 Personal leave\n6.1.1 Apply 24 hours ahead.\n"
            "6.4 On-duty (OD)\n6.4.3 Within 5 working days after the event, submit proof.")
    chunks = rag.chunk_document("KIT Handbook", "2026.1", text)
    assert [h for h, _ in chunks] == ["Chapter 6 · Leave rules › 6.1 Personal leave",
                                      "Chapter 6 · Leave rules › 6.4 On-duty (OD)"]
    _, content = chunks[1]
    assert content.startswith("KIT Handbook (version 2026.1)\nChapter 6 · Leave rules › 6.4 On-duty")
    assert "6.4.3 Within 5 working days" in content


def test_long_section_is_split_at_rules_with_small_overlap():
    rules = [f"6.4.{i} " + ("word " * 60).strip() for i in range(1, 15)]
    chunks = rag.chunk_document("T", "1", "6.4 On-duty\n" + "\n".join(rules))
    assert len(chunks) > 1
    for _, content in chunks:
        assert content.startswith("T (version 1)\n6.4 On-duty\n")
        assert len(content) < rag.MAX_CHUNK_CHARS + rag.OVERLAP_CHARS + 100
    bodies = [c.split("\n", 2)[2] for _, c in chunks]
    assert all(body.split("\n")[0].startswith("6.4.") for body in bodies)


async def test_index_replaces_chunks_and_stores_embeddings(embedder):
    await _index_all()
    await rag.index_document(2)
    rows = await db.pool().fetch(
        "SELECT chunk_index, embedding IS NOT NULL AS has_vec FROM doc_chunks WHERE document_id=2 "
        "ORDER BY chunk_index")
    assert [r["chunk_index"] for r in rows] == list(range(len(rows))) and len(rows) == 3
    assert all(r["has_vec"] for r in rows)


async def test_vector_search_finds_the_rule(embedder):
    await _index_all()
    results = await rag.search("minimum attendance to write the examination", CSE_STUDENT, k=2)
    assert results[0]["document_title"] == "KIT Student Handbook 2026"
    assert "4.1.1" in results[0]["content"]
    assert set(results[0]) == {"document_title", "version", "heading", "content", "score", "ref",
                               "section", "page_start", "page_end", "document_id",
                               "document_type", "effective_date"}


async def test_search_restricted_to_applicable_documents(embedder):
    await _index_all()
    for query in ("on-duty proof working days", "what is a foreign key", "hostel fee"):
        titles = {r["document_title"] for r in await rag.search(query, VISITOR, k=20)}
        assert titles <= {"KIT Admissions Brochure 2026"}
    ece = {r["document_title"] for r in await rag.search("foreign key", ECE_STUDENT, k=20)}
    assert "DBMS Lesson Notes" not in ece and "KIT Student Handbook 2025" not in ece
    cse = await rag.search("foreign key primary key table", CSE_STUDENT, k=1)
    assert cse[0]["document_title"] == "DBMS Lesson Notes"


async def test_full_text_fallback_when_embedding_fails(embedder):
    embedder["fail"] = True
    await insert_documents()
    for doc_id in range(1, 6):
        with pytest.raises(RuntimeError):
            await rag.index_document(doc_id)
    assert await db.pool().fetchval("SELECT count(*) FROM doc_chunks WHERE embedding IS NULL") > 0
    results = await rag.search("how many days to submit on-duty proof", CSE_STUDENT)
    titles = [r["document_title"] for r in results]
    assert "Circular 12/2026" in titles and "KIT Student Handbook 2026" in titles
    assert "KIT Student Handbook 2025" not in titles
    exam = await rag.search("can I take leave on an exam day", CSE_STUDENT)
    assert any("6.1.4" in r["content"] for r in exam)
    assert await rag.search("hostel fee", VISITOR) and all(
        r["document_title"] == "KIT Admissions Brochure 2026"
        for r in await rag.search("on-duty proof", VISITOR))


async def test_failed_embedding_cools_down(embedder):
    embedder["fail"] = True
    with pytest.raises(RuntimeError):
        await rag.embed_texts(["x"], "RETRIEVAL_QUERY")
    with pytest.raises(RuntimeError):
        await rag.embed_texts(["x"], "RETRIEVAL_QUERY")
    assert embedder["calls"] == 1


async def test_embed_missing_backfills_after_recovery(embedder):
    embedder["fail"] = True
    await insert_documents()
    with pytest.raises(RuntimeError):
        await rag.index_document(5)
    embedder["fail"] = False
    rag._embed_down_until = 0.0
    await rag.embed_missing()
    assert await db.pool().fetchval("SELECT count(*) FROM doc_chunks WHERE embedding IS NULL") == 0


async def test_model_choice_prefers_stable_gemini_embedding(monkeypatch):
    models = [SimpleNamespace(name=n, supported_actions=a) for n, a in [
        ("models/gemini-2.5-flash", ["generateContent"]),
        ("models/text-embedding-004", ["embedContent"]),
        ("models/gemini-embedding-exp-03-07", ["embedContent"]),
        ("models/gemini-embedding-001", ["embedContent"]),
    ]]

    async def listing():
        async def gen():
            for m in models:
                yield m
        return gen()

    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(list=listing)))
    monkeypatch.setattr(rag, "_genai_client", lambda: client)
    monkeypatch.setattr(rag, "_model_name", None)
    settings = get_settings()
    monkeypatch.setattr(settings, "embed_model", "")
    assert await rag.embed_model() == "gemini-embedding-001"
    monkeypatch.setattr(settings, "embed_model", "custom-embed")
    assert await rag.embed_model() == "custom-embed"


def test_short_sections_are_merged_not_dropped():
    text = "Chapter 1 · Rules\n1.1 Dress code\n1.1.1 Wear ID.\n1.2 Timing\n1.2.1 Classes start at nine."
    joined = "\n".join(c for _, c in rag.chunk_document("T", "1", text))
    assert "1.1.1 Wear ID." in joined and "1.2.1 Classes start at nine." in joined
    assert rag.chunk_document("T", "1", "5.1 Fines\n5.1.1 Rs 10.")[0][1].endswith("5.1.1 Rs 10.")


async def test_query_with_control_characters_is_safe(embedder):
    await _index_all()
    assert await rag.search("hostel\x00fee\x07", VISITOR)
    embedder["fail"] = True
    assert await rag.search("hostel\x00fee\x07", VISITOR)


async def test_unrelated_query_returns_nothing(embedder):
    await _index_all()
    assert await rag.search("quantum chromodynamics lattice", VISITOR) == []


async def test_concurrent_indexing_of_one_document_leaves_no_duplicates(embedder):
    import asyncio

    await insert_documents()
    await asyncio.gather(*(rag.index_document(2) for _ in range(4)))
    dupes = await db.pool().fetchval(
        "SELECT count(*) FROM (SELECT chunk_index FROM doc_chunks WHERE document_id=2 "
        "GROUP BY chunk_index HAVING count(*) > 1) d")
    assert dupes == 0
