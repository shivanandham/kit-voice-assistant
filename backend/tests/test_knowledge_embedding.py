"""Embedding robustness: model choice, one vector per text, rate-limit pacing, 429 handling, partial corpus."""

from datetime import date
from types import SimpleNamespace

import pytest
from app import db
from app.config import get_settings
from app.knowledge import rag, store
from google.genai import errors

from tests.test_knowledge_fakes import (
    CSE_STUDENT,
    college_date,
    fake_vector,
    insert_documents,
)


def rate_limited(delay: str = "9s") -> errors.ClientError:
    """A Gemini 429 carrying RetryInfo, as the real API returns it."""
    return errors.ClientError(429, {"error": {
        "code": 429, "message": "quota exceeded", "status": "RESOURCE_EXHAUSTED",
        "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}]}})


@pytest.fixture
def clock(monkeypatch):
    """Fake time: sleeping advances the clock instantly; every sleep is recorded."""
    state = {"now": 1000.0, "sleeps": []}

    async def sleep(seconds):
        state["sleeps"].append(seconds)
        state["now"] += seconds

    monkeypatch.setattr(rag, "_clock", lambda: state["now"])
    monkeypatch.setattr(rag, "_sleep", sleep)
    monkeypatch.setattr(rag, "_sent", [])
    monkeypatch.setattr(rag, "_embed_down_until", 0.0)
    monkeypatch.setattr(store, "_college_date", college_date(date(2026, 11, 21)))
    return state


async def _chunks(n: int) -> None:
    """One document with n chunks and no embeddings."""
    await insert_documents()
    await db.pool().execute("DELETE FROM doc_chunks")
    await db.pool().executemany(
        "INSERT INTO doc_chunks (document_id, chunk_index, heading, content) VALUES (2, $1, 'h', $2)",
        [(i, f"chunk number {i} about rule {i}") for i in range(n)],
    )


async def _unembedded() -> int:
    return await db.pool().fetchval("SELECT count(*) FROM doc_chunks WHERE embedding IS NULL")


async def test_default_model_is_one_that_returns_a_vector_per_text(monkeypatch):
    models = [SimpleNamespace(name=f"models/{n}", supported_actions=["embedContent"])
              for n in ("gemini-embedding-001", "gemini-embedding-2-preview", "gemini-embedding-2")]

    async def listing():
        async def gen():
            for m in models:
                yield m
        return gen()

    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(list=listing)))
    monkeypatch.setattr(rag, "_genai_client", lambda: client)
    monkeypatch.setattr(rag, "_model_name", None)
    monkeypatch.setattr(get_settings(), "embed_model", "")
    assert await rag.embed_model() == "gemini-embedding-001"


async def test_model_returning_one_vector_for_a_list_is_embedded_one_text_per_request(clock, monkeypatch):
    calls = []

    async def merges_list(texts, task_type):          # how gemini-embedding-2 behaves
        calls.append(len(texts))
        return [fake_vector(" ".join(texts))]

    monkeypatch.setattr(rag, "_embed", merges_list)
    await _chunks(5)
    await rag.embed_missing()
    assert await _unembedded() == 0
    assert calls[0] == 5 and calls[1:] == [1, 1, 1, 1, 1]
    vectors = await db.pool().fetch("SELECT content, embedding FROM doc_chunks ORDER BY chunk_index")
    for r in vectors:
        assert r["embedding"].to_list() == pytest.approx(fake_vector(r["content"]), abs=1e-6)


async def test_indexing_paces_itself_under_the_per_minute_budget(clock, monkeypatch):
    sent = []

    async def embed(texts, task_type):
        sent.append((clock["now"], len(texts)))
        return [fake_vector(t) for t in texts]

    monkeypatch.setattr(rag, "_api_embed", embed)
    await _chunks(300)
    await rag.embed_missing()
    assert await _unembedded() == 0
    for start, _ in sent:
        in_window = sum(n for t, n in sent if start <= t < start + 60)
        assert in_window <= rag.EMBED_TEXTS_PER_MINUTE
    assert clock["now"] - 1000.0 >= 60 * (300 // rag.EMBED_TEXTS_PER_MINUTE) - 60


async def test_429_waits_the_retry_delay_and_continues_without_cooldown(clock, monkeypatch):
    state = {"calls": 0}

    async def embed(texts, task_type):
        state["calls"] += 1
        if state["calls"] == 2:
            raise rate_limited("9s")
        return [fake_vector(t) for t in texts]

    monkeypatch.setattr(rag, "_embed", embed)
    await _chunks(60)
    await rag.embed_missing()
    assert await _unembedded() == 0
    assert 9 in clock["sleeps"]
    assert rag._embed_down_until == 0.0


async def test_retry_delay_is_bounded(clock, monkeypatch):
    assert rag._retry_delay(rate_limited("3600s")) == rag.MAX_RETRY_DELAY_S
    assert rag._retry_delay(rate_limited("1.5s")) == 1.5
    assert rag._retry_delay(RuntimeError("other")) is None


async def test_progress_is_saved_batch_by_batch_when_embedding_fails(clock, monkeypatch):
    state = {"calls": 0}

    async def embed(texts, task_type):
        state["calls"] += 1
        if state["calls"] > 2:
            raise RuntimeError("gemini down")
        return [fake_vector(t) for t in texts]

    monkeypatch.setattr(rag, "_embed", embed)
    await _chunks(100)
    with pytest.raises(RuntimeError):
        await rag.embed_missing()
    assert await _unembedded() == 100 - 2 * rag.EMBED_BATCH
    assert rag._embed_down_until == 0.0          # indexing failures never disable query embedding


async def test_query_hitting_429_uses_full_text_for_that_query_only(clock, monkeypatch):
    state = {"limited": True, "calls": 0}

    async def embed(texts, task_type):
        state["calls"] += 1
        if state["limited"]:
            raise rate_limited("5s")
        return [fake_vector(t) for t in texts]

    monkeypatch.setattr(rag, "_embed", embed)
    await insert_documents()
    state["limited"] = False
    for doc_id in range(1, 6):
        await rag.index_document(doc_id)
    state["limited"] = True
    results = await rag.search("on-duty proof working days", CSE_STUDENT)
    assert results and clock["sleeps"] == []    # no waiting at query time
    assert rag._embed_down_until == 0.0
    state["limited"] = False
    before = state["calls"]
    assert await rag.search("on-duty proof working days", CSE_STUDENT)
    assert state["calls"] == before + 1         # the next query embeds again


async def test_query_does_not_wait_when_indexing_used_the_budget(clock, monkeypatch):
    async def embed(texts, task_type):
        return [fake_vector(t) for t in texts]

    monkeypatch.setattr(rag, "_api_embed", embed)
    await insert_documents()
    for doc_id in range(1, 6):
        await rag.index_document(doc_id)
    rag._sent.append((clock["now"], rag.EMBED_TEXTS_PER_MINUTE))
    assert await rag.search("foreign key", CSE_STUDENT)
    assert clock["sleeps"] == []


async def test_partially_embedded_corpus_still_finds_unembedded_chunks(clock, monkeypatch):
    async def embed(texts, task_type):
        return [fake_vector(t) for t in texts]

    monkeypatch.setattr(rag, "_embed", embed)
    await insert_documents()
    for doc_id in range(1, 6):
        await rag.index_document(doc_id)
    await db.pool().execute("UPDATE doc_chunks SET embedding = NULL WHERE document_id = 5")
    titles = [r["document_title"] for r in await rag.search("submit on-duty proof", CSE_STUDENT)]
    assert "Circular 12/2026" in titles and "KIT Student Handbook 2026" in titles


async def test_every_sync_backfills_missing_embeddings(monkeypatch):
    calls = []

    async def embed_missing(document_id=None, *, skip_if_busy=False):
        calls.append((document_id, skip_if_busy))

    async def list_remote():
        return []

    monkeypatch.setattr(store, "_list_remote", list_remote)
    monkeypatch.setattr(rag, "embed_missing", embed_missing)
    monkeypatch.setattr(store, "_sync_count", 1)
    await store.sync_documents()
    await store.wait_for_indexing()
    assert calls == [(None, True)]
