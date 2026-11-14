"""Edges of the knowledge module: text extraction, embedding client boundary, cache bound (DOC6, DOC7, DOC8, REL2)."""

import types

import pytest
from app import db
from app.knowledge import mode as mode_mod
from app.knowledge import rag, store
from app.knowledge.mode import knowledge_block

from tests.acc_knowledge_helpers import (
    BRO, CIR_ALL, CIR_UG, CSE, TEACHER_CSE, VISITOR, base_texts, contains_in_order, env, make_pdf, seed,  # noqa: F401
)


# ---------------------------------------------------------------- extraction

def test_DOC6_running_page_header_is_dropped_but_rule_text_kept():
    raw = "KIT Handbook · Edition 1 · page 3 of 11\n6.4.3 Within 5 working days submit proof.\n"
    assert store.clean_page(raw) == ["6.4.3 Within 5 working days submit proof."]


@pytest.mark.parametrize("raw", ["", None, "\n\n  \n"])
def test_DOC6_blank_pages_give_no_lines(raw):
    assert store.clean_page(raw) == []


@pytest.mark.parametrize("raw,expected", [
    ("Chapter\n6\n·\nLeave\n6.4\nOn-duty\n6.4.3 Body text goes here.", ["Chapter 6 · Leave", "6.4 On-duty", "6.4.3 Body text goes here."]),
    ("Annexure\nA\n·\nForms\nTable\n3\n", ["Annexure A · Forms", "Table 3"]),
    ("Unit\n1\n·\nModels\nUnit\n2\n·\nKeys\n", ["Unit 1 · Models", "Unit 2 · Keys"]),
    ("See\nRule\n6.4.3\nand\nRule\n6.4.4\nfor details here.", ["See Rule 6.4.3 and Rule 6.4.4"]),
])
def test_DOC6_split_headings_are_rejoined_and_rule_references_stay_inline(raw, expected):
    out = store.clean_page(raw)
    assert out[:len(expected)] == expected or out == expected


def test_DOC6_extract_text_from_real_pdf_keeps_rule_numbers_with_text():
    pdf = make_pdf(["Chapter 6 - Leave rules", "6.4 On-duty", "6.4.3 Proof must be submitted within five days."])
    text = store.extract_text(pdf)
    assert "6.4.3 Proof must be submitted within five days." in text


def test_DOC6_extract_text_multi_page_concatenates_in_order():
    from io import BytesIO
    from pypdf import PdfReader, PdfWriter
    w = PdfWriter()
    for body in ("first page text here", "second page text here"):
        w.append(PdfReader(BytesIO(make_pdf([body]))))
    buf = BytesIO()
    w.write(buf)
    text = store.extract_text(buf.getvalue())
    assert text.index("first page") < text.index("second page")


@pytest.mark.parametrize("text,low,high", [("", 0, 0), ("a" * 35, 8, 12), ("a" * 3500, 900, 1100)])
def test_DOC5_estimate_tokens_is_conservative_and_proportional(text, low, high):
    assert low <= store.estimate_tokens(text) <= high


def test_DOC5_estimate_never_below_real_word_count():
    prose = "The student must submit proof of participation within five working days after the event."
    assert store.estimate_tokens(prose) >= len(prose.split())


# ---------------------------------------------------------------- embedding boundary (fake client, no network)

class FakeEmbedClient:
    def __init__(self, dim=768, fail=False, wrong=False):
        self.dim, self.fail, self.wrong, self.calls = dim, fail, wrong, []
        self.aio = types.SimpleNamespace(models=types.SimpleNamespace(embed_content=self.embed, list=self.list))

    async def embed(self, model, contents, config):
        self.calls.append((model, list(contents), config))
        if self.fail:
            raise RuntimeError("boom")
        n = len(contents) - (1 if self.wrong else 0)
        return types.SimpleNamespace(embeddings=[types.SimpleNamespace(values=[3.0] + [0.0] * (self.dim - 1)) for _ in range(n)])

    async def list(self):
        async def gen():
            for name, actions in [("models/text-embedding-004", ["embedContent"]),
                                  ("models/gemini-embedding-exp-03-07", ["embedContent"]),
                                  ("models/gemini-embedding-001", ["embedContent"]),
                                  ("models/gemini-2.0-flash", ["generateContent"])]:
                yield types.SimpleNamespace(name=name, supported_actions=actions)
        return gen()


@pytest.fixture
def genai(monkeypatch):
    client = FakeEmbedClient()
    monkeypatch.setattr(rag, "_genai_client", lambda: client)
    monkeypatch.setattr(rag, "_model_name", None)
    monkeypatch.setattr(rag, "_embed_down_until", 0.0)
    monkeypatch.setattr(rag.get_settings(), "embed_model", "")
    return client


async def test_DOC7_embedding_gives_one_normalised_vector_per_text(genai):
    out = await rag.embed_texts([f"t{i}" for i in range(15)], "RETRIEVAL_DOCUMENT")
    assert len(out) == 15
    assert all(abs(sum(x * x for x in v) - 1.0) < 1e-6 for v in out)


async def test_DOC7_embedding_model_prefers_stable_embedding_model(genai):
    assert await rag.embed_model() == "gemini-embedding-001"


async def test_DOC7_configured_embedding_model_wins(genai, monkeypatch):
    monkeypatch.setattr(rag.get_settings(), "embed_model", "my-model")
    assert await rag.embed_model() == "my-model"


async def test_REL1_model_listing_failure_falls_back_to_default(monkeypatch):
    def broken():
        raise RuntimeError("no key")
    monkeypatch.setattr(rag, "_genai_client", broken)
    monkeypatch.setattr(rag, "_model_name", None)
    monkeypatch.setattr(rag.get_settings(), "embed_model", "")
    assert await rag.embed_model() == rag.DEFAULT_EMBED_MODEL


async def test_REL1_embedding_failure_raises_not_returns_garbage(genai):
    genai.fail = True
    with pytest.raises(RuntimeError):
        await rag.embed_texts(["a"], "RETRIEVAL_DOCUMENT")


# ---------------------------------------------------------------- background indexing errors

async def test_REL1_embedding_retry_failure_in_background_is_swallowed(env, monkeypatch):
    async def boom(document_id=None):
        raise RuntimeError("embedding retry failed")
    monkeypatch.setattr(rag, "embed_missing", boom)
    env.college.files = {k: v for k, v in env.college.files.items()}
    monkeypatch.setattr(store, "_sync_count", store.CONTENT_RECHECK_EVERY - 1)
    assert await store.sync_documents() is True
    await store.wait_for_indexing()
    assert await db.pool().fetchval("SELECT count(*) FROM documents") == 9


async def test_REL1_wait_for_indexing_with_nothing_running_returns(env):
    await store.wait_for_indexing()


async def test_REL1_stop_sync_loop_when_never_started_is_harmless(env):
    await store.stop_sync_loop()


# ---------------------------------------------------------------- cache bound

async def test_DOC8_cache_is_bounded_and_text_stays_correct_beyond_the_limit(env):
    await seed()
    first = (await knowledge_block(CSE)).text
    for n in range(mode_mod.CACHE_LIMIT + 5):
        await db.pool().execute("UPDATE documents SET version=$2 WHERE id=$1", CIR_ALL, f"v{n}")
        await knowledge_block(CSE)
    assert len(mode_mod._cache) <= mode_mod.CACHE_LIMIT
    await db.pool().execute("UPDATE documents SET version='13/2026' WHERE id=$1", CIR_ALL)
    assert (await knowledge_block(CSE)).text == first


@pytest.mark.parametrize("only", ["circulars", "no_circulars", "nothing"])
async def test_DOC3_rag_block_renders_for_any_mix_of_documents(env, only):
    await seed()
    if only == "circulars":
        await db.pool().execute("UPDATE documents SET active=false WHERE type <> 'circular'")
    elif only == "no_circulars":
        await db.pool().execute("UPDATE documents SET active=false WHERE type = 'circular'")
    else:
        await db.pool().execute("UPDATE documents SET active=false")
    env.settings.doc_mode = "rag"
    block = await knowledge_block(CSE)
    assert block.mode == "rag"
    assert contains_in_order(block.text, base_texts()[CIR_UG]) == (only == "circulars")


async def test_DOC1_visitor_rag_block_lists_brochure_only_and_no_circular_section(env):
    await seed()
    env.settings.doc_mode = "rag"
    text = (await knowledge_block(VISITOR)).text
    assert "TITLEBROCHUREMARK" in text
    assert "Circulars in force" not in text


# ---------------------------------------------------------------- rate limits from the embedding service

class RateLimited(Exception):
    code = 429
    details = {"error": {"details": [{"retryDelay": "2s"}]}}


async def test_DOC7_rate_limited_query_falls_back_to_text_and_does_not_start_a_cooldown(env):
    from tests.acc_knowledge_helpers import seed_and_index
    await seed_and_index()
    calls = []

    async def limited(texts, task_type):
        calls.append(1)
        raise RateLimited()

    real = rag._embed
    rag._embed = limited
    res = await rag.search("tuition fee hostel", VISITOR)
    assert res and "tuition fee" in res[0]["content"]
    rag._embed = real
    before = env.emb.calls
    await rag.search("tuition fee hostel", VISITOR)
    assert env.emb.calls > before


async def test_DOC6_indexing_waits_out_rate_limits_and_still_completes(env, monkeypatch):
    attempts = {"n": 0}

    async def flaky(texts, task_type):
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise RateLimited()
        return await env.emb(texts, task_type)

    async def no_sleep(_s):
        return None

    monkeypatch.setattr(rag, "_embed", flaky)
    monkeypatch.setattr(rag, "_sleep", no_sleep)
    await seed()
    await rag.index_document(BRO)
    assert await db.pool().fetchval(
        "SELECT count(*) FROM doc_chunks WHERE document_id=$1 AND embedding IS NULL", BRO) == 0


async def test_REL1_persistent_rate_limit_leaves_chunks_searchable_and_sync_alive(env, monkeypatch):
    async def always(texts, task_type):
        raise RateLimited()

    async def no_sleep(_s):
        return None

    monkeypatch.setattr(rag, "_embed", always)
    monkeypatch.setattr(rag, "_sleep", no_sleep)
    assert await store.sync_documents() is True
    await store.wait_for_indexing()
    assert await db.pool().fetchval("SELECT count(*) FROM doc_chunks") > 0
    assert (await rag.search("tuition fee hostel", VISITOR))[0]["content"]
