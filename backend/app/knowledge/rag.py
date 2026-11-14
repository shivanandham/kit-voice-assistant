"""Document search and citations: section chunks, Gemini embeddings in pgvector, full-text fallback."""

import asyncio
import logging
import math
import re
import time

from app import db
from app.auth import SessionCtx
from app.config import get_settings
from app.knowledge.store import applicable_documents
from app.knowledge.structure import (  # noqa: F401  (chunk sizes re-exported for callers)
    MAX_CHUNK_CHARS,
    MIN_BODY_CHARS,
    OVERLAP_CHARS,
    Structure,
    analyse,
)

log = logging.getLogger(__name__)

EMBED_BATCH = 20                # texts per embedding request while indexing
EMBED_TIMEOUT_S = 30
EMBED_TEXTS_PER_MINUTE = 80     # under the free tier's 100 per minute; every text counts as one
QUERY_RESERVE = 10              # part of that budget indexing leaves free for live queries
MAX_RETRY_DELAY_S = 60.0        # cap on the server's 429 retry delay
MAX_RATE_LIMIT_RETRIES = 10     # 429s in a row before an indexing run gives up (next sync resumes)
DEFAULT_EMBED_MODEL = "gemini-embedding-001"   # returns one vector per text in a batch
INDEX_LOCK_NAMESPACE = 7311     # first key of the per-document advisory lock
MIN_VECTOR_SCORE = 0.55         # cosine below this is unrelated text (measured: gemini-embedding-001)
RRF_K = 60                      # reciprocal-rank-fusion constant for a partially embedded corpus

_REF = re.compile(r"^D(\d+):(\S+)$")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

EMBED_COOLDOWN_S = 120     # after a failed query embedding (not a 429), use full-text search for a while

# Words students use mapped to the words the documents use, for full-text search.
SYNONYMS = {
    "exam": "examination assessment test", "exams": "examination assessment test",
    "test": "assessment", "od": "on-duty", "fees": "fee", "cse": "computer science",
    "ece": "electronics communication", "mech": "mechanical", "bunk": "unauthorised absence",
}

_model_name: str | None = None
_client = None
_embed_down_until = 0.0
_clock = time.monotonic
_sleep = asyncio.sleep
_sent: list[tuple[float, int]] = []     # (time, texts) of embedding requests in the last minute
_embed_lock = asyncio.Lock()


class EmbeddingBusy(RuntimeError):
    """The per-minute embedding budget is used up and the caller cannot wait (a live query)."""


# ---------------------------------------------------------------- chunking

def chunk_document(title: str, version: str, text: str) -> list[tuple[str, str]]:
    """Return (heading, content) chunks; content starts with the document title and heading."""
    return [(c.heading, c.content) for c in analyse(0, title, version, "", text).chunks]


def _structure(doc: dict) -> Structure:
    return analyse(doc["id"], doc["title"] or "", doc["version"] or "", doc["type"] or "",
                   doc["text"] or "")


# ---------------------------------------------------------------- embeddings

def _genai_client():
    global _client
    if _client is None:
        from google import genai

        key = get_settings().gemini_api_key
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        _client = genai.Client(api_key=key)
    return _client


async def embed_model() -> str:
    """The embedding model: settings.embed_model, else the best gemini-embedding model this key lists."""
    global _model_name
    configured = get_settings().embed_model
    if configured:
        return configured
    if _model_name is None:
        try:
            names = []
            async for m in await _genai_client().aio.models.list():
                if "embedContent" in (m.supported_actions or []):
                    names.append(m.name.removeprefix("models/"))
            preferred = sorted(
                (n for n in names if n.startswith("gemini-embedding")),
                key=lambda n: ("exp" in n or "preview" in n, [-int(x) for x in re.findall(r"\d+", n)]),
            )
            # gemini-embedding-001 is preferred when listed: newer models (gemini-embedding-2)
            # merge a list of texts into one embedding.
            if DEFAULT_EMBED_MODEL in names:
                preferred.insert(0, DEFAULT_EMBED_MODEL)
            _model_name = (preferred or names or [DEFAULT_EMBED_MODEL])[0]
            log.info("embedding model: %s", _model_name)
        except Exception:
            log.warning("could not list Gemini models; using %s", DEFAULT_EMBED_MODEL)
            _model_name = DEFAULT_EMBED_MODEL
    return _model_name


def _normalise(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def _retry_delay(exc: Exception) -> float | None:
    """Seconds to wait if exc is a Gemini 429 (from its RetryInfo, capped), else None."""
    if getattr(exc, "code", None) != 429:
        return None
    details = getattr(exc, "details", None)
    entries = (details.get("error", {}).get("details", []) if isinstance(details, dict) else [])
    for entry in entries:
        match = re.fullmatch(r"([\d.]+)s", str(entry.get("retryDelay", "")))
        if match:
            return min(float(match.group(1)), MAX_RETRY_DELAY_S)
    return 10.0


async def _reserve(n: int, wait: bool) -> None:
    """Take n texts from the per-minute budget, waiting for room (indexing) or raising (queries).

    Indexing may use the budget minus QUERY_RESERVE, so queries keep some headroom.
    """
    limit = EMBED_TEXTS_PER_MINUTE - (QUERY_RESERVE if wait else 0)
    while True:
        now = _clock()
        _sent[:] = [(t, c) for t, c in _sent if now - t < 60]
        if sum(c for _, c in _sent) + n <= limit:
            _sent.append((now, n))
            return
        if not wait:
            raise EmbeddingBusy("embedding budget for this minute is used up")
        await _sleep(max(0.5, _sent[0][0] + 60 - now))


async def _api_embed(texts: list[str], task_type: str) -> list[list[float]]:
    """One embed_content request; returns whatever embeddings the API sent back, normalised."""
    from google.genai import types

    resp = await asyncio.wait_for(
        _genai_client().aio.models.embed_content(
            model=await embed_model(),
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type, output_dimensionality=get_settings().embed_dim),
        ),
        EMBED_TIMEOUT_S,
    )
    return [_normalise(list(e.values)) for e in resp.embeddings or []]


async def _embed(texts: list[str], task_type: str) -> list[list[float]]:
    """One rate-budgeted request. Document embedding waits for budget; a query raises EmbeddingBusy."""
    await _reserve(len(texts), wait=task_type != "RETRIEVAL_QUERY")
    return await _api_embed(texts, task_type)


async def _embed_request(texts: list[str], task_type: str, wait_on_429: bool) -> list[list[float]]:
    """_embed with 429 handling: wait the server's retry delay and try again (indexing only)."""
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return await _embed(texts, task_type)
        except Exception as exc:
            delay = _retry_delay(exc)
            if delay is None or not wait_on_429 or attempt == MAX_RATE_LIMIT_RETRIES:
                raise
            log.info("embedding rate limited; waiting %.1fs", delay)
            await _sleep(delay)
    raise AssertionError("unreachable")


async def _embed_batch(texts: list[str], task_type: str, wait_on_429: bool) -> list[list[float]]:
    """Exactly one vector per text: if a response has fewer, embed that batch one text per request."""
    vectors = await _embed_request(texts, task_type, wait_on_429)
    if len(vectors) != len(texts) and len(texts) > 1:
        log.info("model returned %d vectors for %d texts; embedding one by one", len(vectors), len(texts))
        vectors = []
        for text in texts:
            vectors.extend((await _embed_request([text], task_type, wait_on_429))[:1])
    dim = get_settings().embed_dim
    if len(vectors) != len(texts) or any(len(v) != dim for v in vectors):
        raise RuntimeError("embedding response has the wrong shape")
    return vectors


async def embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    """Embed texts for a live query; raises at once on failure, never waits on a rate limit.

    A non-rate-limit failure makes later calls fail fast for EMBED_COOLDOWN_S so search does not
    wait on a dead service; a 429 or a used-up budget affects only this call.
    """
    global _embed_down_until
    if _clock() < _embed_down_until:
        raise RuntimeError("embedding service cooling down after a failure")
    try:
        return await _embed_batch(texts, task_type, wait_on_429=False)
    except EmbeddingBusy:
        raise
    except Exception as exc:
        if _retry_delay(exc) is None:
            _embed_down_until = _clock() + EMBED_COOLDOWN_S
        raise


# ---------------------------------------------------------------- indexing

async def index_document(document_id: int) -> None:
    """Re-chunk a document and replace its doc_chunks rows, then embed them.

    Chunks are written first without embeddings so full-text search works at once; if embedding
    fails the error propagates and the chunks stay searchable by full text until a retry.
    """
    row = await db.pool().fetchrow(
        "SELECT id, title, type, version, text FROM documents WHERE id=$1", document_id
    )
    if row is None or not row["text"]:
        return
    chunks = _structure(dict(row)).chunks
    async with db.pool().acquire() as conn, conn.transaction():
        # Serialise re-chunking of one document across tasks and processes; the lock is released
        # at commit, and the DELETE then sees rows committed by an earlier run.
        await conn.execute("SELECT pg_advisory_xact_lock($1, $2)", INDEX_LOCK_NAMESPACE, document_id)
        # Chunks whose text is unchanged keep their embedding, so re-indexing costs no API calls.
        kept = {r["content"]: r["embedding"] for r in await conn.fetch(
            "DELETE FROM doc_chunks WHERE document_id=$1 RETURNING content, embedding", document_id)}
        await conn.executemany(
            """INSERT INTO doc_chunks (document_id, chunk_index, heading, content, embedding,
                                       ref, section, page_start, page_end)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)""",
            [(document_id, c.chunk_index, c.heading, c.content, kept.get(c.content), c.ref,
              c.section, c.page_start, c.page_end) for c in chunks],
        )
    log.info("document %s: %d chunks", document_id, len(chunks))
    await embed_missing(document_id)


async def embed_missing(document_id: int | None = None, *, skip_if_busy: bool = False) -> None:
    """Embed chunks that have no embedding yet (all documents, or one), batch by batch.

    Paced under EMBED_TEXTS_PER_MINUTE; each batch is stored as soon as it is embedded, so progress
    survives a failure; a 429 waits the server's retry delay. Raises on any other failure. With
    skip_if_busy, returns at once when another embedding run is in progress.
    """
    if skip_if_busy and _embed_lock.locked():
        return
    async with _embed_lock:
        rows = await db.pool().fetch(
            """SELECT id, content FROM doc_chunks WHERE embedding IS NULL
                 AND ($1::int IS NULL OR document_id = $1) ORDER BY id""",
            document_id,
        )
        for i in range(0, len(rows), EMBED_BATCH):
            batch = rows[i:i + EMBED_BATCH]
            vectors = await _embed_batch([r["content"] for r in batch], "RETRIEVAL_DOCUMENT",
                                         wait_on_429=True)
            await db.pool().executemany(
                "UPDATE doc_chunks SET embedding=$2 WHERE id=$1 AND embedding IS NULL",
                [(r["id"], v) for r, v in zip(batch, vectors)],
            )
        if rows:
            log.info("embedded %d chunks", len(rows))


# ---------------------------------------------------------------- search

_VECTOR_SQL = """
SELECT c.id, d.title AS document_title, d.version, c.heading, c.content, c.ref, c.section,
       c.page_start, c.page_end, c.document_id, d.type AS document_type, d.effective_date,
       1 - (c.embedding <=> $1) AS score
FROM doc_chunks c JOIN documents d ON d.id = c.document_id
WHERE c.document_id = ANY($2::int[]) AND c.embedding IS NOT NULL
ORDER BY c.embedding <=> $1, c.id
LIMIT $3
"""

_TEXT_SQL = """
WITH q AS (
  SELECT NULLIF(replace(plainto_tsquery('english', $1)::text, ' & ', ' | '), '')::tsquery AS q
)
SELECT c.id, d.title AS document_title, d.version, c.heading, c.content, c.ref, c.section,
       c.page_start, c.page_end, c.document_id, d.type AS document_type, d.effective_date,
       ts_rank_cd(to_tsvector('english', c.content), q.q, 1) AS score
FROM doc_chunks c JOIN documents d ON d.id = c.document_id, q
WHERE c.document_id = ANY($2::int[]) AND q.q IS NOT NULL
  AND to_tsvector('english', c.content) @@ q.q
ORDER BY score DESC, c.id
LIMIT $3
"""


def _expand(query: str) -> str:
    """Add document vocabulary for common student words (full-text search only)."""
    words = re.findall(r"[\w-]+", query.lower())
    return " ".join([query, *(SYNONYMS[w] for w in words if w in SYNONYMS)])


def _fuse(vector_rows: list, text_rows: list, k: int) -> list[dict]:
    """Reciprocal rank fusion of vector and full-text results; score is the fused score."""
    fused: dict[int, dict] = {}
    for rows in (vector_rows, text_rows):
        for rank, r in enumerate(rows):
            entry = fused.setdefault(r["id"], {**dict(r), "score": 0.0})
            entry["score"] += 1.0 / (RRF_K + rank + 1)
    return sorted(fused.values(), key=lambda e: (-e["score"], e["id"]))[:k]


async def search(query: str, ctx: SessionCtx, k: int = 6) -> list[dict]:
    """Top-k chunks for the query from documents this user may read.

    Fully embedded corpus: cosine similarity over Gemini embeddings (score = cosine similarity);
    neighbours below MIN_VECTOR_SCORE that share no words with the query are dropped, so an
    unrelated question returns nothing.
    Partly embedded (a new document still being indexed): vector and full-text results are merged
    by reciprocal rank fusion, so chunks without a vector are not left out (score = fused score).
    No embeddings, or the query cannot be embedded (outage, 429, budget used up): Postgres
    full-text ranking for this query.
    Returns [{document_title, version, heading, content, score}].
    """
    query = _CONTROL.sub(" ", query or "").strip()[:2000]
    k = max(1, min(int(k), 20))
    ids = [d["id"] for d in await applicable_documents(ctx)]
    if not query or not ids:
        return []
    counts = await db.pool().fetchrow(
        "SELECT count(*) AS total, count(embedding) AS embedded FROM doc_chunks "
        "WHERE document_id = ANY($1::int[])", ids)
    complete = counts["embedded"] == counts["total"]
    vector_rows: list = []
    vector_ok = False
    if counts["embedded"]:
        try:
            (vector,) = await embed_texts([query], "RETRIEVAL_QUERY")
            vector_rows = await db.pool().fetch(_VECTOR_SQL, vector, ids, 3 * k)
            vector_ok = True
        except Exception as exc:
            log.warning("query embedding failed (%s); using full-text search", type(exc).__name__)
    text_rows = await db.pool().fetch(_TEXT_SQL, _expand(query), ids, 3 * k)
    if vector_ok:
        # Drop unrelated nearest neighbours, unless the chunk also matches the words of the query.
        text_ids = {r["id"] for r in text_rows}
        vector_rows = [r for r in vector_rows if r["score"] >= MIN_VECTOR_SCORE or r["id"] in text_ids]
        rows = [dict(r) for r in vector_rows[:k]] if complete else _fuse(vector_rows, text_rows, k)
    else:
        rows = [dict(r) for r in text_rows[:k]]
    return [
        {"document_title": r["document_title"], "version": r["version"], "heading": r["heading"],
         "content": r["content"], "score": round(float(r["score"]), 4), "ref": r["ref"],
         "section": r["section"], "page_start": r["page_start"], "page_end": r["page_end"],
         "document_id": r["document_id"], "document_type": r["document_type"],
         "effective_date": r["effective_date"].isoformat() if r["effective_date"] else None}
        for r in rows
    ]


async def resolve_refs(refs: list[str], ctx: SessionCtx) -> list[dict]:
    """Citation details for refs that exist in documents this user may use, in input order.

    Malformed refs, unknown sections and documents outside applicable_documents(ctx) are dropped
    silently; duplicates are returned once.
    Returns [{ref, document_title, document_type, version, effective_date, section, heading,
    page_start, page_end}].
    """
    docs = {d["id"]: d for d in await applicable_documents(ctx)}
    out: list[dict] = []
    seen: set[str] = set()
    for raw in refs or []:
        ref = str(raw).strip() if raw is not None else ""
        match = _REF.match(ref)
        if not match or ref in seen or int(match.group(1)) not in docs:
            continue
        doc = docs[int(match.group(1))]
        info = _structure(doc).refs.get(ref)
        if info is None:
            continue
        seen.add(ref)
        out.append({
            "ref": ref, "document_title": doc["title"], "document_type": doc["type"],
            "version": doc["version"],
            "effective_date": doc["effective_date"].isoformat() if doc["effective_date"] else None,
            **info,
        })
    return out
