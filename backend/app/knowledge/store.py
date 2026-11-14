"""College documents: keep the `documents` table in step with the college API and pick what a user may read."""

import asyncio
import hashlib
import io
import json
import logging
import math
import re
from datetime import date
from typing import Any

from pypdf import PdfReader

from app import db
from app.auth import SessionCtx
from app.config import get_settings
from app.knowledge.structure import strip_page_marks

log = logging.getLogger(__name__)

CHARS_PER_TOKEN = 3.5          # conservative for English prose with the Claude tokenizer
CONTENT_RECHECK_EVERY = 1      # re-download unchanged files every Nth sync to compare hashes (1 = always)
TYPE_ORDER = {"handbook": 0, "circular": 1, "lesson_notes": 2, "brochure": 3}
LOGGED_IN_ROLES = ("student", "teacher")   # every other role string is treated as a visitor

_loop_task: asyncio.Task | None = None
_index_tasks: set[asyncio.Task] = set()
_sync_count = 0
_sync_lock = asyncio.Lock()
_last_college_date: date | None = None

_PAGE_HEADER = re.compile(r"·\s*page \d+ of \d+\s*$", re.IGNORECASE)
_PAGE_NUMBER = re.compile(r"·\s*page (\d+) of \d+\s*$", re.IGNORECASE | re.MULTILINE)
_RUN_BREAK_WORDS = {"Chapter", "Annexure", "Table", "Unit", "Part"}
_RUN_BREAK_NUMBER = re.compile(r"^\d+(\.\d+)+$|^\d+\.$")
_NO_BREAK_AFTER = {"Rule", "(Rule", "Rules", "and", "to", "Table", "Chapter", "section", "Section"}


# ---------------------------------------------------------------- college API (app.kit.client)

async def _list_remote() -> list[dict]:
    """Fetch the document listing from the college API."""
    from app.kit.client import KitClient

    data = await KitClient().documents()
    if isinstance(data, dict):
        data = data.get("documents") or data.get("items") or []
    return list(data)


async def _download(document_id: int) -> bytes:
    """Download one document's PDF from the college API."""
    from app.kit.client import KitClient

    return await KitClient().document_file(document_id)


async def _college_date() -> date | None:
    """The college's own date, which decides which handbook edition is in force.

    Falls back to the last date seen when the college API is unreachable or answers nonsense;
    None if no college date has been seen yet. The server clock is never used.
    """
    global _last_college_date
    from app.kit.client import college_today

    try:
        _last_college_date = date.fromisoformat((await college_today())["date"])
    except Exception:
        log.warning("college date unavailable; using %s", _last_college_date or "document dates")
    return _last_college_date


# ---------------------------------------------------------------- text extraction

def _single_token(line: str) -> bool:
    return bool(line) and " " not in line and len(line) <= 40


def _breaks_run(token: str, previous: str | None) -> bool:
    """True when a one-word line starts a new heading rather than continuing the previous one."""
    if previous is None or previous in _NO_BREAK_AFTER:
        return False
    return token in _RUN_BREAK_WORDS or bool(_RUN_BREAK_NUMBER.match(token))


def clean_page(raw: str) -> list[str]:
    """Drop the running page header and re-join headings that pypdf split into one word per line."""
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if run:
            out.append(" ".join(run))
            run.clear()

    for line in (raw or "").splitlines():
        line = " ".join(line.split())
        if not line or _PAGE_HEADER.search(line):
            continue
        if _single_token(line):
            if run and _breaks_run(line, run[-1]):
                flush()
            run.append(line)
        else:
            flush()
            out.append(line)
    flush()
    return out


def extract_text(pdf: bytes) -> str:
    """Extract readable text from a PDF: headers removed, split headings re-joined.

    Each page starts with a marker line "\\f<page>" (the number printed in the page header
    "page N of M" when present, else the page's position) so chunks can cite pages; see
    structure.strip_page_marks.
    """
    reader = PdfReader(io.BytesIO(pdf))
    lines: list[str] = []
    for position, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        printed = _PAGE_NUMBER.search(raw)
        lines.append(f"\f{int(printed.group(1)) if printed else position}")
        lines.extend(clean_page(raw))
    return "\n".join(lines).rstrip()


def estimate_tokens(text: str) -> int:
    """Rough token count used for the context-or-RAG decision."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


# ---------------------------------------------------------------- sync

def _meta(item: dict) -> dict:
    """The listing fields we store, normalised for comparison."""
    return {
        "title": item.get("title") or f"Document {item['id']}",
        "type": item.get("type") or "other",
        "version": str(item.get("version") or ""),
        "effective_date": date.fromisoformat(item["effective_date"]) if item.get("effective_date") else None,
        "applies_to": item.get("applies_to") or {},
        "is_surprise": bool(item.get("is_surprise")),
    }


def _row_meta(row: Any) -> dict:
    applies = row["applies_to"]
    return {
        "title": row["title"],
        "type": row["type"],
        "version": row["version"],
        "effective_date": row["effective_date"],
        "applies_to": json.loads(applies) if isinstance(applies, str) else (applies or {}),
        "is_surprise": bool(row["is_surprise"]),
    }


async def _upsert(doc_id: int, meta: dict, text: str, content_hash: str) -> None:
    await db.pool().execute(
        """INSERT INTO documents (id, title, type, version, effective_date, applies_to, is_surprise,
                                  text, token_estimate, content_hash, active, fetched_at)
           VALUES ($1,$2,$3,$4,$5,$6::jsonb,$7,$8,$9,$10,true,now())
           ON CONFLICT (id) DO UPDATE SET title=$2, type=$3, version=$4, effective_date=$5,
               applies_to=$6::jsonb, is_surprise=$7, text=$8, token_estimate=$9, content_hash=$10,
               active=true, fetched_at=now()""",
        doc_id, meta["title"], meta["type"], meta["version"], meta["effective_date"],
        json.dumps(meta["applies_to"], sort_keys=True), meta["is_surprise"], text,
        estimate_tokens(text), content_hash,
    )


async def _sync_item(item: dict, row: Any, recheck_content: bool) -> bool:
    """Download one listed document and store it if new or changed. True if stored."""
    doc_id = int(item["id"])
    meta = _meta(item)
    if (row is not None and row["active"] and row["text"] and not recheck_content
            and _row_meta(row) == meta):
        return False
    pdf = await _download(doc_id)
    content_hash = hashlib.sha256(pdf).hexdigest()
    if (row is not None and row["active"] and row["text"] and row["content_hash"] == content_hash
            and _row_meta(row) == meta):
        return False
    if row is not None and row["content_hash"] == content_hash and row["text"]:
        text = row["text"]
    else:
        text = await asyncio.to_thread(extract_text, pdf)
    if not strip_page_marks(text).strip():
        log.warning("document %s has no extractable text; not stored", doc_id)
        return False
    await _upsert(doc_id, meta, text, content_hash)
    log.info("document %s (%s %s) stored", doc_id, meta["title"], meta["version"])
    return True


async def sync_documents() -> bool:
    """Bring `documents` in line with the college listing; index changed documents in the background.

    Every listed file is downloaded (they are small; see CONTENT_RECHECK_EVERY) and compared by SHA-256 and metadata; only new
    or changed ones are extracted and stored. A bad entry (malformed metadata, failed download,
    unreadable or empty PDF) is logged and skipped without affecting the others. Documents that
    vanished from the listing are marked inactive. Returns True if anything changed; returns False,
    keeping the stored documents, if the listing cannot be fetched.
    """
    global _sync_count
    async with _sync_lock:
        _sync_count += 1
        recheck_content = _sync_count % CONTENT_RECHECK_EVERY == 0
        try:
            remote = await _list_remote()
        except Exception as exc:
            log.warning("document listing failed (%s); keeping stored documents", type(exc).__name__)
            return False
        rows = {r["id"]: r for r in await db.pool().fetch("SELECT * FROM documents")}
        changed_ids: list[int] = []
        listed_ids: list[int] = []

        for item in remote:
            try:
                doc_id = int(item["id"])
            except Exception:
                log.warning("listing entry without a usable id skipped: %.200r", item)
                continue
            listed_ids.append(doc_id)
            try:
                if await _sync_item(item, rows.get(doc_id), recheck_content):
                    changed_ids.append(doc_id)
            except Exception as exc:
                log.warning("document %s skipped this sync (%s: %s)", doc_id, type(exc).__name__,
                            str(exc)[:200])

        removed = await db.pool().fetch(
            "UPDATE documents SET active=false WHERE active AND NOT (id = ANY($1::int[])) RETURNING id",
            listed_ids,
        )
        for r in removed:
            log.info("document %s no longer listed; marked inactive", r["id"])

        missing = await db.pool().fetch(
            """SELECT d.id FROM documents d WHERE d.active AND d.text IS NOT NULL
                 AND NOT EXISTS (SELECT 1 FROM doc_chunks c
                                 WHERE c.document_id = d.id AND c.ref IS NOT NULL)"""
        )
        to_index = sorted(set(changed_ids) | {r["id"] for r in missing})
        _index_in_background(to_index)
        return bool(changed_ids or removed)


def _index_in_background(document_ids: list[int]) -> None:
    """Chunk and embed documents without blocking the sync; failures are logged, never raised.

    Afterwards, chunks still without an embedding (an earlier failure, or a large document cut
    short by rate limits) are embedded again, unless another embedding run is already going.
    """
    from app.knowledge import rag

    async def run() -> None:
        for doc_id in document_ids:
            try:
                await rag.index_document(doc_id)
            except Exception as exc:
                log.warning("embedding document %s failed (%s: %s); chunks stay searchable by "
                            "full text", doc_id, type(exc).__name__, str(exc)[:200])
        try:
            await rag.embed_missing(skip_if_busy=True)
        except Exception as exc:
            log.warning("embedding backfill failed (%s); next sync retries", type(exc).__name__)

    task = asyncio.create_task(run())
    _index_tasks.add(task)
    task.add_done_callback(_index_tasks.discard)


async def wait_for_indexing() -> None:
    """Wait until background indexing started by earlier syncs has finished."""
    while _index_tasks:
        await asyncio.gather(*list(_index_tasks), return_exceptions=True)


async def _loop() -> None:
    while True:
        try:
            await sync_documents()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("document sync failed (%s); keeping the documents we have", type(exc).__name__)
        await asyncio.sleep(max(10, get_settings().doc_refresh_seconds))


async def start_sync_loop() -> None:
    """Start the periodic sync (first run immediately). Never raises if the college API is down."""
    global _loop_task
    if _loop_task is None or _loop_task.done():
        _loop_task = asyncio.create_task(_loop(), name="document-sync")


async def stop_sync_loop() -> None:
    """Stop the periodic sync and any background indexing."""
    global _loop_task
    tasks = [t for t in [_loop_task, *_index_tasks] if t is not None]
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    _loop_task = None


# ---------------------------------------------------------------- selection

def _sort_key(doc: dict) -> tuple:
    return (TYPE_ORDER.get(doc["type"], 4), doc["effective_date"] or date.min, doc["id"])


def select_documents(docs: list[dict], role: str, user: dict | None, today: date | None) -> list[dict]:
    """Apply the selection rules to active document rows.

    Visitors, and any role other than exactly "student" or "teacher": brochures only. Logged-in users: the handbook edition with the latest effective date on
    or before the college date, every circular, lesson notes (students: their department only;
    teachers: all), brochures, and any document of a type we do not know yet. Sorted by type
    (handbook, circular, lesson notes, brochure, other), then effective date, then id.

    today None means no college date has been seen yet. The edition is then judged against the
    latest effective date of the other documents, which the college has already issued (a lower
    bound for its date); with no such document, no handbook is chosen rather than guessing.
    """
    if role not in LOGGED_IN_ROLES:
        chosen = [d for d in docs if d["type"] == "brochure"]
    else:
        if today is None:
            issued = [d["effective_date"] for d in docs
                      if d["type"] != "handbook" and d["effective_date"] is not None]
            today = max(issued, default=date.min)
        handbooks = [d for d in docs if d["type"] == "handbook"
                     and d["effective_date"] is not None and d["effective_date"] <= today]
        current = max(handbooks, key=lambda d: (d["effective_date"], d["id"]), default=None)
        department = (user or {}).get("department")
        chosen = []
        for d in docs:
            if d["type"] == "handbook":
                if current is not None and d["id"] == current["id"]:
                    chosen.append(d)
            elif d["type"] == "lesson_notes":
                departments = (d["applies_to"] or {}).get("departments")
                if role == "teacher" or not departments or department in departments:
                    chosen.append(d)
            else:
                chosen.append(d)
    return sorted(chosen, key=_sort_key)


async def active_documents() -> list[dict]:
    """All active document rows with text, applies_to decoded."""
    rows = await db.pool().fetch(
        "SELECT * FROM documents WHERE active AND text IS NOT NULL ORDER BY id"
    )
    out = []
    for r in rows:
        d = dict(r)
        if isinstance(d["applies_to"], str):
            d["applies_to"] = json.loads(d["applies_to"])
        out.append(d)
    return out


async def applicable_documents(ctx: SessionCtx) -> list[dict]:
    """Non-blank document rows this user may use, in a deterministic order (see select_documents)."""
    docs = await active_documents()
    today = await _college_date() if ctx.role in LOGGED_IN_ROLES else None
    # Blank documents are never served; they are dropped after the edition choice so a blank
    # current handbook cannot bring back a superseded one.
    return [d for d in select_documents(docs, ctx.role, ctx.user, today) if strip_page_marks(d["text"]).strip()]
