"""Database layer (CONTRACT.md: Database) and HIS3 (history survives a restart)."""

import uuid

import asyncpg
import pytest
from app import db, history
from app.auth import SessionCtx
from app.db import pool

from tests.conftest import TEST_DATABASE_URL, load_main

CONTRACT_COLUMNS = {
    "sessions": {"id", "owner_key", "role", "user_json", "kit_token", "kit_token_expires_at",
                 "created_at", "expires_at"},
    "conversations": {"id", "owner_key", "role", "title", "started_at", "updated_at"},
    "messages": {"id", "conversation_id", "role", "content", "created_at"},
    "tool_calls": {"id", "conversation_id", "name", "input", "output", "is_error", "created_at"},
    "pending_actions": {"id", "session_id", "conversation_id", "tool_name", "args", "summary",
                        "status", "created_at"},
    "documents": {"id", "title", "type", "version", "effective_date", "applies_to",
                  "is_surprise", "text", "token_estimate", "content_hash", "active",
                  "fetched_at"},
    "doc_chunks": {"id", "document_id", "chunk_index", "heading", "content", "embedding"},
}

NOT_NULL = {
    ("sessions", "owner_key"), ("sessions", "role"), ("sessions", "expires_at"),
    ("conversations", "owner_key"), ("conversations", "role"),
    ("messages", "role"), ("messages", "content"),
}


def ctx(owner="user:a"):
    return SessionCtx("s", owner, "student", None, None)


@pytest.mark.parametrize("table", list(CONTRACT_COLUMNS))
async def test_schema_has_contract_columns(table):
    rows = await pool().fetch(
        "SELECT column_name FROM information_schema.columns WHERE table_name = $1", table)
    assert CONTRACT_COLUMNS[table] <= {r["column_name"] for r in rows}


@pytest.mark.parametrize("table,column", sorted(NOT_NULL))
async def test_schema_not_null_columns(table, column):
    nullable = await pool().fetchval(
        "SELECT is_nullable FROM information_schema.columns WHERE table_name=$1 AND column_name=$2",
        table, column)
    assert nullable == "NO"


async def test_schema_applies_twice_without_error():
    async with pool().acquire() as conn:
        sql = db.SCHEMA_PATH.read_text()
        await conn.execute(sql)
        await conn.execute(sql)


async def test_vector_extension_and_dimension():
    assert await pool().fetchval("SELECT 1 FROM pg_extension WHERE extname = 'vector'") == 1
    await pool().execute("INSERT INTO documents (id, title) VALUES (1, 'd')")
    await pool().execute("INSERT INTO doc_chunks (document_id, embedding) VALUES (1, $1)",
                         [0.5] * 768)
    assert len((await pool().fetchval("SELECT embedding FROM doc_chunks")).to_list()) == 768
    with pytest.raises(asyncpg.DataError):
        await pool().execute("INSERT INTO doc_chunks (document_id, embedding) VALUES (1, $1)",
                             [0.5] * 3)


async def test_deleting_conversation_cascades_messages():
    keep = await history.create_conversation(ctx())
    drop = await history.create_conversation(ctx())
    for cid in (keep, drop):
        await history.add_message(cid, "user", "hi")
    await pool().execute("DELETE FROM conversations WHERE id = $1", uuid.UUID(drop))
    rows = await pool().fetch("SELECT conversation_id FROM messages")
    assert [str(r["conversation_id"]) for r in rows] == [keep]


async def test_deleting_document_cascades_chunks():
    await pool().execute("INSERT INTO documents (id, title) VALUES (7, 'd'), (8, 'e')")
    await pool().execute("INSERT INTO doc_chunks (document_id, content) VALUES (7,'a'),(8,'b')")
    await pool().execute("DELETE FROM documents WHERE id = 7")
    assert await pool().fetchval("SELECT array_agg(document_id) FROM doc_chunks") == [8]


async def test_message_requires_existing_conversation():
    with pytest.raises(asyncpg.ForeignKeyViolationError):
        await pool().execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES ($1, 'user', 'x')",
            uuid.uuid4())


async def test_session_id_is_primary_key():
    q = "INSERT INTO sessions (id, owner_key, role, expires_at) VALUES ('x','o','visitor',now())"
    await pool().execute(q)
    with pytest.raises(asyncpg.UniqueViolationError):
        await pool().execute(q)


@pytest.mark.parametrize("sql", [
    "INSERT INTO sessions (id, role, expires_at) VALUES ('a','visitor',now())",
    "INSERT INTO sessions (id, owner_key, role) VALUES ('a','o','visitor')",
    "INSERT INTO conversations (id, role) VALUES (gen_random_uuid(), 'visitor')",
    "INSERT INTO conversations (id, owner_key) VALUES (gen_random_uuid(), 'o')",
])
async def test_not_null_enforced(sql):
    with pytest.raises(asyncpg.NotNullViolationError):
        await pool().execute(sql)


async def test_defaults():
    cid = await history.create_conversation(ctx())
    row = await pool().fetchrow("SELECT * FROM conversations WHERE id = $1", uuid.UUID(cid))
    assert row["started_at"] is not None and row["updated_at"] is not None
    await pool().execute("INSERT INTO pending_actions (id) VALUES (gen_random_uuid())")
    assert await pool().fetchval("SELECT status FROM pending_actions") == "pending"
    await pool().execute("INSERT INTO documents (id) VALUES (1)")
    assert await pool().fetchval("SELECT active FROM documents") is True


async def test_pool_registers_vector_codec_on_every_connection():
    async with pool().acquire() as conn:
        assert await conn.fetchval("SELECT '[1,2,3]'::vector") is not None


async def test_init_db_is_idempotent():
    first = pool()
    await db.init_db()
    assert pool() is first


async def test_close_and_reinit_lifecycle():
    await db.close_db()
    with pytest.raises(RuntimeError, match="not initialised"):
        pool()
    await db.close_db()  # closing twice is harmless
    await db.init_db()
    assert await pool().fetchval("SELECT 1") == 1


async def test_his3_history_survives_restart():
    cid = await history.create_conversation(ctx("user:arjun"))
    await history.add_message(cid, "user", "remember me")
    await db.close_db()
    await db.init_db()
    assert await history.conversation_owned(ctx("user:arjun"), cid)
    assert await history.recent_messages(cid) == [{"role": "user", "content": "remember me"}]


async def test_his3_app_lifespan_restart_keeps_history_and_closes_everything(fake_college,
                                                                            monkeypatch):
    """The real lifespan: init_db and the sync loop on start; on shutdown stop the sync loop,
    close the college HTTP client and close the database."""
    main = load_main()
    events = []

    async def start():
        events.append("start_sync")

    async def stop():
        events.append("stop_sync")

    real_close_http = main.close_http

    async def close_http():
        events.append("close_http")
        await real_close_http()

    monkeypatch.setattr(main.store, "start_sync_loop", start)
    monkeypatch.setattr(main.store, "stop_sync_loop", stop)
    monkeypatch.setattr(main.get_settings(), "llm_keep_warm", False)
    monkeypatch.setattr(main, "close_http", close_http)
    cid = await history.create_conversation(ctx("user:arjun"))
    await history.add_message(cid, "user", "before restart")
    for _ in range(2):
        async with main.app.router.lifespan_context(main.app):
            assert await pool().fetchval("SELECT 1") == 1
        with pytest.raises(RuntimeError):
            pool()
    assert events == ["start_sync", "stop_sync", "close_http"] * 2
    await db.init_db()
    assert (await history.recent_messages(cid))[0]["content"] == "before restart"


async def test_lifespan_closes_db_even_if_sync_stop_fails(fake_college, monkeypatch):
    main = load_main()

    async def start():
        return None

    async def stop():
        raise RuntimeError("sync stop failed")

    monkeypatch.setattr(main.store, "start_sync_loop", start)
    monkeypatch.setattr(main.store, "stop_sync_loop", stop)
    monkeypatch.setattr(main.get_settings(), "llm_keep_warm", False)
    try:
        async with main.app.router.lifespan_context(main.app):
            pass
    except RuntimeError:
        pass
    with pytest.raises(RuntimeError, match="not initialised"):
        pool()
    await db.init_db()


def test_test_database_is_isolated():
    assert "test" in TEST_DATABASE_URL
    from app.config import get_settings

    assert get_settings().database_url == TEST_DATABASE_URL
