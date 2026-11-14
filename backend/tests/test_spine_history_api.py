"""HIS1 and the history module's contract (CONTRACT.md: History)."""

import asyncio
import json
import uuid

import asyncpg
import pytest
from app import history
from app.auth import SessionCtx
from app.db import pool

from tests.fakes.college import login_as


def ctx(owner: str = "user:a", role: str = "student") -> SessionCtx:
    return SessionCtx("sid", owner, role, None, None)


@pytest.mark.parametrize("role", ["visitor", "student", "teacher"])
async def test_create_conversation_records_owner_and_role(role):
    cid = await history.create_conversation(ctx("x:1", role))
    assert str(uuid.UUID(cid)) == cid
    row = await pool().fetchrow("SELECT * FROM conversations")
    assert row["owner_key"] == "x:1" and row["role"] == role and row["title"] is None


async def test_create_conversation_ids_unique():
    ids = await asyncio.gather(*(history.create_conversation(ctx()) for _ in range(30)))
    assert len(set(ids)) == 30


@pytest.mark.parametrize("value", [None, 123, 1.5, [], {}, b"bytes", True, "", "x", "1234"])
async def test_conversation_owned_rejects_non_uuid_values(value):
    await history.create_conversation(ctx())
    assert await history.conversation_owned(ctx(), value) is False


async def test_conversation_owned_unknown_uuid():
    assert await history.conversation_owned(ctx(), str(uuid.uuid4())) is False


@pytest.mark.parametrize("first,expected", [
    ("How do I apply for leave?", "How do I apply for leave?"),
    ("  lots   of\n\nspace\t here ", "lots of space here"),
    ("x" * 200, "x" * history.TITLE_LIMIT),
    ("தமிழ் கேள்வி", "தமிழ் கேள்வி"),
])
async def test_his1_title_from_first_user_message(first, expected):
    cid = await history.create_conversation(ctx())
    await history.add_message(cid, "user", first)
    await history.add_message(cid, "user", "second question")
    assert await pool().fetchval("SELECT title FROM conversations") == expected


async def test_his1_assistant_first_does_not_set_title():
    cid = await history.create_conversation(ctx())
    await history.add_message(cid, "assistant", "Hello, how can I help?")
    assert await pool().fetchval("SELECT title FROM conversations") is None
    await history.add_message(cid, "user", "Timetable please")
    assert await pool().fetchval("SELECT title FROM conversations") == "Timetable please"


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
async def test_his1_blank_first_user_message_does_not_pin_empty_title(blank):
    cid = await history.create_conversation(ctx())
    await history.add_message(cid, "user", blank)
    await history.add_message(cid, "user", "Real question")
    assert await pool().fetchval("SELECT title FROM conversations") == "Real question"


async def test_his1_messages_saved_in_order_and_updated_at_bumped():
    cid = await history.create_conversation(ctx())
    before = await pool().fetchval("SELECT updated_at FROM conversations")
    for i in range(6):
        await history.add_message(cid, "user" if i % 2 == 0 else "assistant", f"m{i}")
    after = await pool().fetchval("SELECT updated_at FROM conversations")
    assert after > before
    rows = await pool().fetch("SELECT content FROM messages ORDER BY id")
    assert [r["content"] for r in rows] == [f"m{i}" for i in range(6)]


async def test_add_message_unknown_conversation_fails_and_stores_nothing():
    with pytest.raises(asyncpg.ForeignKeyViolationError):
        await history.add_message(str(uuid.uuid4()), "user", "hello")
    assert await pool().fetchval("SELECT count(*) FROM messages") == 0


@pytest.mark.parametrize("limit,expected", [(1, ["m4"]), (3, ["m2", "m3", "m4"]),
                                            (40, [f"m{i}" for i in range(5)]), (0, [])])
async def test_recent_messages_limit_and_order(limit, expected):
    cid = await history.create_conversation(ctx())
    for i in range(5):
        await history.add_message(cid, "user", f"m{i}")
    got = await history.recent_messages(cid, limit=limit)
    assert [m["content"] for m in got] == expected
    assert all(set(m) == {"role", "content"} for m in got)


async def test_recent_messages_default_limit_is_40():
    cid = await history.create_conversation(ctx())
    for i in range(45):
        await history.add_message(cid, "user", f"m{i}")
    got = await history.recent_messages(cid)
    assert len(got) == 40 and got[0]["content"] == "m5"


async def test_recent_messages_unknown_conversation_empty():
    assert await history.recent_messages(str(uuid.uuid4())) == []


async def test_log_tool_call_stores_json_and_odd_values():
    cid = await history.create_conversation(ctx())
    await history.log_tool_call(cid, "t", {"when": uuid.UUID(int=1)}, ["a", 1], True)
    await history.log_tool_call(cid, "t2", {}, None, False)
    await history.log_tool_call(cid, "t3", {"x": 1}, "plain text", False)
    rows = await pool().fetch("SELECT * FROM tool_calls ORDER BY id")
    assert json.loads(rows[0]["input"]) == {"when": str(uuid.UUID(int=1))}
    assert json.loads(rows[0]["output"]) == ["a", 1] and rows[0]["is_error"] is True
    assert json.loads(rows[1]["output"]) is None
    assert json.loads(rows[2]["output"]) == "plain text"


async def test_his1_api_listing_and_detail(app_client, fake_college):
    await login_as(app_client, "arjun")
    from app import auth

    me = await auth._lookup(app_client.cookies["kit_session"])
    old = await history.create_conversation(me)
    await history.add_message(old, "user", "older")
    new = await history.create_conversation(me)
    await history.add_message(new, "user", "newer")
    await history.add_message(new, "assistant", "reply")
    empty = await history.create_conversation(me)
    listing = (await app_client.get("/api/history")).json()
    assert [c["id"] for c in listing] == [new, old]  # newest first, empty skipped
    assert empty not in [c["id"] for c in listing]
    assert listing[0]["message_count"] == 2 and listing[0]["title"] == "newer"
    for c in listing:
        for k in ("started_at", "updated_at"):
            assert "T" in c[k]
    await history.add_message(old, "user", "bump")
    assert [c["id"] for c in (await app_client.get("/api/history")).json()] == [old, new]
    detail = (await app_client.get(f"/api/history/{new}")).json()
    assert detail["id"] == new and detail["title"] == "newer"
    assert [(m["role"], m["content"]) for m in detail["messages"]] == [
        ("user", "newer"), ("assistant", "reply")]
    assert (await app_client.get(f"/api/history/{empty}")).json()["messages"] == []


async def test_his1_empty_history_for_new_visitor(app_client):
    r = await app_client.get("/api/history")
    assert r.status_code == 200 and r.json() == []


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
async def test_history_api_is_read_only(app_client, method):
    assert (await app_client.request(method, "/api/history")).status_code == 405
    r = await app_client.request(method, f"/api/history/{uuid.uuid4()}")
    assert r.status_code == 405
