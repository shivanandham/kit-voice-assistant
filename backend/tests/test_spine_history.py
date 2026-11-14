"""History: ownership isolation, titles, ordering."""

import httpx
import pytest
from app import auth, history
from app.auth import SessionCtx
from app.db import pool
from fastapi import FastAPI


def ctx(owner: str) -> SessionCtx:
    return SessionCtx("sid", owner, "student", None, None)


@pytest.fixture
async def client():
    app = FastAPI()
    app.include_router(auth.router)
    app.include_router(history.router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        yield c


async def test_title_from_first_user_message_only():
    cid = await history.create_conversation(ctx("user:a"))
    await history.add_message(cid, "user", "  How do I  apply for leave? " + "x" * 200)
    await history.add_message(cid, "assistant", "Sure.")
    await history.add_message(cid, "user", "second")
    title = await pool().fetchval("SELECT title FROM conversations")
    assert title.startswith("How do I apply for leave?") and len(title) <= history.TITLE_LIMIT


async def test_ownership():
    cid = await history.create_conversation(ctx("user:a"))
    assert await history.conversation_owned(ctx("user:a"), cid)
    assert not await history.conversation_owned(ctx("user:b"), cid)
    assert not await history.conversation_owned(ctx("user:a"), "not-a-uuid")


async def test_recent_messages_order_and_limit():
    cid = await history.create_conversation(ctx("user:a"))
    for i in range(5):
        await history.add_message(cid, "user" if i % 2 == 0 else "assistant", f"m{i}")
    got = await history.recent_messages(cid, limit=3)
    assert [m["content"] for m in got] == ["m2", "m3", "m4"]
    assert set(got[0]) == {"role", "content"}


async def test_log_tool_call():
    cid = await history.create_conversation(ctx("user:a"))
    await history.log_tool_call(cid, "get_timetable", {"day": "mon"}, {"ok": True}, False)
    row = await pool().fetchrow("SELECT * FROM tool_calls")
    assert row["name"] == "get_timetable" and row["is_error"] is False


async def test_api_isolation(client):
    # Owner A (a logged-in session) creates a conversation.
    token, a = await auth._create_session("user:a", "student")
    cid = await history.create_conversation(a)
    await history.add_message(cid, "user", "hello")
    await history.add_message(cid, "assistant", "hi")

    client.cookies.set("kit_session", token)
    listing = (await client.get("/api/history")).json()
    assert len(listing) == 1 and listing[0]["message_count"] == 2
    detail = (await client.get(f"/api/history/{cid}")).json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["title"] == "hello"

    # Another visitor sees nothing and gets 404.
    other = httpx.AsyncClient(transport=client._transport, base_url="http://t")
    assert (await other.get("/api/history")).json() == []
    assert (await other.get(f"/api/history/{cid}")).status_code == 404
    assert (await other.get("/api/history/garbage")).status_code == 404
    await other.aclose()
