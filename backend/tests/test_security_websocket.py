"""AC7 (and AC3 over the socket): /ws/voice needs a valid session and an allowed Origin; a
conversation can be resumed only by its owner. The socket is driven at the ASGI level in the
session event loop (where the asyncpg pool lives)."""

import asyncio
import json

import pytest
from app import auth, history
from app.config import get_settings
from app.db import pool
from fastapi import FastAPI

ORIGIN = "https://kit.bixops.in"


class WS:
    """Minimal ASGI WebSocket client."""

    def __init__(self, app, path="/ws/voice", origin=ORIGIN, cookie=None):
        headers = [(b"host", b"testserver")]
        if origin is not None:
            headers.append((b"origin", origin.encode()))
        if cookie is not None:
            headers.append((b"cookie", cookie.encode()))
        self.scope = {"type": "websocket", "asgi": {"version": "3.0"}, "http_version": "1.1",
                      "scheme": "wss", "path": path, "raw_path": path.encode(),
                      "query_string": b"", "root_path": "", "headers": headers,
                      "client": ("127.0.0.1", 1234), "server": ("testserver", 443),
                      "subprotocols": []}
        self.app, self.inbox, self.outbox = app, asyncio.Queue(), asyncio.Queue()
        self.task = None

    async def __aenter__(self):
        await self.inbox.put({"type": "websocket.connect"})
        self.task = asyncio.create_task(self.app(self.scope, self.inbox.get, self.outbox.put))
        return self

    async def __aexit__(self, *exc):
        await self.inbox.put({"type": "websocket.disconnect", "code": 1000})
        try:
            await asyncio.wait_for(self.task, 5)
        except Exception:
            self.task.cancel()

    async def recv(self, timeout=5.0):
        return await asyncio.wait_for(self.outbox.get(), timeout)

    async def close_code(self):
        while True:
            msg = await self.recv()
            if msg["type"] == "websocket.close":
                return msg.get("code")

    async def send_json(self, data):
        await self.inbox.put({"type": "websocket.receive", "text": json.dumps(data)})

    async def next_json(self, kind):
        while True:
            msg = await self.recv()
            if msg["type"] == "websocket.close":
                raise AssertionError(f"closed {msg.get('code')} waiting for {kind}")
            if msg["type"] == "websocket.send" and msg.get("text"):
                data = json.loads(msg["text"])
                if data.get("type") == kind:
                    return data


@pytest.fixture
def secure_origin(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "public_origin", ORIGIN)
    monkeypatch.setattr(s, "cookie_secure", True)


@pytest.mark.parametrize("origin", [None, "", "null", "https://evil.com",
                                    "https://kit.bixops.in.evil.com", "http://kit.bixops.in",
                                    "http://localhost:8000"])
async def test_ac7_real_socket_refuses_bad_origin(asgi_app, fake_college, secure_origin, origin):
    token, _ = await fake_college.db_session("arjun")
    async with WS(asgi_app, origin=origin, cookie=f"kit_session={token}") as ws:
        assert await ws.close_code() == 4403


@pytest.mark.parametrize("cookie", [None, "", "kit_session=", "kit_session=forged",
                                    "other=1", "kit_session=" + "x" * 5000])
async def test_ac7_real_socket_refuses_missing_or_bad_session(asgi_app, fake_college,
                                                              secure_origin, cookie):
    async with WS(asgi_app, cookie=cookie) as ws:
        assert await ws.close_code() == 4401


async def test_ac7_real_socket_refuses_expired_session(asgi_app, fake_college, secure_origin):
    token, _ = await fake_college.db_session("arjun")
    await pool().execute("UPDATE sessions SET expires_at = now() - interval '1 second'")
    async with WS(asgi_app, cookie=f"kit_session={token}") as ws:
        assert await ws.close_code() == 4401


async def test_ac7_real_socket_refuses_expired_college_token(asgi_app, fake_college,
                                                             secure_origin):
    token, _ = await fake_college.db_session("arjun")
    await pool().execute("UPDATE sessions SET kit_token_expires_at = now()")
    async with WS(asgi_app, cookie=f"kit_session={token}") as ws:
        assert await ws.close_code() == 4401


async def test_ac7_real_socket_refuses_logged_out_session(asgi_app, fake_college, secure_origin,
                                                          browser):
    c = await browser(base_url="https://testserver")
    from tests.fakes.college import login_as

    await login_as(c, "arjun")
    token = c.cookies["kit_session"]
    await c.post("/api/logout")
    async with WS(asgi_app, cookie=f"kit_session={token}") as ws:
        assert await ws.close_code() == 4401


# ------------------------------------------------------------------ resume ownership


class _Stt:
    async def send_audio(self, pcm):
        return None

    async def events(self):
        if False:
            yield None

    async def close(self):
        return None


class _Tts:
    async def warm(self):
        return None

    async def synthesize(self, text):
        if False:
            yield b""

    async def reset(self):
        return None

    async def close(self):
        return None


async def _no_turn(ctx, conversation_id, text):
    yield {"type": "done", "text": ""}


@pytest.fixture
def voice_app():
    from app.voice.session import build_router

    app = FastAPI()
    app.include_router(auth.router)
    app.include_router(build_router(run_turn=_no_turn, stt_factory=_Stt, tts_factory=_Tts))
    return app


async def _ready(app, token, conversation_id):
    async with WS(app, cookie=f"kit_session={token}") as ws:
        await ws.send_json({"type": "start", "conversation_id": conversation_id})
        return await ws.next_json("ready")


@pytest.mark.parametrize("owner,other", [("arjun", "priya"), ("arjun", "lakshmi"),
                                         ("lakshmi", "arjun"), ("arjun", None), (None, "arjun"),
                                         (None, None)])
async def test_ac7_ac3_resume_only_by_owner(voice_app, fake_college, secure_origin, owner, other):
    owner_token, owner_ctx = await fake_college.db_session(owner)
    other_token, other_ctx = await fake_college.db_session(other)
    cid = await history.create_conversation(owner_ctx)
    await history.add_message(cid, "user", "owner-secret")

    mine = await _ready(voice_app, owner_token, cid)
    assert mine["conversation_id"] == cid

    theirs = await _ready(voice_app, other_token, cid)
    assert theirs["conversation_id"] != cid
    assert "owner-secret" not in json.dumps(theirs)
    assert await history.conversation_owned(other_ctx, theirs["conversation_id"])
    assert not await history.conversation_owned(other_ctx, cid)
    assert await pool().fetchval("SELECT count(*) FROM messages WHERE conversation_id = $1::uuid",
                                 cid) == 1


@pytest.mark.parametrize("bad", [None, "", "garbage", 123, ["x"], {"id": 1}, True])
async def test_ac7_start_with_odd_conversation_id_gets_new_owned_one(voice_app, fake_college,
                                                                     secure_origin, bad):
    token, ctx = await fake_college.db_session("priya")
    ready = await _ready(voice_app, token, bad)
    assert await history.conversation_owned(ctx, ready["conversation_id"])
    assert ready["role"] == "student"


async def test_ac5_ready_message_has_no_secrets(voice_app, fake_college, secure_origin):
    token, ctx = await fake_college.db_session("arjun")
    ready = await _ready(voice_app, token, None)
    text = json.dumps(ready)
    assert ctx.kit_token not in text and token not in text and ctx.session_id not in text
    assert set(ready) == {"type", "conversation_id", "role", "user_name"}
