"""Auth: sessions, login, logout, rotation, rate limit, origin check."""

import sys
import types

import httpx
import pytest
from app import auth
from app.config import get_settings
from app.db import pool
from fastapi import FastAPI

USER = {
    "id": 1,
    "login_id": "arjun",
    "full_name": "Arjun",
    "role": "student",
    "department": "CSE",
    "class_id": 3,
    "class_code": "CSE-A",
    "programme": "B.Tech",
    "year_of_study": 2,
    "notes": None,
}

LOGIN_LIMIT = get_settings().login_limit_per_minute


class FakeKitError(Exception):
    def __init__(self, status, message):
        self.status, self.message, self.code, self.field = status, message, "x", None


@pytest.fixture(autouse=True)
def fake_kit(monkeypatch):
    """Install a stand-in for app.kit.client so the real one is not needed."""
    state = {"fail": None}

    class FakeClient:
        async def login(self, login_id, password):
            if state["fail"]:
                raise state["fail"]
            return {
                "access_token": "jwt-123",
                "token_type": "bearer",
                "expires_in": 3600,
                "user": {**USER, "login_id": login_id},
            }

    mod = types.ModuleType("app.kit.client")
    mod.KitClient = FakeClient
    mod.KitError = FakeKitError
    monkeypatch.setitem(sys.modules, "app.kit.client", mod)
    auth.reset_rate_limit()
    return state


@pytest.fixture
async def client():
    app = FastAPI()
    app.include_router(auth.router)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        yield c


async def test_visitor_session_created(client):
    r = await client.get("/api/me")
    assert r.json() == {"role": "visitor", "user": None}
    cookie = client.cookies.get("kit_session")
    assert cookie
    row = await pool().fetchrow("SELECT * FROM sessions")
    assert row["id"] != cookie and row["owner_key"].startswith("visitor:")
    again = await client.get("/api/me")
    assert again.status_code == 200
    assert await pool().fetchval("SELECT count(*) FROM sessions") == 1
    assert "httponly" in r.headers["set-cookie"].lower()


async def test_login_success_rotates_session(client):
    await client.get("/api/me")
    before = await pool().fetchval("SELECT id FROM sessions")
    r = await client.post("/api/login", json={"login_id": "arjun", "password": "x"})
    assert r.status_code == 200 and r.json()["user"]["login_id"] == "arjun"
    assert "jwt-123" not in r.text
    rows = await pool().fetch("SELECT * FROM sessions")
    assert len(rows) == 1 and rows[0]["id"] != before
    assert rows[0]["owner_key"] == "user:arjun" and rows[0]["role"] == "student"
    assert rows[0]["kit_token"] == "jwt-123"
    me = await client.get("/api/me")
    assert me.json()["role"] == "student"


async def test_login_failure_returns_college_message(client, fake_kit):
    fake_kit["fail"] = FakeKitError(401, "Invalid credentials")
    r = await client.post("/api/login", json={"login_id": "arjun", "password": "bad"})
    assert r.status_code == 401 and r.json()["detail"] == "Invalid credentials"
    assert await pool().fetchval("SELECT count(*) FROM sessions") == 0


async def test_login_college_down_is_503(client, fake_kit):
    fake_kit["fail"] = httpx.ConnectError("boom")
    r = await client.post("/api/login", json={"login_id": "a", "password": "b"})
    assert r.status_code == 503


async def test_logout_deletes_session(client):
    await client.post("/api/login", json={"login_id": "arjun", "password": "x"})
    assert await pool().fetchval("SELECT count(*) FROM sessions") == 1
    r = await client.post("/api/logout")
    assert r.status_code == 200
    assert await pool().fetchval("SELECT count(*) FROM sessions") == 0
    assert (await client.get("/api/me")).json()["role"] == "visitor"


async def test_expired_session_is_replaced(client):
    await client.get("/api/me")
    await pool().execute("UPDATE sessions SET expires_at = now() - interval '1 hour'")
    r = await client.get("/api/me")
    assert r.json()["role"] == "visitor"


async def test_login_rate_limit(client):
    codes = [
        (await client.post("/api/login", json={"login_id": "a", "password": "b"})).status_code
        for _ in range(LOGIN_LIMIT + 2)
    ]
    assert codes[: LOGIN_LIMIT] == [200] * LOGIN_LIMIT
    assert codes[-1] == 429


def test_origin_allowed(monkeypatch):
    from app.config import get_settings

    s = get_settings()
    monkeypatch.setattr(s, "public_origin", "https://kit.example.in")
    monkeypatch.setattr(s, "cookie_secure", True)
    assert auth.origin_allowed("https://kit.example.in")
    assert not auth.origin_allowed("https://evil.com")
    assert not auth.origin_allowed("http://localhost:8000")
    assert not auth.origin_allowed(None)
    monkeypatch.setattr(s, "cookie_secure", False)
    assert auth.origin_allowed("http://localhost:8000")


async def test_expired_college_token_drops_session(client):
    await client.post("/api/login", json={"login_id": "arjun", "password": "x"})
    # Within the 60 s safety margin counts as expired.
    await pool().execute("UPDATE sessions SET kit_token_expires_at = now() + interval '30 seconds'")
    r = await client.get("/api/me")
    assert r.json()["role"] == "visitor"
    rows = await pool().fetch("SELECT owner_key FROM sessions")
    assert [x["owner_key"].split(":")[0] for x in rows] == ["visitor"]


async def test_websocket_lookup_returns_none_for_expired_token(client):
    token, _ = await auth._create_session("user:a", "student", None, "jwt", 3600)
    assert (await auth._lookup(token)) is not None
    await pool().execute("UPDATE sessions SET kit_token_expires_at = now() - interval '1 minute'")
    assert await auth._lookup(token) is None
    assert await pool().fetchval("SELECT count(*) FROM sessions") == 0


async def test_invalidate_session(client):
    token, ctx = await auth._create_session("user:a", "student", None, "jwt", 3600)
    await auth.invalidate_session(ctx.session_id)
    assert await auth._lookup(token) is None
