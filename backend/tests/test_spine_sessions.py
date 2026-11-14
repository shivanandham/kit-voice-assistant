"""AC6 sessions and cookies, through the real app with the college faked."""

import asyncio
import hashlib
from http.cookies import SimpleCookie

import pytest
from app import auth
from app.config import get_settings
from app.db import pool

from tests.fakes.college import login_as

COOKIE = "kit_session"


def parse_set_cookie(header: str) -> dict:
    """Name, value and lower-cased attributes of one Set-Cookie header."""
    jar = SimpleCookie()
    jar.load(header)
    morsel = jar[COOKIE]
    attrs = {k: morsel[k] for k in morsel if morsel[k] not in ("", False)}
    flags = {p.strip().lower() for p in header.split(";")[1:]}
    return {"value": morsel.value, "attrs": attrs, "flags": flags}


async def session_rows():
    return await pool().fetch("SELECT * FROM sessions ORDER BY created_at")


@pytest.mark.parametrize("path", ["/api/me", "/", "/login", "/history", "/api/history"])
async def test_ac6_first_visit_gets_httponly_lax_cookie(app_client, path):
    r = await app_client.get(path)
    c = parse_set_cookie(r.headers["set-cookie"])
    assert "httponly" in c["flags"]
    assert "samesite=lax" in c["flags"]
    assert "path=/" in c["flags"]
    assert "secure" not in c["flags"]
    assert int(c["attrs"]["max-age"]) == get_settings().session_ttl_hours * 3600


@pytest.mark.parametrize("secure", [True, False])
async def test_ac6_secure_flag_follows_cookie_secure(browser, monkeypatch, secure):
    monkeypatch.setattr(get_settings(), "cookie_secure", secure)
    client = await browser(base_url="https://testserver")
    r = await client.get("/api/me")
    assert ("secure" in parse_set_cookie(r.headers["set-cookie"])["flags"]) is secure
    r = await client.post("/api/login", json={"login_id": "arjun", "password": "Kit@2026"})
    assert ("secure" in parse_set_cookie(r.headers["set-cookie"])["flags"]) is secure


async def test_ac6_login_cookie_attributes(app_client, fake_college):
    r = await app_client.post("/api/login", json={"login_id": "arjun", "password": "Kit@2026"})
    c = parse_set_cookie(r.headers["set-cookie"])
    assert {"httponly", "samesite=lax", "path=/"} <= c["flags"]


async def test_ac6_only_hash_of_token_stored(app_client, fake_college):
    await login_as(app_client, "arjun")
    token = app_client.cookies[COOKIE]
    rows = await session_rows()
    assert len(rows) == 1
    assert rows[0]["id"] == hashlib.sha256(token.encode()).hexdigest()
    dump = " ".join(str(v) for v in rows[0].values())
    assert token not in dump


async def test_ac6_token_has_enough_entropy(app_client):
    tokens = set()
    for _ in range(20):
        app_client.cookies.clear()
        await app_client.get("/api/me")
        tokens.add(app_client.cookies[COOKIE])
    assert len(tokens) == 20
    assert all(len(t) >= 32 for t in tokens)


async def test_ac6_existing_visitor_cookie_is_reused(app_client):
    await app_client.get("/api/me")
    r = await app_client.get("/api/me")
    assert "set-cookie" not in r.headers
    assert len(await session_rows()) == 1


@pytest.mark.parametrize("value", [
    "", "garbage", "x" * 8000, "' OR 1=1 --", "%00", "../../etc/passwd",
    "ä" * 10, hashlib.sha256(b"x").hexdigest(),
])
async def test_ac6_unknown_or_garbage_cookie_is_a_fresh_visitor(browser, fake_college, value):
    owner = await browser()
    await login_as(owner, "arjun")
    stranger = await browser()
    r = await stranger.get("/api/me", headers={"cookie": f"{COOKIE}={value}".encode()})
    assert r.status_code == 200
    assert r.json() == {"role": "visitor", "user": None}


async def test_ac6_session_id_hash_as_cookie_does_not_authenticate(browser, fake_college):
    owner = await browser()
    await login_as(owner, "arjun")
    sid = (await session_rows())[0]["id"]
    stranger = await browser()
    r = await stranger.get("/api/me", headers={"cookie": f"{COOKIE}={sid}"})
    assert r.json()["role"] == "visitor"


async def test_ac6_duplicated_cookies_never_mix_identities(browser, fake_college):
    a, b = await browser(), await browser()
    await login_as(a, "arjun")
    await login_as(b, "priya")
    ta, tb = a.cookies[COOKIE], b.cookies[COOKIE]
    c = await browser()
    for header in (f"{COOKIE}={ta}; {COOKIE}=junk", f"{COOKIE}=junk; {COOKIE}={ta}",
                   f"{COOKIE}={ta}; {COOKIE}={tb}"):
        r = await c.get("/api/me", headers={"cookie": header})
        user = r.json()["user"]
        assert user is None or user["login_id"] in {"arjun", "priya"}
        assert r.json()["role"] in {"visitor", "student"}


async def test_ac6_expired_session_is_no_session(app_client, fake_college):
    await login_as(app_client, "arjun")
    await pool().execute("UPDATE sessions SET expires_at = now() - interval '1 second'")
    r = await app_client.get("/api/me")
    assert r.json() == {"role": "visitor", "user": None}
    assert "set-cookie" in r.headers


@pytest.mark.parametrize("seconds_left,still_user", [(-3600, False), (0, False), (30, False),
                                                       (3600, True)])
async def test_ac6_expired_college_token_is_no_session(app_client, fake_college, seconds_left,
                                                      still_user):
    await login_as(app_client, "arjun")
    await pool().execute(
        "UPDATE sessions SET kit_token_expires_at = now() + make_interval(secs => $1)",
        float(seconds_left),
    )
    r = await app_client.get("/api/me")
    assert (r.json()["role"] == "student") is still_user
    if not still_user:
        owners = [row["owner_key"] for row in await session_rows()]
        assert "user:arjun" not in owners


async def test_ac6_session_id_rotates_on_login_fixation(browser, fake_college):
    attacker_known = await browser()
    await attacker_known.get("/")
    planted = attacker_known.cookies[COOKIE]
    await login_as(attacker_known, "arjun")
    assert attacker_known.cookies[COOKIE] != planted
    # The pre-login token no longer works for anybody.
    other = await browser()
    r = await other.get("/api/me", headers={"cookie": f"{COOKIE}={planted}"})
    assert r.json()["role"] == "visitor"
    assert await pool().fetchval(
        "SELECT count(*) FROM sessions WHERE id = $1", hashlib.sha256(planted.encode()).hexdigest()
    ) == 0


async def test_ac6_logout_ends_session_server_side(browser, fake_college):
    a = await browser()
    await login_as(a, "arjun")
    old = a.cookies[COOKIE]
    r = await a.post("/api/logout")
    assert r.status_code == 200
    assert "user:arjun" not in [row["owner_key"] for row in await session_rows()]
    replay = await browser()
    r = await replay.get("/api/me", headers={"cookie": f"{COOKIE}={old}"})
    assert r.json() == {"role": "visitor", "user": None}
    r = await replay.get("/api/history", headers={"cookie": f"{COOKIE}={old}"})
    assert r.json() == []


async def test_ac6_logout_clears_cookie(app_client, fake_college):
    await login_as(app_client, "arjun")
    r = await app_client.post("/api/logout")
    header = r.headers["set-cookie"].lower()
    assert header.startswith(f"{COOKIE}=") and ("max-age=0" in header or "expires=" in header)


@pytest.mark.parametrize("cookie", [None, "garbage"])
async def test_ac6_logout_without_session_is_harmless(browser, cookie):
    c = await browser()
    headers = {"cookie": f"{COOKIE}={cookie}"} if cookie else {}
    r = await c.post("/api/logout", headers=headers)
    assert r.status_code == 200


async def test_ac6_logout_only_ends_own_session(browser, fake_college):
    a, b = await browser(), await browser()
    await login_as(a, "arjun")
    await login_as(b, "arjun")  # same user, second device
    await a.post("/api/logout")
    assert (await b.get("/api/me")).json()["role"] == "student"
    assert (await a.get("/api/me")).json()["role"] == "visitor"


async def test_ac6_login_as_b_after_a_leaves_nothing_of_a(app_client, fake_college):
    from app import history

    await login_as(app_client, "arjun")
    token_a = app_client.cookies[COOKIE]
    ctx_a = await auth._lookup(token_a)
    cid = await history.create_conversation(ctx_a)
    await history.add_message(cid, "user", "arjun secret question")
    await login_as(app_client, "priya")
    me = (await app_client.get("/api/me")).json()
    assert me["user"]["login_id"] == "priya"
    assert await auth._lookup(token_a) is None
    assert (await app_client.get("/api/history")).json() == []
    r = await app_client.get(f"/api/history/{cid}")
    assert r.status_code == 404 and "arjun secret" not in r.text
    assert "arjun" not in (await app_client.get("/api/me")).text


async def test_ac6_failed_login_keeps_existing_session(app_client, fake_college):
    await login_as(app_client, "arjun")
    r = await app_client.post("/api/login", json={"login_id": "priya", "password": "bad"})
    assert r.status_code == 401
    assert (await app_client.get("/api/me")).json()["user"]["login_id"] == "arjun"


async def test_ac6_concurrent_logins_of_same_user(browser, fake_college):
    clients = [await browser() for _ in range(5)]
    results = await asyncio.gather(*(login_as(c, "priya") for c in clients))
    assert all(u["login_id"] == "priya" for u in results)
    tokens = {c.cookies[COOKIE] for c in clients}
    assert len(tokens) == 5
    for c in clients:
        assert (await c.get("/api/me")).json()["user"]["login_id"] == "priya"
    rows = await session_rows()
    assert len(rows) == 5
    assert len({r["kit_token"] for r in rows}) == 5  # each session has its own college token


async def test_ac6_expired_rows_are_pruned_on_login(app_client, fake_college):
    await auth._create_session("visitor:old", "visitor")
    await pool().execute("UPDATE sessions SET expires_at = now() - interval '1 day'")
    await login_as(app_client, "arjun")
    assert [r["owner_key"] for r in await session_rows()] == ["user:arjun"]


async def test_ac6_session_expiry_matches_ttl(app_client, fake_college, monkeypatch):
    monkeypatch.setattr(get_settings(), "session_ttl_hours", 2)
    await login_as(app_client, "arjun")
    row = (await session_rows())[0]
    hours = (row["expires_at"] - row["created_at"]).total_seconds() / 3600
    assert 1.99 < hours < 2.01


async def test_ac6_college_token_expiry_recorded(app_client, fake_college):
    fake_college.token_ttl = 600
    await login_as(app_client, "arjun")
    row = (await session_rows())[0]
    secs = (row["kit_token_expires_at"] - row["created_at"]).total_seconds()
    assert 590 < secs < 610


async def test_ac6_no_expires_in_means_session_ttl_bound(app_client, fake_college):
    fake_college.login_overrides["arjun"] = {"expires_in": None}
    await login_as(app_client, "arjun")
    row = (await session_rows())[0]
    assert row["kit_token_expires_at"] is None
    assert (await app_client.get("/api/me")).json()["role"] == "student"


async def test_ac6_websocket_lookup_needs_cookie(fake_college):
    class FakeWS:
        def __init__(self, cookies):
            self.cookies = cookies

    token, _ = await fake_college.db_session("arjun")
    assert (await auth.session_from_websocket(FakeWS({COOKIE: token}))).owner_key == "user:arjun"
    assert await auth.session_from_websocket(FakeWS({})) is None
    assert await auth.session_from_websocket(FakeWS({COOKIE: "nope"})) is None
    await pool().execute("UPDATE sessions SET expires_at = now() - interval '1 second'")
    assert await auth.session_from_websocket(FakeWS({COOKIE: token})) is None


async def test_ac6_invalidate_session(fake_college):
    token, ctx = await fake_college.db_session("arjun")
    await auth.invalidate_session(ctx.session_id)
    assert await auth._lookup(token) is None
    await auth.invalidate_session("does-not-exist")


async def test_ac6_ctx_from_row_accepts_dict_or_text_json(fake_college):
    token, ctx = await fake_college.db_session("kavya")
    looked = await auth._lookup(token)
    assert looked.user == ctx.user and looked.kit_token == ctx.kit_token
    row = dict(await pool().fetchrow("SELECT * FROM sessions"))
    row["user_json"] = dict(ctx.user)
    assert auth._ctx_from_row(row).user == ctx.user


async def test_ac6_visitor_session_has_no_user_or_token(app_client):
    await app_client.get("/")
    row = (await session_rows())[0]
    assert row["role"] == "visitor" and row["user_json"] is None and row["kit_token"] is None
    assert row["owner_key"].startswith("visitor:") and len(row["owner_key"]) > len("visitor:") + 16
