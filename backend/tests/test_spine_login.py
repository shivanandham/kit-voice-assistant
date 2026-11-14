"""Login: AC4 (role from the college only), AC5 (no secrets out), REL1 (college down), input
validation and the per-IP rate limit."""

import json
import logging

import httpx
import pytest
from app import auth, limits
from app.config import get_settings
from app.db import pool

from tests.conftest import DUMMY_KEYS
from tests.fakes.college import PASSWORD, USERS, login_as, user_summary

GOOD = {"login_id": "arjun", "password": PASSWORD}

LOGIN_LIMIT = get_settings().login_limit_per_minute


def secrets_in(text: str, fake) -> list[str]:
    """Any secret value that appears in `text`."""
    found = [k for k, v in DUMMY_KEYS.items() if v in text]
    found += ["kit_token" for t in fake.tokens if t in text]
    if PASSWORD in text:
        found.append("password")
    return found


def response_text(r: httpx.Response) -> str:
    return r.text + json.dumps(dict(r.headers))


# ------------------------------------------------------------------ AC4 role


@pytest.mark.parametrize("login", list(USERS))
async def test_ac4_role_comes_from_college(app_client, fake_college, login):
    user = await login_as(app_client, login)
    assert user == user_summary(login)
    me = (await app_client.get("/api/me")).json()
    assert me["role"] == USERS[login]["role"]
    row = await pool().fetchrow("SELECT owner_key, role FROM sessions")
    assert row["owner_key"] == f"user:{login}" and row["role"] == USERS[login]["role"]


@pytest.mark.parametrize("extra", [
    {"role": "teacher"}, {"role": "admin"}, {"user": {"role": "teacher", "login_id": "lakshmi"}},
    {"login_id_override": "lakshmi"}, {"owner_key": "user:lakshmi"},
])
async def test_ac4_role_and_identity_in_request_body_ignored(app_client, fake_college, extra):
    r = await app_client.post("/api/login", json={**GOOD, **extra})
    assert r.status_code == 200
    me = (await app_client.get("/api/me")).json()
    assert me["role"] == "student" and me["user"]["login_id"] == "arjun"


async def test_ac4_college_role_change_is_followed(app_client, fake_college):
    fake_college.login_overrides["arjun"] = {"user": {**user_summary("arjun"), "role": "teacher"}}
    await login_as(app_client, "arjun")
    assert (await app_client.get("/api/me")).json()["role"] == "teacher"


async def test_ac4_identity_is_the_colleges_login_id(app_client, fake_college):
    # The owner is whoever the college says logged in, not what was typed.
    fake_college.passwords["ARJUN"] = PASSWORD
    fake_college.login_overrides["ARJUN"] = {"user": user_summary("arjun")}
    r = await app_client.post("/api/login", json={"login_id": "ARJUN", "password": PASSWORD})
    assert r.status_code == 200
    assert await pool().fetchval("SELECT owner_key FROM sessions") == "user:arjun"


@pytest.mark.parametrize("role", ["admin", "visitor", "", None, "Student", "superuser"])
async def test_ac4_unknown_college_role_never_creates_session(app_client, fake_college, role):
    fake_college.login_overrides["arjun"] = {"user": {**user_summary("arjun"), "role": role}}
    r = await app_client.post("/api/login", json=GOOD)
    assert r.status_code in (502, 503)
    assert await pool().fetchval(
        "SELECT count(*) FROM sessions WHERE owner_key LIKE 'user:%'") == 0
    assert (await app_client.get("/api/me")).json()["role"] == "visitor"


@pytest.mark.parametrize("override", [
    {"user": None}, {"user": {"role": "student"}}, {"access_token": None}, {"access_token": ""},
    {"user": "arjun"},
])
async def test_rel1_malformed_college_login_answer_is_503_not_500(app_client, fake_college,
                                                                 override):
    fake_college.login_overrides["arjun"] = override
    r = await app_client.post("/api/login", json=GOOD)
    assert r.status_code == 503
    assert await pool().fetchval(
        "SELECT count(*) FROM sessions WHERE owner_key LIKE 'user:%'") == 0


# ------------------------------------------------------------------ AC2 login carries no token


async def test_ac2_login_call_carries_no_bearer_token(browser, fake_college):
    c = await browser()
    await login_as(c, "arjun")
    await login_as(c, "priya")  # second login from a logged-in browser
    assert [x.token for x in fake_college.calls("POST", "/v1/auth/login")] == [None, None]
    assert all(x.participant_key == DUMMY_KEYS["KIT_PARTICIPANT_KEY"] for x in fake_college.log)


async def test_ac2_stored_token_is_the_one_college_issued_for_that_user(browser, fake_college):
    a, b = await browser(), await browser()
    await login_as(a, "arjun")
    await login_as(b, "lakshmi")
    rows = await pool().fetch("SELECT owner_key, kit_token FROM sessions")
    for row in rows:
        assert f"user:{fake_college.tokens[row['kit_token']][0]}" == row["owner_key"]


# ------------------------------------------------------------------ AC5 secrets


async def test_ac5_login_response_has_no_secrets(app_client, fake_college):
    r = await app_client.post("/api/login", json=GOOD)
    assert r.status_code == 200
    assert secrets_in(response_text(r), fake_college) == []
    assert set(r.json()) == {"user"}
    assert "access_token" not in r.text and "kit_token" not in r.text


async def test_ac5_me_and_history_have_no_secrets(app_client, fake_college):
    await login_as(app_client, "arjun")
    for path in ("/api/me", "/api/history", "/", "/assistant", "/history", "/healthz"):
        r = await app_client.get(path)
        assert secrets_in(response_text(r), fake_college) == [], path


@pytest.mark.parametrize("body", [
    {"password": PASSWORD},
    {"login_id": "arjun", "password": 12345},
    {"login_id": ["arjun"], "password": PASSWORD},
    {"login_id": None, "password": PASSWORD},
    {"login_id": {"x": 1}, "password": PASSWORD},
])
async def test_ac5_validation_errors_do_not_echo_password(app_client, fake_college, body):
    r = await app_client.post("/api/login", json=body)
    assert r.status_code == 422
    assert PASSWORD not in r.text and "12345" not in r.text
    assert fake_college.calls("POST", "/v1/auth/login") == []


async def test_ac5_college_error_does_not_leak_secrets(app_client, fake_college):
    r = await app_client.post("/api/login", json={"login_id": "arjun", "password": "wrong-pw"})
    assert r.status_code == 401 and "wrong-pw" not in r.text
    assert secrets_in(response_text(r), fake_college) == []


async def test_ac5_no_secret_in_log_lines(app_client, fake_college, caplog):
    caplog.set_level(logging.DEBUG)
    await login_as(app_client, "arjun")
    await app_client.get("/api/me")
    await app_client.post("/api/login", json={"login_id": "arjun", "password": "nope"})
    fake_college.down = True
    await app_client.post("/api/login", json=GOOD)
    await app_client.post("/api/logout")
    text = "\n".join(r.getMessage() + str(r.args) for r in caplog.records)
    assert secrets_in(text, fake_college) == []


# ------------------------------------------------------------------ failures and REL1


async def test_cor6_wrong_password_relays_college_message(app_client, fake_college):
    r = await app_client.post("/api/login", json={"login_id": "arjun", "password": "x"})
    assert r.status_code == 401
    assert r.json()["detail"] == "Invalid login ID or password."
    assert await pool().fetchval("SELECT count(*) FROM sessions") == 0


@pytest.mark.parametrize("fault", [
    {"exc": httpx.ConnectError}, {"exc": httpx.ReadTimeout}, {"exc": httpx.ConnectTimeout},
    {"status": 500}, {"status": 502}, {"status": 503},
])
async def test_rel1_college_down_login_is_503_with_clear_message(app_client, fake_college, fault):
    fake_college.fail("/v1/auth/login", **fault)
    r = await app_client.post("/api/login", json=GOOD)
    assert r.status_code == 503
    assert "unavailable" in r.json()["detail"].lower()
    assert await pool().fetchval(
        "SELECT count(*) FROM sessions WHERE owner_key LIKE 'user:%'") == 0


async def test_rel1_college_fully_down(app_client, fake_college):
    fake_college.down = True
    r = await app_client.post("/api/login", json=GOOD)
    assert r.status_code == 503


async def test_rel1_unexpected_exception_in_client_is_503(app_client, fake_college, monkeypatch):
    from app.kit import client

    async def boom(self, *a, **k):
        raise RuntimeError("bug")

    monkeypatch.setattr(client.KitClient, "login", boom)
    r = await app_client.post("/api/login", json=GOOD)
    assert r.status_code == 503 and "bug" not in r.text


# ------------------------------------------------------------------ input validation


@pytest.mark.parametrize("body", [
    {}, {"login_id": "arjun"}, {"password": "x"}, [], "arjun", 42, None,
    {"login_id": 1, "password": "x"}, {"login_id": "arjun", "password": True},
])
async def test_login_rejects_malformed_body(app_client, fake_college, body):
    r = await app_client.post("/api/login", json=body)
    assert r.status_code == 422
    assert fake_college.calls("POST", "/v1/auth/login") == []


@pytest.mark.parametrize("content,ctype", [
    (b"login_id=arjun&password=Kit%402026", "application/x-www-form-urlencoded"),
    (json.dumps(GOOD).encode(), "text/plain"),
    (b"{not json", "application/json"),
    (b"", "application/json"),
])
async def test_login_rejects_non_json_bodies(app_client, fake_college, content, ctype):
    r = await app_client.post("/api/login", content=content, headers={"content-type": ctype})
    assert r.status_code == 422
    assert await pool().fetchval(
        "SELECT count(*) FROM sessions WHERE owner_key LIKE 'user:%'") == 0


@pytest.mark.parametrize("login_id", [
    "a" * 10_000, "arjun' OR '1'='1", "arjun; DROP TABLE sessions;--", "\x00", "ärjun",
    "../../admin", " arjun ", "",
])
async def test_login_odd_strings_are_just_wrong_credentials(app_client, fake_college, login_id):
    r = await app_client.post("/api/login", json={"login_id": login_id, "password": PASSWORD})
    assert r.status_code in (401, 422)
    assert await pool().fetchval("SELECT count(*) FROM sessions") == 0
    assert await pool().fetchval("SELECT to_regclass('sessions') IS NOT NULL")


async def test_login_extra_fields_ignored(app_client, fake_college):
    r = await app_client.post("/api/login", json={**GOOD, "remember": True})
    assert r.status_code == 200
    assert fake_college.calls("POST", "/v1/auth/login")[0].body == GOOD


@pytest.mark.parametrize("method", ["GET", "PUT", "DELETE", "PATCH"])
async def test_login_wrong_method(app_client, method):
    r = await app_client.request(method, "/api/login")
    assert r.status_code == 405


# ------------------------------------------------------------------ rate limit


async def hammer(client, n, **kw):
    return [(await client.post("/api/login", json={"login_id": f"x{i}", "password": "y"}, **kw))
            .status_code for i in range(n)]


async def test_login_rate_limit_per_ip(browser, fake_college):
    c = await browser(client_ip="203.0.113.7")
    codes = await hammer(c, LOGIN_LIMIT + 3)
    assert codes[: LOGIN_LIMIT] == [401] * LOGIN_LIMIT
    assert codes[LOGIN_LIMIT:] == [429] * 3
    assert len(fake_college.calls("POST", "/v1/auth/login")) == LOGIN_LIMIT
    r = await c.post("/api/login", json=GOOD)
    assert r.status_code == 429 and "wait" in r.json()["detail"].lower()


async def test_login_rate_limit_counts_successes_and_bad_bodies(browser, fake_college):
    c = await browser(client_ip="203.0.113.8")
    for _ in range(LOGIN_LIMIT // 2):
        await c.post("/api/login", json=GOOD)
    for _ in range(LOGIN_LIMIT - LOGIN_LIMIT // 2):
        await c.post("/api/login", json={})
    assert (await c.post("/api/login", json=GOOD)).status_code == 429


async def test_login_rate_limit_independent_ips(browser, fake_college):
    a = await browser(client_ip="203.0.113.1")
    b = await browser(client_ip="203.0.113.2")
    await hammer(a, LOGIN_LIMIT)
    assert (await a.post("/api/login", json=GOOD)).status_code == 429
    assert (await b.post("/api/login", json=GOOD)).status_code == 200


async def test_login_rate_limit_window_expires(browser, fake_college, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(limits, "_clock", lambda: now[0])
    c = await browser(client_ip="203.0.113.9")
    await hammer(c, LOGIN_LIMIT)
    assert (await c.post("/api/login", json=GOOD)).status_code == 429
    now[0] += auth.LOGIN_WINDOW_SECONDS + 1
    assert (await c.post("/api/login", json=GOOD)).status_code == 200


async def test_login_rate_limit_behind_proxy_uses_forwarded_client(browser, fake_college):
    proxy = await browser(client_ip="127.0.0.1")  # Caddy on the same host
    await hammer(proxy, LOGIN_LIMIT, headers={"x-forwarded-for": "198.51.100.1"})
    blocked = await proxy.post("/api/login", json=GOOD,
                               headers={"x-forwarded-for": "198.51.100.1"})
    assert blocked.status_code == 429
    other = await proxy.post("/api/login", json=GOOD, headers={"x-forwarded-for": "198.51.100.2"})
    assert other.status_code == 200


async def test_login_rate_limit_forged_leading_forwarded_entries_do_not_dodge(browser,
                                                                            fake_college):
    # Caddy appends the real peer to whatever the client sent; only the last entry is trusted.
    proxy = await browser(client_ip="127.0.0.1")
    for i in range(LOGIN_LIMIT):
        await proxy.post("/api/login", json={"login_id": f"x{i}", "password": "y"},
                         headers={"x-forwarded-for": f"10.0.0.{i}, 198.51.100.5"})
    r = await proxy.post("/api/login", json=GOOD,
                         headers={"x-forwarded-for": "10.9.9.9, 198.51.100.5"})
    assert r.status_code == 429


async def test_login_rate_limit_direct_client_cannot_dodge_with_forged_header(browser,
                                                                            fake_college):
    # A client that reaches the app directly (not via the local proxy) must not choose its IP.
    c = await browser(client_ip="203.0.113.50")
    for i in range(LOGIN_LIMIT):
        await c.post("/api/login", json={"login_id": f"x{i}", "password": "y"},
                     headers={"x-forwarded-for": f"192.0.2.{i}"})
    r = await c.post("/api/login", json=GOOD, headers={"x-forwarded-for": "192.0.2.200"})
    assert r.status_code == 429


@pytest.mark.parametrize("xff", ["", " ", ",", "not-an-ip"])
async def test_login_rate_limit_odd_forwarded_headers_do_not_crash(browser, fake_college, xff):
    c = await browser(client_ip="127.0.0.1")
    r = await c.post("/api/login", json=GOOD, headers={"x-forwarded-for": xff})
    assert r.status_code == 200


def test_client_ip_without_client():
    class R:
        headers: dict = {}
        client = None

    assert auth._client_ip(R()) == "unknown"


async def test_reset_rate_limit(browser, fake_college):
    c = await browser(client_ip="203.0.113.77")
    await hammer(c, LOGIN_LIMIT)
    auth.reset_rate_limit()
    assert (await c.post("/api/login", json=GOOD)).status_code == 200


def test_settings_keys_are_dummies_in_tests():
    s = get_settings()
    assert s.kit_participant_key == DUMMY_KEYS["KIT_PARTICIPANT_KEY"]
    assert s.anthropic_api_key == DUMMY_KEYS["ANTHROPIC_API_KEY"]
