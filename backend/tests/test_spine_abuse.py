"""ABU4, ABU5, ABU9: the spine's limits (new anonymous sessions, HTTP requests, logins) and the
client IP. Limits are settings; exceeding one is a polite 429 that never affects other users."""

import pytest
from app import auth, limits
from app.config import get_settings

from tests.fakes.college import PASSWORD, login_as

GOOD = {"login_id": "arjun", "password": PASSWORD}


@pytest.fixture
def clock(monkeypatch):
    now = [10_000.0]
    monkeypatch.setattr(limits, "_clock", lambda: now[0])
    return now


def test_abu_limits_are_settings_with_defaults():
    s = get_settings()
    assert s.anon_sessions_per_hour == 30
    assert s.http_limit_per_minute == 120
    assert s.login_limit_per_minute == 30
    assert s.login_failures_per_id == 5
    assert s.login_failure_window_seconds == 900


# ------------------------------------------------------------------ ABU4


async def test_abu4_new_anonymous_sessions_limited_per_ip(browser):
    limit = get_settings().anon_sessions_per_hour
    codes = []
    for _ in range(limit + 2):
        c = await browser(client_ip="198.51.100.20")
        codes.append((await c.get("/api/me")).status_code)
    assert codes[:limit] == [200] * limit and codes[limit:] == [429, 429]


async def test_abu4_limit_message_is_clear_and_other_ips_unaffected(browser):
    limit = get_settings().anon_sessions_per_hour
    for _ in range(limit):
        await (await browser(client_ip="198.51.100.21")).get("/")
    r = await (await browser(client_ip="198.51.100.21")).get("/")
    assert r.status_code == 429 and "try again" in r.json()["detail"].lower()
    assert (await (await browser(client_ip="198.51.100.22")).get("/")).status_code == 200


async def test_abu4_existing_session_never_creates_another(browser):
    c = await browser(client_ip="198.51.100.23")
    for _ in range(get_settings().anon_sessions_per_hour + 5):
        assert (await c.get("/api/me")).status_code == 200
    from app.db import pool

    assert await pool().fetchval("SELECT count(*) FROM sessions") == 1


async def test_abu4_logged_in_users_unaffected_by_anonymous_limit(browser, fake_college):
    user = await browser(client_ip="198.51.100.24")
    await login_as(user, "arjun")
    for _ in range(get_settings().anon_sessions_per_hour):
        await (await browser(client_ip="198.51.100.24")).get("/")
    assert (await user.get("/api/me")).json()["role"] == "student"


async def test_abu4_window_is_one_hour(browser, clock):
    limit = get_settings().anon_sessions_per_hour
    for _ in range(limit):
        await (await browser(client_ip="198.51.100.25")).get("/")
    assert (await (await browser(client_ip="198.51.100.25")).get("/")).status_code == 429
    clock[0] += 3601
    assert (await (await browser(client_ip="198.51.100.25")).get("/")).status_code == 200


async def test_abu4_limit_follows_setting(browser, monkeypatch):
    monkeypatch.setattr(get_settings(), "anon_sessions_per_hour", 2)
    codes = [(await (await browser(client_ip="198.51.100.26")).get("/")).status_code
             for _ in range(3)]
    assert codes == [200, 200, 429]


# ------------------------------------------------------------------ ABU5 HTTP


async def test_abu5_http_requests_limited_per_ip(browser):
    limit = get_settings().http_limit_per_minute
    c = await browser(client_ip="198.51.100.30")
    codes = [(await c.get("/healthz")).status_code for _ in range(limit + 1)]
    assert codes[:limit] == [200] * limit and codes[-1] == 429
    other = await browser(client_ip="198.51.100.31")
    assert (await other.get("/healthz")).status_code == 200


async def test_abu5_static_files_excluded(browser, monkeypatch):
    monkeypatch.setattr(get_settings(), "http_limit_per_minute", 3)
    c = await browser(client_ip="198.51.100.32")
    for _ in range(10):
        assert (await c.get("/static/app.css")).status_code == 200
    assert (await c.get("/healthz")).status_code == 200


async def test_abu5_http_limit_message_and_window(browser, monkeypatch, clock):
    monkeypatch.setattr(get_settings(), "http_limit_per_minute", 2)
    c = await browser(client_ip="198.51.100.33")
    await c.get("/healthz")
    await c.get("/healthz")
    r = await c.get("/healthz")
    assert r.status_code == 429 and r.json()["detail"]
    assert r.headers.get("x-content-type-options") == "nosniff"
    clock[0] += 61
    assert (await c.get("/healthz")).status_code == 200


async def test_abu5_logged_in_users_limited_per_login_id_not_ip(browser, fake_college,
                                                                monkeypatch):
    """Two students behind one network each get their own allowance."""
    monkeypatch.setattr(get_settings(), "http_limit_per_minute", 5)
    arjun = await browser(client_ip="198.51.100.34")
    priya = await browser(client_ip="198.51.100.34")
    await login_as(arjun, "arjun")
    await login_as(priya, "priya")
    for _ in range(4):
        assert (await arjun.get("/api/me")).status_code == 200
    assert (await priya.get("/api/me")).status_code == 200


async def test_abu5_logged_in_limit_follows_the_user_across_ips(browser, fake_college,
                                                               monkeypatch):
    monkeypatch.setattr(get_settings(), "http_limit_per_minute", 3)
    first = await browser(client_ip="198.51.100.35")
    await login_as(first, "arjun")
    second = await browser(client_ip="198.51.100.36")
    await login_as(second, "arjun")
    codes = [(await c.get("/api/me")).status_code for c in (first, second, first, second)]
    assert 429 in codes


# ------------------------------------------------------------------ ABU5 logins


async def test_abu5_login_limit_per_ip(browser, fake_college):
    limit = get_settings().login_limit_per_minute
    c = await browser(client_ip="198.51.100.40")
    codes = [(await c.post("/api/login", json={"login_id": f"user{i}", "password": "x"}))
             .status_code for i in range(limit + 1)]
    assert codes[:limit] == [401] * limit and codes[-1] == 429


async def test_abu5_failures_per_login_id(browser, fake_college):
    n = get_settings().login_failures_per_id
    codes = []
    for i in range(n + 1):
        c = await browser(client_ip=f"198.51.100.{50 + i}")  # spread over IPs
        codes.append((await c.post("/api/login", json={"login_id": "arjun", "password": "bad"}))
                     .status_code)
    assert codes[:n] == [401] * n and codes[-1] == 429
    calls = fake_college.calls("POST", "/v1/auth/login")
    assert len(calls) == n  # the locked attempt never reached the college


async def test_abu5_login_id_lock_is_case_and_space_insensitive(browser, fake_college):
    n = get_settings().login_failures_per_id
    for i, spelled in enumerate(["arjun", "ARJUN", " arjun", "Arjun ", "arJun"][:n]):
        c = await browser(client_ip=f"198.51.100.{70 + i}")
        await c.post("/api/login", json={"login_id": spelled, "password": "bad"})
    c = await browser(client_ip="198.51.100.90")
    assert (await c.post("/api/login", json=GOOD)).status_code == 429


async def test_abu5_login_id_lock_does_not_affect_other_accounts(browser, fake_college):
    n = get_settings().login_failures_per_id
    for i in range(n):
        c = await browser(client_ip=f"198.51.100.{100 + i}")
        await c.post("/api/login", json={"login_id": "arjun", "password": "bad"})
    c = await browser(client_ip="198.51.100.120")
    r = await c.post("/api/login", json={"login_id": "priya", "password": PASSWORD})
    assert r.status_code == 200


async def test_abu5_login_id_lock_expires(browser, fake_college, clock):
    s = get_settings()
    for i in range(s.login_failures_per_id):
        c = await browser(client_ip=f"198.51.100.{130 + i}")
        await c.post("/api/login", json={"login_id": "arjun", "password": "bad"})
    c = await browser(client_ip="198.51.100.140")
    assert (await c.post("/api/login", json=GOOD)).status_code == 429
    clock[0] += s.login_failure_window_seconds + 1
    assert (await c.post("/api/login", json=GOOD)).status_code == 200


async def test_abu5_college_outage_is_not_a_login_failure(browser, fake_college):
    fake_college.down = True
    for i in range(get_settings().login_failures_per_id + 1):
        c = await browser(client_ip=f"198.51.100.{150 + i}")
        assert (await c.post("/api/login", json=GOOD)).status_code == 503
    fake_college.down = False
    c = await browser(client_ip="198.51.100.170")
    assert (await c.post("/api/login", json=GOOD)).status_code == 200


async def test_abu5_lock_message_is_polite(browser, fake_college, monkeypatch):
    monkeypatch.setattr(get_settings(), "login_failures_per_id", 1)
    c = await browser(client_ip="198.51.100.180")
    await c.post("/api/login", json={"login_id": "arjun", "password": "bad"})
    r = await c.post("/api/login", json=GOOD)
    assert r.status_code == 429 and "minutes" in r.json()["detail"]
    assert PASSWORD not in r.text


# ------------------------------------------------------------------ ABU9


@pytest.mark.parametrize("peer,xff,expected", [
    ("127.0.0.1", "198.51.100.1", "198.51.100.1"),
    ("127.0.0.1", "10.0.0.1, 198.51.100.1", "198.51.100.1"),
    ("::1", "1.2.3.4,198.51.100.1", "198.51.100.1"),
    ("127.0.0.1", None, "127.0.0.1"),
    ("127.0.0.1", "", "127.0.0.1"),
    ("127.0.0.1", " , ", "127.0.0.1"),
    ("203.0.113.5", "198.51.100.1", "203.0.113.5"),
    ("203.0.113.5", None, "203.0.113.5"),
    ("10.1.1.1", "127.0.0.1", "10.1.1.1"),
])
def test_abu9_client_ip(peer, xff, expected):
    class R:
        headers = {"x-forwarded-for": xff} if xff is not None else {}
        client = type("C", (), {"host": peer})()

    assert auth._client_ip(R()) == expected

