"""ABU5 under the shared-network rule: HTTP requests from logged-in sessions are counted per
login id, never per IP; visitors are counted per IP."""

import pytest
from app import auth, limits

IP = "203.0.113.70"


@pytest.fixture
def clock(monkeypatch):
    t = [1000.0]
    monkeypatch.setattr(limits, "_clock", lambda: t[0])
    return t


async def student(browser, login: str, ip: str = IP):
    """A browser holding a real logged-in session for `login`."""
    user = {"id": f"id-{login}", "login_id": login, "full_name": login, "role": "student"}
    token, _ = await auth._create_session(f"user:{login}", "student", user, "tok", 3600)
    c = await browser(client_ip=ip)
    c.cookies.set(auth.COOKIE_NAME, token)
    return c


async def hits(c, n: int) -> list[int]:
    return [(await c.get("/api/me")).status_code for _ in range(n)]


async def test_abu5_students_on_one_ip_each_get_their_own_allowance(browser, fake_college):
    a = await student(browser, "arjun")
    b = await student(browser, "priya")
    assert await hits(a, 120) == [200] * 120
    assert await hits(b, 120) == [200] * 120  # same IP, separate allowance
    r = await a.get("/api/me")
    assert r.status_code == 429
    assert (await b.get("/api/me")).status_code == 429  # b has used its own 120 too
    c = await student(browser, "kavya")
    assert (await c.get("/api/me")).status_code == 200


async def test_abu5_user_counted_per_login_across_devices(browser, fake_college):
    phone = await student(browser, "rahul", ip="203.0.113.71")
    laptop = await student(browser, "rahul", ip="203.0.113.72")
    await hits(phone, 60)
    await hits(laptop, 60)
    assert (await phone.get("/api/me")).status_code == 429
    assert (await laptop.get("/api/me")).status_code == 429


async def test_abu5_visitor_on_shared_ip_bounded_and_separate(browser, fake_college):
    s = await student(browser, "meena")
    await hits(s, 100)
    visitor = await browser(client_ip=IP)
    assert await hits(visitor, 120) == [200] * 120  # students did not consume it
    assert (await visitor.get("/api/me")).status_code == 429
    assert (await s.get("/api/me")).status_code == 200  # nor the other way round


async def test_abu5_user_window_expires(browser, fake_college, monkeypatch):
    now = [5000.0]
    monkeypatch.setattr(auth.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(limits, "_clock", lambda: now[0])
    s = await student(browser, "divya")
    await hits(s, 120)
    assert (await s.get("/api/me")).status_code == 429
    now[0] += 61
    assert (await s.get("/api/me")).status_code == 200
