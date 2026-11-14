"""ABU10: HTTP request bodies are capped at 64 KB, before the body is read."""

import json

GOOD = {"login_id": "arjun", "password": "Kit@2026"}


def body_of(size: int) -> bytes:
    """A JSON login body padded to exactly `size` bytes."""
    base = json.dumps({**GOOD, "pad": ""}).encode()
    return json.dumps({**GOOD, "pad": "x" * (size - len(base))}).encode()


async def test_abu10_body_at_limit_is_accepted(browser, fake_college):
    c = await browser(client_ip="203.0.113.60")
    data = body_of(64 * 1024)
    assert len(data) == 64 * 1024
    r = await c.post("/api/login", content=data, headers={"content-type": "application/json"})
    assert r.status_code == 200


async def test_abu10_body_over_limit_is_413_and_never_reaches_college(browser, fake_college):
    c = await browser(client_ip="203.0.113.61")
    before = len(fake_college.calls("POST", "/v1/auth/login"))
    r = await c.post(
        "/api/login", content=body_of(64 * 1024 + 1), headers={"content-type": "application/json"}
    )
    assert r.status_code == 413 and "too large" in r.json()["detail"].lower()
    assert r.headers.get("x-content-type-options") == "nosniff"
    assert len(fake_college.calls("POST", "/v1/auth/login")) == before


async def test_abu10_streamed_body_without_length_is_cut_off(browser, fake_college):
    c = await browser(client_ip="203.0.113.62")
    sent = []

    async def chunks():
        for _ in range(100):  # 100 x 4 KB, far over the limit
            sent.append(1)
            yield b"x" * 4096

    r = await c.post("/api/login", content=chunks(), headers={"content-type": "application/json"})
    assert r.status_code == 413
    assert fake_college.calls("POST", "/v1/auth/login") == []


async def test_abu10_bad_content_length_is_refused(browser, fake_college):
    c = await browser(client_ip="203.0.113.63")
    r = await c.post(
        "/api/login",
        content=b"{}",
        headers={"content-length": "999999999", "content-type": "application/json"},
    )
    assert r.status_code == 413
