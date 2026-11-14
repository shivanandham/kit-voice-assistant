"""HTTP hygiene: security headers on every response type, CSP, safe error bodies, static path
traversal, /healthz, and the WebSocket origin check table (AC5, AC7, PLAN Security)."""

import pytest
from app import auth
from app.config import get_settings

from tests.conftest import DUMMY_KEYS, TEST_DATABASE_URL, load_main
from tests.fakes.college import PASSWORD, login_as

REQUIRED_HEADERS = {
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "strict-origin-when-cross-origin",
}

LOGIN_LIMIT = get_settings().login_limit_per_minute


def assert_security_headers(r):
    for k, v in REQUIRED_HEADERS.items():
        assert r.headers.get(k) == v, (k, r.status_code, r.url)
    csp = r.headers.get("content-security-policy", "")
    directives = {d.split()[0]: d.split()[1:] for d in csp.split(";") if d.strip()}
    assert directives.get("default-src") == ["'self'"]
    assert directives.get("frame-ancestors") == ["'none'"]
    assert "object-src" in directives or directives["default-src"] == ["'self'"]
    for d in ("script-src", "default-src"):
        assert "'unsafe-inline'" not in directives.get(d, [])
        assert "'unsafe-eval'" not in directives.get(d, [])
    assert "microphone=(self)" in r.headers.get("permissions-policy", "")


CASES = [
    ("GET", "/", None),  # HTML page
    ("GET", "/login", None),
    ("GET", "/assistant", None),  # redirect for visitors
    ("GET", "/api/me", None),  # JSON
    ("GET", "/healthz", None),
    ("GET", "/api/public/college", None),
    ("GET", "/static/app.css", None),  # static file
    ("GET", "/static/does-not-exist.js", None),  # static 404
    ("GET", "/nope", None),  # router 404
    ("DELETE", "/api/me", None),  # 405
    ("POST", "/api/login", {}),  # 422
    ("POST", "/api/login", {"login_id": "arjun", "password": "wrong"}),  # 401
    ("GET", "/api/history/garbage", None),  # 404 from handler
]


@pytest.mark.parametrize("method,path,body", CASES)
async def test_security_headers_on_every_response_type(app_client, fake_college, method, path,
                                                       body):
    r = await app_client.request(method, path, json=body)
    assert_security_headers(r)


async def test_security_headers_on_429_and_503(browser, fake_college):
    c = await browser(client_ip="203.0.113.99")
    for _ in range(LOGIN_LIMIT):
        await c.post("/api/login", json={"login_id": "x", "password": "y"})
    r = await c.post("/api/login", json={"login_id": "x", "password": "y"})
    assert r.status_code == 429
    assert_security_headers(r)
    fake_college.down = True
    r = await (await browser()).get("/api/public/courses")
    assert r.status_code == 503
    assert_security_headers(r)


async def test_unexpected_500_has_headers_and_no_internals(app_client, monkeypatch):
    async def boom(request, response):
        raise RuntimeError(f"internal detail {TEST_DATABASE_URL}")

    monkeypatch.setattr(auth, "session_from_request", boom)
    r = await app_client.get("/api/me")
    assert r.status_code == 500
    assert_security_headers(r)
    assert "internal detail" not in r.text and "Traceback" not in r.text
    assert "postgresql" not in r.text


@pytest.mark.parametrize("method,path,body", CASES)
async def test_error_bodies_have_no_stack_traces_or_internals(app_client, fake_college, method,
                                                             path, body):
    r = await app_client.request(method, path, json=body)
    for needle in ("Traceback", "File \"", "asyncpg", "postgresql://", "site-packages",
                   *DUMMY_KEYS.values(), PASSWORD):
        assert needle not in r.text, needle


@pytest.mark.parametrize("path,allowed", [
    ("/api/me", {"GET"}), ("/api/login", {"POST"}), ("/api/logout", {"POST"}),
    ("/api/history", {"GET"}), ("/healthz", {"GET"}),
])
@pytest.mark.parametrize("method", ["GET", "POST", "PUT", "DELETE", "PATCH"])
async def test_method_not_allowed(app_client, fake_college, path, allowed, method):
    if method in allowed:
        return
    r = await app_client.request(method, path)
    assert r.status_code == 405
    assert_security_headers(r)


TRAVERSALS = [
    "/static/../.env", "/static/../../.env", "/static/%2e%2e/%2e%2e/.env",
    "/static/..%2f..%2f.env", "/static/..%2F..%2Fbackend%2Fapp%2Fconfig.py",
    "/static/%2e%2e%2f%2e%2e%2f.env", "/static/....//....//.env", "/static//etc/passwd",
    "/static/%252e%252e/.env", "/static/..\\..\\.env", "/static/../index.html",
]


@pytest.mark.parametrize("path", TRAVERSALS)
async def test_static_path_traversal_blocked(app_client, path):
    r = await app_client.get(path)
    assert r.status_code in (400, 404) or "text/html" in r.headers.get("content-type", "")
    for needle in ("KIT_PARTICIPANT_KEY", "DATABASE_URL", "pydantic_settings", "root:"):
        assert needle not in r.text


@pytest.mark.parametrize("raw", [b"/static/../.env", b"/static/../../.env",
                                 b"/static/../../backend/app/config.py"])
async def test_static_path_traversal_raw_asgi(asgi_app, raw):
    """Send an un-normalised path straight to the ASGI app (clients normalise ../ away)."""
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        sent.append(msg)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
             "scheme": "http", "path": raw.decode(), "raw_path": raw, "query_string": b"",
             "root_path": "", "headers": [(b"host", b"testserver")], "client": ("127.0.0.1", 1),
             "server": ("testserver", 80)}
    await asgi_app(scope, receive, send)
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    assert status == 404
    assert b"KIT_PARTICIPANT_KEY" not in body and b"get_settings" not in body


async def test_static_serves_frontend_assets(app_client):
    r = await app_client.get("/static/app.css")
    assert r.status_code == 200 and "text/css" in r.headers["content-type"]


async def test_healthz_shape_and_no_secrets(app_client):
    r = await app_client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"status", "db", "doc_mode"}
    assert body["status"] == "ok" and body["db"] == "ok"
    assert body["doc_mode"] == get_settings().doc_mode
    for needle in ("postgresql", "localhost:5432", "kit_test", *DUMMY_KEYS.values()):
        assert needle not in r.text


async def test_healthz_reports_db_down_without_details(app_client, monkeypatch):
    main = load_main()

    def broken():
        raise RuntimeError(f"cannot connect to {TEST_DATABASE_URL}")

    monkeypatch.setattr(main, "pool", broken)
    r = await app_client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "degraded" and r.json()["db"] == "down"
    assert "postgresql" not in r.text


async def test_unknown_routes_404_json(app_client):
    for path in ("/api/nope", "/admin", "/.env", "/backend/app/config.py", "/api/history/"):
        r = await app_client.get(path)
        assert r.status_code in (404, 307), path
        assert "KIT_PARTICIPANT_KEY" not in r.text


async def test_logged_in_pages_have_headers(app_client, fake_college):
    await login_as(app_client, "arjun")
    for path in ("/", "/assistant", "/history", "/api/me", "/api/history"):
        assert_security_headers(await app_client.get(path))


# ------------------------------------------------------------------ AC7 origin table

PUBLIC_ORIGIN = "https://kit.bixops.in"

ORIGINS = [
    # origin, allowed when secure, allowed when not secure
    ("https://kit.bixops.in", True, True),
    ("https://kit.bixops.in/", True, True),
    ("https://KIT.BIXOPS.IN", True, True),
    ("http://kit.bixops.in", False, False),
    ("wss://kit.bixops.in", False, False),
    ("https://kit.bixops.in:8443", False, False),
    ("https://kit.bixops.in.evil.com", False, False),
    ("https://evilkit.bixops.in", False, False),
    ("https://evil.com/kit.bixops.in", False, False),
    ("https://kit.bixops.in@evil.com", False, False),
    ("https://bixops.in", False, False),
    ("https://sub.kit.bixops.in", False, False),
    ("null", False, False),
    ("", False, False),
    (None, False, False),
    ("file://", False, False),
    ("http://localhost:8000", False, True),
    ("http://localhost", False, True),
    ("http://127.0.0.1:8000", False, True),
    ("http://[::1]:8000", False, True),
    ("http://localhost.evil.com", False, False),
    ("http://127.0.0.1.nip.io", False, False),
    ("http://evil.com#localhost", False, False),
    ("http://localhost@evil.com", False, False),
    ("garbage", False, False),
    ("http://localhost:8000.evil.com", False, False),
    ("http://localhost:evil", False, False),
    ("ftp://localhost", False, False),
    ("ws://localhost:8000", False, False),
    ("javascript://localhost", False, False),
    ("http://localhost:8000/path", False, False),
    ("https://localhost:8443", False, True),
]


@pytest.mark.parametrize("origin,secure_ok,insecure_ok", ORIGINS)
@pytest.mark.parametrize("secure", [True, False])
def test_ac7_origin_allowed_table(monkeypatch, origin, secure_ok, insecure_ok, secure):
    s = get_settings()
    monkeypatch.setattr(s, "public_origin", PUBLIC_ORIGIN)
    monkeypatch.setattr(s, "cookie_secure", secure)
    assert auth.origin_allowed(origin) is (secure_ok if secure else insecure_ok)


@pytest.mark.parametrize("configured", ["https://kit.bixops.in/", "HTTPS://Kit.Bixops.In"])
def test_ac7_origin_allowed_normalises_configured_origin(monkeypatch, configured):
    s = get_settings()
    monkeypatch.setattr(s, "public_origin", configured)
    monkeypatch.setattr(s, "cookie_secure", True)
    assert auth.origin_allowed("https://kit.bixops.in")
