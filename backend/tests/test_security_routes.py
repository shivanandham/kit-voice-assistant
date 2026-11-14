"""AC1: a visitor can never reach student or teacher data. Every route of the real app is
enumerated; a route nobody classified fails the suite, so new routes must be reviewed here."""

import uuid

import pytest
from app import auth, history
from starlette.routing import Mount

from tests.conftest import load_main
from tests.fakes.college import STUDENTS, USERS, login_as

PUBLIC = {
    ("GET", "/"), ("GET", "/login"), ("GET", "/history"), ("GET", "/healthz"),
    ("GET", "/api/public/college"), ("GET", "/api/public/courses"),
    ("POST", "/api/login"), ("POST", "/api/logout"), ("GET", "/api/me"),
}
OWN_SESSION_ONLY = {("GET", "/api/history"), ("GET", "/api/history/{conversation_id}")}
LOGIN_REQUIRED = {("GET", "/assistant")}
WEBSOCKETS = {"/ws/voice"}
MOUNTS = {"/static"}


def all_routes(routes=None, prefix=""):
    """(kind, method, path) for every route, descending into included routers."""
    if routes is None:
        routes = load_main().app.routes
    out = []
    for r in routes:
        if hasattr(r, "original_router"):
            out += all_routes(r.original_router.routes, prefix + (r.include_context.prefix or ""))
        elif isinstance(r, Mount):
            out.append(("mount", None, prefix + r.path))
        elif type(r).__name__.endswith("WebSocketRoute"):
            out.append(("ws", None, prefix + r.path))
        else:
            for m in sorted(r.methods or []):
                if m != "HEAD":
                    out.append(("http", m, prefix + r.path))
    return out


HTTP_ROUTES = [(m, p) for k, m, p in all_routes() if k == "http"]


def test_ac1_every_route_is_classified():
    known = PUBLIC | OWN_SESSION_ONLY | LOGIN_REQUIRED
    unclassified = []
    for kind, method, path in all_routes():
        if kind == "http" and (method, path) not in known:
            unclassified.append((method, path))
        if kind == "ws" and path not in WEBSOCKETS:
            unclassified.append(("WS", path))
        if kind == "mount" and path not in MOUNTS:
            unclassified.append(("MOUNT", path))
    assert unclassified == []


def test_ac1_classified_routes_all_exist():
    present = set(HTTP_ROUTES)
    assert (PUBLIC | OWN_SESSION_ONLY | LOGIN_REQUIRED) <= present
    assert WEBSOCKETS <= {p for k, _, p in all_routes() if k == "ws"}


def test_ac1_no_api_docs_exposed():
    paths = {p for _, _, p in all_routes()}
    assert not paths & {"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}


@pytest.fixture
async def student_world(browser, fake_college):
    """arjun is logged in elsewhere with a conversation holding a secret."""
    a = await browser()
    await login_as(a, "arjun")
    ctx = await auth._lookup(a.cookies["kit_session"])
    cid = await history.create_conversation(ctx)
    await history.add_message(cid, "user", "arjun-private-question")
    await history.add_message(cid, "assistant", "arjun-private-answer")
    return {"cid": cid, "ctx": ctx}


def leaks(text: str, world) -> list[str]:
    bad = ["arjun-private", world["ctx"].kit_token, world["cid"], "user:arjun"]
    bad += [USERS[s]["full_name"] for s in STUDENTS] + [USERS[t]["full_name"]
                                                       for t in ("lakshmi", "senthil")]
    return [b for b in bad if b in text]


@pytest.mark.parametrize("method,path", HTTP_ROUTES)
async def test_ac1_visitor_gets_no_user_data_from_any_route(browser, fake_college, student_world,
                                                            method, path):
    v = await browser()
    url = path.replace("{conversation_id}", student_world["cid"])
    before = [c for c in fake_college.log if c.login is not None]
    r = await v.request(method, url, json={} if method == "POST" else None)
    assert r.status_code < 500
    assert leaks(r.text + str(dict(r.headers)), student_world) == []
    if (method, path) in LOGIN_REQUIRED:
        assert r.status_code in (303, 401, 403)
    after = [c for c in fake_college.log if c.login is not None]
    assert after == before  # no college call was made with anyone's token


@pytest.mark.parametrize("method,path", HTTP_ROUTES)
async def test_ac1_forged_cookie_gets_no_user_data(browser, fake_college, student_world, method,
                                                   path):
    v = await browser()
    url = path.replace("{conversation_id}", student_world["cid"])
    forged = {"cookie": f"kit_session={student_world['ctx'].session_id}"}
    r = await v.request(method, url, headers=forged, json={} if method == "POST" else None)
    assert leaks(r.text, student_world) == []


async def test_ac1_assistant_redirects_visitor_to_login(app_client):
    r = await app_client.get("/assistant")
    assert r.status_code == 303 and r.headers["location"] == "/login"
    assert "kit_session=" in r.headers.get("set-cookie", "")
    assert "<html" not in r.text.lower()


@pytest.mark.parametrize("login", ["arjun", "lakshmi"])
async def test_ac1_assistant_served_to_logged_in_users(app_client, fake_college, login):
    await login_as(app_client, login)
    r = await app_client.get("/assistant")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]


async def test_ac1_assistant_redirects_after_logout(app_client, fake_college):
    await login_as(app_client, "arjun")
    await app_client.post("/api/logout")
    assert (await app_client.get("/assistant")).status_code == 303


@pytest.mark.parametrize("path", ["/", "/login", "/history"])
async def test_ac1_public_pages_served_to_visitors(app_client, path):
    r = await app_client.get(path)
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]


async def test_ac1_me_for_visitor_reveals_nothing(app_client, student_world):
    r = await app_client.get("/api/me")
    assert r.json() == {"role": "visitor", "user": None}


@pytest.mark.parametrize("path,key", [("/api/public/college", "college"),
                                      ("/api/public/courses", "courses")])
async def test_ac1_public_endpoints_return_only_public_college_data(app_client, fake_college,
                                                                   path, key):
    r = await app_client.get(path)
    assert r.status_code == 200
    assert r.json() == getattr(fake_college, key)
    call = fake_college.calls("GET", f"/v1/{key}")[-1]
    assert call.token is None


@pytest.mark.parametrize("path", ["/api/public/college", "/api/public/courses"])
async def test_ac2_public_endpoints_never_use_a_users_token(app_client, fake_college, path):
    await login_as(app_client, "arjun")
    await app_client.get(path)
    assert all(c.token is None for c in fake_college.log)


async def test_public_endpoints_cached_for_five_minutes(app_client, fake_college, monkeypatch):
    main = load_main()
    now = [5000.0]
    monkeypatch.setattr(main.time, "monotonic", lambda: now[0])
    for _ in range(3):
        assert (await app_client.get("/api/public/courses")).status_code == 200
    assert len(fake_college.calls("GET", "/v1/courses")) == 1
    now[0] += main.PUBLIC_CACHE_SECONDS + 1
    fake_college.courses = fake_college.courses[:1]
    r = await app_client.get("/api/public/courses")
    assert len(r.json()) == 1 and len(fake_college.calls("GET", "/v1/courses")) == 2


@pytest.mark.parametrize("path", ["/api/public/college", "/api/public/courses"])
async def test_rel1_public_endpoint_with_college_down_is_clear_503(app_client, fake_college, path):
    fake_college.down = True
    r = await app_client.get(path)
    assert r.status_code == 503
    assert "unavailable" in r.json()["detail"].lower()
    assert "Traceback" not in r.text and "httpx" not in r.text


async def test_rel1_public_endpoint_serves_cache_after_outage_starts(app_client, fake_college):
    await app_client.get("/api/public/college")
    fake_college.down = True
    assert (await app_client.get("/api/public/college")).status_code == 200


@pytest.mark.parametrize("path", ["/", "/login", "/history", "/healthz", "/api/me"])
async def test_rel1_pages_work_with_college_down(app_client, fake_college, path):
    fake_college.down = True
    assert (await app_client.get(path)).status_code == 200


async def test_ac1_visitor_history_detail_for_random_ids(app_client, student_world):
    for cid in (student_world["cid"], str(uuid.uuid4())):
        assert (await app_client.get(f"/api/history/{cid}")).status_code == 404
