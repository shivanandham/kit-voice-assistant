"""College API client (CONTRACT "College API"): headers, error mapping, retries, secrets.

* X-Participant-Key on every call, never in a URL; bearer token only when logged in (AC2, AC5).
* Errors {"error": {code, message, field}} map to KitError fields (COR6).
* GETs retry twice on timeouts and 5xx; POSTs retry only with an idempotency key (COR7, REL1).
* A 401 is never retried.
"""

import httpx
import pytest
import respx
from app.config import get_settings
from app.kit import client as kc
from app.kit.client import KitClient, KitError

BASE = get_settings().kit_base_url
TOKEN = "tok-abc.def.ghi"


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(kc, "RETRY_BACKOFF_SECONDS", 0.0)
    kc.clear_today_cache()
    yield
    kc.clear_today_cache()


@pytest.fixture
def mock():
    with respx.mock(base_url=BASE, assert_all_called=False, assert_all_mocked=True) as m:
        yield m


def key():
    return get_settings().kit_participant_key


# (method name, args, kwargs, HTTP method, path)
ENDPOINTS = [
    ("health", (), {}, "GET", "/v1/health"),
    ("today", (), {}, "GET", "/v1/today"),
    ("college", (), {}, "GET", "/v1/college"),
    ("courses", (), {}, "GET", "/v1/courses"),
    ("documents", (), {}, "GET", "/v1/documents"),
    ("document_file", (3,), {}, "GET", "/v1/documents/3/file"),
    ("submit_admission", ({"full_name": "X"}, "idem-1"), {}, "POST", "/v1/admissions/applications"),
    ("login", ("arjun", "pw-secret"), {}, "POST", "/v1/auth/login"),
    ("me", (), {}, "GET", "/v1/me"),
    ("academics", (), {}, "GET", "/v1/me/academics"),
    ("subjects", (), {}, "GET", "/v1/me/subjects"),
    ("timetable", (), {}, "GET", "/v1/me/timetable"),
    ("calendar", (), {"from_date": "2026-11-21", "to_date": "2026-11-28"}, "GET", "/v1/calendar"),
    ("attendance", (), {}, "GET", "/v1/me/attendance"),
    ("attendance_sessions", (), {"subject": "CS304"}, "GET", "/v1/me/attendance/sessions"),
    ("my_requests", (), {}, "GET", "/v1/me/requests"),
    ("create_request", ({"type": "on_duty"}, "idem-2"), {}, "POST", "/v1/me/requests"),
    ("get_request", (5,), {}, "GET", "/v1/me/requests/5"),
    ("cancel_request", (5,), {}, "POST", "/v1/me/requests/5/cancel"),
    ("placement_drives", (), {}, "GET", "/v1/placements/drives"),
    ("register_drive", (9,), {}, "POST", "/v1/placements/drives/9/register"),
    ("my_placements", (), {}, "GET", "/v1/me/placements"),
    ("teacher_classes", (), {}, "GET", "/v1/teacher/classes"),
    ("teacher_requests", (), {"status": "pending"}, "GET", "/v1/teacher/requests"),
    ("class_attendance", (11,), {}, "GET", "/v1/teacher/classes/11/attendance"),
    ("approve_request", (5,), {"comment": "ok"}, "POST", "/v1/teacher/requests/5/approve"),
    ("reject_request", (5, "no proof"), {}, "POST", "/v1/teacher/requests/5/reject"),
]
IDS = [e[0] for e in ENDPOINTS]
GETS = [e for e in ENDPOINTS if e[3] == "GET"]
POSTS_NO_KEY = [e for e in ENDPOINTS if e[3] == "POST" and e[0] not in
                ("submit_admission", "create_request")]
POSTS_KEY = [e for e in ENDPOINTS if e[0] in ("submit_admission", "create_request")]


async def call(e, token=TOKEN):
    name, args, kwargs, *_ = e
    return await getattr(KitClient(token), name)(*args, **kwargs)


def route(mock, e, *responses, **kw):
    r = mock.route(method=e[3], path=e[4])
    if len(responses) == 1 and not isinstance(responses[0], (list, tuple)):
        r.mock(return_value=responses[0])
    else:
        r.mock(side_effect=list(responses[0] if len(responses) == 1 else responses))
    return r


# ------------------------------------------------------------------ headers


@pytest.mark.parametrize("e", ENDPOINTS, ids=IDS)
async def test_ac5_participant_key_on_every_request_never_in_url(e, mock):
    r = route(mock, e, httpx.Response(200, json={}))
    await call(e)
    req = r.calls.last.request
    assert req.method == e[3]
    assert req.url.path == e[4]
    assert req.headers["x-participant-key"] == key()
    assert key() not in str(req.url)
    assert TOKEN not in str(req.url)


@pytest.mark.parametrize("e", ENDPOINTS, ids=IDS)
async def test_ac2_bearer_token_when_logged_in(e, mock):
    r = route(mock, e, httpx.Response(200, json={}))
    await call(e, TOKEN)
    assert r.calls.last.request.headers["authorization"] == f"Bearer {TOKEN}"


@pytest.mark.parametrize("e", ENDPOINTS, ids=IDS)
@pytest.mark.parametrize("tok", [None, ""])
async def test_ac1_no_authorization_without_login(e, tok, mock):
    r = route(mock, e, httpx.Response(200, json={}))
    await call(e, tok)
    assert "authorization" not in r.calls.last.request.headers


@pytest.mark.parametrize("e", POSTS_KEY, ids=[x[0] for x in POSTS_KEY])
async def test_cor7_idempotency_key_header(e, mock):
    r = route(mock, e, httpx.Response(200, json={}))
    await call(e)
    req = r.calls.last.request
    assert req.headers["idempotency-key"] == e[1][1]
    assert e[1][1] not in str(req.url)


@pytest.mark.parametrize("e", GETS, ids=[x[0] for x in GETS])
async def test_cor7_reads_carry_no_idempotency_key(e, mock):
    r = route(mock, e, httpx.Response(200, json={}))
    await call(e)
    assert "idempotency-key" not in r.calls.last.request.headers


async def test_ac5_login_credentials_in_body_not_url(mock):
    r = mock.post("/v1/auth/login").respond(200, json={"access_token": "t"})
    await KitClient().login("arjun", "pw-secret")
    req = r.calls.last.request
    assert "pw-secret" not in str(req.url)
    assert b"pw-secret" in req.content
    assert "authorization" not in req.headers


async def test_calendar_params_forwarded(mock):
    r = mock.get("/v1/calendar").respond(200, json=[])
    await KitClient(TOKEN).calendar("2026-11-21", "2026-11-28")
    p = r.calls.last.request.url.params
    assert p["from"] == "2026-11-21" and p["to"] == "2026-11-28"


async def test_calendar_without_dates_sends_no_params(mock):
    r = mock.get("/v1/calendar").respond(200, json=[])
    await KitClient(TOKEN).calendar()
    assert dict(r.calls.last.request.url.params) == {}


async def test_attendance_sessions_params(mock):
    r = mock.get("/v1/me/attendance/sessions").respond(200, json=[])
    await KitClient(TOKEN).attendance_sessions("CS304", "2026-11-01", None)
    p = dict(r.calls.last.request.url.params)
    assert p == {"subject": "CS304", "from": "2026-11-01"}


async def test_teacher_requests_without_status(mock):
    r = mock.get("/v1/teacher/requests").respond(200, json=[])
    await KitClient(TOKEN).teacher_requests()
    assert "status" not in r.calls.last.request.url.params


async def test_reject_and_approve_bodies(mock):
    a = mock.post("/v1/teacher/requests/5/approve").respond(200, json={})
    j = mock.post("/v1/teacher/requests/5/reject").respond(200, json={})
    await KitClient(TOKEN).approve_request(5, "fine")
    await KitClient(TOKEN).reject_request(5, "no proof")
    import json

    assert json.loads(a.calls.last.request.content)["comment"] == "fine"
    assert json.loads(j.calls.last.request.content)["reason"] == "no proof"


@pytest.mark.parametrize("bad", ["5/../../admin", "5?x=1", "abc"])
async def test_ac2_path_ids_cannot_be_injected(bad, mock):
    mock.route().respond(200, json={})
    with pytest.raises((ValueError, TypeError, KitError)):
        await KitClient(TOKEN).get_request(bad)


# ------------------------------------------------------------------ responses


async def test_parsed_json_returned(mock):
    mock.get("/v1/courses").respond(200, json=[{"code": "BTECH-CSE"}])
    assert await KitClient().courses() == [{"code": "BTECH-CSE"}]


async def test_document_file_returns_bytes(mock):
    mock.get("/v1/documents/3/file").respond(200, content=b"%PDF-1.4 x",
                                            headers={"content-type": "application/pdf"})
    assert await KitClient().document_file(3) == b"%PDF-1.4 x"


async def test_empty_body_returns_none(mock):
    mock.post("/v1/me/requests/5/cancel").respond(204)
    assert await KitClient(TOKEN).cancel_request(5) is None


# ------------------------------------------------------------------ error mapping (COR6)

ERRORS = [
    (400, {"error": {"code": "bad_request", "message": "Bad dates.", "field": "from_date"}},
     ("bad_request", "Bad dates.", "from_date")),
    (401, {"error": {"code": "unauthorized", "message": "Login token has expired.", "field": None}},
     ("unauthorized", "Login token has expired.", None)),
    (403, {"error": {"code": "not_allowed", "message": "Only students can access this endpoint."}},
     ("not_allowed", "Only students can access this endpoint.", None)),
    (404, {"error": {"code": "not_found", "message": "Request with id 5 not found in this sandbox.",
                     "field": None}},
     ("not_found", "Request with id 5 not found in this sandbox.", None)),
    (409, {"error": {"code": "idempotency_conflict",
                     "message": "Idempotency-Key was already used with a different request body.",
                     "field": None}},
     ("idempotency_conflict", "Idempotency-Key was already used with a different request body.",
      None)),
    (422, {"error": {"code": "invalid_input", "message": "Field required", "field": "reason"}},
     ("invalid_input", "Field required", "reason")),
    (429, {"error": {"code": "rate_limited", "message": "Slow down.", "field": None}},
     ("rate_limited", "Slow down.", None)),
    (422, {"error": {"code": "invalid_input", "message": "Ungültig — «x»", "field": "x"}},
     ("invalid_input", "Ungültig — «x»", "x")),
]


@pytest.mark.parametrize("status,payload,expected", ERRORS)
async def test_cor6_error_shape_maps_to_kiterror(status, payload, expected, mock):
    mock.get("/v1/me").respond(status, json=payload)
    with pytest.raises(KitError) as ei:
        await KitClient(TOKEN).me()
    e = ei.value
    assert (e.status, e.code, e.message, e.field) == (status, *expected)
    assert str(e) == expected[1]


@pytest.mark.parametrize("status,payload,expected", ERRORS)
async def test_cor6_error_shape_on_post(status, payload, expected, mock):
    mock.post("/v1/me/requests").respond(status, json=payload)
    with pytest.raises(KitError) as ei:
        await KitClient(TOKEN).create_request({"type": "on_duty"}, "k1")
    assert (ei.value.code, ei.value.message, ei.value.field) == expected


async def test_cor6_fastapi_validation_detail_maps(mock):
    mock.post("/v1/me/requests").respond(422, json={"detail": [
        {"loc": ["body", "from_date"], "msg": "invalid date format", "type": "value_error"}]})
    with pytest.raises(KitError) as ei:
        await KitClient(TOKEN).create_request({}, "k")
    assert ei.value.status == 422
    assert ei.value.message == "invalid date format"
    assert ei.value.field == "from_date"


async def test_cor6_detail_without_loc(mock):
    mock.get("/v1/me").respond(422, json={"detail": [{"msg": "bad"}]})
    with pytest.raises(KitError) as ei:
        await KitClient(TOKEN).me()
    assert ei.value.message == "bad" and ei.value.field is None


@pytest.mark.parametrize("status,content", [(400, b"<html>Bad</html>"), (404, b""),
                                            (403, b"Forbidden"), (418, b"{not json")])
async def test_cor6_non_json_error_is_kiterror(status, content, mock):
    mock.get("/v1/me").respond(status, content=content)
    with pytest.raises(KitError) as ei:
        await KitClient(TOKEN).me()
    assert ei.value.status == status
    assert isinstance(ei.value.message, str) and ei.value.message
    assert "<html>" not in ei.value.message


@pytest.mark.parametrize("payload", [{"error": "flat string"}, {"error": {}}, {"message": "x"},
                                     [], {"error": {"code": None, "message": None}}])
async def test_cor6_odd_error_bodies_still_kiterror(payload, mock):
    mock.get("/v1/me").respond(400, json=payload)
    with pytest.raises(KitError) as ei:
        await KitClient(TOKEN).me()
    assert ei.value.status == 400
    assert ei.value.code and ei.value.message


def test_cor6_kiterror_to_dict():
    e = KitError(409, "invalid_state", "Request is already approved.", "status")
    assert e.to_dict()["error"] == "Request is already approved."
    assert e.to_dict()["code"] == "invalid_state"
    assert e.to_dict()["field"] == "status"
    assert "field" not in KitError(409, "x", "y").to_dict()


# ------------------------------------------------------------------ retries


@pytest.mark.parametrize("status", [500, 502, 503, 504])
@pytest.mark.parametrize("e", GETS, ids=[x[0] for x in GETS])
async def test_rel1_get_retries_twice_on_5xx(status, e, mock):
    r = route(mock, e, httpx.Response(status, json={"error": {"code": "down",
                                                              "message": "Down for a moment."}}))
    with pytest.raises(KitError) as ei:
        await call(e)
    assert r.call_count == 3
    assert ei.value.status == status
    assert ei.value.message == "Down for a moment."


@pytest.mark.parametrize("e", GETS, ids=[x[0] for x in GETS])
async def test_rel1_get_recovers_on_third_attempt(e, mock):
    r = route(mock, e, [httpx.Response(503), httpx.Response(502), httpx.Response(200, json={"ok": 1})])
    out = await call(e)
    assert r.call_count == 3
    if e[0] == "document_file":
        assert out == b'{"ok":1}'
    else:
        assert out == {"ok": 1}


@pytest.mark.parametrize("exc", [httpx.ReadTimeout, httpx.ConnectTimeout, httpx.PoolTimeout,
                                 httpx.WriteTimeout])
async def test_rel1_get_retries_twice_on_timeout(exc, mock):
    r = mock.get("/v1/me").mock(side_effect=exc("slow"))
    with pytest.raises(KitError) as ei:
        await KitClient(TOKEN).me()
    assert r.call_count == 3
    assert ei.value.status == 0
    assert TOKEN not in str(ei.value) and key() not in str(ei.value)


@pytest.mark.parametrize("exc", [httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError])
async def test_rel1_get_transport_error_is_kiterror(exc, mock):
    r = mock.get("/v1/me").mock(side_effect=exc("boom"))
    with pytest.raises(KitError) as ei:
        await KitClient(TOKEN).me()
    assert ei.value.status == 0
    assert r.call_count == 3


async def test_rel1_get_timeout_then_success(mock):
    r = mock.get("/v1/me").mock(side_effect=[httpx.ReadTimeout("x"),
                                             httpx.Response(200, json={"login_id": "arjun"})])
    assert (await KitClient(TOKEN).me())["login_id"] == "arjun"
    assert r.call_count == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422])
@pytest.mark.parametrize("e", GETS[:6], ids=[x[0] for x in GETS[:6]])
async def test_rel1_get_4xx_not_retried(status, e, mock):
    r = route(mock, e, httpx.Response(status, json={"error": {"code": "x", "message": "m"}}))
    with pytest.raises(KitError):
        await call(e)
    assert r.call_count == 1


@pytest.mark.parametrize("e", POSTS_NO_KEY + [("login", ("a", "b"), {}, "POST", "/v1/auth/login")],
                         ids=[x[0] for x in POSTS_NO_KEY] + ["login"])
@pytest.mark.parametrize("fault", ["503", "timeout", "connect"])
async def test_cor7_post_without_key_not_retried(e, fault, mock):
    if fault == "503":
        r = route(mock, e, httpx.Response(503))
    else:
        exc = httpx.ReadTimeout if fault == "timeout" else httpx.ConnectError
        r = mock.route(method="POST", path=e[4]).mock(side_effect=exc("x"))
    with pytest.raises(KitError):
        await call(e)
    assert r.call_count == 1


@pytest.mark.parametrize("e", POSTS_KEY, ids=[x[0] for x in POSTS_KEY])
@pytest.mark.parametrize("fault", ["503", "timeout"])
async def test_cor7_post_with_key_retried_with_same_key(e, fault, mock):
    first = httpx.Response(503) if fault == "503" else httpx.ReadTimeout("x")
    r = mock.route(method="POST", path=e[4]).mock(
        side_effect=[first, first, httpx.Response(200, json={"id": 1})])
    assert await call(e) == {"id": 1}
    assert r.call_count == 3
    keys = {c.request.headers.get("idempotency-key") for c in r.calls}
    assert keys == {e[1][1]}
    bodies = {c.request.content for c in r.calls}
    assert len(bodies) == 1


@pytest.mark.parametrize("e", POSTS_KEY, ids=[x[0] for x in POSTS_KEY])
async def test_cor7_post_with_key_gives_up_after_three(e, mock):
    r = route(mock, e, httpx.Response(503))
    with pytest.raises(KitError):
        await call(e)
    assert r.call_count == 3


@pytest.mark.parametrize("e", ENDPOINTS, ids=IDS)
async def test_ac6_401_never_retried(e, mock):
    r = route(mock, e, httpx.Response(401, json={"error": {"code": "unauthorized",
                                                           "message": "Login token has expired."}}))
    with pytest.raises(KitError) as ei:
        await call(e)
    assert r.call_count == 1
    assert ei.value.status == 401


# ------------------------------------------------------------------ shared client and today


async def test_close_http_then_reuse(mock):
    mock.get("/v1/health").respond(200, json={"status": "ok"})
    await KitClient().health()
    await kc.close_http()
    await kc.close_http()
    assert (await KitClient().health())["status"] == "ok"


async def test_cor1_college_today_cached_60s(mock, monkeypatch):
    r = mock.get("/v1/today").respond(200, json={"date": "2026-11-21", "weekday": 6,
                                                 "weekday_name": "Saturday",
                                                 "timezone": "Asia/Kolkata"})
    t1 = await kc.college_today()
    t2 = await kc.college_today()
    assert t1 == t2 and t1["date"] == "2026-11-21"
    assert r.call_count == 1
    real = kc.time.monotonic
    monkeypatch.setattr(kc.time, "monotonic", lambda: real() + 61)
    await kc.college_today()
    assert r.call_count == 2


async def test_cor1_college_today_is_public_call(mock):
    r = mock.get("/v1/today").respond(200, json={"date": "2026-11-21", "weekday": 6,
                                                 "weekday_name": "Saturday", "timezone": "IST"})
    await kc.college_today()
    req = r.calls.last.request
    assert "authorization" not in req.headers
    assert req.headers["x-participant-key"] == key()


async def test_cor1_college_today_failure_not_cached(mock):
    r = mock.get("/v1/today").mock(side_effect=[httpx.Response(503)] * 3 + [
        httpx.Response(200, json={"date": "2026-11-21", "weekday": 6, "weekday_name": "Saturday",
                                  "timezone": "IST"})])
    with pytest.raises(KitError):
        await kc.college_today()
    assert (await kc.college_today())["date"] == "2026-11-21"


async def test_ac5_kiterror_repr_has_no_secret(mock):
    mock.get("/v1/me").respond(500, json={"error": {"code": "x", "message": "oops"}})
    with pytest.raises(KitError) as ei:
        await KitClient(TOKEN).me()
    for text in (str(ei.value), repr(ei.value), str(ei.value.to_dict())):
        assert TOKEN not in text and key() not in text
