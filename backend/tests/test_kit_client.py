"""College API client: error mapping, headers, retries, college date cache."""

import httpx
import pytest
import respx
from app.config import get_settings
from app.kit import client as kc
from app.kit.client import KitClient, KitError, college_today

BASE = get_settings().kit_base_url


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(kc, "RETRY_BACKOFF_SECONDS", 0)
    kc.clear_today_cache()
    yield
    kc.clear_today_cache()


def err(code: str, message: str, field=None) -> dict:
    return {"error": {"code": code, "message": message, "field": field}}


@respx.mock(base_url=BASE)
async def test_error_body_maps_to_kit_error(respx_mock):
    respx_mock.get("/v1/me/requests/9").respond(
        403, json=err("not_allowed", "You can only view your own requests."))
    with pytest.raises(KitError) as e:
        await KitClient("tok").get_request(9)
    assert (e.value.status, e.value.code, e.value.message) == (
        403, "not_allowed", "You can only view your own requests.")


@respx.mock(base_url=BASE)
async def test_field_and_validation_shapes(respx_mock):
    respx_mock.get("/v1/calendar").respond(
        422, json=err("invalid_input", "Input should be a valid date", "from"))
    with pytest.raises(KitError) as e:
        await KitClient("t").calendar("bad")
    assert e.value.field == "from"
    respx_mock.get("/v1/courses").respond(422, json={"detail": [{"loc": ["query", "x"],
                                                                "msg": "bad x"}]})
    with pytest.raises(KitError) as e:
        await KitClient().courses()
    assert (e.value.message, e.value.field) == ("bad x", "x")
    respx_mock.get("/v1/college").respond(418, text="teapot")
    with pytest.raises(KitError) as e:
        await KitClient().college()
    assert e.value.code == "http_418"


@respx.mock(base_url=BASE)
async def test_headers(respx_mock):
    route = respx_mock.post("/v1/me/requests").respond(201, json={"id": 1})
    await KitClient("jwt").create_request({"type": "on_duty"}, "key-1")
    req = route.calls.last.request
    assert req.headers["Authorization"] == "Bearer jwt"
    assert req.headers["Idempotency-Key"] == "key-1"
    assert req.headers["X-Participant-Key"] == get_settings().kit_participant_key
    visitor = respx_mock.get("/v1/courses").respond(200, json=[])
    await KitClient().courses()
    assert "Authorization" not in visitor.calls.last.request.headers


@respx.mock(base_url=BASE)
async def test_get_retries_5xx_and_timeouts(respx_mock):
    route = respx_mock.get("/v1/college").mock(side_effect=[
        httpx.Response(503, json=err("unavailable", "down")),
        httpx.ReadTimeout("slow"),
        httpx.Response(200, json={"name": "KIT"}),
    ])
    assert (await KitClient().college())["name"] == "KIT"
    assert route.call_count == 3


@respx.mock(base_url=BASE)
async def test_get_gives_up_after_two_retries(respx_mock):
    route = respx_mock.get("/v1/college").respond(500, json=err("boom", "Server error"))
    with pytest.raises(KitError) as e:
        await KitClient().college()
    assert route.call_count == 3 and e.value.status == 500


@respx.mock(base_url=BASE)
async def test_4xx_not_retried(respx_mock):
    route = respx_mock.get("/v1/me").respond(401, json=err("unauthorized", "Invalid login token."))
    with pytest.raises(KitError):
        await KitClient("x").me()
    assert route.call_count == 1


@respx.mock(base_url=BASE)
async def test_post_without_key_not_retried(respx_mock):
    route = respx_mock.post("/v1/placements/drives/7/register").mock(
        side_effect=httpx.ConnectError("reset"))
    with pytest.raises(KitError) as e:
        await KitClient("t").register_drive(7)
    assert route.call_count == 1 and e.value.status == 0


@respx.mock(base_url=BASE)
async def test_post_with_key_retried_with_same_key(respx_mock):
    route = respx_mock.post("/v1/admissions/applications").mock(side_effect=[
        httpx.ReadTimeout("slow"), httpx.Response(201, json={"id": 5}),
    ])
    assert (await KitClient().submit_admission({"a": 1}, "k"))["id"] == 5
    assert route.call_count == 2
    assert {c.request.headers["Idempotency-Key"] for c in route.calls} == {"k"}


@respx.mock(base_url=BASE)
async def test_college_today_cached(respx_mock):
    route = respx_mock.get("/v1/today").respond(200, json={
        "date": "2026-11-21", "weekday": 6, "weekday_name": "Saturday", "timezone": "Asia/Kolkata"})
    assert (await college_today())["date"] == "2026-11-21"
    await college_today()
    assert route.call_count == 1
