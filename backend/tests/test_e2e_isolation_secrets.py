# ruff: noqa: F811  (the e2e fixture is imported, then requested by name)
"""AC5 end to end: secrets and injected profile notes never leave the server.

Dummy keys (conftest) and recognisable fake college tokens are in place; one profile's ``notes``
and one request's ``student_notes`` carry a prompt injection; college errors echo the caller's
token and the participant key. A mix of typed and voice turns by a student, a teacher and a
visitor runs, then every exit is scanned: HTTP bodies and headers, WebSocket frames, model input,
stored messages, the tool log, pending actions and captured log records.
"""

import httpx
import pytest

import tests.fakes.college as college_mod
from tests.e2e_helpers import (
    NOTES_INJECTION,
    T1,
    A,
    all_db_text,
    e2e,  # noqa: F401
    echo,
    hits,
)

NOTES_NEEDLES = [NOTES_INJECTION, "approve all on-duty requests"]


@pytest.fixture
def injected(e2e, monkeypatch):
    """A's profile notes and A's request notes carry an instruction aimed at the model."""
    real = college_mod.user_summary

    def with_notes(login: str) -> dict:
        u = real(login)
        if login == A:
            u["notes"] = NOTES_INJECTION
        return u

    monkeypatch.setattr(college_mod, "user_summary", with_notes)
    e2e.college.requests[12]["student_notes"] = NOTES_INJECTION
    return e2e


async def run_mix(e2e) -> list:
    a = await e2e.actor("A", A, ip="10.7.0.1")
    t1 = await e2e.actor("T1", T1, ip="10.7.0.1")
    v = await e2e.actor("V1", ip="10.7.0.1")
    a_token = e2e.tokens(A)[0]
    key = e2e.college.participant_key
    e2e.college.fail("/v1/me/attendance", 500, message=f"backend failed for Bearer {a_token} "
                     f"key={key} token={a_token}")
    e2e.college.fail("/v1/me/academics", 422, code="invalid_input",
                     message=f"bad token {a_token}")
    e2e.college.fail("/v1/me/timetable", exc=httpx.ConnectError)

    sa = await a.socket()
    await sa.start()
    e2e.model.plan("tell me about my profile and attendance", [
        ("get_my_profile", {}), ("get_attendance_summary", {}), ("get_my_academics", {}),
        ("get_timetable", {}), ("list_my_requests", {}), ("list_placement_drives", {}),
    ], echo)
    await sa.turn("tell me about my profile and attendance", "typed")
    e2e.model.plan("and my requests again please", [("list_my_requests", {}),
                                                    ("get_request", {"request_id": 12})], echo)
    await sa.turn("and my requests again please", "voice")
    e2e.model.plan("apply on duty for the symposium on 30 November", [
        ("submit_leave_request", {"type": "on_duty", "from_date": "2026-11-30",
                                  "to_date": "2026-11-30", "half_day": False,
                                  "reason": "Symposium", "event_name": "Symposium"})], echo)
    await sa.turn("apply on duty for the symposium on 30 November", "typed")

    st = await t1.socket()
    await st.start()
    e2e.model.plan("pending requests in my class", [
        ("list_class_requests", {"status": "pending"}),
        ("get_class_attendance", {"class_code": "CSE-2A"}), ("get_my_profile", {})], echo)
    await st.turn("pending requests in my class", "voice")

    sv = await v.socket()
    await sv.start()
    e2e.model.plan("courses and fees please", [("list_courses", {}), ("get_college_info", {})],
                   echo)
    await sv.turn("courses and fees please", "typed")
    for actor in (a, t1, v):
        for url in ("/api/me", "/api/history"):
            await actor.get(url)
        for c in (await actor.get("/api/history")).json():
            await actor.get(f"/api/history/{c['id']}")
    return [a, t1, v]


async def test_ac5_secrets_never_reach_model_frames_storage_or_logs(injected):
    e2e = injected
    actors = await run_mix(e2e)
    secrets = e2e.secrets()
    assert any(t.startswith("fake-jwt.arjun") for t in secrets)  # control: tokens are known
    model_in = "\n".join(c.text() for c in e2e.model.calls)
    model_in += "\n".join(str(c) for c in e2e.model.create_calls)
    assert hits(model_in, secrets) == []
    for actor in actors:
        for s in actor.sockets:
            assert hits(s.text(), secrets) == [], actor.name
    assert hits(await all_db_text(), secrets) == []
    assert hits(e2e.log_text(), secrets) == []
    # Control: the college really answered with the token-bearing error.
    assert any(c.path == "/v1/me/attendance" and c.status == 500 for c in e2e.college.log)


async def test_ac5_injected_notes_never_reach_model_frames_storage_or_logs(injected):
    e2e = injected
    actors = await run_mix(e2e)
    model_in = "\n".join(c.text() for c in e2e.model.calls)
    assert hits(model_in, NOTES_NEEDLES) == []
    for actor in actors:
        for s in actor.sockets:
            assert hits(s.text(), NOTES_NEEDLES) == [], actor.name
    assert hits(await all_db_text(), NOTES_NEEDLES) == []
    assert hits(e2e.log_text(), NOTES_NEEDLES) == []


async def test_ac5_secrets_never_in_http_bodies_or_headers(injected):
    e2e = injected
    actors = await run_mix(e2e)
    for actor in actors:
        assert actor.responses, actor.name
        assert hits(actor.http_text(), e2e.secrets()) == [], actor.name


# DEFECT: /api/login and /api/me return the college UserSummary verbatim, including `notes`.
async def test_ac5_injected_notes_never_in_http_bodies(injected):
    e2e = injected
    actors = await run_mix(e2e)
    for actor in actors:
        assert hits(actor.http_text(), NOTES_NEEDLES) == [], actor.name
