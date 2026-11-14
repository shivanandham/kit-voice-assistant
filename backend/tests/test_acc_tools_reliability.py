"""REL1, REL2, AGT6 (as it applies to tools), AC6 for tools.

With the college down every tool answers with a short, clear error; malformed college data never
crashes a tool or leaks a stack trace; a 401 for a logged-in session ends that session.
"""

import uuid

import httpx
import pytest
from app.db import pool
from app.kit import client as kc

from tests.acc_tools_helpers import (
    ALLOWED,
    ROLE_LOGIN,
    SAMPLE_ARGS,
    SEARCH,
    mentions,
    new_conv,
    override,
    run,
)

STACK_MARKERS = ("Traceback", "File \"", "KeyError", "TypeError", "AttributeError",
                 "ConnectError", "httpx", "asyncpg", ", line ")


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr(kc, "RETRY_BACKOFF_SECONDS", 0.0)


def _cases(exclude=()):
    for role in ("visitor", "student", "teacher"):
        for name in sorted(ALLOWED[role] - {SEARCH} - set(exclude)):
            yield role, name


NO_COLLEGE = {"confirm_pending_action", "cancel_pending_action"}


@pytest.mark.parametrize("role,name", list(_cases(NO_COLLEGE)))
async def test_rel1_college_down_clear_error(role, name, fake_college):
    ctx = fake_college.ctx(ROLE_LOGIN[role])
    cid = await new_conv(ctx)
    fake_college.down = True
    r, b = await run(name, SAMPLE_ARGS[name], ctx, cid)
    assert r.is_error
    assert isinstance(b, dict)
    for m in STACK_MARKERS:
        assert m not in r.content, m
    assert mentions(r.content, "college", "reach", "unavailable", "try again")
    assert len(r.content) < 600


@pytest.mark.parametrize("exc", [httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout,
                                 httpx.RemoteProtocolError, httpx.PoolTimeout])
@pytest.mark.parametrize("name", ["get_my_profile", "get_timetable", "list_my_requests",
                                  "list_placement_drives"])
async def test_rel1_transport_faults_are_clear_errors(exc, name, fake_college):
    ctx = fake_college.ctx("arjun")
    cid = await new_conv(ctx)
    fake_college.fail("^/v1/.*", exc=exc)
    r, b = await run(name, SAMPLE_ARGS[name], ctx, cid)
    assert r.is_error
    for m in STACK_MARKERS:
        assert m not in r.content


@pytest.mark.parametrize("status", [500, 502, 503, 504])
@pytest.mark.parametrize("role,name", [("visitor", "list_courses"), ("visitor", "get_college_info"),
                                       ("student", "get_my_profile"),
                                       ("teacher", "list_my_classes")])
async def test_rel1_5xx_is_clear_error(status, role, name, fake_college):
    ctx = fake_college.ctx(ROLE_LOGIN[role])
    cid = await new_conv(ctx)
    fake_college.fail("^/v1/.*", status)
    r, b = await run(name, SAMPLE_ARGS[name], ctx, cid)
    assert r.is_error
    for m in STACK_MARKERS:
        assert m not in r.content


@pytest.mark.parametrize("name", ["confirm_pending_action", "cancel_pending_action"])
async def test_rel1_gate_tools_work_with_college_down(name, fake_college):
    ctx = fake_college.ctx("arjun")
    cid = await new_conv(ctx)
    fake_college.down = True
    r, b = await run(name, {"action_id": str(uuid.uuid4())}, ctx, cid)
    assert r.is_error
    assert "Traceback" not in r.content


async def test_rel1_college_recovers(fake_college):
    ctx = fake_college.ctx("arjun")
    cid = await new_conv(ctx)
    fake_college.down = True
    r1, _ = await run("get_my_profile", {}, ctx, cid)
    fake_college.down = False
    r2, _ = await run("get_my_profile", {}, ctx, cid)
    assert r1.is_error and not r2.is_error


async def test_rel1_flaky_get_recovers_by_retry(fake_college):
    fake_college.fail("/v1/me", 503, times=2)
    ctx = fake_college.ctx("arjun")
    r, b = await run("get_my_profile", {}, ctx, await new_conv(ctx))
    assert not r.is_error, b
    assert len(fake_college.calls("GET", "/v1/me")) == 3


# ------------------------------------------------------------------ malformed college data (AGT6)

MALFORMED = [
    ("get_attendance_summary", "/v1/me/attendance", {}),
    ("get_attendance_summary", "/v1/me/attendance", []),
    ("get_attendance_summary", "/v1/me/attendance", None),
    ("get_my_profile", "/v1/me", "not a dict"),
    ("get_my_subjects", "/v1/me/subjects", [{"wrong": 1}]),
    ("get_timetable", "/v1/me/timetable", [{"weekday": "x"}]),
    ("list_my_requests", "/v1/me/requests", {"not": "a list"}),
    ("list_placement_drives", "/v1/placements/drives", [{"id": 1}]),
    ("get_attendance_sessions", "/v1/me/attendance/sessions", [{"date": "bad"}]),
    ("get_calendar", "/v1/calendar", [{"date": None}]),
]


@pytest.mark.parametrize("name,path,payload", MALFORMED)
async def test_agt6_malformed_college_payload_is_safe_error(name, path, payload, fake_college):
    override(fake_college, "GET", path, lambda call, orig: (200, payload))
    ctx = fake_college.ctx("arjun")
    r, b = await run(name, SAMPLE_ARGS[name], ctx, await new_conv(ctx))
    assert r.is_error
    assert isinstance(b, dict)
    for m in STACK_MARKERS:
        assert m not in r.content
    assert len(r.content) < 400


@pytest.mark.parametrize("name,path", [("list_my_classes", "/v1/teacher/classes"),
                                       ("list_class_requests", "/v1/teacher/requests")])
async def test_agt6_malformed_teacher_payload_is_safe_error(name, path, fake_college):
    override(fake_college, "GET", path, lambda call, orig: (200, [{"nothing": True}]))
    ctx = fake_college.ctx("lakshmi")
    r, b = await run(name, SAMPLE_ARGS[name], ctx, await new_conv(ctx))
    assert isinstance(b, dict)
    for m in STACK_MARKERS:
        assert m not in r.content


async def test_agt6_tool_log_failure_does_not_break_result(fake_college, monkeypatch):
    from app import history

    async def broken(*a, **kw):
        raise RuntimeError("db down")

    monkeypatch.setattr(history, "log_tool_call", broken)
    ctx = fake_college.ctx("arjun")
    r, b = await run("get_today", {}, ctx, await new_conv(ctx))
    assert not r.is_error
    assert "2026-11-21" in r.content


async def test_agt6_every_result_is_logged(fake_college):
    ctx = fake_college.ctx("arjun")
    cid = await new_conv(ctx)
    fake_college.down = True
    await run("get_my_profile", {}, ctx, cid)
    rows = await pool().fetch("SELECT name, is_error FROM tool_calls")
    assert [(x["name"], x["is_error"]) for x in rows] == [("get_my_profile", True)]


# ------------------------------------------------------------------ 401 ends the session (AC6)

STUDENT_READS = ["get_my_profile", "get_my_academics", "get_my_subjects", "get_attendance_summary",
                 "get_attendance_sessions", "get_timetable", "get_calendar", "list_my_requests",
                 "get_request", "list_placement_drives", "list_my_placements", "cancel_request",
                 "submit_leave_request", "register_for_placement_drive"]
TEACHER_READS = ["get_my_profile", "list_my_classes", "get_class_attendance",
                 "list_class_requests", "approve_request", "reject_request"]


# DEFECT: submit_leave_request swallows the 401 from its pre-checks and stores a pending action.
@pytest.mark.parametrize("login,name", [("arjun", n) for n in STUDENT_READS] +
                         [("lakshmi", n) for n in TEACHER_READS])
async def test_ac6_401_invalidates_session_not_retried(login, name, fake_college):
    token, ctx = await fake_college.db_session(login)
    cid = await new_conv(ctx)
    fake_college.expire_token(ctx.kit_token)
    r, b = await run(name, SAMPLE_ARGS[name], ctx, cid)
    assert r.is_error
    assert await pool().fetchval("SELECT count(*) FROM sessions WHERE id = $1",
                                 ctx.session_id) == 0
    unauth = [c for c in fake_college.log if c.status == 401]
    assert len(unauth) == 1, [(c.path, c.status) for c in fake_college.log]
    assert await pool().fetchval("SELECT count(*) FROM pending_actions") == 0
    assert mentions(r.content, "log in", "login", "expired")


async def test_ac6_401_invalidates_only_that_session(fake_college):
    _, a = await fake_college.db_session("arjun")
    _, b = await fake_college.db_session("priya")
    fake_college.expire_token(a.kit_token)
    await run("get_my_profile", {}, a, await new_conv(a))
    r, _ = await run("get_my_profile", {}, b, await new_conv(b))
    assert not r.is_error
    assert await pool().fetchval("SELECT count(*) FROM sessions WHERE id = $1", b.session_id) == 1


async def test_ac6_visitor_401_is_reported_without_crash(fake_college):
    _, v = await fake_college.db_session(None)
    fake_college.fail("/v1/courses", 401, code="unauthorized",
                      message="Invalid X-Participant-Key provided.")
    r, b = await run("list_courses", {}, v, await new_conv(v))
    assert r.is_error
    assert fake_college.participant_key not in r.content


# ------------------------------------------------------------------ REL2


async def test_rel2_all_college_traffic_hits_the_fake(fake_college, network_guard):
    ctx = fake_college.ctx("arjun")
    cid = await new_conv(ctx)
    for name in sorted(ALLOWED["student"] - {SEARCH}):
        await run(name, SAMPLE_ARGS[name], ctx, cid)
    assert fake_college.log
    assert network_guard == []


def test_rel2_base_url_is_not_contacted_for_real(fake_college):
    from app.config import get_settings

    assert get_settings().kit_participant_key == "test-participant-key-not-real"
