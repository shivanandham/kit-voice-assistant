"""Tool gating: role tool sets, strict schemas, argument validation, 401 handling, audit log."""

import json

import pytest
import respx
from app import history
from app.agent import tools as T
from app.auth import SessionCtx
from app.config import get_settings
from app.db import pool
from app.kit import client as kc

BASE = get_settings().kit_base_url
STUDENT_ONLY = {"get_my_academics", "get_my_subjects", "get_attendance_summary",
                "get_attendance_sessions", "get_timetable", "list_my_requests", "get_request",
                "submit_leave_request", "cancel_request", "list_placement_drives",
                "register_for_placement_drive", "list_my_placements"}
TEACHER_ONLY = {"list_my_classes", "get_class_attendance", "list_class_requests",
                "approve_request", "reject_request"}
COMMON = ["get_today", "get_college_info", "list_courses", "confirm_pending_action",
          "cancel_pending_action"]


@pytest.fixture(autouse=True)
def _reset():
    kc.clear_today_cache()


def ctx(role: str, token: str | None = "tok", sid: str = "sid") -> SessionCtx:
    user = {"id": 28, "login_id": "arjun", "department": "CSE", "year_of_study": 2,
            "programme": "UG"} if role != "visitor" else None
    return SessionCtx(sid, f"{role}:x", role, user, token if role != "visitor" else None)


def names(role: str, search: bool = False) -> list[str]:
    return [t["name"] for t in T.tool_specs(role, search)]


def test_role_tool_sets():
    assert names("visitor") == COMMON + ["submit_admission_application"]
    assert set(names("student")) == set(COMMON) | STUDENT_ONLY | {"get_my_profile", "get_calendar"}
    assert set(names("teacher")) == set(COMMON) | TEACHER_ONLY | {"get_my_profile"}
    assert "search_documents" not in names("student")
    assert "search_documents" in names("visitor", True)
    assert names("student") == names("student")  # stable order


def test_schemas_are_strict():
    for role in ("visitor", "student", "teacher"):
        for spec in T.tool_specs(role, True):
            s = spec["input_schema"]
            assert spec["strict"] is True and s["additionalProperties"] is False
            assert set(s["required"]) <= set(s["properties"])
            assert not {"user_id", "student_id", "login_id", "teacher_id"} & set(s["properties"])


async def test_visitor_cannot_run_student_or_teacher_tools():
    cid = await history.create_conversation(ctx("visitor"))
    for name in STUDENT_ONLY | TEACHER_ONLY | {"get_my_profile"}:
        r = await T.run_tool(name, {}, ctx("visitor"), cid)
        assert r.is_error and "not available" in r.content, name


async def test_student_cannot_run_teacher_tools_and_vice_versa():
    cid = await history.create_conversation(ctx("student"))
    for name in TEACHER_ONLY:
        r = await T.run_tool(name, {"request_id": 1, "reason": "x"}, ctx("student"), cid)
        assert r.is_error and "not available" in r.content, name
    teacher_cid = await history.create_conversation(ctx("teacher"))
    for name in STUDENT_ONLY:
        r = await T.run_tool(name, {}, ctx("teacher"), teacher_cid)
        assert r.is_error and "not available" in r.content, name
    assert (await T.run_tool("submit_admission_application", {}, ctx("student"), cid)).is_error


async def test_unknown_and_extra_args_rejected():
    cid = await history.create_conversation(ctx("student"))
    r = await T.run_tool("get_request", {"request_id": 1, "student_id": 99}, ctx("student"), cid)
    assert r.is_error and "student_id" in r.content
    r = await T.run_tool("get_timetable", {"date": "23/11/2026"}, ctx("student"), cid)
    assert r.is_error and "date" in r.content
    r = await T.run_tool("get_request", {"request_id": "12"}, ctx("student"), cid)
    assert r.is_error and "integer" in r.content
    r = await T.run_tool("submit_leave_request", {"type": "vacation", "from_date": "2026-11-24",
                         "to_date": "2026-11-24", "half_day": False, "reason": "x"},
                         ctx("student"), cid)
    assert r.is_error and "one of" in r.content
    assert (await T.run_tool("nope", {}, ctx("student"), cid)).is_error


async def test_every_call_is_logged():
    cid = await history.create_conversation(ctx("student"))
    await T.run_tool("list_my_classes", {}, ctx("student"), cid)
    row = await pool().fetchrow("SELECT name, is_error, output FROM tool_calls")
    assert row["name"] == "list_my_classes" and row["is_error"]
    assert "not available" in json.loads(row["output"])["error"]


@respx.mock(base_url=BASE)
async def test_college_error_returned_verbatim(respx_mock):
    respx_mock.get("/v1/me/requests/15").respond(403, json={"error": {
        "code": "not_allowed", "message": "You can only view your own requests.", "field": None}})
    respx_mock.get("/v1/today").respond(200, json={"date": "2026-11-21", "weekday": 6,
                                                   "weekday_name": "Saturday", "timezone": "IST"})
    cid = await history.create_conversation(ctx("student"))
    r = await T.run_tool("get_request", {"request_id": 15}, ctx("student"), cid)
    assert r.is_error and json.loads(r.content)["error"] == "You can only view your own requests."


@respx.mock(base_url=BASE)
async def test_college_401_expires_the_session(respx_mock):
    await pool().execute("INSERT INTO sessions (id, owner_key, role, expires_at)"
                         " VALUES ('sid', 'user:arjun', 'student', now() + interval '1 hour')")
    respx_mock.get("/v1/me/academics").respond(401, json={"error": {
        "code": "unauthorized", "message": "Invalid login token.", "field": None}})
    cid = await history.create_conversation(ctx("student"))
    r = await T.run_tool("get_my_academics", {}, ctx("student"), cid)
    assert r.is_error and "log in again" in r.content
    assert await pool().fetchval("SELECT count(*) FROM sessions") == 0


async def test_logged_in_tool_without_token():
    cid = await history.create_conversation(ctx("student"))
    r = await T.run_tool("get_my_academics", {}, ctx("student", token=None), cid)
    assert r.is_error and "log in again" in r.content


async def test_ac4_teacher_set_contains_only_tools_that_work_for_teachers():
    assert "get_calendar" not in names("teacher") and "get_calendar" in names("student")
    cid = await history.create_conversation(ctx("teacher"))
    r = await T.run_tool("get_calendar", {}, ctx("teacher"), cid)
    assert r.is_error and "not available" in r.content
