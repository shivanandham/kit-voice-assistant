"""Confirm-before-write gate: next-turn confirmation, session isolation, no double submit."""

import json

import httpx
import pytest
import respx
from app import history
from app.agent import tools as T
from app.auth import SessionCtx
from app.config import get_settings
from app.db import pool
from app.kit import client as kc

BASE = get_settings().kit_base_url
LEAVE = {"type": "medical_leave", "from_date": "2026-11-24", "to_date": "2026-11-24",
         "half_day": False, "reason": "Fever"}
CREATED = {"id": 40, "student_id": 28, "student_name": "Arjun K", "class_code": "CSE-2A",
           "type": "medical_leave", "from_date": "2026-11-24", "to_date": "2026-11-24",
           "half_day": False, "reason": "Fever", "event_name": None, "status": "pending",
           "decided_by": None, "decision_note": None, "created_at": "x", "student_notes": None}


def student(sid: str = "sid-a") -> SessionCtx:
    return SessionCtx(sid, "user:arjun", "student", {"id": 28, "department": "CSE",
                      "year_of_study": 2, "programme": "UG"}, "tok")


@pytest.fixture
def college():
    kc.clear_today_cache()
    with respx.mock(base_url=BASE, assert_all_called=False) as m:
        m.get("/v1/today").respond(200, json={"date": "2026-11-21", "weekday": 6,
                                              "weekday_name": "Saturday", "timezone": "IST"})
        m.get("/v1/calendar").respond(200, json=[])
        m.get("/v1/me/requests").respond(200, json=[])
        m.get("/v1/me/attendance").respond(200, json={
            "as_of_date": "2026-11-21", "overall_percentage": 76.0, "total_classes_held": 50,
            "total_classes_attended": 38, "subjects": []})
        m.post_route = m.post("/v1/me/requests").respond(201, json=CREATED)
        yield m


async def propose(ctx: SessionCtx) -> tuple[str, str]:
    cid = await history.create_conversation(ctx)
    await history.add_message(cid, "user", "I need medical leave on Tuesday")
    r = await T.run_tool("submit_leave_request", LEAVE, ctx, cid)
    body = json.loads(r.content)
    assert not r.is_error and body["status"] == "needs_confirmation"
    return cid, body["action_id"]


async def test_write_tool_does_not_call_college(college):
    await propose(student())
    assert college.post_route.call_count == 0
    assert await pool().fetchval("SELECT status FROM pending_actions") == "pending"


async def test_cannot_confirm_in_same_turn(college):
    ctx = student()
    cid, aid = await propose(ctx)
    r = await T.run_tool("confirm_pending_action", {"action_id": aid}, ctx, cid)
    assert r.is_error and "not confirmed" in r.content
    assert college.post_route.call_count == 0


async def test_confirm_next_turn_submits_once_with_idempotency_key(college):
    ctx = student()
    cid, aid = await propose(ctx)
    await history.add_message(cid, "user", "yes")
    r = await T.run_tool("confirm_pending_action", {"action_id": aid}, ctx, cid)
    body = json.loads(r.content)
    assert not r.is_error and body["status"] == "done" and body["result"]["id"] == 40
    assert "student_notes" not in body["result"]
    req = college.post_route.calls.last.request
    assert req.headers["Idempotency-Key"] == aid
    assert json.loads(req.content) == LEAVE
    again = await T.run_tool("confirm_pending_action", {"action_id": aid}, ctx, cid)
    assert json.loads(again.content)["status"] == "already_done"
    assert college.post_route.call_count == 1
    assert await pool().fetchval("SELECT status FROM pending_actions") == "done"


async def test_other_session_or_conversation_cannot_confirm(college):
    cid, aid = await propose(student("sid-a"))
    await history.add_message(cid, "user", "yes")
    r = await T.run_tool("confirm_pending_action", {"action_id": aid}, student("sid-b"), cid)
    assert r.is_error and "No such pending action" in r.content
    other = await history.create_conversation(student("sid-a"))
    r = await T.run_tool("confirm_pending_action", {"action_id": aid}, student("sid-a"), other)
    assert r.is_error
    r = await T.run_tool("confirm_pending_action", {"action_id": "not-a-uuid"}, student(), cid)
    assert r.is_error
    assert college.post_route.call_count == 0


async def test_college_rejection_is_verbatim_and_marks_failed(college):
    college.post_route.respond(409, json={"error": {"code": "conflict", "message":
                               "Personal leave is not allowed on internal assessment days.",
                               "field": "from_date"}})
    ctx = student()
    cid, aid = await propose(ctx)
    await history.add_message(cid, "user", "yes please")
    r = await T.run_tool("confirm_pending_action", {"action_id": aid}, ctx, cid)
    body = json.loads(r.content)
    assert r.is_error and body["status"] == "failed"
    assert body["error"] == "Personal leave is not allowed on internal assessment days."
    assert await pool().fetchval("SELECT status FROM pending_actions") == "failed"


async def test_college_outage_keeps_action_pending(college):
    college.post_route.mock(side_effect=[httpx.ConnectError("x")] * 3)
    ctx = student()
    cid, aid = await propose(ctx)
    await history.add_message(cid, "user", "yes")
    r = await T.run_tool("confirm_pending_action", {"action_id": aid}, ctx, cid)
    assert r.is_error and "still pending" in r.content
    assert await pool().fetchval("SELECT status FROM pending_actions") == "pending"


async def test_cancel_and_pending_summary(college):
    ctx = student()
    cid, aid = await propose(ctx)
    summary = await T.pending_summary(ctx, cid)
    assert aid in summary and "medical leave" in summary
    assert await T.pending_summary(student("sid-b"), cid) is None
    r = await T.run_tool("cancel_pending_action", {"action_id": aid}, ctx, cid)
    assert not r.is_error and json.loads(r.content)["status"] == "cancelled"
    assert await T.pending_summary(ctx, cid) is None
    await history.add_message(cid, "user", "yes")
    r = await T.run_tool("confirm_pending_action", {"action_id": aid}, ctx, cid)
    assert r.is_error and college.post_route.call_count == 0


async def test_same_proposal_reuses_pending_action(college):
    ctx = student()
    cid, aid = await propose(ctx)
    r = await T.run_tool("submit_leave_request", LEAVE, ctx, cid)
    assert json.loads(r.content)["action_id"] == aid
    assert await pool().fetchval("SELECT count(*) FROM pending_actions") == 1


async def test_teacher_reject_flow(college):
    teacher = SessionCtx("sid-t", "user:lakshmi", "teacher", {"id": 25}, "tok")
    college.get("/v1/teacher/requests").respond(200, json=[{**CREATED, "id": 13,
                                                           "type": "on_duty"}])
    reject = college.post("/v1/teacher/requests/13/reject").respond(
        200, json={**CREATED, "id": 13, "status": "rejected", "decision_note": "No proof"})
    cid = await history.create_conversation(teacher)
    await history.add_message(cid, "user", "reject 13, no proof")
    r = await T.run_tool("reject_request", {"request_id": 13, "reason": "No proof"}, teacher, cid)
    aid = json.loads(r.content)["action_id"]
    await history.add_message(cid, "user", "yes")
    r = await T.run_tool("confirm_pending_action", {"action_id": aid}, teacher, cid)
    assert json.loads(r.content)["result"]["status"] == "rejected"
    assert json.loads(reject.calls.last.request.content) == {"reason": "No proof"}
    r = await T.run_tool("approve_request", {"request_id": 99}, teacher, cid)
    assert r.is_error and "not among" in r.content


async def test_concurrent_confirms_submit_once(college):
    import asyncio

    ctx = student()
    cid, aid = await propose(ctx)
    await history.add_message(cid, "user", "yes")
    results = await asyncio.gather(*[
        T.run_tool("confirm_pending_action", {"action_id": aid}, ctx, cid) for _ in range(3)])
    statuses = sorted(json.loads(r.content)["status"] for r in results)
    assert statuses == ["already_done", "already_done", "done"]
    assert college.post_route.call_count == 1
