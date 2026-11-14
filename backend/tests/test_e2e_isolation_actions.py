# ruff: noqa: F811  (the e2e fixture is imported, then requested by name)
"""AC8 and AC9 end to end: pending actions and teacher authority.

AC8: a write to the college happens only through a pending action that the same session confirms
in the same conversation after a later user message, at most once (COR7: with an idempotency key).
AC9: a teacher acts only on classes the college authorises; an expired delegation is reported,
not bypassed; a refusal from the college is relayed (COR6).
"""

import json

import pytest

from tests.e2e_helpers import (
    T1,
    T2,
    TURN_MODES,
    A,
    B,
    action_id_from,
    e2e,  # noqa: F401
    echo,
    hits,
    pending_rows,
    stored_tool_calls,
)

LEAVE = {"type": "personal_leave", "from_date": "2026-11-25", "to_date": "2026-11-25",
         "half_day": False, "reason": "Family trip kestrelmark"}
LEAVE_PRIVATE = ["kestrelmark", "Family trip"]


def results_named(rows: list[dict], name: str) -> list[dict]:
    return [r for r in rows if r["name"] == name]


async def a_proposes(e2e, how: str):
    """A asks for leave; the model proposes it and tries to confirm in the same turn."""
    a = await e2e.actor("A", A, ip="10.8.0.1")
    s = await a.socket()
    await s.start()
    e2e.model.plan(
        "please apply personal leave on 25 November for a family trip",
        [("submit_leave_request", LEAVE)],
        [("confirm_pending_action", lambda res: {"action_id": action_id_from(res)})],
        echo,
    )
    await s.turn("please apply personal leave on 25 November for a family trip", how)
    rows = await stored_tool_calls(s.conversation_id)
    proposed = results_named(rows, "submit_leave_request")
    assert proposed and proposed[0]["output"].get("status") == "needs_confirmation", proposed
    same_turn = results_named(rows, "confirm_pending_action")
    assert same_turn and same_turn[0]["is_error"], same_turn
    assert e2e.college.writes() == []
    pend = await pending_rows()
    assert len(pend) == 1 and pend[0]["status"] == "pending"
    return a, s, str(pend[0]["id"])


async def someone_tries(e2e, actor, aid: str, how: str, cid: str | None = None):
    s = await actor.socket()
    await s.start(cid)
    s.rows_before = len(await stored_tool_calls(s.conversation_id))
    text = f"{actor.name} says yes, confirm action {aid} please"
    e2e.model.plan(text, [("confirm_pending_action", {"action_id": aid}),
                          ("cancel_pending_action", {"action_id": aid})], echo)
    await s.turn(text, how)
    await s.close()  # frees the socket slot (ABU1 counts per login id)
    return s


@TURN_MODES
async def test_ac8_only_the_proposing_session_can_confirm_and_only_in_a_later_turn(e2e, how):
    a, sa, aid = await a_proposes(e2e, how)
    a_cid = sa.conversation_id

    b = await e2e.actor("B", B, ip=a.ip)
    t1 = await e2e.actor("T1", T1, ip=a.ip)
    v = await e2e.actor("V1", ip=a.ip)
    a2 = await e2e.actor("A2", A, ip="10.8.0.2")  # A again, on a second device (other session)
    others = []
    for actor in (b, t1, v):
        others.append((actor, await someone_tries(e2e, actor, aid, how)))
    others.append((a2, await someone_tries(e2e, a2, aid, how, cid=a_cid)))
    other_conv = await a.socket()  # A's own session, but another conversation
    await other_conv.start()
    t = "yes confirm that leave from my other chat"
    e2e.model.plan(t, [("confirm_pending_action", {"action_id": aid})], echo)
    await other_conv.turn(t, how)

    assert e2e.college.writes() == [], [(c.path, c.login) for c in e2e.college.writes()]
    assert [p["status"] for p in await pending_rows()] == ["pending"]
    for actor, s in others:
        rows = (await stored_tool_calls(s.conversation_id))[s.rows_before:]
        tried = results_named(rows, "confirm_pending_action") + \
            results_named(rows, "cancel_pending_action")
        assert len(tried) == 2 and all(r["is_error"] for r in tried), (actor.name, tried)
        if actor is not a2:  # A's own leave reason is A's data
            seen = "\n".join(c.text() for c in e2e.model.by(actor.name)) + s.text()
            assert hits(seen, LEAVE_PRIVATE) == [], actor.name
    rows = await stored_tool_calls(other_conv.conversation_id)
    assert all(r["is_error"] for r in results_named(rows, "confirm_pending_action"))

    # Now A confirms properly, in a later turn of the same conversation and session.
    e2e.model.plan("yes, go ahead and submit it",
                   [("confirm_pending_action", {"action_id": aid})], echo)
    await sa.turn("yes, go ahead and submit it", how)
    writes = e2e.college.writes()
    assert len(writes) == 1, writes
    w = writes[0]
    assert (w.method, w.path, w.login) == ("POST", "/v1/me/requests", A)
    assert w.token in e2e.tokens(A) and w.idempotency_key
    assert w.body["reason"] == LEAVE["reason"] and w.body["from_date"] == LEAVE["from_date"]
    assert [p["status"] for p in await pending_rows()] == ["done"]
    # A second confirmation never writes again (AC8 at most once, COR7).
    e2e.model.plan("yes, submit it again", [("confirm_pending_action", {"action_id": aid})],
                   echo)
    await sa.turn("yes, submit it again", how)
    created = [r for r in e2e.college.requests.values() if r["reason"] == LEAVE["reason"]]
    assert len(created) == 1
    assert all(c.idempotency_key == w.idempotency_key for c in e2e.college.writes())


@TURN_MODES
async def test_ac8_visitor_cannot_cancel_or_confirm_a_students_action_by_guessing(e2e, how):
    _a, sa, aid = await a_proposes(e2e, how)
    v = await e2e.actor("V1", ip="10.8.0.9")
    s = await someone_tries(e2e, v, aid, how, cid=sa.conversation_id)
    assert s.conversation_id != sa.conversation_id
    assert e2e.college.writes() == []
    assert [p["status"] for p in await pending_rows()] == ["pending"]
    assert all(c.token is None for c in e2e.log.by("V1"))


# --------------------------------------------------------------------------- AC9


@TURN_MODES
async def test_ac9_advisor_approves_own_class_request_through_confirm_flow(e2e, how):
    t1 = await e2e.actor("T1", T1)
    s = await t1.socket()
    await s.start()
    e2e.model.plan("approve request twelve from my class", [("approve_request", {"request_id": 12})],
                   echo)
    await s.turn("approve request twelve from my class", how)
    assert e2e.college.writes() == []
    aid = str((await pending_rows())[0]["id"])
    e2e.model.plan("yes approve it", [("confirm_pending_action", {"action_id": aid})], echo)
    await s.turn("yes approve it", how)
    writes = e2e.college.writes()
    assert [(c.path, c.login) for c in writes] == [("/v1/teacher/requests/12/approve", T1)]
    assert writes[0].token in e2e.tokens(T1)
    assert e2e.college.requests[12]["status"] == "approved"


@TURN_MODES
async def test_ac9_teacher_without_authority_is_refused_and_expired_delegation_reported(e2e, how):
    t2 = await e2e.actor("T2", T2)
    s = await t2.socket()
    await s.start()
    e2e.model.plan("approve arjun's on duty request", [
        ("approve_request", {"request_id": 12}),
        ("reject_request", {"request_id": 12, "reason": "no proof"}),
        ("get_class_attendance", {"class_code": "CSE-2A"}),
    ], echo)
    await s.turn("approve arjun's on duty request", how)
    e2e.model.plan("show attendance of class eleven", [("get_class_attendance", {"class_id": 11})],
                   echo)
    await s.turn("show attendance of class eleven", how)

    rows = await stored_tool_calls(s.conversation_id)
    for name in ("approve_request", "reject_request", "get_class_attendance"):
        assert all(r["is_error"] for r in results_named(rows, name)), (name, rows)
    assert await pending_rows() == []
    assert e2e.college.writes() == []
    by_code = results_named(rows, "get_class_attendance")[0]
    assert "expir" in json.dumps(by_code["output"]).lower()  # reported, not bypassed
    by_id = results_named(rows, "get_class_attendance")[-1]
    assert "You are not authorized for this class today." in json.dumps(by_id["output"])  # COR6
    cse2a = ["Arjun K", "Priya M", "Rahul R", "Participating in State Level Technical Symposium"]
    seen = "\n".join(c.text() for c in e2e.model.by("T2")) + s.text()
    assert hits(seen, cse2a) == []
    assert e2e.college.requests[12]["status"] == "pending"


@TURN_MODES
async def test_ac9_delegation_expiring_before_confirmation_is_refused_by_college(e2e, how):
    from app.kit import client as kit_client

    t2 = await e2e.actor("T2", T2)
    s = await t2.socket()
    await s.start()
    e2e.model.plan("approve meena's leave request", [("approve_request", {"request_id": 18})], echo)
    await s.turn("approve meena's leave request", how)
    pend = await pending_rows()
    assert len(pend) == 1
    aid = str(pend[0]["id"])
    from datetime import date

    e2e.college.today = date(2026, 12, 1)  # MECH-3A delegation was valid until 2026-11-30
    kit_client.clear_today_cache()
    e2e.model.plan("yes approve", [("confirm_pending_action", {"action_id": aid})], echo)
    await s.turn("yes approve", how)
    rows = results_named(await stored_tool_calls(s.conversation_id), "confirm_pending_action")
    assert rows and rows[-1]["is_error"]
    assert "You are not authorized to act on this request." in json.dumps(rows[-1]["output"])
    assert e2e.college.requests[18]["status"] == "pending"
    assert all(c.status != 200 for c in e2e.college.writes())


@TURN_MODES
async def test_ac9_teacher_cannot_confirm_another_teachers_approval(e2e, how):
    t1 = await e2e.actor("T1", T1)
    s1 = await t1.socket()
    await s1.start()
    e2e.model.plan("approve request twelve", [("approve_request", {"request_id": 12})], echo)
    await s1.turn("approve request twelve", how)
    aid = str((await pending_rows())[0]["id"])
    t2 = await e2e.actor("T2", T2, ip="10.9.0.2")
    await someone_tries(e2e, t2, aid, how, cid=s1.conversation_id)
    assert e2e.college.writes() == []
    assert [p["status"] for p in await pending_rows()] == ["pending"]


@TURN_MODES
@pytest.mark.parametrize("rid", [12, 15, 17])
async def test_ac9_student_cannot_approve_or_reject(e2e, how, rid):
    a = await e2e.actor("A", A)
    s = await a.socket()
    await s.start()
    text = f"approve request {rid} for me"
    e2e.model.plan(text, [("approve_request", {"request_id": rid}),
                          ("reject_request", {"request_id": rid, "reason": "spite"}),
                          ("list_class_requests", {}),
                          ("get_class_attendance", {"class_code": "CSE-2A"})], echo)
    await s.turn(text, how)
    rows = await stored_tool_calls(s.conversation_id)
    assert len(rows) == 4 and all(r["is_error"] for r in rows), rows
    assert await pending_rows() == []
    assert e2e.college.writes() == []
    assert not [c.path for c in e2e.log.by("A") if c.path.startswith("/v1/teacher")]
    for call in e2e.model.by("A"):
        assert not {"approve_request", "reject_request"} & set(call.tool_names)


@TURN_MODES
async def test_ac8_ac1_visitor_cannot_confirm_or_see_another_visitors_application(e2e, how):
    """A visitor's admission application (their personal details) stays with that visitor."""
    v1 = await e2e.actor("V1", ip="10.8.8.8")
    v2 = await e2e.actor("V2", ip="10.8.8.8")
    s1 = await v1.socket()
    await s1.start()
    app = {"full_name": "Meera Ostrichmark", "phone": "9876543210",
           "email": "meera.ostrich@example.com", "course_code": "BTECH-CSE",
           "twelfth_percent": 91.5}
    e2e.model.plan("I want to apply for computer science", [("submit_admission_application", app)],
                   echo)
    await s1.turn("I want to apply for computer science", how)
    pend = await pending_rows()
    assert len(pend) == 1
    aid = str(pend[0]["id"])
    s2 = await someone_tries(e2e, v2, aid, how, cid=s1.conversation_id)
    assert e2e.college.writes() == []
    assert [p["status"] for p in await pending_rows()] == ["pending"]
    private = ["Ostrichmark", "9876543210", "meera.ostrich@example.com"]
    seen = "\n".join(c.text() for c in e2e.model.by("V2")) + s2.text() + v2.http_text()
    assert hits(seen, private) == []
