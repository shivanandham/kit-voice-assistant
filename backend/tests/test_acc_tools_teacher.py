"""AC9: a teacher acts only on classes the college authorises; the app never widens that and an
expired delegation is reported, not bypassed.

lakshmi: advisor of CSE-2A (11) and CSE-4A (14).
senthil: advisor of ECE-2B (12), delegated MECH-3A (13) until 2026-11-30, and an expired delegation
for CSE-2A (valid_until 2026-11-20; college date 2026-11-21).
Requests: 12 arjun CSE-2A pending, 15 priya CSE-2A approved, 16 rahul CSE-2A rejected,
17 kavya ECE-2B pending, 18 meena MECH-3A pending, 19 priya CSE-2A pending.
"""

import pytest

from tests.acc_tools_helpers import (
    action_status,
    mentions,
    new_conv,
    pending_count,
    propose,
    run,
    user_says,
    writes,
)

CSE2A_NAMES = ("Arjun", "Priya", "Rahul")


async def trun(fake, login, name, args):
    ctx = fake.ctx(login)
    cid = await new_conv(ctx)
    r, b = await run(name, args, ctx, cid)
    return ctx, cid, r, b


# ------------------------------------------------------------------ classes and attendance


async def test_ac9_list_classes_lakshmi(fake_college):
    _, _, r, b = await trun(fake_college, "lakshmi", "list_my_classes", {})
    assert {k["class_code"] for k in b["classes"]} == {"CSE-2A", "CSE-4A"}


async def test_ac9_list_classes_senthil_excludes_expired_delegation(fake_college):
    _, _, r, b = await trun(fake_college, "senthil", "list_my_classes", {})
    assert {k["class_code"] for k in b["classes"]} == {"ECE-2B", "MECH-3A"}
    mech = next(k for k in b["classes"] if k["class_code"] == "MECH-3A")
    assert mech["valid_until"] == "2026-11-30"


@pytest.mark.parametrize("login,code", [("lakshmi", "CSE-2A"), ("lakshmi", "CSE-4A"),
                                        ("senthil", "ECE-2B"), ("senthil", "MECH-3A"),
                                        ("lakshmi", "cse-2a")])
async def test_ac9_attendance_of_authorised_class(login, code, fake_college):
    _, _, r, b = await trun(fake_college, login, "get_class_attendance", {"class_code": code})
    assert not r.is_error, b
    assert b["students"]


@pytest.mark.parametrize("login,cid", [("lakshmi", 11), ("lakshmi", 14), ("senthil", 12),
                                       ("senthil", 13)])
async def test_ac9_attendance_by_class_id_authorised(login, cid, fake_college):
    _, _, r, b = await trun(fake_college, login, "get_class_attendance", {"class_id": cid})
    assert not r.is_error, b


@pytest.mark.parametrize("login,code", [("lakshmi", "ECE-2B"), ("lakshmi", "MECH-3A"),
                                        ("senthil", "CSE-4A"), ("senthil", "CSE-2A")])
async def test_ac9_attendance_of_unauthorised_class_by_code_refused(login, code, fake_college):
    _, _, r, b = await trun(fake_college, login, "get_class_attendance", {"class_code": code})
    assert r.is_error
    for n in ("Arjun", "Priya", "Rahul", "Kavya", "Meena", "Karthik"):
        assert n not in r.content
    assert all(c.status >= 400 for c in fake_college.log if "attendance" in c.path)


@pytest.mark.parametrize("login,cid", [("lakshmi", 12), ("lakshmi", 13), ("senthil", 14),
                                       ("senthil", 11)])
async def test_ac9_attendance_of_unauthorised_class_by_id_relays_college_refusal(login, cid,
                                                                                 fake_college):
    _, _, r, b = await trun(fake_college, login, "get_class_attendance", {"class_id": cid})
    assert r.is_error
    assert "You are not authorized for this class today." in r.content  # COR6 verbatim
    for n in ("Arjun", "Priya", "Rahul", "Kavya", "Meena", "Karthik"):
        assert n not in r.content


async def test_ac9_expired_delegation_reported(fake_college):
    _, _, r, b = await trun(fake_college, "senthil", "get_class_attendance",
                            {"class_code": "CSE-2A"})
    assert r.is_error
    assert mentions(r.content, "expire", "delegat", "not authori", "not among")


async def test_ac9_unknown_class_id_relays_not_found(fake_college):
    _, _, r, b = await trun(fake_college, "lakshmi", "get_class_attendance", {"class_id": 99})
    assert r.is_error
    assert "Class with id 99 not found." in r.content


async def test_ac9_class_attendance_needs_a_class(fake_college):
    _, _, r, b = await trun(fake_college, "lakshmi", "get_class_attendance", {})
    assert r.is_error


async def test_ac9_class_code_and_id_disagree_never_widens(fake_college):
    """An authorised code with an unauthorised id must not reveal the other class."""
    _, _, r, b = await trun(fake_college, "lakshmi", "get_class_attendance",
                            {"class_code": "CSE-2A", "class_id": 12})
    assert "Kavya" not in r.content


async def test_ac9_delegation_removed_takes_effect_immediately(fake_college):
    fake_college.grants["senthil"] = [(12, "advisor", None)]
    _, _, r, b = await trun(fake_college, "senthil", "get_class_attendance",
                            {"class_code": "MECH-3A"})
    assert r.is_error
    assert "Meena" not in r.content


# ------------------------------------------------------------------ requests


@pytest.mark.parametrize("login,visible,hidden", [
    ("lakshmi", {12, 15, 16, 19}, {17, 18}),
    ("senthil", {17, 18}, {12, 15, 16, 19}),
])
async def test_ac9_class_requests_scoped(login, visible, hidden, fake_college):
    _, _, r, b = await trun(fake_college, login, "list_class_requests", {})
    ids = {x["id"] for x in b["requests"]}
    assert ids == visible
    assert not ids & hidden


@pytest.mark.parametrize("status", ["pending", "approved", "rejected", "cancelled"])
async def test_ac9_class_requests_status_filter_forwarded(status, fake_college):
    _, _, r, b = await trun(fake_college, "lakshmi", "list_class_requests", {"status": status})
    assert not r.is_error
    assert all(x["status"] == status for x in b["requests"])
    assert fake_college.calls("GET", "/v1/teacher/requests")[-1].params.get("status") == status


@pytest.mark.parametrize("tool,args", [("approve_request", {"request_id": 17}),
                                       ("approve_request", {"request_id": 18}),
                                       ("reject_request", {"request_id": 17, "reason": "No"}),
                                       ("reject_request", {"request_id": 18, "reason": "No"})])
async def test_ac9_lakshmi_cannot_act_on_other_classes(tool, args, fake_college):
    _, _, r, b = await trun(fake_college, "lakshmi", tool, args)
    assert r.is_error
    assert await pending_count() == 0
    assert writes(fake_college) == []
    assert "Kavya" not in r.content and "Meena" not in r.content


@pytest.mark.parametrize("tool,args", [("approve_request", {"request_id": rid})
                                       for rid in (12, 15, 16, 19)] +
                         [("reject_request", {"request_id": rid, "reason": "No"})
                          for rid in (12, 15, 16, 19)])
async def test_ac9_expired_delegation_cannot_approve_or_reject(tool, args, fake_college):
    _, _, r, b = await trun(fake_college, "senthil", tool, args)
    assert r.is_error
    assert await pending_count() == 0
    assert writes(fake_college) == []
    for n in CSE2A_NAMES:
        assert n not in r.content


@pytest.mark.parametrize("rid", [9999, 0, -1])
async def test_ac9_unknown_request_refused(rid, fake_college):
    _, _, r, b = await trun(fake_college, "lakshmi", "approve_request", {"request_id": rid})
    assert r.is_error
    assert await pending_count() == 0


async def test_ac9_reject_needs_reason(fake_college):
    for reason in ("", "   "):
        _, _, r, b = await trun(fake_college, "lakshmi", "reject_request",
                                {"request_id": 12, "reason": reason})
        assert r.is_error
    assert await pending_count() == 0


async def test_ac9_delegated_teacher_can_act_on_active_delegation(fake_college):
    ctx, cid, aid = await propose(fake_college, "approve_request", ctx=fake_college.ctx("senthil"),
                                  args={"request_id": 18})
    await user_says(cid)
    r, b = await run("confirm_pending_action", {"action_id": aid}, ctx, cid)
    assert not r.is_error, b
    assert fake_college.requests[18]["status"] == "approved"


async def test_ac9_authority_lost_before_confirm_relays_college_refusal(fake_college):
    ctx, cid, aid = await propose(fake_college, "approve_request", ctx=fake_college.ctx("senthil"),
                                  args={"request_id": 18})
    fake_college.grants["senthil"] = [(12, "advisor", None)]   # delegation withdrawn
    await user_says(cid)
    r, b = await run("confirm_pending_action", {"action_id": aid}, ctx, cid)
    assert r.is_error
    assert "You are not authorized to act on this request." in r.content
    assert fake_college.requests[18]["status"] == "pending"
    assert await action_status(aid) != "done"


async def test_ac9_already_decided_relays_college_message(fake_college):
    ctx, cid, aid = await propose(fake_college, "approve_request", ctx=fake_college.ctx("lakshmi"),
                                  args={"request_id": 16})
    await user_says(cid)
    r, b = await run("confirm_pending_action", {"action_id": aid}, ctx, cid)
    assert r.is_error
    assert "Request is already rejected." in r.content


async def test_ac9_already_decided_warned_before_confirm(fake_college):
    _, _, r, b = await trun(fake_college, "lakshmi", "approve_request", {"request_id": 15})
    assert not r.is_error
    assert any(mentions(w, "already") for w in b["warnings"])


@pytest.mark.parametrize("login", [None, "arjun", "priya", "karthik"])
@pytest.mark.parametrize("tool,args", [("approve_request", {"request_id": 12}),
                                       ("reject_request", {"request_id": 12, "reason": "x"}),
                                       ("list_class_requests", {}),
                                       ("get_class_attendance", {"class_id": 11}),
                                       ("list_my_classes", {})])
async def test_ac9_visitor_and_student_never_reach_teacher_tools(login, tool, args, fake_college):
    _, _, r, b = await trun(fake_college, login, tool, args)
    assert r.is_error
    assert fake_college.log == []
    assert await pending_count() == 0


@pytest.mark.parametrize("login", [None, "arjun", "priya"])
async def test_ac9_student_cannot_confirm_teachers_approval(login, fake_college):
    ctx, cid, aid = await propose(fake_college, "approve_request")
    await user_says(cid)
    other = fake_college.ctx(login)
    r, b = await run("confirm_pending_action", {"action_id": aid}, other, cid)
    assert r.is_error
    assert writes(fake_college) == []
    assert fake_college.requests[12]["status"] == "pending"


async def test_ac9_other_teacher_cannot_confirm_approval(fake_college):
    ctx, cid, aid = await propose(fake_college, "approve_request")
    await user_says(cid)
    r, b = await run("confirm_pending_action", {"action_id": aid}, fake_college.ctx("senthil"), cid)
    assert r.is_error
    assert writes(fake_college) == []


async def test_ac9_class_attendance_lowest_first(fake_college):
    _, _, r, b = await trun(fake_college, "lakshmi", "get_class_attendance",
                            {"class_code": "CSE-2A"})
    pcts = [s["percentage"] for s in b["students"]]
    assert pcts == sorted(pcts)
    assert {s["name"] for s in b["students"]} == {"Arjun K", "Priya M", "Rahul R"}
