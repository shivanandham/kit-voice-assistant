"""AC2: every college call for a session carries that session's token and nobody else's.

Two students (A = arjun, B = priya, same class) and a teacher (lakshmi) run every tool in their
role. The fake college maps each token to exactly one user, so the recorded login of every call
proves whose identity was used.
"""

import asyncio

import pytest
from app.db import pool

from tests.acc_tools_helpers import (
    ALLOWED,
    SAMPLE_ARGS,
    SEARCH,
    body,
    new_conv,
    run,
    user_says,
    writes,
)

STUDENT_TOOLS = sorted(ALLOWED["student"] - {SEARCH})
TEACHER_TOOLS = sorted(ALLOWED["teacher"] - {SEARCH})
STUFFED = {
    "student_id": 29, "login_id": "priya", "token": "stolen-token", "authorization":
    "Bearer stolen-token", "user_id": 29, "user": {"login_id": "priya"}, "kit_token": "stolen",
    "as_user": "priya", "Authorization": "Bearer stolen-token",
}


def _assert_only(fake, login, token):
    authed = [c for c in fake.log if c.token is not None]
    for c in authed:
        assert c.token == token, (c.path, c.login)
        assert c.login == login, (c.path, c.login)


@pytest.mark.parametrize("name", STUDENT_TOOLS)
async def test_ac2_student_tool_sends_only_own_token(name, fake_college):
    a, b = fake_college.ctx("arjun"), fake_college.ctx("priya")
    cid = await new_conv(a)
    await run(name, SAMPLE_ARGS[name], a, cid)
    _assert_only(fake_college, "arjun", a.kit_token)
    assert b.kit_token not in {c.token for c in fake_college.log}


@pytest.mark.parametrize("name", STUDENT_TOOLS)
async def test_ac2_student_b_tool_sends_only_b_token(name, fake_college):
    a, b = fake_college.ctx("arjun"), fake_college.ctx("priya")
    args = dict(SAMPLE_ARGS[name])
    if name in ("get_request", "cancel_request"):
        args["request_id"] = 15  # priya's own request
    cid = await new_conv(b)
    await run(name, args, b, cid)
    _assert_only(fake_college, "priya", b.kit_token)
    assert a.kit_token not in {c.token for c in fake_college.log}


@pytest.mark.parametrize("name", TEACHER_TOOLS)
async def test_ac2_teacher_tool_sends_only_own_token(name, fake_college):
    t, s = fake_college.ctx("lakshmi"), fake_college.ctx("arjun")
    cid = await new_conv(t)
    await run(name, SAMPLE_ARGS[name], t, cid)
    _assert_only(fake_college, "lakshmi", t.kit_token)
    assert s.kit_token not in {c.token for c in fake_college.log}


@pytest.mark.parametrize("name", [n for n in STUDENT_TOOLS if n in {
    "get_my_profile", "get_my_academics", "get_my_subjects", "get_attendance_summary",
    "get_attendance_sessions", "list_my_requests", "list_my_placements",
    "list_placement_drives", "get_timetable", "get_calendar"}])
async def test_ac2_concurrent_students_get_their_own_data(name, fake_college):
    """Interleaved runs for A and B: each output is computed from its own identity only."""
    a, b = fake_college.ctx("arjun"), fake_college.ctx("kavya")
    ca, cb = await new_conv(a), await new_conv(b)
    runs = []
    for _ in range(4):
        runs += [run(name, SAMPLE_ARGS[name], a, ca), run(name, SAMPLE_ARGS[name], b, cb)]
    results = await asyncio.gather(*runs)
    for i, (r, _) in enumerate(results):
        mine, other = ("arjun", "kavya") if i % 2 == 0 else ("kavya", "arjun")
        assert not r.is_error, r.content
        assert other.capitalize() not in r.content
        assert '"' + other + '"' not in r.content
    tokens = {c.token: c.login for c in fake_college.log if c.token}
    assert set(tokens.values()) <= {"arjun", "kavya"}
    assert tokens.get(a.kit_token, "arjun") == "arjun"
    assert tokens.get(b.kit_token, "kavya") == "kavya"


async def test_ac2_profile_of_a_and_b_differ(fake_college):
    a, b = fake_college.ctx("arjun"), fake_college.ctx("priya")
    ra, _ = await run("get_my_profile", {}, a, await new_conv(a))
    rb, _ = await run("get_my_profile", {}, b, await new_conv(b))
    assert "Arjun" in ra.content and "Priya" not in ra.content
    assert "Priya" in rb.content and "Arjun" not in rb.content


async def test_ac2_academics_are_own(fake_college):
    a, r_ = fake_college.ctx("arjun"), fake_college.ctx("rahul")
    ra, ba = await run("get_my_academics", {}, a, await new_conv(a))
    rr, br = await run("get_my_academics", {}, r_, await new_conv(r_))
    assert "8.2" in ra.content and "6.1" not in ra.content
    assert "6.1" in rr.content and "8.2" not in rr.content


def _stuffed_cases():
    roles = [("arjun", n) for n in STUDENT_TOOLS] + [("lakshmi", n) for n in TEACHER_TOOLS] + \
            [(None, n) for n in sorted(ALLOWED["visitor"])]
    for login, name in roles:
        for key in STUFFED:
            yield login, name, key


@pytest.mark.parametrize("login,name,key", list(_stuffed_cases()))
async def test_ac2_identity_fields_in_args_never_forwarded(login, name, key, fake_college):
    ctx = fake_college.ctx(login)
    cid = await new_conv(ctx)
    args = {**SAMPLE_ARGS[name], key: STUFFED[key]}
    r, b = await run(name, args, ctx, cid)
    # Rejected or ignored; never forwarded and never changes whose identity is used.
    for c in fake_college.log:
        assert c.token in (None, ctx.kit_token), (c.path, c.token)
        assert "stolen" not in str(c.params) and "stolen" not in str(c.body)
        assert "priya" not in str(c.params) and "priya" not in str(c.body)
        if c.login is not None:
            assert c.login == login
    for row in await pool().fetch("SELECT args FROM pending_actions"):
        assert key not in str(row["args"])
        assert "stolen" not in str(row["args"])


@pytest.mark.parametrize("key", list(STUFFED))
async def test_ac2_identity_field_rejected_for_write(key, fake_college):
    """A write tool carrying an identity field is refused outright (strict schema)."""
    ctx = fake_college.ctx("arjun")
    cid = await new_conv(ctx)
    r, _ = await run("submit_leave_request", {**SAMPLE_ARGS["submit_leave_request"],
                                               key: STUFFED[key]}, ctx, cid)
    assert r.is_error
    assert await pool().fetchval("SELECT count(*) FROM pending_actions") == 0


@pytest.mark.parametrize("rid", [15, 19])  # priya's requests
async def test_ac2_get_request_of_other_student_refused_by_college(rid, fake_college):
    a = fake_college.ctx("arjun")
    r, b = await run("get_request", {"request_id": rid}, a, await new_conv(a))
    assert r.is_error
    assert f"Request with id {rid} not found in this sandbox." in r.content  # COR6 verbatim
    for leak in ("Priya", "Family function", "Dental", "approved"):
        assert leak not in r.content
    _assert_only(fake_college, "arjun", a.kit_token)


@pytest.mark.parametrize("rid", [15, 16, 17, 18, 19])
async def test_ac2_cancel_other_students_request_creates_nothing(rid, fake_college):
    a = fake_college.ctx("arjun")
    cid = await new_conv(a)
    r, b = await run("cancel_request", {"request_id": rid}, a, cid)
    assert r.is_error
    assert await pool().fetchval("SELECT count(*) FROM pending_actions") == 0
    for leak in ("Priya", "Rahul", "Kavya", "Meena", "Family", "Fever", "wedding", "Dental",
                 "hackathon"):
        assert leak not in r.content
    assert writes(fake_college) == []


async def test_ac2_confirmed_write_carries_session_token_only(fake_college):
    a, b = fake_college.ctx("arjun"), fake_college.ctx("priya")
    cid = await new_conv(a)
    await user_says(cid, "medical leave please")
    r, bd = await run("submit_leave_request", SAMPLE_ARGS["submit_leave_request"], a, cid)
    await user_says(cid)
    await run("confirm_pending_action", {"action_id": bd["action_id"]}, a, cid)
    w = writes(fake_college)
    assert len(w) == 1 and w[0].token == a.kit_token and w[0].login == "arjun"
    created = [r for r in fake_college.requests.values() if r["id"] > 100]
    assert len(created) == 1 and created[0]["student_id"] == 28


async def test_ac2_student_cannot_confirm_other_students_action(fake_college):
    a, b = fake_college.ctx("arjun"), fake_college.ctx("priya")
    cid = await new_conv(a)
    await user_says(cid, "leave")
    _, bd = await run("submit_leave_request", SAMPLE_ARGS["submit_leave_request"], a, cid)
    await user_says(cid)
    r, _ = await run("confirm_pending_action", {"action_id": bd["action_id"]}, b, cid)
    assert r.is_error
    assert writes(fake_college) == []
    assert "Fever" not in r.content


async def test_ac2_ctx_user_dict_does_not_override_token_identity(fake_college):
    """A ctx whose user dict names someone else still only speaks with its own token."""
    from app.auth import SessionCtx

    from tests.fakes.college import user_summary

    base = fake_college.ctx("arjun")
    ctx = SessionCtx(base.session_id, base.owner_key, "student", user_summary("priya"),
                     base.kit_token)
    cid = await new_conv(ctx)
    for name in ("get_my_profile", "get_my_academics", "list_my_requests"):
        await run(name, {}, ctx, cid)
    _assert_only(fake_college, "arjun", base.kit_token)


@pytest.mark.parametrize("name", sorted(ALLOWED["visitor"]))
async def test_ac1_visitor_never_makes_authenticated_call(name, fake_college):
    a = fake_college.ctx("arjun")  # a logged-in user exists in the same process
    await run("get_my_profile", {}, a, await new_conv(a))
    fake_college.log.clear()
    v = fake_college.ctx(None)
    await run(name, SAMPLE_ARGS[name], v, await new_conv(v))
    assert all(c.token is None for c in fake_college.log)


async def test_ac2_tool_result_body_is_json(fake_college):
    a = fake_college.ctx("arjun")
    r, b = await run("get_my_profile", {}, a, await new_conv(a))
    assert isinstance(b, dict)
    assert body(r) == b


# DEFECT: run_tool never checks that conversation_id belongs to ctx; A's tool runs and its output is logged into B's conversation (tools.run_tool).
@pytest.mark.parametrize("name", ["get_my_profile", "get_my_academics", "list_my_requests",
                                  "submit_leave_request"])
async def test_ac3_tool_call_in_other_users_conversation_refused(name, fake_college):
    """A's session using B's conversation id gets nothing and leaves nothing in B's conversation."""
    import uuid as _uuid

    a, b = fake_college.ctx("arjun"), fake_college.ctx("priya")
    b_cid = await new_conv(b)
    r, _ = await run(name, SAMPLE_ARGS[name], a, b_cid)
    assert r.is_error
    assert "Arjun" not in r.content and "8.2" not in r.content
    rows = await pool().fetch("SELECT output::text AS o FROM tool_calls WHERE conversation_id = $1",
                              _uuid.UUID(b_cid))
    for row in rows:
        assert "Arjun" not in row["o"] and "8.2" not in row["o"]
    assert await pool().fetchval("SELECT count(*) FROM pending_actions") == 0
