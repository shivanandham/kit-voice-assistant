"""AC8, COR6, COR7: the confirm-before-write gate.

A write reaches the college only through a pending action confirmed by the same session in the
same conversation after a later user message; it executes at most once and always carries an
Idempotency-Key; a college refusal is relayed with the college's own message.
"""

import asyncio
import json
import uuid

import httpx
import pytest
from app import history
from app.agent import confirm as gate
from app.agent import tools as T
from app.auth import SessionCtx
from app.db import pool

from tests.acc_tools_helpers import (
    SAMPLE_ARGS,
    WRITE_LOGIN,
    WRITE_PATH,
    WRITE_TOOLS,
    action_status,
    new_conv,
    override,
    pending_count,
    propose,
    run,
    secrets_for,
    user_says,
    writes,
)


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    from app.kit import client as kc

    monkeypatch.setattr(kc, "RETRY_BACKOFF_SECONDS", 0.0)


async def confirm(ctx, cid, aid):
    return await run("confirm_pending_action", {"action_id": aid}, ctx, cid)


# ------------------------------------------------------------------ proposing never writes


@pytest.mark.parametrize("tool", WRITE_TOOLS)
async def test_ac8_write_tool_creates_pending_and_no_college_write(tool, fake_college):
    ctx, cid, aid = await propose(fake_college, tool)
    assert writes(fake_college) == []
    assert await action_status(aid) == "pending"
    row = await pool().fetchrow("SELECT * FROM pending_actions WHERE id = $1", uuid.UUID(aid))
    assert row["session_id"] == ctx.session_id
    assert str(row["conversation_id"]) == cid
    assert row["tool_name"] == tool


@pytest.mark.parametrize("tool", WRITE_TOOLS)
async def test_ac8_needs_confirmation_shape(tool, fake_college):
    ctx = fake_college.ctx(WRITE_LOGIN[tool])
    cid = await new_conv(ctx)
    r, b = await run(tool, SAMPLE_ARGS[tool], ctx, cid)
    assert not r.is_error
    assert b["status"] == "needs_confirmation"
    assert uuid.UUID(b["action_id"])
    assert isinstance(b["summary"], str) and b["summary"]
    assert isinstance(b["warnings"], list)


@pytest.mark.parametrize("tool", WRITE_TOOLS)
async def test_ac8_repeated_proposal_never_writes(tool, fake_college):
    ctx = fake_college.ctx(WRITE_LOGIN[tool])
    cid = await new_conv(ctx)
    for _ in range(3):
        await run(tool, SAMPLE_ARGS[tool], ctx, cid)
    assert writes(fake_college) == []


# ------------------------------------------------------------------ valid confirmation


@pytest.mark.parametrize("tool", WRITE_TOOLS)
async def test_ac8_valid_confirm_writes_exactly_once(tool, fake_college):
    ctx, cid, aid = await propose(fake_college, tool)
    await user_says(cid)
    r, b = await confirm(ctx, cid, aid)
    assert not r.is_error, b
    w = writes(fake_college)
    assert [c.path for c in w] == [WRITE_PATH[tool]]
    assert w[0].status < 400
    assert await action_status(aid) == "done"


# DEFECT: cancel/register/approve/reject are POSTed with no Idempotency-Key (confirm.py executors, client.py methods take none).
@pytest.mark.parametrize("tool", WRITE_TOOLS)
async def test_cor7_confirmed_write_carries_idempotency_key(tool, fake_college):
    ctx, cid, aid = await propose(fake_college, tool)
    await user_says(cid)
    await confirm(ctx, cid, aid)
    w = writes(fake_college)
    assert len(w) == 1
    assert w[0].idempotency_key, f"{tool} was sent without an Idempotency-Key"


@pytest.mark.parametrize("tool", WRITE_TOOLS)
async def test_ac2_confirmed_write_carries_own_token(tool, fake_college):
    ctx, cid, aid = await propose(fake_college, tool)
    await user_says(cid)
    await confirm(ctx, cid, aid)
    w = writes(fake_college)[0]
    assert w.token == ctx.kit_token
    assert w.login == WRITE_LOGIN[tool]


# DEFECT: same cause: cancel/register/approve/reject writes carry no Idempotency-Key.
@pytest.mark.parametrize("tool", WRITE_TOOLS)
async def test_cor7_idempotency_keys_differ_between_actions(tool, fake_college):
    keys = []
    for _ in range(2):
        fake_college_state = dict(fake_college.requests)
        ctx, cid, aid = await propose(fake_college, tool)
        await user_says(cid)
        await confirm(ctx, cid, aid)
        keys.append(writes(fake_college)[-1].idempotency_key)
        fake_college.requests.update(fake_college_state)
        for r in fake_college.requests.values():
            if r["id"] == 12:
                r["status"] = "pending"
        fake_college.registrations.clear()
    assert keys[0] and keys[1] and keys[0] != keys[1]


@pytest.mark.parametrize("tool", WRITE_TOOLS)
async def test_ac8_confirm_result_is_relayed(tool, fake_college):
    ctx, cid, aid = await propose(fake_college, tool)
    await user_says(cid)
    r, b = await confirm(ctx, cid, aid)
    assert b.get("status") == "done"


async def test_ac8_leave_payload_forwarded_exactly(fake_college):
    args = {"type": "on_duty", "from_date": "2026-11-25", "to_date": "2026-11-26",
            "half_day": False, "reason": "Symposium", "event_name": "TechFest 2026"}
    ctx, cid, aid = await propose(fake_college, "submit_leave_request", args=args)
    await user_says(cid)
    await confirm(ctx, cid, aid)
    sent = writes(fake_college)[0].body
    for k in ("type", "from_date", "to_date", "reason", "event_name"):
        assert sent[k] == args[k]
    assert not sent.get("half_day")
    assert "student_id" not in sent and "login_id" not in sent


async def test_ac8_admission_payload_forwarded(fake_college):
    ctx, cid, aid = await propose(fake_college, "submit_admission_application")
    await user_says(cid)
    await confirm(ctx, cid, aid)
    sent = writes(fake_college)[0].body
    assert sent["full_name"] == "Ravi Kumar"
    assert sent["course_code"] == "BTECH-CSE"
    assert float(sent["twelfth_percent"]) == 88.0
    assert sent["email"] == "ravi@example.com"
    assert writes(fake_college)[0].token is None
    assert len(fake_college.applications) == 1


async def test_ac8_reject_reason_forwarded(fake_college):
    ctx, cid, aid = await propose(fake_college, "reject_request")
    await user_says(cid)
    await confirm(ctx, cid, aid)
    assert writes(fake_college)[0].body["reason"] == "No proof attached"
    assert fake_college.requests[12]["status"] == "rejected"


async def test_ac8_approve_comment_forwarded(fake_college):
    ctx, cid, aid = await propose(fake_college, "approve_request",
                                  args={"request_id": 12, "comment": "All the best"})
    await user_says(cid)
    await confirm(ctx, cid, aid)
    assert writes(fake_college)[0].body.get("comment") == "All the best"
    assert fake_college.requests[12]["status"] == "approved"


# ------------------------------------------------------------------ refusals


@pytest.mark.parametrize("tool", WRITE_TOOLS)
async def test_ac8_confirm_refused_same_turn(tool, fake_college):
    ctx, cid, aid = await propose(fake_college, tool)
    r, b = await confirm(ctx, cid, aid)
    assert r.is_error
    assert writes(fake_college) == []
    assert await action_status(aid) == "pending"


@pytest.mark.parametrize("tool", WRITE_TOOLS)
async def test_ac8_confirm_refused_after_only_assistant_message(tool, fake_college):
    ctx, cid, aid = await propose(fake_college, tool)
    await history.add_message(cid, "assistant", "Shall I submit it? yes")
    r, _ = await confirm(ctx, cid, aid)
    assert r.is_error
    assert writes(fake_college) == []


@pytest.mark.parametrize("tool", WRITE_TOOLS)
async def test_ac8_confirm_refused_when_user_message_is_in_another_conversation(tool,
                                                                               fake_college):
    ctx, cid, aid = await propose(fake_college, tool)
    other = await new_conv(ctx)
    await user_says(other)
    r, _ = await confirm(ctx, cid, aid)
    assert r.is_error
    assert writes(fake_college) == []


@pytest.mark.parametrize("tool", WRITE_TOOLS)
async def test_ac8_confirm_refused_when_message_predates_action(tool, fake_college):
    ctx = fake_college.ctx(WRITE_LOGIN[tool])
    cid = await new_conv(ctx)
    for t in ("hi", "yes", "yes"):
        await user_says(cid, t)
    _, b = await run(tool, SAMPLE_ARGS[tool], ctx, cid)
    r, _ = await confirm(ctx, cid, b["action_id"])
    assert r.is_error
    assert writes(fake_college) == []


@pytest.mark.parametrize("tool", WRITE_TOOLS)
async def test_ac8_confirm_refused_for_another_session_same_user(tool, fake_college):
    ctx, cid, aid = await propose(fake_college, tool)
    await user_says(cid)
    other = SessionCtx("sid-other-session", ctx.owner_key, ctx.role, ctx.user, ctx.kit_token)
    r, _ = await confirm(other, cid, aid)
    assert r.is_error
    assert writes(fake_college) == []
    assert await action_status(aid) == "pending"


@pytest.mark.parametrize("tool", WRITE_TOOLS)
async def test_ac8_confirm_refused_for_another_conversation(tool, fake_college):
    ctx, cid, aid = await propose(fake_college, tool)
    await user_says(cid)
    other = await new_conv(ctx)
    await user_says(other)
    r, _ = await confirm(ctx, other, aid)
    assert r.is_error
    assert writes(fake_college) == []
    assert await action_status(aid) == "pending"


@pytest.mark.parametrize("tool", WRITE_TOOLS)
@pytest.mark.parametrize("other_login", [None, "priya", "senthil", "karthik"])
async def test_ac8_confirm_refused_for_another_user(tool, other_login, fake_college):
    ctx, cid, aid = await propose(fake_college, tool)
    await user_says(cid)
    other = fake_college.ctx(other_login)
    r, _ = await confirm(other, cid, aid)
    assert r.is_error
    assert writes(fake_college) == []
    assert await action_status(aid) == "pending"


@pytest.mark.parametrize("tool", WRITE_TOOLS)
async def test_ac8_cancel_prevents_later_confirm(tool, fake_college):
    ctx, cid, aid = await propose(fake_college, tool)
    r, b = await run("cancel_pending_action", {"action_id": aid}, ctx, cid)
    assert not r.is_error, b
    assert await action_status(aid) == "cancelled"
    await user_says(cid)
    r, _ = await confirm(ctx, cid, aid)
    assert r.is_error
    assert writes(fake_college) == []


@pytest.mark.parametrize("tool", WRITE_TOOLS)
async def test_ac8_cancel_by_other_session_refused(tool, fake_college):
    ctx, cid, aid = await propose(fake_college, tool)
    other = fake_college.ctx("priya")
    r, _ = await run("cancel_pending_action", {"action_id": aid}, other, cid)
    assert r.is_error
    assert await action_status(aid) == "pending"


@pytest.mark.parametrize("tool", WRITE_TOOLS)
async def test_ac8_cancel_in_other_conversation_refused(tool, fake_college):
    ctx, cid, aid = await propose(fake_college, tool)
    r, _ = await run("cancel_pending_action", {"action_id": aid}, ctx, await new_conv(ctx))
    assert r.is_error
    assert await action_status(aid) == "pending"


@pytest.mark.parametrize("tool", WRITE_TOOLS)
async def test_ac8_cancel_twice_second_is_error(tool, fake_college):
    ctx, cid, aid = await propose(fake_college, tool)
    await run("cancel_pending_action", {"action_id": aid}, ctx, cid)
    r, _ = await run("cancel_pending_action", {"action_id": aid}, ctx, cid)
    assert r.is_error


@pytest.mark.parametrize("tool", WRITE_TOOLS)
async def test_ac8_cancel_after_done_does_not_undo(tool, fake_college):
    ctx, cid, aid = await propose(fake_college, tool)
    await user_says(cid)
    await confirm(ctx, cid, aid)
    r, _ = await run("cancel_pending_action", {"action_id": aid}, ctx, cid)
    assert r.is_error
    assert await action_status(aid) == "done"


@pytest.mark.parametrize("status", ["cancelled", "done", "failed"])
@pytest.mark.parametrize("tool", WRITE_TOOLS)
async def test_ac8_confirm_refused_for_terminal_status(status, tool, fake_college):
    ctx, cid, aid = await propose(fake_college, tool)
    await pool().execute("UPDATE pending_actions SET status = $1 WHERE id = $2", status,
                         uuid.UUID(aid))
    await user_says(cid)
    r, b = await confirm(ctx, cid, aid)
    assert writes(fake_college) == []
    assert await action_status(aid) == status
    if status != "done":
        assert r.is_error
    else:
        assert b.get("status") != "done", "a done action must not be reported as freshly done"


@pytest.mark.parametrize("bad", ["", "abc", "not-a-uuid", "1", "' OR 1=1 --",
                                 "00000000-0000-0000-0000-000000000000",
                                 str(uuid.uuid4()), "{" + str(uuid.uuid4()) + "}x",
                                 "x" * 5000, "\x00", "../../etc/passwd"])
async def test_ac8_confirm_refused_for_unknown_or_malformed_id(bad, fake_college):
    ctx, cid, aid = await propose(fake_college, "submit_leave_request")
    await user_says(cid)
    r, b = await confirm(ctx, cid, bad)
    assert r.is_error
    assert isinstance(b, dict)
    assert writes(fake_college) == []
    assert await action_status(aid) == "pending"


@pytest.mark.parametrize("bad", [123, None, True, ["x"], {"id": 1}, 1.5])
async def test_ac8_confirm_refused_for_non_string_id(bad, fake_college):
    ctx, cid, aid = await propose(fake_college, "submit_leave_request")
    await user_says(cid)
    r, _ = await confirm(ctx, cid, bad)
    assert r.is_error
    assert writes(fake_college) == []


@pytest.mark.parametrize("bad", ["", "abc", str(uuid.uuid4())])
async def test_ac8_cancel_unknown_or_malformed_id(bad, fake_college):
    ctx, cid, aid = await propose(fake_college, "submit_leave_request")
    r, _ = await run("cancel_pending_action", {"action_id": bad}, ctx, cid)
    assert r.is_error
    assert await action_status(aid) == "pending"


@pytest.mark.parametrize("bad_cid", ["", "nope", str(uuid.uuid4())])
async def test_ac8_confirm_with_bad_conversation_refused(bad_cid, fake_college):
    ctx, cid, aid = await propose(fake_college, "submit_leave_request")
    await user_says(cid)
    payload, is_error = await gate.confirm(ctx, bad_cid, aid)
    assert is_error
    assert writes(fake_college) == []
    payload, is_error = await gate.cancel(ctx, bad_cid, aid)
    assert is_error
    assert await action_status(aid) == "pending"


async def test_ac8_unknown_tool_name_in_pending_row_never_executes(fake_college):
    ctx = fake_college.ctx("arjun")
    cid = await new_conv(ctx)
    aid = await gate.create_pending(ctx, cid, "drop_all_tables", {"x": 1}, "Do something odd")
    await user_says(cid)
    r, _ = await confirm(ctx, cid, aid)
    assert r.is_error
    assert writes(fake_college) == []


# ------------------------------------------------------------------ at most once


@pytest.mark.parametrize("tool", WRITE_TOOLS)
async def test_cor7_double_confirm_writes_once(tool, fake_college):
    ctx, cid, aid = await propose(fake_college, tool)
    await user_says(cid)
    await confirm(ctx, cid, aid)
    await user_says(cid, "yes yes")
    r, b = await confirm(ctx, cid, aid)
    assert len(writes(fake_college)) == 1
    assert b.get("status") != "done", "a second confirm must not report a fresh submission"


@pytest.mark.parametrize("tool", WRITE_TOOLS)
async def test_cor7_concurrent_confirms_write_once(tool, fake_college):
    ctx, cid, aid = await propose(fake_college, tool)
    await user_says(cid)
    results = await asyncio.gather(*[confirm(ctx, cid, aid) for _ in range(6)])
    assert len(writes(fake_college)) == 1
    fresh = [b for r, b in results if not r.is_error and b.get("status") == "done"]
    assert len(fresh) == 1
    assert await action_status(aid) == "done"


async def test_cor7_concurrent_confirms_create_one_leave_record(fake_college):
    ctx, cid, aid = await propose(fake_college, "submit_leave_request")
    await user_says(cid)
    await asyncio.gather(*[confirm(ctx, cid, aid) for _ in range(5)])
    created = [r for r in fake_college.requests.values() if r["id"] > 100]
    assert len(created) == 1


@pytest.mark.parametrize("tool", WRITE_TOOLS)
async def test_cor7_repeat_proposal_reuses_or_never_doubles(tool, fake_college):
    """The model proposing the same action twice and both being confirmed yields one record."""
    ctx = fake_college.ctx(WRITE_LOGIN[tool])
    cid = await new_conv(ctx)
    await user_says(cid, "do it")
    _, b1 = await run(tool, SAMPLE_ARGS[tool], ctx, cid)
    _, b2 = await run(tool, SAMPLE_ARGS[tool], ctx, cid)
    await user_says(cid)
    await confirm(ctx, cid, b1["action_id"])
    await confirm(ctx, cid, b2["action_id"])
    assert len([w for w in writes(fake_college) if w.status < 400]) == 1


@pytest.mark.parametrize("tool", ["submit_leave_request", "submit_admission_application"])
async def test_cor7_lost_response_retry_creates_one_record(tool, fake_college):
    """The college stores the record but the response is lost; the retry must not duplicate."""
    state = {"n": 0}

    def flaky(call, orig):
        out = orig()
        state["n"] += 1
        if state["n"] == 1:
            raise httpx.ReadTimeout("lost response")
        return out

    override(fake_college, "POST", WRITE_PATH[tool], flaky)
    ctx, cid, aid = await propose(fake_college, tool)
    await user_says(cid)
    r, b = await confirm(ctx, cid, aid)
    w = writes(fake_college, WRITE_PATH[tool])
    assert len(w) >= 2
    assert len({c.idempotency_key for c in w}) == 1 and w[0].idempotency_key
    if tool == "submit_leave_request":
        assert len([x for x in fake_college.requests.values() if x["id"] > 100]) == 1
    else:
        assert len(fake_college.applications) == 1
    assert not r.is_error
    assert await action_status(aid) == "done"


# DEFECT: same cause: cancel/register/approve/reject writes carry no Idempotency-Key, so a retry is not deduplicated.
@pytest.mark.parametrize("tool", WRITE_TOOLS)
async def test_cor7_retry_after_outage_reuses_same_key(tool, fake_college):
    """College down at confirm time: nothing marked done; a later confirm uses the same key."""
    fake_college.fail(WRITE_PATH[tool], 503, method="POST")
    ctx, cid, aid = await propose(fake_college, tool)
    await user_says(cid)
    r, b = await confirm(ctx, cid, aid)
    assert r.is_error
    assert await action_status(aid) != "done"
    first_keys = {c.idempotency_key for c in writes(fake_college)}
    fake_college.clear_faults()
    await user_says(cid, "try again")
    r, b = await confirm(ctx, cid, aid)
    assert not r.is_error, b
    keys = {c.idempotency_key for c in writes(fake_college)}
    assert None not in keys
    assert keys == first_keys


# ------------------------------------------------------------------ college refusals (COR6)

REFUSALS = [
    (409, "invalid_state", "Request is already approved."),
    (403, "not_allowed", "You are not authorized to act on this request."),
    (422, "invalid_input", "event_name is required for on-duty."),
    (404, "not_found", "Request with id 12 not found in this sandbox."),
    (409, "no_seats", "No seats left."),
    (403, "not_eligible", "Not eligible: cgpa."),
    (400, "bad_request", "Le système a refusé: «date» invalide."),
]


@pytest.mark.parametrize("status,code,message", REFUSALS)
@pytest.mark.parametrize("tool", WRITE_TOOLS)
async def test_cor6_college_refusal_relayed_verbatim(status, code, message, tool, fake_college):
    fake_college.fail(WRITE_PATH[tool], status, method="POST", code=code, message=message)
    ctx, cid, aid = await propose(fake_college, tool)
    await user_says(cid)
    r, b = await confirm(ctx, cid, aid)
    assert r.is_error
    assert message in json.dumps(b, ensure_ascii=False)
    assert await action_status(aid) != "done"


@pytest.mark.parametrize("tool", WRITE_TOOLS)
async def test_cor6_refusal_not_retried(tool, fake_college):
    fake_college.fail(WRITE_PATH[tool], 409, method="POST", code="invalid_state",
                      message="Request is already approved.")
    ctx, cid, aid = await propose(fake_college, tool)
    await user_says(cid)
    await confirm(ctx, cid, aid)
    assert len(writes(fake_college)) == 1


@pytest.mark.parametrize("tool", WRITE_TOOLS)
async def test_cor6_real_college_refusal_from_state(tool, fake_college):
    """The fake's own state-based refusal (no scripted fault) is relayed too."""
    ctx, cid, aid = await propose(fake_college, tool)
    # change the college's state behind the assistant's back
    if tool == "submit_admission_application":
        fake_college.courses = [c for c in fake_college.courses if c["code"] != "BTECH-CSE"]
        expected = "Unknown course code."
    elif tool == "submit_leave_request":
        fake_college.fail("/v1/me/requests", 422, method="POST", code="invalid_input",
                          message="to_date cannot be before from_date.")
        expected = "to_date cannot be before from_date."
    elif tool == "cancel_request":
        fake_college.requests[12]["status"] = "rejected"
        expected = "Request is already rejected."
    elif tool == "register_for_placement_drive":
        fake_college.drives = [{**d, "seats_left": 0} for d in fake_college.drives]
        expected = "No seats left."
    else:
        fake_college.requests[12]["status"] = "approved"
        expected = "Request is already approved."
    await user_says(cid)
    r, b = await confirm(ctx, cid, aid)
    assert r.is_error
    assert expected in r.content
    assert await action_status(aid) != "done"


@pytest.mark.parametrize("exc", [httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout])
@pytest.mark.parametrize("tool", WRITE_TOOLS)
async def test_rel1_write_transport_failure_reported_not_done(exc, tool, fake_college):
    fake_college.fail(WRITE_PATH[tool], method="POST", exc=exc)
    ctx, cid, aid = await propose(fake_college, tool)
    await user_says(cid)
    r, b = await confirm(ctx, cid, aid)
    assert r.is_error
    assert "Traceback" not in r.content
    assert await action_status(aid) != "done"


@pytest.mark.parametrize("tool", [t for t in WRITE_TOOLS if WRITE_LOGIN[t]])
async def test_ac6_401_at_confirm_invalidates_session_and_not_done(tool, fake_college):
    token, ctx = await fake_college.db_session(WRITE_LOGIN[tool])
    cid = await new_conv(ctx)
    await user_says(cid, "do it")
    _, b = await run(tool, SAMPLE_ARGS[tool], ctx, cid)
    aid = b["action_id"]
    fake_college.expire_token(ctx.kit_token)
    await user_says(cid)
    r, _ = await confirm(ctx, cid, aid)
    assert r.is_error
    assert await action_status(aid) != "done"
    assert await pool().fetchval("SELECT count(*) FROM sessions WHERE id = $1",
                                 ctx.session_id) == 0
    assert len(writes(fake_college)) == 1  # not retried


# ------------------------------------------------------------------ pending_summary


async def test_ac8_pending_summary_none_without_actions(fake_college):
    ctx = fake_college.ctx("arjun")
    assert await T.pending_summary(ctx, await new_conv(ctx)) is None


@pytest.mark.parametrize("bad", ["", "nope", str(uuid.uuid4())])
async def test_ac8_pending_summary_bad_conversation_is_none(bad, fake_college):
    await propose(fake_college, "submit_leave_request")
    assert await T.pending_summary(fake_college.ctx("arjun"), bad) is None


async def test_ac8_pending_summary_only_this_session_and_conversation(fake_college):
    a = fake_college.ctx("arjun")
    a2 = SessionCtx("sid-arjun-second", a.owner_key, a.role, a.user, a.kit_token)
    b = fake_college.ctx("priya")
    _, c1, mine = await propose(fake_college, "submit_leave_request", ctx=a)
    _, c2, other_conv = await propose(fake_college, "submit_leave_request", ctx=a,
                                      args={**SAMPLE_ARGS["submit_leave_request"],
                                            "reason": "Other conversation"})
    other_sess_cid = await new_conv(a2)
    _, bd = await run("submit_leave_request", {**SAMPLE_ARGS["submit_leave_request"],
                                               "reason": "Other session"}, a2, other_sess_cid)
    _, c3, other_user = await propose(fake_college, "submit_leave_request", ctx=b,
                                      args={**SAMPLE_ARGS["submit_leave_request"],
                                            "reason": "Priya private"})
    text = await T.pending_summary(a, c1)
    assert text and mine in text
    for foreign in (other_conv, bd["action_id"], other_user, "Other conversation",
                    "Other session", "Priya private"):
        assert foreign not in text
    # the same conversation id seen from another session shows nothing of A's
    assert await T.pending_summary(b, c1) is None
    assert await T.pending_summary(a2, c1) is None


async def test_ac8_pending_summary_excludes_cancelled_done_failed(fake_college):
    ctx = fake_college.ctx("arjun")
    cid = await new_conv(ctx)

    async def make(reason):
        _, b = await run("submit_leave_request", {**SAMPLE_ARGS["submit_leave_request"],
                                                  "reason": reason}, ctx, cid)
        return b["action_id"]

    keep = await make("Keep")
    cancelled = await make("Cancel me")
    await run("cancel_pending_action", {"action_id": cancelled}, ctx, cid)
    done = await make("Done me")
    await pool().execute("UPDATE pending_actions SET status='done' WHERE id=$1", uuid.UUID(done))
    failed = await make("Fail me")
    await pool().execute("UPDATE pending_actions SET status='failed' WHERE id=$1",
                         uuid.UUID(failed))
    text = await T.pending_summary(ctx, cid)
    assert keep in text
    for i in (cancelled, done, failed):
        assert i not in text
    for word in ("Cancel me", "Done me", "Fail me"):
        assert word not in text


@pytest.mark.parametrize("tool", WRITE_TOOLS)
async def test_ac5_pending_summary_has_no_secrets(tool, fake_college):
    ctx, cid, aid = await propose(fake_college, tool)
    text = await T.pending_summary(ctx, cid)
    assert text and aid in text
    for s in secrets_for(fake_college, ctx):
        assert s not in text


@pytest.mark.parametrize("tool", WRITE_TOOLS)
async def test_ac5_confirm_results_have_no_secrets(tool, fake_college):
    ctx, cid, aid = await propose(fake_college, tool)
    r1, _ = await confirm(ctx, cid, aid)
    await user_says(cid)
    r2, _ = await confirm(ctx, cid, aid)
    r3, _ = await confirm(ctx, cid, aid)
    for r in (r1, r2, r3):
        for s in secrets_for(fake_college, ctx):
            assert s not in r.content
    rows = await pool().fetch("SELECT args::text AS a, summary FROM pending_actions")
    for row in rows:
        for s in secrets_for(fake_college, ctx):
            assert s not in row["a"] and s not in row["summary"]


async def test_ac8_pending_count_untouched_by_refused_confirms(fake_college):
    ctx, cid, aid = await propose(fake_college, "submit_leave_request")
    for bad in ("x", str(uuid.uuid4())):
        await confirm(ctx, cid, bad)
    assert await pending_count() == 1


# DEFECT: concurrent identical proposals race in confirm.create_pending (SELECT then INSERT, no lock or unique key), giving two actions with different Idempotency-Keys and two records (leave always, admission intermittently).
@pytest.mark.parametrize("tool", WRITE_TOOLS)
async def test_cor7_concurrent_identical_proposals_then_confirms_write_once(tool, fake_college):
    """The model firing the same write twice at once, then both confirmed: one record."""
    ctx = fake_college.ctx(WRITE_LOGIN[tool])
    cid = await new_conv(ctx)
    await user_says(cid, "do it")
    results = await asyncio.gather(*[run(tool, SAMPLE_ARGS[tool], ctx, cid) for _ in range(3)])
    ids = {b["action_id"] for _, b in results if "action_id" in b}
    await user_says(cid)
    for aid in ids:
        await confirm(ctx, cid, aid)
    assert len([w for w in writes(fake_college) if w.status < 400]) == 1


@pytest.mark.parametrize("bad_cid", ["not-a-uuid", "", "1234"])
@pytest.mark.parametrize("tool", WRITE_TOOLS)
async def test_agt6_write_with_malformed_conversation_is_safe(bad_cid, tool, fake_college):
    ctx = fake_college.ctx(WRITE_LOGIN[tool])
    r = await T.run_tool(tool, SAMPLE_ARGS[tool], ctx, bad_cid)
    assert r.is_error
    assert "Traceback" not in r.content
    assert await pending_count() == 0
    assert writes(fake_college) == []
