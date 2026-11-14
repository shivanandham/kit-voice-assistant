"""ABU6: caps on pending actions and on admission applications."""

import pytest
from app import limits

from tests.acc_tools_helpers import ADMISSION, LEAVE, new_conv, run, user_says, writes

ADMISSIONS = "/v1/admissions/applications"


@pytest.fixture
def clock(monkeypatch):
    t = [1000.0]
    monkeypatch.setattr(limits, "_clock", lambda: t[0])
    return t


@pytest.fixture
def ip():
    """Run the tools as if called from a voice socket at this IP."""
    token = limits.client_ip_var.set("198.51.100.1")
    yield
    limits.client_ip_var.reset(token)


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    from app.kit import client as kc

    monkeypatch.setattr(kc, "RETRY_BACKOFF_SECONDS", 0.0)


async def leave(ctx, cid, reason: str):
    return await run("submit_leave_request", {**LEAVE, "reason": reason}, ctx, cid)


async def test_abu6_pending_actions_per_conversation(fake_college):
    ctx = fake_college.ctx("arjun")
    cid = await new_conv(ctx)
    ids = []
    for n in range(3):
        r, b = await leave(ctx, cid, f"Fever day {n}")
        assert not r.is_error and b["status"] == "needs_confirmation"
        ids.append(b["action_id"])
    r, b = await leave(ctx, cid, "Fever day 3")
    assert r.is_error and "pending" in b["error"].lower()
    other = await new_conv(ctx)
    r, _ = await leave(ctx, other, "Fever elsewhere")
    assert not r.is_error  # another conversation is unaffected
    r, _ = await run("cancel_pending_action", {"action_id": ids[0]}, ctx, cid)
    assert not r.is_error
    r, _ = await leave(ctx, cid, "Fever day 3")
    assert not r.is_error


async def test_abu6_same_pending_action_again_is_not_refused(fake_college):
    ctx = fake_college.ctx("arjun")
    cid = await new_conv(ctx)
    for n in range(3):
        await leave(ctx, cid, f"Fever day {n}")
    r, b = await leave(ctx, cid, "Fever day 1")  # identical: the existing one is reused
    assert not r.is_error and b["status"] == "needs_confirmation"


async def submit(ctx, n: int):
    """Propose and confirm one admission; returns the confirm result (or the refusal)."""
    cid = await new_conv(ctx)
    await user_says(cid, "I want to apply")
    args = {**ADMISSION, "email": f"ravi{n}@example.com"}
    r, b = await run("submit_admission_application", args, ctx, cid)
    if r.is_error:
        return r, b
    await user_says(cid)
    return await run("confirm_pending_action", {"action_id": b["action_id"]}, ctx, cid)


async def test_abu6_admissions_per_visitor_session_per_day(fake_college, clock, ip):
    ctx = fake_college.ctx(None)
    for n in range(2):
        r, b = await submit(ctx, n)
        assert not r.is_error and b["status"] == "done"
    r, b = await submit(ctx, 2)
    assert r.is_error and "application" in b["error"].lower()
    assert len(writes(fake_college, ADMISSIONS)) == 2
    clock[0] += 86401
    r, b = await submit(ctx, 3)
    assert not r.is_error


async def test_abu6_admissions_per_ip_per_day(fake_college, clock, ip):
    done = 0
    for _ in range(3):  # three visitor sessions from one IP
        ctx = fake_college.ctx(None)
        for _ in range(2):
            r, _ = await submit(ctx, done)
            if not r.is_error:
                done += 1
    assert done == 5
    assert len(writes(fake_college, ADMISSIONS)) == 5
    token = limits.client_ip_var.set("198.51.100.2")
    try:
        r, _ = await submit(fake_college.ctx(None), 99)  # another IP is unaffected
        assert not r.is_error
    finally:
        limits.client_ip_var.reset(token)


async def test_abu6_admission_cap_also_checked_at_confirm(fake_college, clock, ip):
    ctx = fake_college.ctx(None)
    pending = []
    for n in range(3):  # three proposals before any confirmation
        cid = await new_conv(ctx)
        await user_says(cid, "apply")
        r, b = await run(
            "submit_admission_application", {**ADMISSION, "email": f"p{n}@example.com"}, ctx, cid
        )
        assert not r.is_error
        pending.append((cid, b["action_id"]))
    results = []
    for cid, aid in pending:
        await user_says(cid)
        r, _ = await run("confirm_pending_action", {"action_id": aid}, ctx, cid)
        results.append(r.is_error)
    assert results == [False, False, True]
    assert len(writes(fake_college, ADMISSIONS)) == 2
