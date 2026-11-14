# ruff: noqa: F811  (the e2e fixture is imported, then requested by name)
"""ABU2 (isolation aspect) end to end: one person's turn allowance never affects another.

ABU2: a visitor gets a daily number of turns counted per IP; a logged-in user gets a daily number
counted per login id, across sessions and devices. The ABU rule of thumb: IP limits apply only to
visitors, so people sharing a network never use up each other's allowance. The allowances are
lowered through the settings overrides so the scenario stays short.
"""

import pytest
from app import limits

from tests.e2e_helpers import (
    TURN_MODES,
    A,
    B,
    e2e,  # noqa: F401
    events,
)

USER_CAP, VISITOR_CAP = 3, 2
IP = "10.4.0.1"


@pytest.fixture
def small_allowances(e2e, monkeypatch):
    monkeypatch.setitem(limits.DEFAULTS, "limit_user_turns_per_day", USER_CAP)
    monkeypatch.setitem(limits.DEFAULTS, "limit_visitor_turns_per_ip_per_day", VISITOR_CAP)
    return e2e


async def turns(e2e, actor, n: int, how: str, tag: str) -> list[bool]:
    """Run n turns on a fresh socket; True for each that was answered by the assistant."""
    s = await actor.socket()
    await s.start()
    out = []
    for i in range(n):
        text = f"{actor.name} {tag} question number {i} about courses"
        e2e.model.plan(text, "Here is the answer.")
        before = len([c for c in e2e.model.by(actor.name)])
        frames = await s.turn(text, how)
        answered = bool(events(frames, "assistant_done"))
        if not answered:
            assert events(frames, "error"), frames  # a clear message, not silence
            assert len(e2e.model.by(actor.name)) == before  # refused before the model
        out.append(answered)
    await s.close()
    return out


@TURN_MODES
async def test_abu2_student_exhausting_allowance_does_not_block_others_on_same_ip(
        small_allowances, how):
    e2e = small_allowances
    a = await e2e.actor("A", A, ip=IP)
    b = await e2e.actor("B", B, ip=IP)
    v = await e2e.actor("V1", ip=IP)
    assert await turns(e2e, a, USER_CAP + 1, how, "a") == [True] * USER_CAP + [False]
    assert await turns(e2e, b, USER_CAP, how, "b") == [True] * USER_CAP
    assert await turns(e2e, v, VISITOR_CAP, how, "v") == [True] * VISITOR_CAP
    a2 = await e2e.actor("A2", A, ip="10.4.9.9")  # A on another device and network
    assert await turns(e2e, a2, 1, how, "a2") == [False]


@TURN_MODES
async def test_abu2_visitor_neither_consumes_nor_is_covered_by_student_allowance(
        small_allowances, how):
    e2e = small_allowances
    v1 = await e2e.actor("V1", ip=IP)
    a = await e2e.actor("A", A, ip=IP)
    assert await turns(e2e, v1, VISITOR_CAP, how, "first") == [True] * VISITOR_CAP
    assert await turns(e2e, a, USER_CAP, how, "a") == [True] * USER_CAP  # not consumed
    assert await turns(e2e, v1, 1, how, "again") == [False]  # not covered by A's allowance
    v2 = await e2e.actor("V2", ip=IP)  # new cookie, same IP: still the IP's allowance
    assert await turns(e2e, v2, 1, how, "v2") == [False]
    v3 = await e2e.actor("V3", ip="10.4.0.2")
    assert await turns(e2e, v3, 1, how, "v3") == [True]
