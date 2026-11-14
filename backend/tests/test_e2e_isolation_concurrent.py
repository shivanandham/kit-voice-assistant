"""AC2/AC3 end to end: concurrent users each get only their own data.

Students A and B (same class), C (other class and department), teachers T1 (advisor of A and B's
class) and T2 (no authority over it) and two visitors talk at the same time on their own sockets.
Every college call is attributed to the connection that caused it (see e2e_helpers.ACTOR), so
"every college call for A's turns carries A's token only" is checked exactly, not inferred.
"""

import asyncio

import pytest

from tests.e2e_helpers import (
    TURN_MODES,
    A,
    B,
    C,
    T1,
    T2,
    db_text,
    e2e,  # noqa: F401
    echo,
    events,
    forbidden_for,
    hits,
    owner_of,
    stored_messages,
)
from tests.fakes.college import USERS, user_summary

STUDENT_TOOLS = [("get_my_profile", {}), ("get_attendance_summary", {}), ("list_my_requests", {}),
                 ("get_timetable", {}), ("list_placement_drives", {}), ("get_my_academics", {}),
                 ("list_my_placements", {})]
VISITOR_TOOLS = [("list_courses", {}), ("get_college_info", {})]


def teacher_tools(login: str) -> list:
    own = {"lakshmi": "CSE-2A", "senthil": "ECE-2B"}[login]
    return [("get_my_profile", {}), ("list_my_classes", {}), ("list_class_requests", {}),
            ("get_class_attendance", {"class_code": own})]


def plan_for(login: str | None) -> list:
    if login is None:
        return VISITOR_TOOLS
    return STUDENT_TOOLS if USERS[login]["role"] == "student" else teacher_tools(login)


async def converse(e2e, actor, turns: list[tuple[str, str]]) -> list:
    """Open a socket, start a conversation and run the given (text, how) turns."""
    s = await actor.socket()
    ready = await s.start()
    assert ready and ready["type"] == "ready", ready
    out = []
    for text, how in turns:
        out.append(await s.turn(text, how))
    return out


def check_actor(e2e, actor) -> None:
    """Every exit of this actor carries only this actor's data and identity."""
    login = actor.login
    my_tokens = set(e2e.tokens(login)) if login else set()
    other_tokens = [t for t in e2e.tokens() if t not in my_tokens]
    forbidden = forbidden_for(login) + other_tokens
    calls = e2e.log.by(actor.name)
    authed = [c for c in calls if c.token is not None and c.path != "/v1/auth/login"]
    if login is None:
        assert authed == [], [(c.path, c.login) for c in authed]
    else:
        assert authed, "no authenticated college call was made for this user"
        assert {c.login for c in authed} == {login}, [(c.path, c.login) for c in authed]
        assert {c.token for c in authed} <= my_tokens
    model_text = "\n".join(c.text() for c in e2e.model.by(actor.name))
    assert e2e.model.by(actor.name), "the model was never called for this actor"
    assert hits(model_text, forbidden) == []
    for s in actor.sockets:
        assert hits(s.text(), forbidden) == []
    assert hits(actor.http_text(), forbidden) == []


async def check_storage(actor) -> None:
    forbidden = forbidden_for(actor.login)
    for s in actor.sockets:
        cid = s.conversation_id
        expected_owner = f"user:{actor.login}" if actor.login else None
        owner = await owner_of(cid)
        if expected_owner:
            assert owner == expected_owner
        else:
            assert owner.startswith("visitor:")
        assert hits(await db_text(cid), forbidden) == []


@TURN_MODES
async def test_ac2_ac3_two_students_same_class_concurrent_sockets(e2e, how):
    a = await e2e.actor("A", A, ip="10.1.1.1")
    b = await e2e.actor("B", B, ip="10.1.1.1")  # same network, as in a lab
    turns = {}
    for actor in (a, b):
        t1 = e2e.model.plan(f"{actor.name} asks: how is my attendance and my requests",
                            plan_for(actor.login), echo)
        t2 = e2e.model.plan(f"{actor.name} asks again: my timetable and placements",
                            plan_for(actor.login), echo)
        turns[actor.name] = [(t1, how), (t2, how)]
    await asyncio.gather(converse(e2e, a, turns["A"]), converse(e2e, b, turns["B"]))

    for actor in (a, b):
        check_actor(e2e, actor)
        await check_storage(actor)
    # Positive control: each one's own data did reach them.
    assert "Participating in State Level Technical Symposium" in a.sockets[0].text()
    assert "Family function in hometown" in b.sockets[0].text()
    # Histories are disjoint.
    ha = [c["id"] for c in (await a.get("/api/history")).json()]
    hb = [c["id"] for c in (await b.get("/api/history")).json()]
    assert ha == [a.sockets[0].conversation_id] and hb == [b.sockets[0].conversation_id]
    assert (await a.get(f"/api/history/{hb[0]}")).status_code == 404
    assert (await b.get(f"/api/history/{ha[0]}")).status_code == 404


async def test_ac2_ac3_everyone_at_once_students_teachers_visitors(e2e):
    people = [("A", A, "10.2.0.1"), ("B", B, "10.2.0.1"), ("C", C, "10.2.0.2"),
              ("T1", T1, "10.2.0.1"), ("T2", T2, "10.2.0.3"), ("V1", None, "10.2.0.1"),
              ("V2", None, "10.2.0.4")]
    actors = [await e2e.actor(n, login, ip=ip) for n, login, ip in people]
    jobs = []
    for actor in actors:
        t1 = e2e.model.plan(f"{actor.name} typed: tell me about my records", plan_for(actor.login),
                            echo)
        t2 = e2e.model.plan(f"{actor.name} spoke: and what else do you have", plan_for(actor.login),
                            echo)
        jobs.append(converse(e2e, actor, [(t1, "typed"), (t2, "voice")]))
    results = await asyncio.gather(*jobs)
    for actor, frames_per_turn in zip(actors, results, strict=True):
        for frames in frames_per_turn:
            assert events(frames, "assistant_done"), (actor.name, frames)
        check_actor(e2e, actor)
        await check_storage(actor)
    # Global cross-check: every authenticated college call was caused by its token's owner.
    by_name = {a.name: a.login for a in actors}
    for actor_name, call in e2e.log.pairs():
        if call.login is not None and call.path != "/v1/auth/login":
            assert by_name.get(actor_name) == call.login, (actor_name, call.path, call.login)
    # Positive controls.
    assert user_summary(C)["full_name"] in e2e.actors["C"].sockets[0].text()
    assert "Priya M" in e2e.actors["T1"].sockets[0].text()  # T1 advises CSE-2A
    assert "Kavya S" in e2e.actors["T2"].sockets[0].text()  # T2 advises ECE-2B


@pytest.mark.parametrize("victim", [B, C])
async def test_ac2_model_cannot_pass_another_identity_to_a_tool(e2e, victim):
    """AC2: no tool accepts a user id, login id or token from the model."""
    a = await e2e.actor("A", A)
    vic = await e2e.actor("X", victim, ip="10.3.0.2")
    victim_token = e2e.tokens(victim)[0]
    bogus = [
        ("get_attendance_summary", {"login_id": victim}),
        ("get_attendance_summary", {"user_id": USERS[victim]["id"]}),
        ("list_my_requests", {"token": victim_token}),
        ("get_my_profile", {"student_id": USERS[victim]["id"], "login_id": victim}),
        ("get_request", {"request_id": 15}),  # priya's request, by number
        ("get_request", {"request_id": 17}),  # kavya's request, by number
        ("cancel_request", {"request_id": 15}),
    ]
    text = e2e.model.plan("show my attendance and requests", bogus, echo)
    s = await a.socket()
    await s.start()
    await s.turn(text)
    assert {c.login for c in e2e.log.by("A") if c.token} <= {A}
    assert victim_token not in {c.token for c in e2e.log.by("A")}
    # (the victim's token itself appears in model input only because the model wrote it)
    forbidden = forbidden_for(A)
    assert hits("\n".join(c.text() for c in e2e.model.by("A")), forbidden) == []
    assert hits(s.text(), forbidden) == []
    assert hits(await db_text(s.conversation_id), forbidden) == []
    assert [c for c in e2e.college.writes()] == []
    msgs = await stored_messages(s.conversation_id)
    assert msgs[0]["content"] == text
