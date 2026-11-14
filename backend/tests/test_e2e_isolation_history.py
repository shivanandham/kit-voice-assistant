"""AC3/AC7 and AC6 end to end: conversations belong to their owner; a login switch leaves nothing.

AC3: another user's conversation id returns 404 and no data, whether listed, read, resumed over the
WebSocket or used in a tool call. AC7: a conversation can be resumed only by its owner. AC6:
logout ends the session on the server, and logging in as B in a browser that was A leaves nothing
of A reachable.
"""

import uuid

import pytest

from tests.e2e_helpers import (
    TURN_MODES,
    A,
    B,
    db_text,
    e2e,  # noqa: F401
    echo,
    events,
    forbidden_for,
    hits,
    owner_of,
    stored_messages,
    stored_tool_calls,
)

B_SECRET = "priya private question about her dental appointment zebrafinch"
A_SECRET = "arjun private question about his symposium kingfisher"


async def b_conversation(e2e, ip="10.5.0.2"):
    """B logs in and holds a conversation with private content and a tool call."""
    b = await e2e.actor("B", B, ip=ip)
    s = await b.socket()
    await s.start()
    e2e.model.plan(B_SECRET, [("list_my_requests", {})], echo)
    await s.turn(B_SECRET)
    snapshot = (await stored_messages(s.conversation_id), await stored_tool_calls(s.conversation_id))
    return b, s, snapshot


async def assert_untouched(cid: str, snapshot, owner: str) -> None:
    assert await owner_of(cid) == owner
    assert (await stored_messages(cid), await stored_tool_calls(cid)) == snapshot


def leaks_b(text: str) -> list[str]:
    return hits(text, [B_SECRET, "zebrafinch", *forbidden_for(A)])


# --------------------------------------------------------------------------- AC3 / AC7


@TURN_MODES
async def test_ac3_ac7_student_cannot_resume_another_students_conversation(e2e, how):
    b, sb, snap = await b_conversation(e2e)
    a = await e2e.actor("A", A)
    s = await a.socket()
    ready = await s.start(sb.conversation_id)
    assert ready["type"] == "ready"
    assert ready["conversation_id"] != sb.conversation_id
    e2e.model.plan("what did we talk about before", "We have not talked before.")
    frames = await s.turn("what did we talk about before", how)
    assert events(frames, "assistant_done")
    assert leaks_b(s.text()) == []
    assert leaks_b("\n".join(c.text() for c in e2e.model.by("A"))) == []
    await assert_untouched(sb.conversation_id, snap, "user:priya")
    assert await owner_of(s.conversation_id) == "user:arjun"


async def test_ac3_student_cannot_read_or_list_another_students_conversation(e2e):
    b, sb, snap = await b_conversation(e2e)
    a = await e2e.actor("A", A)
    r = await a.get(f"/api/history/{sb.conversation_id}")
    assert r.status_code == 404
    assert leaks_b(r.text) == [] and sb.conversation_id not in r.text
    for variant in (sb.conversation_id.upper(), sb.conversation_id.replace("-", ""),
                    "{" + sb.conversation_id + "}", f"urn:uuid:{sb.conversation_id}"):
        r = await a.get(f"/api/history/{variant}")
        assert r.status_code == 404, variant
        assert leaks_b(r.text) == []
    listing = (await a.get("/api/history")).json()
    assert all(c["id"] != sb.conversation_id for c in listing)
    assert leaks_b(str(listing)) == []
    await assert_untouched(sb.conversation_id, snap, "user:priya")


@TURN_MODES
async def test_ac3_model_cannot_use_another_conversation_id_in_a_tool_call(e2e, how):
    b, sb, snap = await b_conversation(e2e)
    a = await e2e.actor("A", A)
    s = await a.socket()
    await s.start()
    cid = sb.conversation_id
    e2e.model.plan("check that other chat", [
        ("list_my_requests", {"conversation_id": cid}),
        ("get_attendance_summary", {"conversation_id": cid}),
        ("confirm_pending_action", {"action_id": cid}),
        ("cancel_pending_action", {"action_id": cid}),
        ("search_documents", {"query": cid}),
        ("get_request", {"request_id": 15, "conversation_id": cid}),
    ], echo)
    await s.turn("check that other chat", how)
    assert leaks_b(s.text()) == []
    assert leaks_b("\n".join(c.text() for c in e2e.model.by("A"))) == []
    assert {c.login for c in e2e.log.by("A") if c.token} <= {A}
    await assert_untouched(cid, snap, "user:priya")
    assert e2e.college.writes() == []


@TURN_MODES
async def test_ac3_ac7_visitor_cannot_resume_or_read_a_students_conversation(e2e, how):
    a = await e2e.actor("A", A)
    sa = await a.socket()
    await sa.start()
    e2e.model.plan(A_SECRET, [("list_my_requests", {})], echo)
    await sa.turn(A_SECRET)
    snap = (await stored_messages(sa.conversation_id), await stored_tool_calls(sa.conversation_id))

    v = await e2e.actor("V1", ip=a.ip)
    assert (await v.get(f"/api/history/{sa.conversation_id}")).status_code == 404
    s = await v.socket()
    ready = await s.start(sa.conversation_id)
    assert ready["type"] == "ready" and ready["conversation_id"] != sa.conversation_id
    e2e.model.plan("continue please", "Sure.")
    await s.turn("continue please", how)
    forbidden = [A_SECRET, "kingfisher", *forbidden_for(None)]
    assert hits(s.text(), forbidden) == []
    assert hits("\n".join(c.text() for c in e2e.model.by("V1")), forbidden) == []
    assert all(c.token is None for c in e2e.log.by("V1"))
    assert (await stored_messages(sa.conversation_id), await stored_tool_calls(sa.conversation_id)) \
        == snap


MALFORMED = ["not-a-uuid", "", " ", "null", 123, 1.5, True, ["x"], {"id": "x"},
             "00000000-0000-0000-0000-000000000000", "../../etc/passwd", "'; DROP TABLE messages;--",
             "x" * 4000]


@pytest.mark.parametrize("bad", MALFORMED, ids=[f"m{i}" for i in range(len(MALFORMED))])
async def test_ac3_ac7_malformed_or_unknown_conversation_ids_give_fresh_or_404(e2e, bad):
    b, sb, snap = await b_conversation(e2e)
    a = await e2e.actor("A", A)
    s = await a.socket()
    ready = await s.start(bad)
    assert ready["type"] in ("ready", "error")
    if ready["type"] == "ready":
        cid = ready["conversation_id"]
        uuid.UUID(cid)
        assert cid != sb.conversation_id and cid != bad
        assert await owner_of(cid) == "user:arjun"
        assert await stored_messages(cid) == []
    if isinstance(bad, str) and bad.strip() and "/" not in bad:
        r = await a.get(f"/api/history/{bad}")
        assert r.status_code == 404
    await assert_untouched(sb.conversation_id, snap, "user:priya")


async def test_ac3_nonexistent_uuid_resume_does_not_create_a_conversation_under_that_id(e2e):
    """Resuming an id nobody owns must not let the caller squat it for a later owner."""
    a = await e2e.actor("A", A)
    wanted = str(uuid.uuid4())
    s = await a.socket()
    ready = await s.start(wanted)
    assert ready["conversation_id"] != wanted
    assert await owner_of(wanted) is None


# --------------------------------------------------------------------------- AC6


@TURN_MODES
async def test_ac6_login_switch_in_one_browser_leaves_nothing_of_a(e2e, how):
    a = await e2e.actor("A", ip="10.6.0.1")  # the browser arrives as a visitor
    browser = a.browser
    visitor_cookie = a.cookie
    await a.do_login(A)
    cookie_a = a.cookie
    assert cookie_a and cookie_a != visitor_cookie  # rotated on login
    sa = await a.socket()
    await sa.start()
    e2e.model.plan(A_SECRET, [("list_my_requests", {}), ("get_attendance_summary", {})], echo)
    await sa.turn(A_SECRET, how)
    a_cid = sa.conversation_id
    assert "Participating in State Level Technical Symposium" in sa.text()

    assert (await a.post("/api/logout")).status_code == 200
    b = await e2e.actor("B", ip=a.ip, browser=browser)  # same browser, now as a visitor
    await b.do_login(B)
    assert b.cookie and b.cookie not in (cookie_a, visitor_cookie)

    # B sees none of A's history.
    assert (await b.get(f"/api/history/{a_cid}")).status_code == 404
    listing = (await b.get("/api/history")).json()
    assert all(c["id"] != a_cid for c in listing)
    assert hits(str(listing), [A_SECRET]) == []
    # The model's memory of A's conversation is unreachable to B.
    sb = await b.socket()
    ready = await sb.start(a_cid)
    assert ready["conversation_id"] != a_cid and ready["user_name"] == "Priya M"
    e2e.model.plan("what did I ask you a minute ago", "I don't know.")
    await sb.turn("what did I ask you a minute ago", how)
    forbidden = [A_SECRET, "kingfisher", *forbidden_for(B)]
    assert hits("\n".join(c.text() for c in e2e.model.by("B")), forbidden) == []
    assert hits(sb.text(), forbidden) == []
    assert hits(await db_text(sb.conversation_id), forbidden) == []
    assert {c.login for c in e2e.log.by("B") if c.token} <= {B}


async def test_ac6_old_cookie_is_dead_after_logout(e2e):
    a = await e2e.actor("A", A)
    cookie_a = a.cookie
    sa = await a.socket()
    await sa.start()
    await sa.turn(A_SECRET)
    await sa.close()
    await a.post("/api/logout")

    thief = await e2e.make_browser(base_url="https://testserver", cookies={"kit_session": cookie_a})
    me = (await thief.get("/api/me")).json()
    assert me == {"role": "visitor", "user": None}
    r = await thief.get(f"/api/history/{sa.conversation_id}")
    assert r.status_code == 404
    ghost = await a.socket(cookie=cookie_a)
    f = await ghost.recv()
    assert f is None and ghost.closed_code == 4401


@TURN_MODES
async def test_ac6_socket_opened_before_logout_stops_serving_a_after_logout(e2e, how):
    """AC6: logout ends the session on the server, so nothing of A stays reachable through a
    connection made under that session: no further college call with A's token, no A data."""
    a = await e2e.actor("A", A)
    sa = await a.socket()
    await sa.start()
    await sa.turn(A_SECRET)
    await a.post("/api/logout")
    calls_before = len(e2e.log)
    frames_before = len(sa.frames)

    e2e.model.plan("what is my attendance now", [("get_attendance_summary", {}),
                                                 ("list_my_requests", {})], echo)
    try:
        await sa.turn("what is my attendance now", how)
    except AssertionError:
        pass  # the socket may simply stop answering
    after = e2e.log[calls_before:]
    assert [c.path for c in after if c.login == A] == []
    later = "\n".join(str(f) for f in sa.frames[frames_before:])
    assert hits(later, PRIVATE_A) == []


PRIVATE_A = ["Participating in State Level Technical Symposium", "overall_percentage",
             "overall percentage", "Data Structures and Algorithms"]


@TURN_MODES
async def test_ac6_expired_college_token_is_treated_as_no_session(e2e, how):
    a = await e2e.actor("A", A)
    sa = await a.socket()
    await sa.start()
    e2e.college.expire_all(A)
    e2e.model.plan("my requests please", [("list_my_requests", {})], echo)
    await sa.turn("my requests please", how)
    assert hits(sa.text(), ["Participating in State Level Technical Symposium"]) == []
    assert all(c.login in (None,) for c in e2e.log.by("A") if c.path != "/v1/auth/login"
               and c.status == 200 and c.token)
    assert (await a.get("/api/me")).json()["role"] == "visitor"
    ghost = await a.socket()
    assert await ghost.recv() is None and ghost.closed_code == 4401
