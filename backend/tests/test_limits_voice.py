"""Voice limits over a real local server: ABU1 sockets, ABU2/ABU7/ABU8 turns, ABU3 audio."""

import asyncio
import uuid

import pytest
import websockets
from app import history, limits
from app.db import pool
from app.voice import session as voice
from websockets.exceptions import ConnectionClosed

from tests import acc_voice_helpers
from tests.acc_voice_helpers import (
    ORIGIN,
    Agent,
    Client,
    close_code,
    make_session,
    wait_for,
)

make_server = acc_voice_helpers.make_server  # the real-server fixture
IP = "198.51.100.1"


@pytest.fixture
def clock(monkeypatch):
    t = [1000.0]
    monkeypatch.setattr(limits, "_clock", lambda: t[0])
    return t


def quick_agent() -> Agent:
    return Agent(deltas=["Okay."], tool=False, save=False)


async def connect(server, token: str, ip: str = IP) -> Client:
    """Connect through the 'proxy': the server sees 127.0.0.1 plus X-Forwarded-For."""
    ws = await websockets.connect(
        server.url,
        additional_headers={"Cookie": f"kit_session={token}", "X-Forwarded-For": ip},
        origin=ORIGIN,
        max_size=None,
        ping_interval=None,
        open_timeout=5,
    )
    return Client(ws)


async def turn(c: Client, text: str = "hello") -> dict:
    """Send typed text; return the final event (audio_end, assistant_done or error)."""
    await c.send_json({"type": "text", "text": text})
    while True:
        item = await c.recv(5)
        if isinstance(item, dict) and item["type"] in ("audio_end", "error"):
            return item


async def started(server, token: str, ip: str = IP) -> Client:
    c = await connect(server, token, ip)
    await c.start()
    return c


async def next_event(c: Client) -> dict:
    """The next JSON event, skipping audio and `interrupted` (a reply may still be playing)."""
    while True:
        item = await c.recv()
        if isinstance(item, dict) and item["type"] != "interrupted":
            return item


async def is_open(c: Client) -> bool:
    try:
        await c.recv(0.15)
    except TimeoutError:
        return True
    except ConnectionClosed:
        return False
    return True


# --------------------------------------------------------------------------- ABU1


async def test_abu1_visitor_ip_holds_three_sockets(make_server):
    server = await make_server(quick_agent())
    tokens = [(await make_session())[0] for _ in range(5)]
    open_ = [await started(server, t) for t in tokens[:3]]
    fourth = await connect(server, tokens[3])
    assert await close_code(fourth) == 4429
    other_ip = await started(server, tokens[4], ip="198.51.100.2")  # other IPs unaffected
    assert await turn(other_ip) == {"type": "audio_end"}
    await open_[0].close()
    await wait_for(lambda: limits.CONNECTIONS.total == 3, what="release")
    again = await started(server, tokens[3])
    for c in [*open_[1:], other_ip, again]:
        await c.close()


async def test_abu1_logged_in_user_two_sockets_per_login_never_per_ip(make_server):
    server = await make_server(quick_agent())
    phone, _ = await make_session("student", login="arjun")
    laptop, _ = await make_session("student", login="arjun")
    tablet, _ = await make_session("student", login="arjun")
    a = await started(server, phone)
    b = await started(server, laptop, ip="198.51.100.7")
    third = await connect(server, tablet, ip="198.51.100.8")  # a new IP does not help
    assert await close_code(third) == 4429
    visitors = [await started(server, (await make_session())[0]) for _ in range(3)]
    priya, _ = await make_session("student", login="priya")
    p = await started(server, priya)  # same IP as three visitors and arjun: still allowed
    assert await turn(p) == {"type": "audio_end"}
    for c in [a, b, p, *visitors]:
        await c.close()


async def test_abu1_forty_sockets_overall(make_server):
    server = await make_server(quick_agent())
    tokens = [(await make_session("student", login=f"s{i:02d}"))[0] for i in range(41)]
    open_ = [await started(server, t) for t in tokens[:40]]
    extra = await connect(server, tokens[40])
    assert await close_code(extra) == 4429
    await open_[0].close()
    await wait_for(lambda: limits.CONNECTIONS.total == 39, what="release")
    late = await started(server, tokens[40])
    for c in [*open_[1:], late]:
        await c.close()


async def test_abu1_abu2_thirty_students_behind_one_ip(make_server, clock):
    agent = quick_agent()
    server = await make_server(agent)
    students = []
    for i in range(30):
        token, _ = await make_session("student", login=f"lab{i:02d}")
        c = await started(server, token)  # all from IP
        assert await turn(c) == {"type": "audio_end"}
        students.append(c)
    assert len(agent.calls) == 30
    # one student uses up the daily allowance; the others are unaffected
    for _ in range(99):
        limits.TURNS_USER.hit("login:lab00")
    assert (await turn(students[0]))["type"] == "error"
    assert await turn(students[1]) == {"type": "audio_end"}
    for c in students:
        await c.close()


async def test_abu1_abu2_visitor_and_students_on_one_ip_do_not_share(make_server, clock):
    agent = quick_agent()
    server = await make_server(agent)
    student_token, _ = await make_session("student", login="hostel1")
    student = await started(server, student_token)
    for _ in range(25):  # more than a visitor's allowance, on the same IP
        assert await turn(student) == {"type": "audio_end"}
    visitor_token, _ = await make_session()
    visitor = await started(server, visitor_token)
    for _ in range(20):  # the students' turns did not consume the visitor allowance
        assert await turn(visitor) == {"type": "audio_end"}
    assert (await turn(visitor))["type"] == "error"  # visitors are still bounded
    assert await turn(student) == {"type": "audio_end"}  # and that does not touch the student
    for c in (student, visitor):
        await c.close()


# --------------------------------------------------------------------------- ABU2


async def test_abu2_visitor_turns_per_day_per_ip(make_server, clock):
    agent = quick_agent()
    server = await make_server(agent)
    token, _ = await make_session()
    c = await started(server, token)
    for _ in range(20):
        assert await turn(c) == {"type": "audio_end"}
    refused = await turn(c)
    assert refused["type"] == "error" and "today's limit" in refused["message"].lower()
    assert len(agent.calls) == 20 and len(server.ttss[0].requested) == 20
    assert await is_open(c)  # the socket stays open
    clock[0] += 86401
    assert await turn(c) == {"type": "audio_end"}
    await c.close()


async def test_abu2_clearing_cookies_does_not_reset_visitor_allowance(make_server, clock):
    agent = quick_agent()
    server = await make_server(agent)
    for n in (12, 8):  # two "cleared cookie" visitor sessions from one IP share 20 turns
        token, _ = await make_session()
        c = await started(server, token)
        for _ in range(n):
            assert await turn(c) == {"type": "audio_end"}
        await c.close()
    token, _ = await make_session()
    c = await started(server, token)
    assert (await turn(c))["type"] == "error"
    assert len(agent.calls) == 20
    other_token, _ = await make_session()
    other = await started(server, other_token, ip="198.51.100.2")
    assert await turn(other) == {"type": "audio_end"}  # other IPs are unaffected
    user_token, _ = await make_session("student")
    student = await started(server, user_token)  # logged-in users have their own allowance
    assert await turn(student) == {"type": "audio_end"}
    for x in (c, other, student):
        await x.close()


async def test_abu2_logged_in_turns_per_day_per_login_across_sessions(make_server, clock):
    agent = quick_agent()
    server = await make_server(agent)
    phone, _ = await make_session("student", login="arjun")
    laptop, _ = await make_session("student", login="arjun")
    a = await started(server, phone)
    b = await started(server, laptop, ip="198.51.100.3")  # another device and network
    for _ in range(60):
        assert await turn(a) == {"type": "audio_end"}
    for _ in range(40):
        assert await turn(b) == {"type": "audio_end"}
    assert (await turn(a))["type"] == "error"
    assert (await turn(b))["type"] == "error"
    assert len(agent.calls) == 100
    priya, _ = await make_session("student", login="priya")
    p = await started(server, priya)
    assert await turn(p) == {"type": "audio_end"}  # another user is unaffected
    clock[0] += 86401
    assert await turn(a) == {"type": "audio_end"}
    for x in (a, b, p):
        await x.close()


async def test_abu2_refused_utterance_is_not_a_turn(make_server, clock):
    agent = quick_agent()
    server = await make_server(agent)
    token, _ = await make_session()
    c = await started(server, token)
    for _ in range(20):
        await turn(c)
    await c.mic(b"Uone more question please")
    while True:
        item = await c.recv(5)
        if isinstance(item, dict) and item["type"] == "error":
            break
    assert len(agent.calls) == 20
    await c.close()


# --------------------------------------------------------------------------- ABU7


async def test_abu7_conversation_turn_cap(make_server, clock):
    agent = Agent(deltas=["Okay."], tool=False, save=True)
    server = await make_server(agent)
    token, ctx = await make_session("student")
    cid = await history.create_conversation(ctx)
    await pool().executemany(
        "INSERT INTO messages (conversation_id, role, content) VALUES ($1, 'user', $2)",
        [(uuid.UUID(cid), f"q{i}") for i in range(199)],
    )
    c = await connect(server, token)
    await c.start(cid)
    assert await turn(c) == {"type": "audio_end"}  # turn 200
    refused = await turn(c)
    assert refused["type"] == "error" and "new conversation" in refused["message"].lower()
    assert len(agent.calls) == 1
    await c.send_json({"type": "start", "conversation_id": None})
    await c.recv()
    assert await turn(c) == {"type": "audio_end"}  # a new conversation works
    await c.close()


# --------------------------------------------------------------------------- ABU8


def spend_global(n: int) -> None:
    for _ in range(n):
        limits.TURNS_GLOBAL.hit("all")


async def test_abu8_global_budget_visitors_then_users(make_server, clock):
    agent = quick_agent()
    server = await make_server(agent)
    vt, _ = await make_session()
    ut, _ = await make_session("teacher")
    visitor = await started(server, vt)
    user = await started(server, ut, ip="198.51.100.2")
    spend_global(4999)
    assert await turn(visitor) == {"type": "audio_end"}  # turn 5000
    busy = await turn(visitor)
    assert busy["type"] == "error" and "busy" in busy["message"].lower()
    assert "today's limit" in busy["message"].lower()  # same style as ABU2
    assert await turn(user) == {"type": "audio_end"}  # users keep working
    spend_global(8000 - 5001 - 1)
    assert await turn(user) == {"type": "audio_end"}  # turn 8000
    assert (await turn(user))["type"] == "error"
    assert len(agent.calls) == 3
    clock[0] += 86401
    assert await turn(visitor) == {"type": "audio_end"}
    for x in (visitor, user):
        await x.close()


# --------------------------------------------------------------------------- ABU3


async def test_abu3_utterance_capped_at_30_seconds(make_server):
    agent = quick_agent()
    server = await make_server(agent)
    token, _ = await make_session()
    c = await started(server, token)
    await c.mic(b"S")
    await c.mic(b"Ihello there")
    assert (await c.recv())["text"] == "hello there"
    second = b"\0" * 32000
    for _ in range(29):
        await c.send_bytes(second)
    await asyncio.sleep(0.2)
    assert agent.calls == []
    await c.send_bytes(second)  # 30 s of speech: the utterance is ended here
    final = await c.recv()
    assert final == {"type": "transcript", "final": True, "text": "hello there"}
    await wait_for(lambda: len(agent.calls) == 1, what="turn")
    await c.recv()  # audio_start etc. follow
    # the rest of the capped utterance is dropped; the next one is a normal turn
    await c.mic(b"Uhello there and on and on")
    await c.mic(b"Unext question")
    await wait_for(lambda: len(agent.calls) == 2, what="next turn")
    assert agent.texts == ["hello there", "next question"]
    await c.close()


async def test_abu3_idle_socket_closed_after_3_minutes(make_server, clock, monkeypatch):
    monkeypatch.setattr(voice, "WATCHDOG_INTERVAL_S", 0.02)
    server = await make_server(quick_agent())
    token, _ = await make_session()
    c = await started(server, token)
    clock[0] += 170
    assert await turn(c) == {"type": "audio_end"}  # typed text is activity
    clock[0] += 179
    assert await is_open(c)
    clock[0] += 2
    err = await next_event(c)
    assert err["type"] == "error" and "few minutes" in err["message"].lower()
    assert await close_code(c) == 1000


async def test_abu3_speech_counts_as_activity(make_server, clock, monkeypatch):
    monkeypatch.setattr(voice, "WATCHDOG_INTERVAL_S", 0.02)
    server = await make_server(quick_agent())
    token, _ = await make_session()
    c = await started(server, token)
    clock[0] += 170
    await c.mic(b"Iam I")
    await c.recv()
    clock[0] += 170
    assert await is_open(c)
    await c.close()


async def test_abu3_visitor_socket_lifetime_and_resume(make_server, clock, monkeypatch):
    monkeypatch.setattr(voice, "WATCHDOG_INTERVAL_S", 0.02)
    server = await make_server(quick_agent())
    token, _ = await make_session()
    c = await connect(server, token)
    cid = (await c.start())["conversation_id"]
    for _ in range(8):
        clock[0] += 112
        assert await turn(c) == {"type": "audio_end"}
    clock[0] += 3  # 899 s
    assert await is_open(c)
    clock[0] += 2
    err = await next_event(c)
    assert err["type"] == "error" and "reconnect" in err["message"].lower()
    assert await close_code(c) == 1000
    again = await connect(server, token)
    assert (await again.start(cid))["conversation_id"] == cid
    await again.close()


async def test_abu3_logged_in_socket_lifetime(make_server, clock, monkeypatch):
    monkeypatch.setattr(voice, "WATCHDOG_INTERVAL_S", 0.02)
    server = await make_server(quick_agent())
    token, _ = await make_session("student")
    c = await started(server, token)
    for _ in range(35):
        clock[0] += 100
        assert await turn(c) == {"type": "audio_end"}
    clock[0] += 99  # 3599 s
    assert await is_open(c)
    clock[0] += 2
    assert (await next_event(c))["type"] == "error"
    assert await close_code(c) == 1000


# --------------------------------------------------------------------------- ABU10


async def test_abu10_typed_text_at_most_1000_characters(make_server, clock):
    agent = quick_agent()
    server = await make_server(agent)
    token, _ = await make_session()
    c = await started(server, token)
    assert await turn(c, "a" * 1000) == {"type": "audio_end"}
    refused = await turn(c, "b" * 1001)
    assert refused["type"] == "error" and "1000" in refused["message"]
    assert agent.texts == ["a" * 1000]
    assert len(server.ttss[0].requested) == 1
    assert await is_open(c)
    await c.close()


async def test_abu10_audio_frame_at_most_32_kb(make_server):
    server = await make_server(quick_agent())
    token, _ = await make_session()
    c = await started(server, token)
    await c.send_bytes(b"\0" * 32 * 1024)
    await c.send_bytes(b"\0" * (32 * 1024 + 2))
    err = await c.recv()
    assert err["type"] == "error" and "audio" in err["message"].lower()
    await wait_for(lambda: len(server.stts[0].frames) == 1, what="frame")
    assert len(server.stts[0].frames[0]) == 32 * 1024
    await c.close()


async def test_abu10_control_message_at_most_8_kb(make_server):
    agent = quick_agent()
    server = await make_server(agent)
    token, _ = await make_session()
    c = await started(server, token)
    await c.send_text('{"type":"text","text":"' + "x" * (8 * 1024) + '"}')
    assert (await c.recv())["type"] == "error"
    assert agent.calls == []
    await c.close()
