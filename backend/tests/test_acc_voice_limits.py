"""Session-side enforcement of ABU1, ABU2, ABU3 and ABU10 on /ws/voice (limits are read from
app.limits.DEFAULTS, which the tests lower to keep them fast)."""

import asyncio

import pytest
from app import limits

from tests.acc_voice_helpers import (
    Agent,
    close_code,
    events_of,
    make_server,  # noqa: F401
    make_session,
    types,
    wait_for,
)


async def _open(make_server, role="student", agent=None):
    agent = agent or Agent(["Ok."])
    srv = await make_server(agent)
    token, ctx = await make_session(role, None, "Pat") if role != "visitor" else await make_session("visitor")
    return srv, agent, token


# ------------------------------------------------------------------------- ABU1


async def test_abu1_third_socket_of_one_login_is_refused_with_4429_even_across_sessions(make_server):
    srv = await make_server(Agent())
    t1, _ = await make_session("student", "same")
    t2, _ = await make_session("student", "same")
    a, b = await srv.connect(t1), await srv.connect(t2)
    await a.start()
    await b.start()
    c = await srv.connect(t1)
    assert await close_code(c) == 4429
    await a.close()
    await b.close()


async def test_abu1_slot_is_freed_when_a_socket_closes(make_server):
    srv = await make_server(Agent())
    token, _ = await make_session("student", "free1")
    a, b = await srv.connect(token), await srv.connect(token)
    await a.start()
    await b.start()
    await a.close()
    await wait_for(lambda: limits.CONNECTIONS.total == 1, 5, "slot freed")
    async with await srv.connect(token) as c:
        assert (await c.start())["type"] == "ready"
    await b.close()


async def test_abu1_fourth_visitor_socket_from_one_ip_is_refused(make_server):
    srv = await make_server(Agent())
    toks = [(await make_session("visitor"))[0] for _ in range(4)]
    cs = [await srv.connect(t) for t in toks[:3]]
    for c in cs:
        await c.start()
    assert await close_code(await srv.connect(toks[3])) == 4429
    for c in cs:
        await c.send_json({"type": "text", "text": "still fine"})
        await c.until("audio_end")
        await c.close()


async def test_abu1_many_logged_in_users_behind_one_ip_can_all_connect(make_server):
    srv = await make_server(Agent())
    clients = []
    for i in range(6):
        t, _ = await make_session("student", f"lan{i}")
        c = await srv.connect(t)
        assert (await c.start())["type"] == "ready"
        clients.append(c)
    for c in clients:
        await c.close()


async def test_abu1_one_users_limit_does_not_affect_another_user(make_server):
    srv = await make_server(Agent())
    ta, _ = await make_session("student", "lim_a")
    tb, _ = await make_session("student", "lim_b")
    a1, a2 = await srv.connect(ta), await srv.connect(ta)
    await a1.start()
    await a2.start()
    assert await close_code(await srv.connect(ta)) == 4429
    async with await srv.connect(tb) as b:
        assert (await b.start())["type"] == "ready"
    await a1.close()
    await a2.close()


async def test_abu1_aborted_sockets_release_their_slots(make_server):
    srv = await make_server(Agent())
    token, _ = await make_session("student", "abort1")
    for _ in range(2):
        c = await srv.connect(token)
        await c.start()
        c.abort()
    await wait_for(lambda: limits.CONNECTIONS.total == 0, 5, "slots released")
    async with await srv.connect(token) as c:
        assert (await c.start())["type"] == "ready"


async def test_abu1_overall_cap(make_server, monkeypatch):
    monkeypatch.setitem(limits.DEFAULTS, "limit_ws_total", 2)
    srv = await make_server(Agent())
    toks = [(await make_session("student", f"tot{i}"))[0] for i in range(3)]
    a, b = await srv.connect(toks[0]), await srv.connect(toks[1])
    await a.start()
    await b.start()
    assert await close_code(await srv.connect(toks[2])) == 4429
    await a.close()
    await b.close()


async def test_abu1_refused_socket_creates_no_upstream_and_no_conversation(make_server):
    srv = await make_server(Agent())
    t, _ = await make_session("student", "ref1")
    a, b = await srv.connect(t), await srv.connect(t)
    await a.start()
    await b.start()
    n = len(srv.stts)
    assert await close_code(await srv.connect(t)) == 4429
    assert len(srv.stts) == n or all(not s.frames for s in srv.stts[n:])
    await a.close()
    await b.close()


# ------------------------------------------------------------------------- ABU10


async def test_abu10_typed_text_over_1000_chars_is_refused_not_truncated(make_server):
    srv, agent, token = await _open(make_server)
    async with await srv.connect(token) as c:
        await c.start()
        await c.send_json({"type": "text", "text": "a" * 1001})
        msg = await c.recv()
        assert msg["type"] == "error" and "1000" in msg["message"]
        await c.send_json({"type": "text", "text": "ok now"})
        await c.until("audio_end")
    assert agent.texts == ["ok now"]


async def test_abu10_typed_text_of_exactly_1000_chars_is_accepted(make_server):
    srv, agent, token = await _open(make_server)
    async with await srv.connect(token) as c:
        await c.start()
        await c.send_json({"type": "text", "text": "b" * 1000})
        await c.until("audio_end")
    assert agent.texts == ["b" * 1000]


async def test_abu10_control_message_over_8kb_refused(make_server):
    srv, agent, token = await _open(make_server)
    async with await srv.connect(token) as c:
        await c.start()
        await c.send_text('{"type":"text","text":"' + "c" * 9000 + '"}')
        assert (await c.recv())["type"] == "error"
    assert agent.calls == []


async def test_abu10_control_message_of_8kb_minus_something_accepted(make_server):
    srv, agent, token = await _open(make_server)
    async with await srv.connect(token) as c:
        await c.start()
        pad = "e" * 7000
        await c.send_json({"type": "interrupt", "pad": pad})
        got = await c.drain(0.4)
    assert "error" not in types(got)


async def test_abu10_multibyte_text_message_size_counts_bytes(make_server):
    srv, agent, token = await _open(make_server)
    async with await srv.connect(token) as c:
        await c.start()
        # 3000 Tamil characters = 9000 UTF-8 bytes in the JSON body
        await c.send_json({"type": "text", "text": "அ" * 900, "pad": "அ" * 2200})
        assert (await c.recv())["type"] == "error"
    assert agent.calls == []


async def test_abu10_audio_frame_over_32kb_refused_not_relayed(make_server):
    srv, _, token = await _open(make_server)
    async with await srv.connect(token) as c:
        await c.start()
        await c.send_bytes(b"\x00\x01" * 16_385)
        msg = await c.recv()
        assert msg["type"] == "error"
        await c.send_bytes(b"\x00\x01" * 16_384)
        await c.drain(0.3)
    assert [len(f) for s in srv.stts for f in s.frames] == [32768]


# ------------------------------------------------------------------------- ABU2


async def test_abu2_visitor_turn_cap_per_ip_gives_polite_error_and_no_model_call(make_server, monkeypatch):
    monkeypatch.setitem(limits.DEFAULTS, "limit_visitor_turns_per_ip_per_day", 2)
    srv, agent, _ = await _open(make_server, "visitor")
    tok1 = (await make_session("visitor"))[0]
    async with await srv.connect(tok1) as c:
        await c.start()
        for i in range(2):
            await c.send_json({"type": "text", "text": f"q{i}"})
            await c.until("audio_end")
        await c.send_json({"type": "text", "text": "q2"})
        msg = await c.until("error")
        assert msg[-1]["message"] and "Traceback" not in msg[-1]["message"]
    # clearing cookies (a brand-new session) does not reset the count
    tok2 = (await make_session("visitor"))[0]
    async with await srv.connect(tok2) as c:
        await c.start()
        await c.send_json({"type": "text", "text": "q3"})
        await c.until("error")
    assert agent.texts == ["q0", "q1"]


async def test_abu2_user_cap_counts_per_login_across_sessions(make_server, monkeypatch):
    monkeypatch.setitem(limits.DEFAULTS, "limit_user_turns_per_day", 2)
    srv = await make_server(Agent(["Ok."]))
    t1, _ = await make_session("student", "cap1")
    t2, _ = await make_session("student", "cap1")
    other, _ = await make_session("student", "cap2")
    async with await srv.connect(t1) as a:
        await a.start()
        await a.send_json({"type": "text", "text": "one"})
        await a.until("audio_end")
    async with await srv.connect(t2) as b:
        await b.start()
        await b.send_json({"type": "text", "text": "two"})
        await b.until("audio_end")
        await b.send_json({"type": "text", "text": "three"})
        await b.until("error")
        # the socket remains usable for non-turn messages
        await b.send_json({"type": "interrupt"})
    async with await srv.connect(other) as c:
        await c.start()
        await c.send_json({"type": "text", "text": "other user"})
        await c.until("audio_end")


async def test_abu7_conversation_turn_cap_asks_for_a_new_conversation(make_server, monkeypatch):
    monkeypatch.setitem(limits.DEFAULTS, "limit_turns_per_conversation", 2)
    srv, agent, token = await _open(make_server)
    async with await srv.connect(token) as c:
        await c.start()
        for i in range(2):
            await c.send_json({"type": "text", "text": f"t{i}"})
            await c.until("audio_end")
        await c.send_json({"type": "text", "text": "t2"})
        msg = (await c.until("error"))[-1]
        assert "new conversation" in msg["message"].lower()
        # a fresh conversation works again
        await c.send_json({"type": "start", "conversation_id": None})
        await c.until("ready")
        await c.send_json({"type": "text", "text": "t3"})
        await c.until("audio_end")
    assert agent.texts == ["t0", "t1", "t3"]


async def test_abu8_global_budget_gives_busy_message_to_visitors_only(make_server, monkeypatch):
    monkeypatch.setitem(limits.DEFAULTS, "limit_daily_turns", 1)
    srv = await make_server(Agent(["Ok."]))
    v1, _ = await make_session("visitor")
    v2, _ = await make_session("visitor")
    stu, _ = await make_session("student", "bud1")
    async with await srv.connect(v1) as c:
        await c.start()
        await c.send_json({"type": "text", "text": "one"})
        await c.until("audio_end")
    async with await srv.connect(v2) as c:
        await c.start()
        await c.send_json({"type": "text", "text": "two"})
        assert "busy" in (await c.until("error"))[-1]["message"].lower()
    async with await srv.connect(stu) as c:
        await c.start()
        await c.send_json({"type": "text", "text": "student still works"})
        await c.until("audio_end")


# ------------------------------------------------------------------------- ABU3


async def test_abu3_idle_socket_is_closed_with_an_explanation(make_server, monkeypatch):
    monkeypatch.setitem(limits.DEFAULTS, "limit_idle_seconds", 0.5)
    srv, _, token = await _open(make_server)
    c = await srv.connect(token)
    await c.start()
    msg = await c.until("error", timeout=6)
    assert msg[-1]["message"]
    assert await close_code(c, 3) == 1000
    await wait_for(lambda: srv.stts[0].closed and srv.ttss[0].closed, 5, "upstreams closed")


async def test_abu3_activity_keeps_the_socket_alive(make_server, monkeypatch):
    monkeypatch.setitem(limits.DEFAULTS, "limit_idle_seconds", 3)
    srv, _, token = await _open(make_server)
    async with await srv.connect(token) as c:
        await c.start()
        for i in range(4):
            await asyncio.sleep(1.0)
            await c.send_json({"type": "text", "text": f"alive{i}"})
            await c.until("audio_end")


async def test_abu3_visitor_socket_lifetime(make_server, monkeypatch):
    monkeypatch.setitem(limits.DEFAULTS, "limit_visitor_socket_seconds", 1)
    srv, _, token = await _open(make_server, "visitor")
    c = await srv.connect(token)
    await c.start()
    msg = await c.until("error", timeout=6)
    assert "time limit" in msg[-1]["message"].lower()
    assert await close_code(c, 3) == 1000


async def test_abu3_user_socket_outlives_the_visitor_limit(make_server, monkeypatch):
    monkeypatch.setitem(limits.DEFAULTS, "limit_visitor_socket_seconds", 1)
    srv, _, token = await _open(make_server, "student")
    async with await srv.connect(token) as c:
        await c.start()
        await asyncio.sleep(2.5)
        await c.send_json({"type": "text", "text": "still here"})
        await c.until("audio_end")


async def test_abu3_utterance_over_the_cap_is_ended_and_becomes_a_turn(make_server, monkeypatch):
    monkeypatch.setitem(limits.DEFAULTS, "limit_utterance_seconds", 0.5)  # 16000 bytes
    srv, agent, token = await _open(make_server)
    async with await srv.connect(token) as c:
        await c.start()
        await c.mic(b"S")
        await c.mic(b"Ia very long monologue indeed")
        assert (await c.recv())["type"] == "transcript"  # the interim has been processed
        for _ in range(10):
            await c.send_bytes(b"\x00\x01" * 1600)
        frames = await c.until("audio_end")
    assert {"type": "transcript", "final": True, "text": "a very long monologue indeed"} in frames
    assert agent.texts == ["a very long monologue indeed"]


async def test_abu3_late_events_of_a_capped_utterance_are_not_a_second_turn(make_server, monkeypatch):
    monkeypatch.setitem(limits.DEFAULTS, "limit_utterance_seconds", 0.5)
    srv, agent, token = await _open(make_server)
    async with await srv.connect(token) as c:
        await c.start()
        await c.mic(b"S")
        await c.mic(b"Ia very long monologue indeed")
        assert (await c.recv())["type"] == "transcript"  # the interim has been processed
        for _ in range(10):
            await c.send_bytes(b"\x00\x01" * 1600)
        await c.until("audio_end")
        from app.voice.deepgram import SttEvent

        srv.stts[0].push(SttEvent("interim", "a very long monologue indeed and more"))
        srv.stts[0].push(SttEvent("utterance", "a very long monologue indeed and more"))
        await c.drain(0.6)
    assert agent.texts == ["a very long monologue indeed"]
