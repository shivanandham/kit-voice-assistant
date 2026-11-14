"""AC3, AC1 and cleanup: sockets of different users or conversations never share anything."""

import asyncio

import pytest

from tests.acc_voice_helpers import (
    Agent,
    FakeTts,
    audio_bytes,
    events_of,
    history_rows,
    make_server,  # noqa: F401
    relaxed_limits,  # noqa: F401
    make_session,
    settle_tasks,
    types,
    wait_for,
)


def reply_agent(delay: float = 0.0) -> Agent:
    def script(t):
        body = f"Reply to {t}. "
        return [("sleep", delay), {"type": "tool", "name": f"tool-for-{t}", "state": "start"},
                ("sleep", delay), {"type": "text_delta", "text": body}, ("sleep", delay),
                {"type": "text_delta", "text": f"More for {t}."}, {"type": "done", "text": body + f"More for {t}."}]

    return Agent(script=script)


def blob(frames) -> str:
    return str([f for f in frames if isinstance(f, dict)]) + audio_bytes(frames).decode(errors="ignore")


async def _two(make_server, agent, roles=("student", "student"), tts=None):
    srv = await make_server(agent, tts=(tts if tts else None))
    ta, ca = await make_session(roles[0], "isoA" if roles[0] != "visitor" else None, "Alice")
    tb, cb = await make_session(roles[1], "isoB" if roles[1] != "visitor" else None, "Bob")
    a, b = await srv.connect(ta), await srv.connect(tb)
    ra, rb = await a.start(), await b.start()
    return srv, a, b, ra, rb, ca, cb


@pytest.mark.parametrize("roles", [("student", "student"), ("student", "teacher"), ("visitor", "student"), ("visitor", "visitor"), ("teacher", "teacher")])
async def test_ac3_two_users_at_once_never_see_each_others_events_or_audio(make_server, roles):
    agent = reply_agent(0.01)
    srv, a, b, ra, rb, ca, cb = await _two(make_server, agent, roles)
    async with a, b:
        assert ra["conversation_id"] != rb["conversation_id"]

        async def user(c, tag):
            await c.send_json({"type": "text", "text": f"typedMARK{tag}"})
            await c.until("audio_end")
            await c.mic(f"Ispokenpart{tag} words here".encode())
            await c.mic(f"Uvoiceutter{tag} spoken words".encode())
            await c.until("audio_end")
            await c.drain(0.5)

        await asyncio.gather(user(a, "AAAA"), user(b, "BBBB"))
    assert "BBBB" not in blob(a.frames) and "bbbb" not in blob(a.frames).lower().replace("bbbbb", "")
    assert "AAAA" not in blob(b.frames)
    assert "Bob" not in blob(a.frames) and "Alice" not in blob(b.frames)
    for ctx, cid, text in agent.calls:
        who = ca if "AAAA" in text else cb
        mine = ra if "AAAA" in text else rb
        assert ctx.owner_key == who.owner_key and cid == mine["conversation_id"]
    # each socket got its own upstream speech clients
    assert len({id(s) for s in srv.stts}) == 2 and len({id(t) for t in srv.ttss}) == 2


async def test_ac3_history_of_each_user_holds_only_their_messages(make_server):
    agent = reply_agent()
    srv, a, b, ra, rb, *_ = await _two(make_server, agent)
    async with a, b:
        await asyncio.gather(
            a.send_json({"type": "text", "text": "alice question"}),
            b.send_json({"type": "text", "text": "bob question"}),
        )
        await a.until("audio_end")
        await b.until("audio_end")
    assert all("bob" not in t for _, t in await history_rows(ra["conversation_id"]))
    assert all("alice" not in t for _, t in await history_rows(rb["conversation_id"]))


async def test_ac3_stt_events_of_one_socket_never_reach_the_other(make_server):
    srv, a, b, *_ = await _two(make_server, Agent())
    async with a, b:
        await a.mic(b"S")
        await a.mic(b"Isecret words alice")
        await b.mic(b"S")
        got_a = await a.drain(0.4)
        got_b = await b.drain(0.4)
        await a.mic(b"Ualice said something private")
        await a.until("audio_end")
        got_b += await b.drain(0.5)
    assert "alice" not in blob(got_b).lower()
    assert "transcript" not in types(got_b)


async def test_ac3_barge_in_and_interrupt_of_one_user_do_not_touch_the_other(make_server):
    agent = Agent(["Slow words. "] * 25, delay=0.05)
    srv, a, b, *_ = await _two(make_server, agent)
    async with a, b:
        await a.send_json({"type": "text", "text": "alice long"})
        await b.send_json({"type": "text", "text": "bob long"})
        await a.until("assistant_delta")
        await b.until("assistant_delta")
        await a.send_json({"type": "interrupt"})
        await a.until("interrupted")
        await a.mic(b"Iinterrupt me with alice words")
        got_b = await b.until("assistant_done")
        got_b += await b.until("audio_end")
    assert "interrupted" not in types(got_b)
    assert "bob long" in agent.completed and "alice long" in agent.stopped_early


async def test_ac3_failure_on_one_socket_does_not_error_the_other(make_server):
    agent = Agent(script=lambda t: [("raise", RuntimeError("x"))] if t == "bad" else [{"type": "text_delta", "text": "Fine. "}, {"type": "done", "text": "Fine. "}])
    srv, a, b, *_ = await _two(make_server, agent)
    async with a, b:
        await a.send_json({"type": "text", "text": "bad"})
        await a.until("error")
        await b.send_json({"type": "text", "text": "good"})
        got = await b.until("audio_end") + await b.drain(0.3)
    assert "error" not in types(got)


async def test_ac3_same_user_two_sockets_two_conversations_stay_separate(make_server):
    agent = reply_agent(0.01)
    srv = await make_server(agent)
    token, ctx = await make_session("student", "twin", "Twin")
    a, b = await srv.connect(token), await srv.connect(token)
    async with a, b:
        ra, rb = await a.start(), await b.start()
        assert ra["conversation_id"] != rb["conversation_id"]
        await asyncio.gather(a.send_json({"type": "text", "text": "firstsocketMARK"}), b.send_json({"type": "text", "text": "secondsocketMARK"}))
        fa = await a.until("audio_end") + await a.drain(0.3)
        fb = await b.until("audio_end") + await b.drain(0.3)
    assert "secondsocketMARK" not in blob(fa) and "firstsocketMARK" not in blob(fb)
    assert [t for r, t in await history_rows(ra["conversation_id"]) if r == "user"] == ["firstsocketMARK"]
    assert [t for r, t in await history_rows(rb["conversation_id"]) if r == "user"] == ["secondsocketMARK"]
    assert srv.stts[0] is not srv.stts[1]


async def test_ac3_same_user_voice_on_one_socket_not_heard_on_the_other(make_server):
    srv = await make_server(Agent())
    token, _ = await make_session("student", "twin2")
    a, b = await srv.connect(token), await srv.connect(token)
    async with a, b:
        await a.start()
        await b.start()
        await a.mic(b"Ionly socket a hears this")
        await a.mic(b"Uonly socket a hears this")
        await a.until("audio_end")
        got = await b.drain(0.5)
    assert got == []


async def test_ac3_same_user_two_sockets_same_conversation_deliver_only_to_originating_socket(make_server):
    agent = reply_agent()
    srv = await make_server(agent)
    token, _ = await make_session("student", "twin3")
    a, b = await srv.connect(token), await srv.connect(token)
    async with a, b:
        cid = (await a.start())["conversation_id"]
        await b.start(cid)
        await a.send_json({"type": "text", "text": "onlyForA"})
        await a.until("audio_end")
        got_b = await b.drain(0.5)
    assert got_b == []


async def test_ac3_visitor_socket_receives_nothing_of_a_students_turn(make_server):
    agent = reply_agent(0.01)
    srv, a, b, ra, rb, *_ = await _two(make_server, agent, ("student", "visitor"))
    async with a, b:
        await a.send_json({"type": "text", "text": "my attendance marker STUDENTONLY"})
        await a.until("audio_end")
        got = await b.drain(0.6)
    assert got == []


async def test_ac3_many_users_concurrently_no_cross_talk(make_server, relaxed_limits):
    agent = reply_agent(0.005)
    srv = await make_server(agent)
    users = []
    for i in range(6):
        t, ctx = await make_session("student", f"mass{i}", f"User{i}")
        c = await srv.connect(t)
        users.append((i, c, await c.start(), ctx))

    async def go(i, c):
        for k in range(3):
            await c.send_json({"type": "text", "text": f"uniq{i}x{k}"})
            await c.until("audio_end")
        await c.drain(0.3)

    try:
        await asyncio.gather(*(go(i, c) for i, c, _, _ in users))
    finally:
        for _, c, _, _ in users:
            await c.close()
    for i, c, ready, ctx in users:
        text = blob(c.frames)
        for j in range(6):
            if j != i:
                assert f"uniq{j}x" not in text, f"user {i} saw user {j}"
        assert all(f"uniq{i}x{k}" in text for k in range(3))
    for ctx, cid, text in agent.calls:
        i = int(text[4])
        assert ctx.owner_key == f"user:mass{i}" and cid == users[i][2]["conversation_id"]


# ------------------------------------------------------------------------- cleanup


async def test_cleanup_disconnecting_one_closes_only_its_own_upstreams_and_tasks(make_server):
    agent = Agent(["Slow words. "] * 30, delay=0.05)
    srv, a, b, *_ = await _two(make_server, agent)
    await a.send_json({"type": "text", "text": "alice turn"})
    await b.send_json({"type": "text", "text": "bob turn"})
    await a.until("assistant_delta")
    await b.until("assistant_delta")
    await a.close()
    await wait_for(lambda: srv.stts[0].closed and srv.ttss[0].closed, 5, "alice upstreams closed")
    assert not srv.stts[1].closed and not srv.ttss[1].closed
    await wait_for(lambda: "alice turn" in agent.stopped_early, 5, "alice turn cancelled")
    got = await b.until("assistant_done")
    got += await b.until("audio_end")
    assert "bob turn" in agent.completed
    await b.close()
    await wait_for(lambda: srv.stts[1].closed and srv.ttss[1].closed, 5, "bob upstreams closed")


@pytest.mark.parametrize("how", ["graceful", "abort"])
@pytest.mark.parametrize("stage", ["generating", "synthesising", "sending", "idle"])
async def test_cleanup_disconnect_mid_turn_cancels_turn_and_leaks_no_task(make_server, how, stage):
    if stage == "synthesising":
        tts = FakeTts(delay=lambda t: 5.0)
        agent = Agent(["A sentence to speak."])
    elif stage == "sending":
        tts = FakeTts(chunks=200, chunk_bytes=960, chunk_delay=0.02)
        agent = Agent(["A sentence to speak."])
    else:
        tts = FakeTts()
        agent = Agent(["Slow words. "] * 40, delay=0.05)
    srv = await make_server(agent, tts=lambda: tts)
    token, _ = await make_session("student", None, "Pat")
    await asyncio.sleep(0.1)
    before = set(asyncio.all_tasks())
    c = await srv.connect(token)
    await c.start()
    if stage == "idle":
        await c.mic(b"Iwhat is the")  # starts the STT pump only
        await c.recv()
    else:
        await c.send_json({"type": "text", "text": "start a turn"})
        if stage == "generating":
            await c.until("assistant_delta")
        elif stage == "synthesising":
            await c.until("assistant_done")
        else:
            while not any(isinstance(f, bytes) for f in c.frames):
                await c.recv()
    if how == "abort":
        c.abort()
    else:
        await c.close()
    await wait_for(lambda: srv.stts[0].closed and tts.closed, 5, "upstreams closed")
    if stage != "idle":
        await wait_for(lambda: agent.active == 0, 5, "turn cancelled")
        if stage == "generating":
            assert "start a turn" in agent.stopped_early
    leftover = await settle_tasks(before, 6)
    assert not leftover, f"leaked tasks: {leftover}"


async def test_cleanup_after_disconnect_no_more_tts_requests_or_agent_calls(make_server):
    tts = FakeTts(delay=lambda t: 0.01)
    agent = Agent(["One sentence. ", "Two sentence. ", "Three sentence. "] * 4, delay=0.1)
    srv = await make_server(agent, tts=lambda: tts)
    token, _ = await make_session("student", None, "Pat")
    c = await srv.connect(token)
    await c.start()
    await c.send_json({"type": "text", "text": "go"})
    await c.until("assistant_delta")
    c.abort()
    await wait_for(lambda: agent.active == 0, 5, "stop")
    seen = (len(agent.calls), len(tts.requested))
    await asyncio.sleep(0.6)
    assert (len(agent.calls), len(tts.requested)) == seen


async def test_cleanup_many_connect_disconnect_cycles_leave_nothing_behind(make_server, relaxed_limits):
    agent = Agent(["Slow words. "] * 30, delay=0.03)
    srv = await make_server(agent)
    token, _ = await make_session("student", None, "Pat")
    await asyncio.sleep(0.1)
    before = set(asyncio.all_tasks())
    for i in range(12):
        c = await srv.connect(token)
        await c.start()
        await c.send_json({"type": "text", "text": f"cycle{i}"})
        await c.until("assistant_delta")
        if i % 2:
            c.abort()
        else:
            await c.close()
    await wait_for(lambda: all(s.closed and t.closed for s, t in zip(srv.stts, srv.ttss)) and len(srv.stts) == 12, 8, "all upstreams closed")
    await wait_for(lambda: agent.active == 0, 5, "turns stopped")
    leftover = await settle_tasks(before, 6)
    assert not leftover, f"leaked tasks: {leftover}"


async def test_cleanup_disconnect_before_start_closes_upstreams(make_server):
    srv = await make_server(Agent())
    token, _ = await make_session("student", None, "Pat")
    await asyncio.sleep(0.1)
    before = set(asyncio.all_tasks())
    c = await srv.connect(token)
    await c.close()
    await wait_for(lambda: srv.stts and srv.stts[0].closed and srv.ttss[0].closed, 5, "closed")
    assert not await settle_tasks(before, 5)


async def test_cleanup_rejected_connections_create_no_upstreams_and_leak_nothing(make_server, relaxed_limits):
    srv = await make_server(Agent())
    await asyncio.sleep(0.1)
    before = set(asyncio.all_tasks())
    for _ in range(5):
        c = await srv.connect(cookie="nonsense")
        await c.close()
    assert srv.stts == [] and srv.ttss == []
    assert not await settle_tasks(before, 5)


async def test_cleanup_disconnect_during_stt_utterance_closes_stt(make_server):
    srv = await make_server(Agent())
    token, _ = await make_session("student", None, "Pat")
    c = await srv.connect(token)
    await c.start()
    await c.mic(b"S")
    await c.mic(b"Iwhat is the placement")
    await c.recv()
    c.abort()
    await wait_for(lambda: srv.stts[0].closed, 5, "stt closed")
