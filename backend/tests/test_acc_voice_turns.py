"""VOX3, VOX5, HIS1: turn flow, ordering of audio, one turn at a time, history order."""

import asyncio
import re

import pytest
from app.db import pool

from tests.acc_voice_helpers import (
    Agent,
    FakeTts,
    audio_bytes,
    check_audio_protocol,
    events_of,
    history_rows,
    make_server,  # noqa: F401
    make_session,
    types,
    wait_for,
)


async def _open(make_server, agent=None, tts=None, role="student", name="Pat Student"):
    agent = agent or Agent()
    srv = await make_server(agent, tts=(lambda: tts) if tts else None)
    token, ctx = await make_session(role, None, name) if role != "visitor" else await make_session("visitor")
    c = await srv.connect(token)
    ready = await c.start()
    return srv, agent, c, ready


def _spoken(frames) -> str:
    return re.sub(r"\s+", "", audio_bytes(frames).decode())


# ------------------------------------------------------------------------- VOX3: audio framing


async def test_vox3_one_audio_pair_per_turn_with_binary_only_between(make_server):
    agent = Agent(["First sentence. ", "Second sentence. ", "Third one."])
    _, _, c, _ = await _open(make_server, agent)
    async with c:
        await c.send_json({"type": "text", "text": "go"})
        frames = await c.until("audio_end")
        frames += await c.drain(0.3)
    check_audio_protocol(frames)
    assert types(frames).count("audio_start") == 1 and types(frames).count("audio_end") == 1
    i, j = types(frames).index("audio_start"), types(frames).index("audio_end")
    assert all(t == "audio" or k in (i, j) for k, t in enumerate(types(frames)) if i <= k <= j)
    assert all(not isinstance(f, bytes) for f in frames[:i]) and all(not isinstance(f, bytes) for f in frames[j:])


async def test_vox3_two_turns_two_pairs(make_server):
    _, _, c, _ = await _open(make_server, Agent(["Hello. ", "World."]))
    async with c:
        allf = []
        for t in ("one", "two"):
            await c.send_json({"type": "text", "text": t})
            allf += await c.until("audio_end")
    check_audio_protocol(allf)
    assert types(allf).count("audio_start") == 2 and types(allf).count("audio_end") == 2


async def test_vox3_sentences_spoken_in_order_even_if_later_ones_finish_first(make_server):
    delays = {"Alpha": 0.5, "Bravo": 0.25, "Charlie": 0.01, "Delta": 0.0}
    tts = FakeTts(delay=lambda t: delays[t.split()[0].strip(".")])
    agent = Agent(["Alpha one. ", "Bravo two. ", "Charlie three. ", "Delta four."])
    _, _, c, _ = await _open(make_server, agent, tts)
    async with c:
        await c.send_json({"type": "text", "text": "go"})
        frames = await c.until("audio_end")
    assert _spoken(frames) == "<Alphaone.><Bravotwo.><Charliethree.><Deltafour.>"
    assert tts.requested[0].startswith("Alpha") and tts.requested[-1].startswith("Delta")


async def test_vox3_sentence_split_for_speech_follows_abbreviations(make_server):
    tts = FakeTts()
    agent = Agent(["We offer B.Tech and M.Tech. ", "Fees are Rs. 1.2 lakh. ", "Dr. Rao leads CSE."])
    _, _, c, _ = await _open(make_server, agent, tts)
    async with c:
        await c.send_json({"type": "text", "text": "go"})
        await c.until("audio_end")
    assert tts.requested == ["We offer B.Tech and M.Tech.", "Fees are Rs. 1.2 lakh.", "Dr. Rao leads CSE."]


@pytest.mark.parametrize(
    "deltas",
    [
        ["Hello world. How are you today? I am fine!"],
        list("Hello world. How are you today? I am fine!"),
        ["Hel", "lo wor", "ld. How are y", "ou today? I am fine!"],
        ["The CGPA cutoff is 7.5 for Zoho. ", "Seats left: 1. ", "Register soon."],
    ],
)
async def test_vox3_all_of_the_reply_is_spoken_regardless_of_delta_size(make_server, deltas):
    tts = FakeTts()
    _, _, c, _ = await _open(make_server, Agent(deltas, tool=False), tts)
    async with c:
        await c.send_json({"type": "text", "text": "go"})
        frames = await c.until("audio_end")
    assert "".join(tts.requested).replace(" ", "") == "".join(deltas).replace(" ", "")
    assert _spoken(frames) == "".join(f"<{s}>" for s in tts.requested).replace(" ", "")


async def test_vox3_empty_reply_has_no_audio_frames_or_stray_audio_end(make_server):
    agent = Agent(script=lambda t: [{"type": "done", "text": ""}])
    _, _, c, _ = await _open(make_server, agent)
    async with c:
        await c.send_json({"type": "text", "text": "x"})
        frames = await c.until("assistant_done") + await c.drain(0.4)
    assert "audio_start" not in types(frames) and "audio_end" not in types(frames) and "audio" not in types(frames)


async def test_vox3_tool_only_turn_then_text_still_speaks(make_server):
    agent = Agent(["Done."], tool=True)
    _, _, c, _ = await _open(make_server, agent)
    async with c:
        await c.send_json({"type": "text", "text": "x"})
        frames = await c.until("audio_end")
    assert types(frames).index("tool") < types(frames).index("audio_start")


async def test_vox3_whitespace_only_deltas_are_not_synthesised(make_server):
    tts = FakeTts()
    agent = Agent(["   ", "\n", "Real words here."], tool=False)
    _, _, c, _ = await _open(make_server, agent, tts)
    async with c:
        await c.send_json({"type": "text", "text": "x"})
        await c.until("audio_end")
    assert tts.requested == ["Real words here."]


async def test_vox3_long_reply_without_punctuation_is_spoken_in_pieces(make_server):
    tts = FakeTts()
    words = ("word " * 400).strip()
    _, _, c, _ = await _open(make_server, Agent([words], tool=False), tts)
    async with c:
        await c.send_json({"type": "text", "text": "x"})
        await c.until("audio_end")
    assert len(tts.requested) > 1 and all(len(s) <= 1_000 for s in tts.requested)
    assert " ".join(tts.requested).split() == words.split()


async def test_vox3_audio_chunks_are_whole_samples(make_server):
    tts = FakeTts(chunks=5, chunk_bytes=480)
    _, _, c, _ = await _open(make_server, Agent(["One sentence."]), tts)
    async with c:
        await c.send_json({"type": "text", "text": "x"})
        frames = await c.until("audio_end")
    assert all(len(f) % 2 == 0 for f in frames if isinstance(f, bytes))
    assert len(audio_bytes(frames)) == 5 * 480


# ------------------------------------------------------------------------- VOX5: one turn at a time


async def test_vox5_typed_message_during_turn_cancels_it_and_runs_next(make_server):
    agent = Agent(["Slow words. "] * 20, delay=0.05)
    _, _, c, _ = await _open(make_server, agent)
    async with c:
        await c.send_json({"type": "text", "text": "first"})
        await c.until("assistant_delta")
        await c.send_json({"type": "text", "text": "second"})
        frames = await c.until("interrupted")
        await c.until("assistant_delta")
    assert agent.texts == ["first", "second"]
    assert agent.max_active == 1
    assert "first" in agent.stopped_early


async def test_vox5_rapid_fire_typed_messages_never_overlap_and_run_in_order(make_server):
    agent = Agent(["Short answer. "] * 3, delay=0.02)
    _, _, c, _ = await _open(make_server, agent)
    async with c:
        for i in range(8):
            await c.send_json({"type": "text", "text": f"m{i}"})
        await wait_for(lambda: "m7" in agent.texts and agent.active == 0, 10, "last turn")
        await c.drain(0.5)
    # a message superseded before the model started may never run; the rest run in order, the last one runs
    assert agent.texts == sorted(agent.texts, key=lambda t: int(t[1:])) and agent.texts[-1] == "m7"
    assert agent.max_active == 1


async def test_vox5_utterance_during_typed_turn_does_not_overlap(make_server):
    agent = Agent(["Slow words. "] * 20, delay=0.05)
    _, _, c, _ = await _open(make_server, agent)
    async with c:
        await c.send_json({"type": "text", "text": "typed one"})
        await c.until("assistant_delta")
        await c.mic(b"Utell me about hostels now")
        await c.until("interrupted")
        await c.until("assistant_delta")
    assert agent.texts == ["typed one", "tell me about hostels now"]
    assert agent.max_active == 1


async def test_vox5_typed_during_voice_turn_does_not_overlap(make_server):
    agent = Agent(["Slow words. "] * 20, delay=0.05)
    _, _, c, _ = await _open(make_server, agent)
    async with c:
        await c.mic(b"Utell me about fees today")
        await c.until("assistant_delta")
        await c.send_json({"type": "text", "text": "typed two"})
        await c.until("interrupted")
        await c.until("assistant_delta")
    assert agent.texts == ["tell me about fees today", "typed two"]
    assert agent.max_active == 1


async def test_vox5_simultaneous_voice_and_typed_input(make_server):
    agent = Agent(["Slow words. "] * 10, delay=0.03)
    _, _, c, _ = await _open(make_server, agent)
    async with c:
        await c.mic(b"S")
        await asyncio.gather(
            c.mic(b"Uspoken question number one"),
            c.send_json({"type": "text", "text": "typed question"}),
        )
        await wait_for(lambda: len(agent.calls) >= 1, 5, "a turn")
        await c.drain(1.0)
    # one of the two may be superseded before the model starts, but never two at once
    assert agent.max_active == 1
    assert set(agent.texts) <= {"spoken question number one", "typed question"}


async def test_vox5_after_cancel_events_of_the_new_turn_only(make_server):
    agent = Agent(script=lambda t: [{"type": "sleep"} if False else ("sleep", 0.05), {"type": "text_delta", "text": f"Reply to {t}. "}] * 4 + [{"type": "done", "text": f"Reply to {t}."}])
    _, _, c, _ = await _open(make_server, agent)
    async with c:
        await c.send_json({"type": "text", "text": "AAA"})
        await c.until("assistant_delta")
        await c.send_json({"type": "text", "text": "BBB"})
        pre = await c.until("interrupted")
        rest = await c.until("assistant_done")
    assert all("BBB" in f["text"] for f in events_of(rest, "assistant_delta"))
    assert events_of(rest, "assistant_done")[0]["text"] == "Reply to BBB."


async def test_vox5_idle_socket_accepts_turn_after_turn(make_server):
    agent = Agent(["Ok."])
    _, _, c, _ = await _open(make_server, agent)
    async with c:
        for i in range(5):
            await c.send_json({"type": "text", "text": f"q{i}"})
            frames = await c.until("audio_end")
            assert "error" not in types(frames)
    assert agent.texts == [f"q{i}" for i in range(5)] and agent.max_active == 1


async def test_vox5_agent_events_after_done_do_not_break_the_turn(make_server):
    agent = Agent(script=lambda t: [
        {"type": "text_delta", "text": "Fine. "},
        {"type": "done", "text": "Fine. "},
        {"type": "text_delta", "text": "Late. "},
    ])
    _, _, c, _ = await _open(make_server, agent)
    async with c:
        await c.send_json({"type": "text", "text": "x"})
        frames = await c.until("audio_end")
        await c.send_json({"type": "text", "text": "y"})
        await c.until("audio_end")
    check_audio_protocol(frames)


# ------------------------------------------------------------------------- HIS1


async def test_his1_voice_and_typed_turns_saved_in_order_and_titled(make_server):
    agent = Agent(["Reply."])
    _, _, c, ready = await _open(make_server, agent)
    cid = ready["conversation_id"]
    async with c:
        await c.send_json({"type": "text", "text": "typed first question"})
        await c.until("audio_end")
        await c.mic(b"Uspoken second question")
        await c.until("audio_end")
        await c.send_json({"type": "text", "text": "typed third question"})
        await c.until("audio_end")
    rows = await history_rows(cid)
    assert [r for r in rows if r[0] == "user"] == [
        ("user", "typed first question"),
        ("user", "spoken second question"),
        ("user", "typed third question"),
    ]
    assert [r[0] for r in rows] == ["user", "assistant"] * 3
    assert await pool().fetchval("SELECT title FROM conversations WHERE id=$1::uuid", cid) == "typed first question"


async def test_his1_voice_first_conversation_is_titled_from_the_voice_message(make_server):
    _, _, c, ready = await _open(make_server, Agent(["Ok."]))
    async with c:
        await c.mic(b"Uwhen is the placement drive")
        await c.until("audio_end")
        await c.send_json({"type": "text", "text": "later typed"})
        await c.until("audio_end")
    assert await pool().fetchval("SELECT title FROM conversations WHERE id=$1::uuid", ready["conversation_id"]) == "when is the placement drive"


async def test_his1_rapid_turns_keep_message_order(make_server):
    agent = Agent(["Short. "], delay=0.01)
    _, _, c, ready = await _open(make_server, agent)
    async with c:
        for i in range(6):
            await c.send_json({"type": "text", "text": f"n{i}"})
        await wait_for(lambda: "n5" in agent.texts and agent.active == 0, 10, "last turn")
        await c.drain(0.5)
    users = [t for r, t in await history_rows(ready["conversation_id"]) if r == "user"]
    assert users == sorted(users, key=lambda t: int(t[1:])) and users[-1] == "n5"
    assert users == agent.texts


async def test_his1_turns_go_to_the_announced_conversation(make_server):
    agent = Agent(["Ok."])
    _, _, c, ready = await _open(make_server, agent)
    async with c:
        await c.send_json({"type": "text", "text": "a"})
        await c.until("audio_end")
        await c.mic(b"Ub b b")
        await c.until("audio_end")
    assert {cid for _, cid, _ in agent.calls} == {ready["conversation_id"]}


async def test_his1_resumed_conversation_continues_in_order(make_server):
    agent = Agent(["Ok."])
    srv = await make_server(agent)
    token, _ = await make_session("student", "his9")
    async with await srv.connect(token) as c:
        cid = (await c.start())["conversation_id"]
        await c.send_json({"type": "text", "text": "before"})
        await c.until("audio_end")
    async with await srv.connect(token) as c:
        assert (await c.start(cid))["conversation_id"] == cid
        await c.mic(b"Uafter reconnect")
        await c.until("audio_end")
    assert [t for r, t in await history_rows(cid) if r == "user"] == ["before", "after reconnect"]


async def test_his1_visitor_turns_saved_under_visitor_owner(make_server):
    agent = Agent(["Ok."])
    srv = await make_server(agent)
    token, ctx = await make_session("visitor")
    async with await srv.connect(token) as c:
        cid = (await c.start())["conversation_id"]
        await c.send_json({"type": "text", "text": "visitor question"})
        await c.until("audio_end")
    assert agent.calls[0][0].owner_key == ctx.owner_key and ctx.owner_key.startswith("visitor:")
    assert ("user", "visitor question") in await history_rows(cid)


async def test_his1_turn_receives_the_session_context_of_its_socket(make_server):
    agent = Agent(["Ok."])
    srv = await make_server(agent)
    token, ctx = await make_session("teacher", "tt1", "Teach T")
    async with await srv.connect(token) as c:
        await c.start()
        await c.send_json({"type": "text", "text": "x"})
        await c.until("audio_end")
    got = agent.calls[0][0]
    assert got.owner_key == "user:tt1" and got.role == "teacher" and got.session_id == ctx.session_id
