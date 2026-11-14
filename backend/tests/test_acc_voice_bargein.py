"""VOX2 (and VOX3 no-audio-after-interrupted): barge-in by speech and by `interrupt`."""

import asyncio

import pytest

from tests.acc_voice_helpers import (
    Agent,
    FakeTts,
    check_audio_protocol,
    events_of,
    make_server,  # noqa: F401
    make_session,
    types,
    wait_for,
)

SLOW = ["Slow words. "] * 40


async def _open(make_server, agent, tts=None):
    srv = await make_server(agent, tts=(lambda: tts) if tts else None)
    token, _ = await make_session("student", None, "Pat")
    c = await srv.connect(token)
    await c.start()
    return srv, c


async def _quiet_after(c, frames_after_interrupt_window=0.6):
    """Frames arriving after an `interrupted`: none may be assistant/audio of the old turn."""
    return await c.drain(frames_after_interrupt_window)


# ------------------------------------------------------------------------- stage: generating


async def test_vox2_speech_during_text_generation_cancels_turn_and_starts_next(make_server):
    agent = Agent(SLOW, delay=0.05)
    srv, c = await _open(make_server, agent)
    async with c:
        await c.send_json({"type": "text", "text": "tell me everything"})
        await c.until("assistant_delta")
        await c.mic(b"Iwait what about hostels")
        frames = await c.until("interrupted")
        assert events_of(frames, "interrupted") == [{"type": "interrupted"}]
        await c.mic(b"Uwait what about hostels")
        later = await c.until("transcript")
        while not (isinstance(later[-1], dict) and later[-1].get("final")):
            later += await c.until("transcript")
        # between `interrupted` and the new final transcript nothing of the old turn is delivered
        between = [f for f in later if isinstance(f, dict) and f["type"] in ("assistant_delta", "assistant_done", "tool", "audio_start")] + [f for f in later if isinstance(f, bytes)]
        assert between == []
        await c.until("assistant_delta")
    assert agent.texts == ["tell me everything", "wait what about hostels"]
    assert agent.stopped_early[0] == "tell me everything"
    assert srv.ttss[0].resets >= 1


async def test_vox2_final_utterance_without_interim_also_interrupts(make_server):
    agent = Agent(SLOW, delay=0.05)
    _, c = await _open(make_server, agent)
    async with c:
        await c.send_json({"type": "text", "text": "long answer"})
        await c.until("assistant_delta")
        await c.mic(b"Utell me about hostel fees")
        frames = await c.until("interrupted")
        frames += await c.until("assistant_delta")
    assert {"type": "transcript", "final": True, "text": "tell me about hostel fees"} in frames
    assert agent.texts == ["long answer", "tell me about hostel fees"]


async def test_vox2_cancellation_stops_the_model_stream(make_server):
    agent = Agent(SLOW, delay=0.05)
    _, c = await _open(make_server, agent)
    async with c:
        await c.send_json({"type": "text", "text": "go on"})
        await c.until("assistant_delta")
        await c.send_json({"type": "interrupt"})
        await c.until("interrupted")
        await wait_for(lambda: agent.active == 0, 3, "model stream to stop")
        await c.drain(0.5)
    assert agent.stopped_early == ["go on"] and agent.completed == []


async def test_vox2_nothing_from_old_turn_after_interrupted(make_server):
    agent = Agent(SLOW, delay=0.02)
    _, c = await _open(make_server, agent)
    async with c:
        await c.send_json({"type": "text", "text": "go on"})
        head = await c.until("audio_start")
        await c.send_json({"type": "interrupt"})
        frames = head + await c.until("interrupted")
        tail = await c.drain(0.8)
    assert tail == []
    check_audio_protocol(frames + tail)


@pytest.mark.parametrize("blip", [b"Iuh", b"Ihmm", b"Ium"])
async def test_vox2_a_blip_is_not_a_barge_in(make_server, blip):
    agent = Agent(SLOW, delay=0.05)
    _, c = await _open(make_server, agent)
    async with c:
        await c.send_json({"type": "text", "text": "tell me"})
        await c.until("assistant_delta")
        await c.mic(blip)
        got = await c.drain(0.5)
    assert "interrupted" not in types(got) and agent.stopped_early == []


# ------------------------------------------------------------------------- stage: synthesising


async def test_vox2_speech_during_synthesis_cancels_turn(make_server):
    tts = FakeTts(delay=lambda t: 3.0)
    agent = Agent(["Quick reply that is long enough."])
    srv, c = await _open(make_server, agent, tts)
    async with c:
        await c.send_json({"type": "text", "text": "ask"})
        await c.until("assistant_done")
        await c.mic(b"Iactually tell me something else")
        frames = await c.until("interrupted")
        assert "audio_start" not in types(frames)
        await c.mic(b"Uactually tell me something else")
        await c.until("assistant_done")
    assert agent.texts == ["ask", "actually tell me something else"]
    assert tts.abandoned, "pending synthesis must be abandoned"
    assert tts.resets >= 1


async def test_vox2_interrupt_during_synthesis(make_server):
    tts = FakeTts(delay=lambda t: 3.0)
    agent = Agent(["Quick reply that is long enough."])
    _, c = await _open(make_server, agent, tts)
    async with c:
        await c.send_json({"type": "text", "text": "ask"})
        await c.until("assistant_done")
        await c.send_json({"type": "interrupt"})
        await c.until("interrupted")
        tail = await c.drain(0.6)
    assert "audio_start" not in types(tail) and "audio" not in types(tail)
    assert tts.resets >= 1


# ------------------------------------------------------------------------- stage: sending audio


def _streaming_tts():
    return FakeTts(chunks=60, chunk_bytes=960, chunk_delay=0.02)


async def test_vox2_speech_during_audio_sending_cancels_and_stops_audio(make_server):
    tts = _streaming_tts()
    agent = Agent(["A long sentence that is being spoken right now."])
    _, c = await _open(make_server, agent, tts)
    async with c:
        await c.send_json({"type": "text", "text": "speak"})
        got = []
        while sum(isinstance(f, bytes) for f in got) < 3:
            got.append(await c.recv())
        await c.mic(b"Iexcuse me one moment please")
        frames = await c.until("interrupted")
        tail = await c.drain(0.8)
    check_audio_protocol(got + frames + tail)
    after = [f for f in tail if isinstance(f, bytes)]
    assert after == [], "no audio after interrupted"
    assert "audio_end" not in types(tail)


async def test_vox2_interrupt_during_audio_sending_no_audio_afterwards(make_server):
    tts = _streaming_tts()
    agent = Agent(["Sentence one is long enough. ", "Sentence two follows it."])
    _, c = await _open(make_server, agent, tts)
    async with c:
        await c.send_json({"type": "text", "text": "speak"})
        got = []
        while sum(isinstance(f, bytes) for f in got) < 5:
            got.append(await c.recv())
        await c.send_json({"type": "interrupt"})
        frames = await c.until("interrupted")
        tail = await c.drain(1.0)
    check_audio_protocol(got + frames + tail)
    assert not [f for f in tail if isinstance(f, bytes)]
    # the unspoken second sentence is never played
    assert "Sentence two follows it." not in tts.completed


async def test_vox2_new_utterance_after_barge_in_speaks_its_own_reply_only(make_server):
    tts = _streaming_tts()
    agent = Agent(script=lambda t: [{"type": "text_delta", "text": f"Answer about {t.split()[-1]} follows now."}, {"type": "done", "text": f"Answer about {t.split()[-1]} follows now."}])
    _, c = await _open(make_server, agent, tts)
    async with c:
        await c.send_json({"type": "text", "text": "question alpha"})
        got = []
        while sum(isinstance(f, bytes) for f in got) < 3:
            got.append(await c.recv())
        await c.mic(b"Uno wait question beta")
        rest = await c.until("audio_end")
    allf = got + rest
    check_audio_protocol(allf)
    assert agent.texts == ["question alpha", "no wait question beta"]
    assert types(allf).count("interrupted") == 1
    # after `interrupted` the next audio_start must precede any binary frame
    idx = types(allf).index("interrupted")
    nxt = types(allf)[idx + 1 :]
    assert nxt.index("audio_start") < nxt.index("audio")


async def test_vox2_speech_while_client_is_still_playing_reply_interrupts(make_server):
    """Server finished sending 1 s of audio; the client is still playing it: barge-in must say so."""
    tts = FakeTts(chunks=1, chunk_bytes=48_000, delay=lambda t: 0.01)
    agent = Agent(["One short sentence."])
    _, c = await _open(make_server, agent, tts)
    async with c:
        await c.send_json({"type": "text", "text": "speak"})
        await c.until("audio_end")
        await asyncio.sleep(0.1)
        await c.mic(b"Istop talking and listen to me")
        frames = await c.until("interrupted")
    assert events_of(frames, "interrupted")


async def test_vox2_speech_long_after_playback_is_just_a_transcript(make_server):
    tts = FakeTts(chunks=1, chunk_bytes=480, delay=lambda t: 0.01)
    _, c = await _open(make_server, Agent(["Short."]), tts)
    async with c:
        await c.send_json({"type": "text", "text": "speak"})
        await c.until("audio_end")
        await asyncio.sleep(0.8)
        await c.mic(b"Isomething completely new now")
        got = await c.until("transcript")
        tail = await c.drain(0.3)
    assert "interrupted" not in types(got + tail)


# ------------------------------------------------------------------------- interrupt command


async def test_vox2_interrupt_when_idle_is_harmless(make_server):
    agent = Agent(["Ok."])
    _, c = await _open(make_server, agent)
    async with c:
        for _ in range(3):
            await c.send_json({"type": "interrupt"})
        got = await c.drain(0.4)
        assert "error" not in types(got) and "audio_start" not in types(got)
        await c.send_json({"type": "text", "text": "after idle interrupt"})
        await c.until("audio_end")
    assert agent.texts == ["after idle interrupt"]


async def test_vox2_interrupt_after_reply_fully_played_is_harmless(make_server):
    agent = Agent(["Ok."])
    _, c = await _open(make_server, agent)
    async with c:
        await c.send_json({"type": "text", "text": "x"})
        await c.until("audio_end")
        await asyncio.sleep(0.6)
        await c.send_json({"type": "interrupt"})
        got = await c.drain(0.4)
        assert "error" not in types(got)
        await c.send_json({"type": "text", "text": "y"})
        await c.until("audio_end")


async def test_vox2_double_interrupt_sends_one_interrupted_and_no_error(make_server):
    agent = Agent(SLOW, delay=0.05)
    _, c = await _open(make_server, agent)
    async with c:
        await c.send_json({"type": "text", "text": "x"})
        await c.until("assistant_delta")
        await c.send_json({"type": "interrupt"})
        await c.send_json({"type": "interrupt"})
        got = await c.until("interrupted") + await c.drain(0.5)
    assert types(got).count("interrupted") == 1 and "error" not in types(got)


async def test_vox2_interrupt_during_tool_call(make_server):
    agent = Agent(script=lambda t: [
        {"type": "tool", "name": "get_attendance", "state": "start", "is_error": False},
        ("hang",),
    ])
    _, c = await _open(make_server, agent)
    async with c:
        await c.send_json({"type": "text", "text": "attendance"})
        await c.until("tool")
        await c.send_json({"type": "interrupt"})
        await c.until("interrupted")
        await wait_for(lambda: agent.active == 0, 3, "tool turn to stop")
    assert agent.stopped_early == ["attendance"]


async def test_vox2_speech_while_model_is_silent_cancels(make_server):
    """No delta yet (model thinking): real speech still cancels."""
    agent = Agent(script=lambda t: [("hang",)] if t == "think" else [{"type": "text_delta", "text": "Okay then."}, {"type": "done", "text": "Okay then."}])
    _, c = await _open(make_server, agent)
    async with c:
        await c.send_json({"type": "text", "text": "think"})
        await wait_for(lambda: agent.calls, 3, "turn start")
        await c.mic(b"Uplease answer something different")
        await c.until("interrupted")
        await c.until("audio_end")
    assert agent.texts == ["think", "please answer something different"]


async def test_vox2_repeated_barge_ins_keep_working(make_server):
    agent = Agent(SLOW, delay=0.04)
    _, c = await _open(make_server, agent)
    async with c:
        await c.send_json({"type": "text", "text": "turn zero"})
        for i in range(1, 4):
            await c.until("assistant_delta")
            await c.mic(f"Iinterrupting you number {i}".encode())
            await c.until("interrupted")
            await c.mic(f"Uinterrupting you number {i}".encode())
            await c.until("assistant_delta")
    assert agent.texts == ["turn zero"] + [f"interrupting you number {i}" for i in range(1, 4)]
    assert agent.max_active == 1


# ------------------------------------------------------------------------- echo


@pytest.mark.parametrize("echo", [b"Iadmissions are open", b"IAdmissions, ARE open!", b"Iopen admissions", b"Iadmissions are open now"])
async def test_vox2_echo_of_own_reply_does_not_interrupt_during_generation(make_server, echo):
    agent = Agent(["Admissions are open now. "] * 30, delay=0.05)
    _, c = await _open(make_server, agent)
    async with c:
        await c.send_json({"type": "text", "text": "admissions?"})
        await c.until("assistant_delta")
        await c.mic(echo)
        got = await c.drain(0.6)
    assert "interrupted" not in types(got)
    assert agent.stopped_early == []


async def test_vox2_echo_final_utterance_does_not_start_a_turn(make_server):
    agent = Agent(["Admissions are open now. "] * 30, delay=0.05)
    _, c = await _open(make_server, agent)
    async with c:
        await c.send_json({"type": "text", "text": "admissions?"})
        await c.until("assistant_delta")
        await c.mic(b"Uadmissions are open now")
        got = await c.drain(0.6)
    assert "interrupted" not in types(got)
    assert agent.texts == ["admissions?"]


async def test_vox2_echo_during_synthesis_does_not_interrupt(make_server):
    tts = FakeTts(delay=lambda t: 1.0)
    agent = Agent(["Admissions are open now."])
    _, c = await _open(make_server, agent, tts)
    async with c:
        await c.send_json({"type": "text", "text": "admissions?"})
        await c.until("assistant_done")
        await c.mic(b"Iadmissions are open now")
        got = await c.drain(0.5)
        assert "interrupted" not in types(got)
        await c.until("audio_end")
    assert agent.texts == ["admissions?"]


async def test_vox2_echo_during_audio_sending_does_not_interrupt(make_server):
    tts = _streaming_tts()
    agent = Agent(["Admissions are open now."])
    _, c = await _open(make_server, agent, tts)
    async with c:
        await c.send_json({"type": "text", "text": "admissions?"})
        await c.until("audio_start")
        await c.mic(b"Iadmissions are open now")
        frames = await c.until("audio_end")
    assert "interrupted" not in types(frames)


async def test_vox2_reply_words_plus_one_new_word_is_real_speech(make_server):
    agent = Agent(["Admissions are open now. "] * 30, delay=0.05)
    _, c = await _open(make_server, agent)
    async with c:
        await c.send_json({"type": "text", "text": "admissions?"})
        await c.until("assistant_delta")
        await c.mic(b"Iadmissions are open but hostels")
        await c.until("interrupted")


async def test_vox2_echo_words_of_previous_turn_do_not_whitelist_next_turn(make_server):
    """Words of an earlier reply are not 'echo' for a later reply that never said them."""
    agent = Agent(script=lambda t: [("sleep", 0.05), {"type": "text_delta", "text": "Alpha beta gamma. " if t == "one" else "Delta epsilon zeta. "}] * 20 + [{"type": "done", "text": "x"}])
    _, c = await _open(make_server, agent)
    async with c:
        await c.send_json({"type": "text", "text": "one"})
        await c.until("assistant_delta")
        await c.send_json({"type": "text", "text": "two"})
        await c.until("interrupted")
        await c.until("assistant_delta")
        await c.mic(b"Ialpha beta gamma")
        await c.until("interrupted")
