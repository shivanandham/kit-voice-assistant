"""AGT6, REL1, REL2, VOX5, AC5: failing speech services or assistant give a safe `error` event
and leave the socket usable; no secret or exception text ever reaches the client."""

import asyncio
import socket

import httpx
import pytest
from app.config import get_settings
from app.voice import deepgram, session as voice
from app.voice.deepgram import SttEvent

from tests.acc_voice_helpers import (
    DG_KEY,
    OTHER_KEYS,
    SECRET_KIT_TOKEN,
    Agent,
    FakeStt,
    FakeTts,
    all_text,
    check_audio_protocol,
    events_of,
    make_server,  # noqa: F401
    make_session,
    types,
    wait_for,
)

MARK = "BOOM-INTERNAL-7731"


def _leaky(kind=RuntimeError) -> Exception:
    return kind(f"{MARK} wss://api.deepgram.com/v1/speak Authorization: Token {DG_KEY} kit={SECRET_KIT_TOKEN}")


def assert_safe(msg: dict, token: str = "") -> None:
    """A safe error event: exact shape, short human text, no exception text, no secrets."""
    assert set(msg) == {"type", "message"} and msg["type"] == "error"
    text = msg["message"]
    assert isinstance(text, str) and text.strip() and len(text) <= 200
    for bad in (MARK, DG_KEY, SECRET_KIT_TOKEN, *OTHER_KEYS, "Traceback", 'File "', "api.deepgram.com", token):
        if bad:
            assert bad not in text, f"error message leaks {bad!r}"


def assert_clean(frames: list, token: str = "") -> None:
    blob = all_text(frames)
    for bad in (MARK, DG_KEY, SECRET_KIT_TOKEN, *OTHER_KEYS, token):
        if bad:
            assert bad not in blob


async def _open(make_server, agent=None, stt=None, tts=None):
    agent = agent or Agent(["Recovered fine."])
    srv = await make_server(agent, stt=stt, tts=tts)
    token, _ = await make_session("student", None, "Pat")
    c = await srv.connect(token)
    await c.start()
    return srv, agent, c, token


async def _next_turn_works(c, agent, text="are you still there"):
    await c.send_json({"type": "text", "text": text})
    frames = await c.until("audio_end")
    assert text in agent.texts
    assert events_of(frames, "assistant_done")
    return frames


# ------------------------------------------------------------------------- STT failures


async def test_rel1_stt_connect_failure_gives_error_event_and_usable_socket(make_server):
    def failing():
        s = FakeStt()
        s.fail_send = _leaky(ConnectionError)
        return s

    srv, agent, c, token = await _open(make_server, stt=failing)
    async with c:
        await c.send_bytes(b"\x00\x01" * 320)
        msg = await c.until("error")
        assert_safe(msg[-1], token)
        await _next_turn_works(c, agent)
    assert_clean(c.frames, token)


@pytest.mark.parametrize("kind", [ConnectionError, TimeoutError, OSError, RuntimeError, ValueError])
async def test_agt6_stt_send_failure_variants_are_safe(make_server, kind):
    def failing():
        s = FakeStt()
        s.fail_send = _leaky(kind)
        return s

    _, agent, c, token = await _open(make_server, stt=failing)
    async with c:
        for _ in range(3):
            await c.send_bytes(b"\x00\x01" * 320)
        errs = events_of(await c.drain(0.5), "error")
        assert errs
        for e in errs:
            assert_safe(e, token)
        await _next_turn_works(c, agent)


async def test_rel1_stt_recovers_when_upstream_comes_back(make_server):
    holder = {}

    def flaky():
        s = FakeStt()
        s.fail_send = _leaky(ConnectionError)
        holder["stt"] = s
        return s

    srv, agent, c, _ = await _open(make_server, stt=flaky)
    async with c:
        await c.send_bytes(b"\x00\x01" * 320)
        await c.until("error")
        holder["stt"].fail_send = None
        await c.mic(b"Uhello again after outage")
        await c.until("audio_end")
    assert "hello again after outage" in agent.texts


# DEFECT: when the STT event stream raises mid-utterance the session swallows it: no `error` event and voice input is dead (VoiceSession._pump_stt) (medium).
async def test_rel1_stt_dropping_mid_utterance_gives_error_event(make_server):
    srv, agent, c, token = await _open(make_server)
    async with c:
        await c.mic(b"Iwhat is the")
        await c.recv()
        srv.stts[0].push(_leaky(ConnectionError))
        msg = await c.until("error")
        assert_safe(msg[-1], token)
        await _next_turn_works(c, agent)
    assert_clean(c.frames, token)


# DEFECT: after the STT event stream dies the pump is never restarted, so later speech is ignored silently (VoiceSession._pump_stt/on_audio) (medium).
async def test_rel1_voice_works_again_after_stt_stream_failure(make_server):
    srv, agent, c, _ = await _open(make_server)
    async with c:
        await c.mic(b"Iwhat is the")
        await c.recv()
        srv.stts[0].push(RuntimeError("stream died"))
        await c.drain(0.3)
        await c.mic(b"Uhello once more please")
        await c.until("audio_end")
    assert "hello once more please" in agent.texts


async def test_rel1_stt_error_event_becomes_error_and_socket_usable(make_server):
    srv, agent, c, token = await _open(make_server)
    async with c:
        await c.mic(b"S")
        srv.stts[0].push(SttEvent("error", "Speech recognition disconnected."))
        msg = await c.until("error")
        assert_safe(msg[-1], token)
        await _next_turn_works(c, agent)


# DEFECT: STT error text is forwarded to the client verbatim, so a secret in it would reach the browser (VoiceSession.on_stt) (low).
async def test_ac5_stt_error_text_with_a_secret_is_not_forwarded(make_server):
    srv, agent, c, token = await _open(make_server)
    async with c:
        await c.mic(b"S")
        srv.stts[0].push(SttEvent("error", _leaky().args[0]))
        msg = await c.until("error")
        assert_safe(msg[-1], token)


async def test_rel1_stt_error_without_text_still_gives_a_message(make_server):
    srv, agent, c, _ = await _open(make_server)
    async with c:
        await c.mic(b"S")
        srv.stts[0].push(SttEvent("error", ""))
        msg = await c.until("error")
        assert_safe(msg[-1])


# ------------------------------------------------------------------------- TTS failures


def _fail_first_turn(exc_factory, only_first_sentence=True):
    state = {"armed": True}

    def fail(text, n):
        if state["armed"] and (n == 0 or not only_first_sentence):
            return exc_factory()
        return None

    return state, fail


TTS_ERRORS = [
    lambda: _leaky(RuntimeError),
    lambda: _leaky(ConnectionError),
    lambda: httpx.ReadTimeout(f"{MARK} {DG_KEY}"),
    lambda: httpx.HTTPStatusError(f"401 {MARK} {DG_KEY}", request=httpx.Request("POST", "https://api.deepgram.com/v1/speak"), response=httpx.Response(401)),
    lambda: TimeoutError(MARK),
    lambda: asyncio.TimeoutError(MARK),
]


# DEFECT: a TTS failure on the first sentence is only logged; the client never receives an `error` event (VoiceSession._speak) (high).
@pytest.mark.parametrize("exc_factory", TTS_ERRORS, ids=["runtime", "conn", "readtimeout", "http401", "timeout", "asynctimeout"])
async def test_agt6_tts_failure_on_first_sentence_gives_error_event_and_usable_socket(make_server, exc_factory):
    state, fail = _fail_first_turn(exc_factory)
    tts = FakeTts(fail=fail)
    srv, agent, c, token = await _open(make_server, Agent(["Only sentence here."]), tts=lambda: tts)
    async with c:
        await c.send_json({"type": "text", "text": "speak"})
        frames = await c.until("assistant_done")
        frames += await c.drain(1.0)
        errs = events_of(frames, "error")
        assert errs, "a failed speech service must produce an error event"
        assert_safe(errs[0], token)
        check_audio_protocol(frames)
        state["armed"] = False
        await _next_turn_works(c, agent)
    assert_clean(c.frames, token)


# DEFECT: a TTS failure on a later sentence is only logged; no `error` event is sent (VoiceSession._speak) (high).
@pytest.mark.parametrize("exc_factory", TTS_ERRORS[:3], ids=["runtime", "conn", "readtimeout"])
async def test_agt6_tts_failure_on_later_sentence_gives_error_event(make_server, exc_factory):
    tts = FakeTts(fail=lambda t, n: exc_factory() if n == 1 else None)
    agent = Agent(["First sentence fine. ", "Second sentence breaks. ", "Third is ok."])
    srv, _, c, token = await _open(make_server, agent, tts=lambda: tts)
    async with c:
        await c.send_json({"type": "text", "text": "speak"})
        frames = await c.until("assistant_done")
        frames += await c.drain(1.0)
        errs = events_of(frames, "error")
        assert errs
        assert_safe(errs[0], token)
        check_audio_protocol(frames)
        # audio that was started is closed (audio_end or interrupted), never left open
        t = types(frames)
        if "audio_start" in t:
            assert "audio_end" in t or "interrupted" in t
    assert_clean(c.frames, token)


async def test_rel1_tts_failure_on_every_sentence_leaves_socket_usable(make_server):
    state, fail = _fail_first_turn(lambda: _leaky(), only_first_sentence=False)
    tts = FakeTts(fail=fail)
    srv, agent, c, token = await _open(make_server, Agent(["One. ", "Two. ", "Three."]), tts=lambda: tts)
    async with c:
        await c.send_json({"type": "text", "text": "speak"})
        await c.until("assistant_done")
        await c.drain(0.8)
        state["armed"] = False
        await _next_turn_works(c, agent)
    assert_clean(c.frames, token)


async def test_agt6_tts_hang_does_not_block_interrupt_or_next_turn(make_server):
    state = {"hang": True}
    tts = FakeTts(hang=lambda t: state["hang"])
    srv, agent, c, _ = await _open(make_server, Agent(["Stuck sentence here."]), tts=lambda: tts)
    async with c:
        await c.send_json({"type": "text", "text": "speak"})
        await c.until("assistant_done")
        await c.send_json({"type": "interrupt"})
        await c.until("interrupted")
        state["hang"] = False
        await _next_turn_works(c, agent)


async def test_agt6_tts_hang_then_new_typed_message_supersedes(make_server):
    state = {"hang": True}
    tts = FakeTts(hang=lambda t: state["hang"])
    srv, agent, c, _ = await _open(make_server, Agent(["Stuck sentence here."]), tts=lambda: tts)
    async with c:
        await c.send_json({"type": "text", "text": "speak"})
        await c.until("assistant_done")
        state["hang"] = False
        await c.send_json({"type": "text", "text": "second try"})
        frames = await c.until("audio_end")
    assert "second try" in agent.texts and events_of(frames, "assistant_done")


# ------------------------------------------------------------------------- agent failures


def _raising_agent(before_text=True, sync=False):
    if sync:
        def run(ctx, cid, text):
            raise _leaky()
        return run
    return Agent(script=lambda t: ([] if before_text else [{"type": "text_delta", "text": "Partial reply. "}]) + [("raise", _leaky())] if t == "boom" else [{"type": "text_delta", "text": "Recovered fine."}, {"type": "done", "text": "Recovered fine."}])


@pytest.mark.parametrize("variant", ["immediate", "after_delta", "sync"])
async def test_agt6_run_turn_raising_gives_safe_error_and_usable_socket(make_server, variant):
    agent = _raising_agent(before_text=variant == "immediate", sync=variant == "sync")
    follow = Agent(["Recovered fine."])
    calls = {"n": 0}

    if variant == "sync":
        async def good(ctx, cid, text):
            yield {"type": "text_delta", "text": "Recovered fine."}
            yield {"type": "done", "text": "Recovered fine."}

        def run(ctx, cid, text):
            calls["n"] += 1
            if text == "boom":
                raise _leaky()
            return good(ctx, cid, text)

        agent = run

    srv = await make_server(agent)
    token, _ = await make_session("student", None, "Pat")
    async with await srv.connect(token) as c:
        await c.start()
        await c.send_json({"type": "text", "text": "boom"})
        frames = await c.until("error")
        assert_safe(frames[-1], token)
        check_audio_protocol(frames)
        await c.send_json({"type": "text", "text": "fine now"})
        ok = await c.until("audio_end")
        assert events_of(ok, "assistant_done")[0]["text"] == "Recovered fine."
    assert_clean(c.frames, token)


async def test_agt6_run_turn_failure_does_not_leave_audio_open(make_server):
    agent = Agent(script=lambda t: [{"type": "text_delta", "text": "Partial sentence. "}, ("raise", _leaky())])
    _, _, c, token = await _open(make_server, agent)
    async with c:
        await c.send_json({"type": "text", "text": "x"})
        frames = await c.until("error") + await c.drain(1.0)
    check_audio_protocol(frames)
    t = types(frames)
    if "audio_start" in t:
        assert "audio_end" in t or "interrupted" in t


async def test_agt6_run_turn_error_event_is_forwarded_and_socket_usable(make_server):
    msg = "The assistant is unavailable right now. Please try again."
    agent = Agent(script=lambda t: [{"type": "error", "message": msg}] if t == "fail" else [{"type": "text_delta", "text": "Back again."}, {"type": "done", "text": "Back again."}])
    _, _, c, token = await _open(make_server, agent)
    async with c:
        await c.send_json({"type": "text", "text": "fail"})
        frames = await c.until("error")
        assert frames[-1] == {"type": "error", "message": msg}
        await c.send_json({"type": "text", "text": "retry"})
        await c.until("audio_end")
    assert agent.texts == ["fail", "retry"]


async def test_agt6_run_turn_error_event_without_message_still_safe(make_server):
    agent = Agent(script=lambda t: [{"type": "error"}])
    _, _, c, _ = await _open(make_server, agent)
    async with c:
        await c.send_json({"type": "text", "text": "x"})
        frames = await c.until("error")
        assert_safe(frames[-1])


async def test_agt6_agent_hanging_forever_can_be_interrupted(make_server):
    agent = Agent(script=lambda t: [("hang",)] if t == "hang" else [{"type": "text_delta", "text": "Ok."}, {"type": "done", "text": "Ok."}])
    _, _, c, _ = await _open(make_server, agent)
    async with c:
        await c.send_json({"type": "text", "text": "hang"})
        await wait_for(lambda: agent.calls, 3, "turn")
        await c.send_json({"type": "interrupt"})
        await c.until("interrupted")
        await c.send_json({"type": "text", "text": "next"})
        await c.until("audio_end")


async def test_agt6_repeated_failures_keep_socket_usable(make_server):
    agent = Agent(script=lambda t: [("raise", _leaky())] if t.startswith("bad") else [{"type": "text_delta", "text": "Fine."}, {"type": "done", "text": "Fine."}])
    _, _, c, token = await _open(make_server, agent)
    async with c:
        for i in range(5):
            await c.send_json({"type": "text", "text": f"bad{i}"})
            assert_safe((await c.until("error"))[-1], token)
        await c.send_json({"type": "text", "text": "good"})
        await c.until("audio_end")
    assert_clean(c.frames, token)


async def test_agt6_failed_turn_then_voice_turn_works(make_server):
    agent = Agent(script=lambda t: [("raise", _leaky())] if t == "bad" else [{"type": "text_delta", "text": "Fine."}, {"type": "done", "text": "Fine."}])
    _, _, c, _ = await _open(make_server, agent)
    async with c:
        await c.send_json({"type": "text", "text": "bad"})
        await c.until("error")
        await c.mic(b"Uspoken turn after failure")
        await c.until("audio_end")
    assert agent.texts == ["bad", "spoken turn after failure"]


# ------------------------------------------------------------------------- AC5 sweep


async def test_ac5_no_secret_in_any_frame_of_a_busy_session(make_server):
    state, fail = _fail_first_turn(lambda: _leaky(), only_first_sentence=False)
    tts = FakeTts(fail=fail)
    agent = Agent(script=lambda t: [("raise", _leaky())] if t == "bad" else [
        {"type": "tool", "name": "get_attendance", "state": "start", "is_error": False},
        {"type": "text_delta", "text": "Your attendance is high. "},
        {"type": "done", "text": "Your attendance is high. "},
    ])
    srv, _, c, token = await _open(make_server, agent, tts=lambda: tts)
    async with c:
        await c.send_text("{broken")
        await c.send_json({"type": "nope"})
        await c.send_json({"type": "text", "text": "bad"})
        await c.drain(0.6)
        await c.send_json({"type": "text", "text": "attendance"})
        await c.drain(0.8)
        state["armed"] = False
        await c.send_json({"type": "text", "text": "attendance again"})
        await c.drain(0.8)
    assert_clean(c.frames, token)
    assert len(c.frames) > 5


# ------------------------------------------------------------------------- REL2


def test_rel2_default_factories_make_no_network_connection(network_guard):
    """Building the production speech clients opens nothing; keys are the test dummies."""
    assert get_settings().deepgram_api_key == DG_KEY
    stt = voice._default_stt()
    tts = voice._default_tts()
    assert isinstance(stt, deepgram.DeepgramSTT) and isinstance(tts, deepgram.DeepgramTTS)
    assert network_guard == []


def test_rel2_guard_blocks_real_deepgram_host(network_guard):
    with pytest.raises(OSError):
        socket.getaddrinfo("api.deepgram.com", 443)
    network_guard.clear()  # the violation was the intended probe


async def test_rel2_acceptance_server_uses_only_injected_fakes(make_server, network_guard):
    srv, agent, c, _ = await _open(make_server)
    async with c:
        await c.mic(b"Uhello")
        await c.until("audio_end")
    assert srv.factory_calls == 1 and srv.stts and srv.ttss
    assert network_guard == []
