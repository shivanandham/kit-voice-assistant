"""ABU11: only speech is forwarded to Deepgram, and forwarded voice time is capped per day."""

import math
import struct

import pytest
from app import limits
from app.voice.gate import GATE_HANGOVER_S, GATE_PREROLL_S, EnergyGate

FRAME_S = 0.02
FRAME_BYTES = 640


def tone(amplitude: float, seconds: float = FRAME_S, hz: float = 220.0) -> bytes:
    """PCM16 sine at the given peak amplitude."""
    n = int(16000 * seconds)
    return struct.pack(
        f"<{n}h", *(int(amplitude * math.sin(2 * math.pi * hz * i / 16000)) for i in range(n))
    )


def silence(seconds: float = FRAME_S) -> bytes:
    return b"\0" * (int(16000 * seconds) * 2)


def run(gate: EnergyGate, frames: list[bytes]) -> bytes:
    return b"".join(b"".join(gate.feed(f)) for f in frames)


def test_abu11_silence_is_never_forwarded():
    gate = EnergyGate()
    assert run(gate, [silence()] * 500) == b""  # 10 s of silence


def test_abu11_speech_opens_gate_with_pre_roll_and_hang_over():
    gate = EnergyGate()
    quiet = [tone(30)] * 50  # 1 s of room noise
    speech = [tone(8000)] * 25  # 0.5 s of speech
    tail = [tone(30)] * 150  # 3 s of room noise afterwards
    assert run(gate, quiet) == b""
    out = run(gate, speech)
    pre = int(GATE_PREROLL_S / FRAME_S) * FRAME_BYTES
    assert len(out) >= len(b"".join(speech)) + pre - 4 * FRAME_BYTES  # pre-roll at opening
    after = run(gate, tail)
    hang = int(GATE_HANGOVER_S / FRAME_S) * FRAME_BYTES
    assert hang - 2 * FRAME_BYTES <= len(after) <= hang + 2 * FRAME_BYTES
    assert not gate.open


def test_abu11_one_loud_blip_does_not_open_the_gate():
    gate = EnergyGate()
    run(gate, [tone(30)] * 50)
    assert run(gate, [tone(8000)] + [tone(30)] * 50) == b""


def test_abu11_constant_noise_cannot_hold_the_gate_open():
    gate = EnergyGate()
    out = [len(b"".join(gate.feed(tone(6000)))) for _ in range(30 * 50)]  # 30 s of loud hum
    assert sum(out[-10 * 50 :]) == 0  # the floor has risen; the last 10 s are not forwarded
    assert sum(out) < 15 * 50 * FRAME_BYTES


def test_abu11_speech_after_noise_still_gets_through():
    gate = EnergyGate()
    for _ in range(20 * 50):
        gate.feed(tone(1500))  # steady fan noise raises the floor
    assert run(gate, [tone(12000)] * 25) != b""  # a voice well above it opens the gate


def test_abu11_odd_and_tiny_frames_are_safe():
    gate = EnergyGate()
    assert gate.feed(b"") == []
    assert gate.feed(b"\x01") == []


@pytest.fixture
def clock(monkeypatch):
    t = [1000.0]
    monkeypatch.setattr(limits, "_clock", lambda: t[0])
    return t


def test_abu11_voice_seconds_per_visitor_ip_and_per_login(clock):
    visitor = type("C", (), {"role": "visitor", "user": None, "owner_key": "visitor:x"})()
    student = type("C", (), {"role": "student", "user": {"login_id": "arjun"}, "owner_key": "u"})()
    for _ in range(899):
        assert limits.take_voice_seconds(visitor, "ip1", 1.0)
    assert limits.take_voice_seconds(visitor, "ip1", 1.0)  # 900 s = 15 min
    assert not limits.take_voice_seconds(visitor, "ip1", 1.0)
    assert limits.take_voice_seconds(visitor, "ip2", 1.0)  # another IP
    for _ in range(3600):
        assert limits.take_voice_seconds(student, "ip1", 1.0)  # same IP, own allowance
    assert not limits.take_voice_seconds(student, "ip9", 1.0)  # per login id, any IP
    clock[0] += 86401
    assert limits.take_voice_seconds(visitor, "ip1", 1.0)
    assert limits.take_voice_seconds(student, "ip1", 1.0)


# --------------------------------------------------------------------------- in the session

from app.config import get_settings

from tests import acc_voice_helpers
from tests.acc_voice_helpers import Agent, make_session, wait_for

make_server = acc_voice_helpers.make_server  # the real-server fixture


@pytest.fixture
def gate_on(monkeypatch):
    monkeypatch.setattr(get_settings(), "voice_gate_enabled", True)


async def started(server, token):
    c = await server.connect(token)
    await c.start()
    return c


async def test_abu11_session_forwards_speech_not_silence(make_server, gate_on):
    server = await make_server(Agent(["Okay."], tool=False, save=False))
    token, _ = await make_session()
    c = await started(server, token)
    for _ in range(100):  # 2 s of silence
        await c.send_bytes(silence())
    await c.send_bytes(tone(8000, 0.5))  # speech
    await wait_for(lambda: server.stts[0].frames, what="speech forwarded")
    forwarded = b"".join(server.stts[0].frames)
    assert 0 < len(forwarded) <= len(tone(8000, 0.5)) + 2 * 12800
    await c.close()


async def test_abu11_voice_time_cap_stops_audio_but_typing_works(make_server, gate_on, monkeypatch):
    monkeypatch.setitem(limits.DEFAULTS, "limit_voice_seconds_visitor_ip_per_day", 1)
    agent = Agent(["Okay."], tool=False, save=False)
    server = await make_server(agent)
    token, _ = await make_session()
    c = await started(server, token)
    for _ in range(100):  # 2 s of speech
        await c.send_bytes(tone(8000))
    err = await c.until("error")
    assert "voice time" in err[-1]["message"] and "type" in err[-1]["message"]
    sent = sum(len(f) for f in server.stts[0].frames)
    assert sent <= 32000
    for _ in range(50):
        await c.send_bytes(tone(8000))
    await c.drain(0.3)
    assert sum(len(f) for f in server.stts[0].frames) == sent  # nothing more forwarded
    await c.send_json({"type": "text", "text": "typed still works"})
    await c.until("audio_end")
    assert agent.texts == ["typed still works"]
    await c.close()


async def test_abu11_gate_off_forwards_everything(make_server):
    server = await make_server(Agent())
    token, _ = await make_session()
    c = await started(server, token)
    for _ in range(10):
        await c.send_bytes(silence())
    await wait_for(lambda: len(server.stts[0].frames) == 10, what="frames")
    await c.close()
