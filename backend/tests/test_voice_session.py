"""The /ws/voice protocol with Deepgram and the agent loop faked."""

import asyncio
import json
import time

import pytest
from app import auth, history
from app.voice import session as voice
from app.voice.deepgram import SttEvent
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

ORIGIN = {"origin": "http://localhost:8000"}
CTX = auth.SessionCtx(
    session_id="s1",
    owner_key="visitor:x",
    role="visitor",
    user=None,
    kit_token=None,
)


class FakeStt:
    """Audio frames are scripts: b'S' speech started, b'I<text>' interim, b'U<text>' utterance."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue | None = None
        self.closed = False
        self.frames = 0

    def _q(self) -> asyncio.Queue:
        if self.queue is None:
            self.queue = asyncio.Queue()
        return self.queue

    async def send_audio(self, pcm: bytes) -> None:
        self.frames += 1
        text = pcm[1:].decode().strip()
        kind = {"S": "speech_started", "I": "interim", "U": "utterance"}.get(chr(pcm[0]))
        if kind:
            self._q().put_nowait(SttEvent(kind, text))

    async def events(self):
        while True:
            yield await self._q().get()

    async def close(self) -> None:
        self.closed = True


class FakeTts:
    """Synthesises `<text>` as bytes; sentences named 'Slow...' take longer to produce."""

    def __init__(self) -> None:
        self.started: list[tuple[str, float]] = []
        self.resets = 0
        self.closed = False

    async def warm(self) -> None:
        return None

    def synthesize(self, text: str):
        self.started.append((text, time.monotonic()))

        async def gen():
            data = f"<{text}>".encode()
            data += b" " * (len(data) % 2)
            await asyncio.sleep(0.15 if text.startswith("Slow") else 0.01)
            half = (len(data) // 4) * 2
            yield data[:half]
            await asyncio.sleep(0.01)
            yield data[half:]

        return gen()

    async def reset(self) -> None:
        self.resets += 1

    async def close(self) -> None:
        self.closed = True


class Agent:
    """Fake run_turn: streams the given deltas, optionally slowly; records cancellation."""

    def __init__(self, deltas: list[str], delay: float = 0.0) -> None:
        self.deltas = deltas
        self.delay = delay
        self.calls: list[str] = []
        self.cancelled = 0

    async def __call__(self, ctx, conversation_id, text):
        self.calls.append(text)
        try:
            yield {"type": "tool", "name": "list_courses", "state": "start", "is_error": False}
            yield {"type": "tool", "name": "list_courses", "state": "end", "is_error": False}
            for d in self.deltas:
                await asyncio.sleep(self.delay)
                yield {"type": "text_delta", "text": d}
            yield {"type": "done", "text": "".join(self.deltas)}
        except asyncio.CancelledError:
            self.cancelled += 1
            raise


@pytest.fixture
def setup(monkeypatch):
    """Build an app around fakes; returns a factory taking the agent."""

    async def session_ok(ws):
        return CTX

    async def create(ctx):
        return "conv-new"

    async def owned(ctx, cid):
        return cid == "conv-mine"

    monkeypatch.setattr(auth, "session_from_websocket", session_ok)
    monkeypatch.setattr(history, "create_conversation", create)
    monkeypatch.setattr(history, "conversation_owned", owned)

    def make(agent: Agent):
        stt, tts = FakeStt(), FakeTts()
        app = FastAPI()
        app.include_router(voice.build_router(agent, lambda: stt, lambda: tts))
        return TestClient(app), stt, tts

    return make


def frame(script: bytes) -> bytes:
    """Pad a scripted fake-mic frame to whole 16-bit samples."""
    return script + b" " * (len(script) % 2)


def recv_until(ws, kind: str, limit: int = 200) -> list:
    """Collect frames (dicts or bytes) up to and including the first JSON event of `kind`."""
    out = []
    for _ in range(limit):
        msg = ws.receive()
        item = msg.get("bytes") if msg.get("bytes") is not None else json.loads(msg["text"])
        out.append(item)
        if isinstance(item, dict) and item.get("type") == kind:
            return out
    raise AssertionError(f"no {kind} event")


def events(frames: list) -> list[str]:
    return [f["type"] for f in frames if isinstance(f, dict)]


def test_happy_path_voice_turn(setup):
    agent = Agent(["We offer B.Tech ", "and M.Tech. ", "Fees are Rs. 1.2 lakh."])
    client, stt, tts = setup(agent)
    with client.websocket_connect("/ws/voice", headers=ORIGIN) as ws:
        ws.send_json({"type": "start", "conversation_id": None})
        ready = ws.receive_json()
        assert ready == {
            "type": "ready",
            "conversation_id": "conv-new",
            "role": "visitor",
            "user_name": None,
        }
        ws.send_bytes(frame(b"S"))
        ws.send_bytes(frame(b"IWhat courses"))
        assert ws.receive_json() == {"type": "transcript", "final": False, "text": "What courses"}
        ws.send_bytes(frame(b"UWhat courses do you offer?"))
        frames = recv_until(ws, "audio_end")
    kinds = events(frames)
    assert frames[0] == {"type": "transcript", "final": True, "text": "What courses do you offer?"}
    assert kinds.index("tool") < kinds.index("assistant_delta") < kinds.index("audio_start")
    assert "assistant_done" in kinds
    audio = b"".join(f for f in frames if isinstance(f, bytes)).decode()
    assert audio.replace(" ", "") == "<WeofferB.TechandM.Tech.><FeesareRs.1.2lakh.>"
    assert agent.calls == ["What courses do you offer?"]
    assert stt.closed and tts.closed


def test_resume_only_owned_conversation(setup):
    client, _, _ = setup(Agent([]))
    with client.websocket_connect("/ws/voice", headers=ORIGIN) as ws:
        ws.send_json({"type": "start", "conversation_id": "conv-mine"})
        assert ws.receive_json()["conversation_id"] == "conv-mine"
        ws.send_json({"type": "start", "conversation_id": "conv-someone-else"})
        assert ws.receive_json()["conversation_id"] == "conv-new"


def test_typed_text_runs_turn_without_echo(setup):
    agent = Agent(["Hello there."])
    client, _, _ = setup(agent)
    with client.websocket_connect("/ws/voice", headers=ORIGIN) as ws:
        ws.send_json({"type": "start", "conversation_id": None})
        ws.receive_json()
        ws.send_json({"type": "text", "text": "  hi  "})
        frames = recv_until(ws, "audio_end")
    assert "transcript" not in events(frames)
    assert agent.calls == ["hi"]


def test_barge_in_cancels_turn(setup):
    agent = Agent(["Slow first part. ", "More words ", "keep coming ", "for a while. "] * 5, 0.05)
    client, _, tts = setup(agent)
    with client.websocket_connect("/ws/voice", headers=ORIGIN) as ws:
        ws.send_json({"type": "start", "conversation_id": None})
        ws.receive_json()
        ws.send_json({"type": "text", "text": "tell me everything"})
        recv_until(ws, "assistant_delta")
        ws.send_bytes(frame(b"Iuh"))  # one-word blip: ignored, not forwarded
        ws.send_bytes(frame(b"Iwait what about hostels"))
        frames = recv_until(ws, "interrupted")
        assert {"type": "transcript", "final": False, "text": "uh"} not in frames
        assert ws.receive_json() == {
            "type": "transcript",
            "final": False,
            "text": "wait what about hostels",
        }
        ws.send_bytes(frame(b"Uwait what about hostels"))
        later = recv_until(ws, "transcript")
        assert later[-1]["final"] is True
        recv_until(ws, "assistant_delta")
    assert agent.cancelled >= 1 and tts.resets >= 1
    assert agent.calls == ["tell me everything", "wait what about hostels"]


def test_echo_of_reply_does_not_barge_in(setup):
    agent = Agent(["Slow start. ", "Admissions are open now. "] + ["Still talking. "] * 10, 0.05)
    client, _, _ = setup(agent)
    with client.websocket_connect("/ws/voice", headers=ORIGIN) as ws:
        ws.send_json({"type": "start", "conversation_id": None})
        ws.receive_json()
        ws.send_json({"type": "text", "text": "admissions?"})
        recv_until(ws, "assistant_delta")
        recv_until(ws, "assistant_delta")
        ws.send_bytes(frame(b"Iadmissions are open"))
        frames = recv_until(ws, "audio_end")
    assert "interrupted" not in events(frames) and "transcript" not in events(frames)
    assert len(agent.calls) == 1


def test_client_interrupt_and_typed_text_interrupts(setup):
    agent = Agent(["Slow words. "] * 20, 0.05)
    client, _, _ = setup(agent)
    with client.websocket_connect("/ws/voice", headers=ORIGIN) as ws:
        ws.send_json({"type": "start", "conversation_id": None})
        ws.receive_json()
        ws.send_json({"type": "text", "text": "one"})
        recv_until(ws, "assistant_delta")
        ws.send_json({"type": "text", "text": "two"})
        recv_until(ws, "interrupted")
        recv_until(ws, "assistant_delta")
        ws.send_json({"type": "interrupt"})
        recv_until(ws, "interrupted")
    assert agent.calls == ["one", "two"] and agent.cancelled == 2


def test_audio_in_order_and_pipelined(setup):
    agent = Agent(["Slow sentence one. ", "Fast sentence two. ", "Three."])
    client, _, tts = setup(agent)
    with client.websocket_connect("/ws/voice", headers=ORIGIN) as ws:
        ws.send_json({"type": "start", "conversation_id": None})
        ws.receive_json()
        ws.send_json({"type": "text", "text": "go"})
        frames = recv_until(ws, "audio_end")
    audio = b"".join(f for f in frames if isinstance(f, bytes)).decode()
    assert audio.index("Slow") < audio.index("Fast") < audio.index("Three")
    first_end = frames.index(next(f for f in frames if isinstance(f, bytes) and b">" in f))
    assert events(frames).count("audio_start") == 1
    # sentence two started synthesising before sentence one's audio finished
    assert tts.started[1][1] - tts.started[0][1] < 0.15 and first_end > 0


def test_unauthenticated_socket_rejected(setup, monkeypatch):
    client, _, _ = setup(Agent([]))

    async def no_session(ws):
        return None

    monkeypatch.setattr(auth, "session_from_websocket", no_session)
    with (
        client.websocket_connect("/ws/voice", headers=ORIGIN) as ws,
        pytest.raises(WebSocketDisconnect) as exc,
    ):
        ws.receive_json()
    assert exc.value.code == 4401


def test_bad_origin_rejected(setup):
    client, _, _ = setup(Agent([]))
    with (
        client.websocket_connect("/ws/voice", headers={"origin": "https://evil.example"}) as ws,
        pytest.raises(WebSocketDisconnect) as exc,
    ):
        ws.receive_json()
    assert exc.value.code == 4403


def test_bad_messages_and_audio_before_start(setup):
    client, stt, _ = setup(Agent([]))
    with client.websocket_connect("/ws/voice", headers=ORIGIN) as ws:
        ws.send_bytes(frame(b"Uignored"))
        ws.send_text("not json")
        assert ws.receive_json() == {"type": "error", "message": "Invalid JSON."}
        ws.send_text("x" * 10_000)
        assert ws.receive_json() == {"type": "error", "message": "Message too large."}
        ws.send_json({"type": "text", "text": "hi"})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "start", "conversation_id": None})
        assert ws.receive_json()["type"] == "ready"
    assert stt.frames == 0


def test_agent_error_becomes_error_event(setup):
    async def broken(ctx, cid, text):
        yield {"type": "text_delta", "text": "Partial "}
        raise RuntimeError("boom")

    client, _, _ = setup(broken)
    with client.websocket_connect("/ws/voice", headers=ORIGIN) as ws:
        ws.send_json({"type": "start", "conversation_id": None})
        ws.receive_json()
        ws.send_json({"type": "text", "text": "hi"})
        frames = recv_until(ws, "error")
        assert frames[-1]["message"] == "Sorry, something went wrong."
        ws.send_json({"type": "start", "conversation_id": "conv-mine"})
        assert recv_until(ws, "ready")[-1]["conversation_id"] == "conv-mine"


SOURCES = [
    {
        "kind": "document",
        "ref": "D2:6.4.3",
        "document_title": "Handbook",
        "version": "2026",
        "section": "6.4.3",
        "heading": "On-duty",
        "page_start": 4,
        "page_end": 4,
    },
    {"kind": "college_record", "label": "College system: attendance"},
]


def cited_agent(delay: float = 0.0):
    """run_turn that emits citations (and an unknown future event) just before done."""
    calls: list[str] = []

    async def run(ctx, cid, text):
        calls.append(text)
        yield {"type": "text_delta", "text": "On-duty needs proof. "}
        await asyncio.sleep(delay)
        yield {"type": "something_new", "x": 1}
        yield {"type": "citations", "sources": SOURCES}
        yield {"type": "done", "text": "On-duty needs proof.", "citations": SOURCES}

    run.calls = calls
    return run


@pytest.mark.parametrize("spoken", [True, False])
def test_cit2_citations_forwarded_before_done(setup, spoken):
    client, _, tts = setup(cited_agent())
    with client.websocket_connect("/ws/voice", headers=ORIGIN) as ws:
        ws.send_json({"type": "start", "conversation_id": None})
        ws.receive_json()
        if spoken:
            ws.send_bytes(frame(b"UWhat about on duty?"))
        else:
            ws.send_json({"type": "text", "text": "What about on duty?"})
        frames = recv_until(ws, "audio_end")
    kinds = events(frames)
    assert {"type": "citations", "sources": SOURCES} in frames
    assert kinds.index("citations") < kinds.index("assistant_done")
    assert "something_new" not in kinds  # unknown events are ignored, not forwarded
    spoken_text = " ".join(t for t, _ in tts.started)
    assert "D2" not in spoken_text and "College system" not in spoken_text


def test_cit2_citations_dropped_when_interrupted(setup):
    client, _, _ = setup(cited_agent(delay=0.5))
    with client.websocket_connect("/ws/voice", headers=ORIGIN) as ws:
        ws.send_json({"type": "start", "conversation_id": None})
        ws.receive_json()
        ws.send_json({"type": "text", "text": "on duty?"})
        recv_until(ws, "assistant_delta")
        ws.send_json({"type": "interrupt"})
        frames = recv_until(ws, "interrupted")
        ws.send_json({"type": "start", "conversation_id": "conv-mine"})
        frames += recv_until(ws, "ready")
    assert "citations" not in events(frames)


def test_ac6_turn_after_logout_is_refused_and_socket_closed(setup, monkeypatch):
    agent = Agent(["Hello."])
    client, _, tts = setup(agent)
    alive = [True]

    async def session(ws):
        return CTX if alive[0] else None

    monkeypatch.setattr(auth, "session_from_websocket", session)
    with client.websocket_connect("/ws/voice", headers=ORIGIN) as ws:
        ws.send_json({"type": "start", "conversation_id": None})
        ws.receive_json()
        ws.send_json({"type": "text", "text": "first"})
        recv_until(ws, "audio_end")
        alive[0] = False  # logged out elsewhere
        ws.send_json({"type": "text", "text": "second"})
        err = recv_until(ws, "error")[-1]  # `interrupted` may come first: audio is playing
        assert "sign in" in err["message"]
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
    assert exc.value.code == 4401
    assert agent.calls == ["first"] and len(tts.started) == 1


def test_ac6_idle_socket_closed_after_logout(setup, monkeypatch):
    monkeypatch.setattr(voice, "WATCHDOG_INTERVAL_S", 0.01)
    monkeypatch.setattr(voice, "SESSION_RECHECK_S", 0.0)
    client, _, _ = setup(Agent([]))
    alive = [True]

    async def session(ws):
        return CTX if alive[0] else None

    monkeypatch.setattr(auth, "session_from_websocket", session)
    with client.websocket_connect("/ws/voice", headers=ORIGIN) as ws:
        ws.send_json({"type": "start", "conversation_id": None})
        ws.receive_json()
        alive[0] = False
        assert ws.receive_json()["type"] == "error"
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
    assert exc.value.code == 4401


COURSE_REPLY = [
    "Slow: KIT offers B.Tech in computer science, electronics and mechanical. ",
    "The fee for computer science is one lakh twenty five thousand per year. ",
    "What else would you like to know about the hostel? ",
] + ["The hostel fee is sixty thousand. "] * 10


def test_vox2_follow_up_made_of_reply_words_interrupts_and_becomes_a_turn(setup):
    agent = Agent(COURSE_REPLY, 0.05)
    client, _, _ = setup(agent)
    with client.websocket_connect("/ws/voice", headers=ORIGIN) as ws:
        ws.send_json({"type": "start", "conversation_id": None})
        ws.receive_json()
        ws.send_json({"type": "text", "text": "What courses do you offer and what are the fees?"})
        recv_until(ws, "assistant_delta")
        recv_until(ws, "assistant_delta")
        recv_until(ws, "assistant_delta")
        ws.send_bytes(frame(b"Iwhat is the fee for"))
        frames = recv_until(ws, "interrupted")
        assert {"type": "transcript", "final": False, "text": "what is the fee for"} in frames + [
            ws.receive_json()
        ]
        ws.send_bytes(frame(b"Uwhat is the fee for computer science"))
        later = recv_until(ws, "assistant_delta")
    assert {
        "type": "transcript",
        "final": True,
        "text": "what is the fee for computer science",
    } in later
    assert agent.calls[-1] == "what is the fee for computer science"


def test_vox2_final_follow_up_without_interim_is_not_dropped(setup):
    agent = Agent(COURSE_REPLY, 0.05)
    client, _, _ = setup(agent)
    with client.websocket_connect("/ws/voice", headers=ORIGIN) as ws:
        ws.send_json({"type": "start", "conversation_id": None})
        ws.receive_json()
        ws.send_json({"type": "text", "text": "courses?"})
        for _ in range(3):
            recv_until(ws, "assistant_delta")
        ws.send_bytes(frame(b"Uwhat about the hostel fee"))
        frames = recv_until(ws, "transcript")
        recv_until(ws, "assistant_delta")
    assert frames[-1] == {"type": "transcript", "final": True, "text": "what about the hostel fee"}
    assert agent.calls[-1] == "what about the hostel fee"


def test_vox2_verbatim_echo_of_the_reply_is_still_ignored(setup):
    agent = Agent(COURSE_REPLY, 0.05)
    client, _, _ = setup(agent)
    with client.websocket_connect("/ws/voice", headers=ORIGIN) as ws:
        ws.send_json({"type": "start", "conversation_id": None})
        ws.receive_json()
        ws.send_json({"type": "text", "text": "courses?"})
        for _ in range(3):
            recv_until(ws, "assistant_delta")
        ws.send_bytes(frame(b"Ithe fee for computer science is one lakh"))
        ws.send_bytes(frame(b"Uthe fee for computer science is one lakh"))
        frames = recv_until(ws, "audio_end", limit=400)
    assert "interrupted" not in events(frames) and "transcript" not in events(frames)
    assert len(agent.calls) == 1


@pytest.mark.parametrize("word", ["stop", "wait", "no", "okay", "Priya"])
def test_vox2_one_real_word_interrupts(setup, word):
    agent = Agent(["Slow words. "] * 20, 0.05)
    client, _, _ = setup(agent)
    with client.websocket_connect("/ws/voice", headers=ORIGIN) as ws:
        ws.send_json({"type": "start", "conversation_id": None})
        ws.receive_json()
        ws.send_json({"type": "text", "text": "talk"})
        recv_until(ws, "assistant_delta")
        ws.send_bytes(frame(b"I" + word.encode()))
        recv_until(ws, "interrupted")
    assert agent.cancelled == 1


@pytest.mark.parametrize("sound", sorted(voice.FILLERS) + ["uh um", "Hmm."])
def test_vox2_bare_hesitation_does_not_interrupt(setup, sound):
    agent = Agent(["Slow start. "] + ["Still talking. "] * 6, 0.05)
    client, _, _ = setup(agent)
    with client.websocket_connect("/ws/voice", headers=ORIGIN) as ws:
        ws.send_json({"type": "start", "conversation_id": None})
        ws.receive_json()
        ws.send_json({"type": "text", "text": "talk"})
        recv_until(ws, "assistant_delta")
        ws.send_bytes(frame(b"I" + sound.encode()))
        frames = recv_until(ws, "audio_end")
    assert "interrupted" not in events(frames) and agent.cancelled == 0
