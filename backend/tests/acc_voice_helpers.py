"""Shared fakes and a real in-process server for the voice acceptance tests.

The server is a real uvicorn instance on the test loop (so the real auth/history code and the
shared asyncpg pool are used); clients are real WebSocket clients speaking to 127.0.0.1.
Deepgram and the agent are replaced by the fakes below. Nothing here decides what is "right":
assertions live in the tests, derived from ACCEPTANCE.md and CONTRACT.md.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import uvicorn
import websockets
from app import auth, history
from app.db import pool
from app.voice import session as voice
from app.voice.deepgram import SttEvent
from fastapi import FastAPI
from websockets.exceptions import ConnectionClosed, InvalidHandshake

ORIGIN = "http://localhost:8000"

# Values planted in sessions and fake exceptions; none may ever reach a client frame.
DG_KEY = "test-deepgram-key-not-real"
SECRET_KIT_TOKEN = "KITJWT.secret-college-token.XYZ"
OTHER_KEYS = (
    "test-participant-key-not-real",
    "test-anthropic-key-not-real",
    "test-gemini-key-not-real",
)


# --------------------------------------------------------------------------- sessions


async def make_session(
    role: str = "visitor",
    login: str | None = None,
    name: str | None = None,
    kit_expires_in: int | None = None,
) -> tuple[str, auth.SessionCtx]:
    """Insert a real session row; returns (cookie token, ctx). Users own `user:<login>`."""
    if role == "visitor":
        return await auth._create_session(f"visitor:{uuid.uuid4().hex[:12]}", "visitor")
    login = login or f"{role[0]}{uuid.uuid4().hex[:6]}"
    user = {
        "id": f"id-{login}",
        "login_id": login,
        "full_name": name or f"Name {login}",
        "role": role,
        "department": "CSE",
    }
    return await auth._create_session(
        f"user:{login}", role, user, SECRET_KIT_TOKEN, kit_expires_in or 3600
    )


async def expire_session(ctx: auth.SessionCtx) -> None:
    """Make a stored session expired."""
    await pool().execute(
        "UPDATE sessions SET expires_at = now() - interval '1 minute' WHERE id = $1",
        ctx.session_id,
    )


async def expire_kit_token(ctx: auth.SessionCtx) -> None:
    """Make the college token of a stored session expired."""
    await pool().execute(
        "UPDATE sessions SET kit_token_expires_at = $2 WHERE id = $1",
        ctx.session_id,
        datetime.now(UTC) - timedelta(minutes=5),
    )


async def history_rows(cid: str) -> list[tuple[str, str]]:
    """(role, content) of every stored message of a conversation, oldest first."""
    rows = await pool().fetch("SELECT role, content FROM messages WHERE conversation_id=$1 ORDER BY id", uuid.UUID(cid))
    return [(r["role"], r["content"]) for r in rows]


# --------------------------------------------------------------------------- fake STT


class FakeStt:
    """Fake speech-to-text. Mic frames are scripts: b'S' speech started, b'I<text>' interim,
    b'U<text>' utterance, b'E<text>' error. Anything else is just recorded."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue()
        self.closed = 0
        self.frames: list[bytes] = []
        self.fail_send: Exception | None = None

    async def send_audio(self, pcm: bytes) -> None:
        if self.fail_send is not None:
            raise self.fail_send
        self.frames.append(pcm)
        kind = {"S": "speech_started", "I": "interim", "U": "utterance", "E": "error"}.get(
            chr(pcm[0]) if pcm else ""
        )
        if kind:
            self.queue.put_nowait(SttEvent(kind, pcm[1:].decode().strip()))

    def push(self, event: SttEvent | Exception) -> None:
        """Queue an event; an Exception is raised out of events() when reached."""
        self.queue.put_nowait(event)

    async def events(self):
        while True:
            item = await self.queue.get()
            if isinstance(item, Exception):
                raise item
            yield item

    async def close(self) -> None:
        self.closed += 1


def frame(script: bytes) -> bytes:
    """Pad a scripted fake-mic frame to whole 16-bit samples."""
    return script + b" " * (len(script) % 2)


# --------------------------------------------------------------------------- fake TTS


class FakeTts:
    """Fake synthesiser mimicking the real one: synthesis starts at synthesize(), not at first
    iteration, and audio comes back as chunks of `<text>` bytes.

    delay(text) -> seconds before the first chunk; fail(text, n) -> exception to raise instead
    of audio (n = how many sentences were requested before this one); hang(text) -> never
    produce anything.
    """

    def __init__(
        self,
        delay: Callable[[str], float] | None = None,
        fail: Callable[[str, int], Exception | None] | None = None,
        hang: Callable[[str], bool] | None = None,
        chunks: int = 2,
        chunk_delay: float = 0.0,
        chunk_bytes: int | None = None,
    ) -> None:
        self.delay = delay or (lambda t: 0.01)
        self.fail = fail or (lambda t, n: None)
        self.hang = hang or (lambda t: False)
        self.chunks = chunks
        self.chunk_delay = chunk_delay
        self.chunk_bytes = chunk_bytes
        self.requested: list[str] = []
        self.completed: list[str] = []
        self.abandoned: list[str] = []
        self.resets = 0
        self.warms = 0
        self.closed = 0
        self._tasks: set[asyncio.Task] = set()

    async def warm(self) -> None:
        self.warms += 1

    def synthesize(self, text: str):
        n = len(self.requested)
        self.requested.append(text)
        q: asyncio.Queue = asyncio.Queue()

        async def produce() -> None:
            try:
                if self.hang(text):
                    await asyncio.sleep(3600)
                await asyncio.sleep(self.delay(text))
                if (exc := self.fail(text, n)) is not None:
                    q.put_nowait(exc)
                    return
                if self.chunk_bytes:
                    parts = [b"\x01\x00" * (self.chunk_bytes // 2) for _ in range(self.chunks)]
                else:
                    data = f"<{text}>".encode()
                    data += b" " * (len(data) % 2)
                    step = max(2, (len(data) // self.chunks) // 2 * 2)
                    parts = [data[i : i + step] for i in range(0, len(data), step)]
                for part in parts:
                    q.put_nowait(part)
                    if self.chunk_delay:
                        await asyncio.sleep(self.chunk_delay)
                q.put_nowait(None)
            except asyncio.CancelledError:
                self.abandoned.append(text)
                raise

        task = asyncio.create_task(produce())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

        async def gen():
            try:
                while (item := await q.get()) is not None:
                    if isinstance(item, Exception):
                        raise item
                    yield item
                self.completed.append(text)
            finally:
                if not task.done():
                    task.cancel()

        return gen()

    def _cancel_all(self) -> None:
        for t in list(self._tasks):
            t.cancel()

    async def reset(self) -> None:
        self.resets += 1
        self._cancel_all()

    async def close(self) -> None:
        self.closed += 1
        self._cancel_all()


# --------------------------------------------------------------------------- fake agent


class Agent:
    """Fake run_turn following the contract: saves the user message first, streams events,
    saves the reply on done (or the partial reply on cancellation).

    `script(text)` returns steps: an event dict, ("sleep", seconds), ("raise", exc),
    ("hang",). Default: tool start/end then the deltas, then done.
    """

    def __init__(
        self,
        deltas: list[str] | None = None,
        delay: float = 0.0,
        script: Callable[[str], list] | None = None,
        save: bool = True,
        tool: bool = True,
    ) -> None:
        self.deltas = deltas if deltas is not None else ["Hello there."]
        self.delay = delay
        self.script = script
        self.save = save
        self.tool = tool
        self.calls: list[tuple[auth.SessionCtx, str, str]] = []
        self.active = 0
        self.max_active = 0
        self.stopped_early: list[str] = []  # texts whose turn ended without `done`
        self.completed: list[str] = []

    @property
    def texts(self) -> list[str]:
        return [c[2] for c in self.calls]

    def _steps(self, text: str) -> list:
        if self.script:
            return self.script(text)
        steps: list = []
        if self.tool:
            steps.append({"type": "tool", "name": "list_courses", "state": "start", "is_error": False})
            steps.append({"type": "tool", "name": "list_courses", "state": "end", "is_error": False})
        for d in self.deltas:
            steps += [("sleep", self.delay), {"type": "text_delta", "text": d}]
        steps.append({"type": "done", "text": "".join(self.deltas)})
        return steps

    async def __call__(self, ctx, conversation_id, text):
        self.calls.append((ctx, conversation_id, text))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        said = ""
        done = False
        try:
            if self.save:
                await history.add_message(conversation_id, "user", text)
            for step in self._steps(text):
                if isinstance(step, tuple):
                    if step[0] == "sleep":
                        if step[1]:
                            await asyncio.sleep(step[1])
                    elif step[0] == "raise":
                        raise step[1]
                    elif step[0] == "hang":
                        await asyncio.sleep(3600)
                    continue
                if step.get("type") == "text_delta":
                    said += step["text"]
                if step.get("type") == "done":
                    done = True
                    if self.save:
                        await history.add_message(conversation_id, "assistant", step["text"])
                yield step
            if done:
                self.completed.append(text)
        finally:
            self.active -= 1
            if not done:
                self.stopped_early.append(text)
                if self.save and said:
                    with contextlib.suppress(Exception):
                        await asyncio.shield(history.add_message(conversation_id, "assistant", said))


def echo_history_agent() -> Agent:
    """An agent that answers with everything stored in the conversation it was given (so any
    cross-conversation leak shows up in frames)."""

    async def run(ctx, cid, text):
        await history.add_message(cid, "user", text)
        stored = await history.recent_messages(cid)
        reply = " ".join(f"{m['content']}." for m in stored)
        yield {"type": "text_delta", "text": reply}
        yield {"type": "done", "text": reply}
        await history.add_message(cid, "assistant", reply)

    return run  # type: ignore[return-value]


# --------------------------------------------------------------------------- server


class VoiceServer:
    """A real uvicorn server on a free local port, serving only the voice router."""

    def __init__(self, run_turn, stt=None, tts=None) -> None:
        self.run_turn = run_turn
        self.stts: list[FakeStt] = []
        self.ttss: list[FakeTts] = []
        self._stt_make = stt or FakeStt
        self._tts_make = tts or FakeTts
        self.factory_calls = 0
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task | None = None
        self.port = 0

    def _new_stt(self):
        self.factory_calls += 1
        s = self._stt_make()
        self.stts.append(s)
        return s

    def _new_tts(self):
        t = self._tts_make()
        self.ttss.append(t)
        return t

    async def start(self) -> None:
        app = FastAPI()
        app.include_router(voice.build_router(self.run_turn, self._new_stt, self._new_tts))
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        cfg = uvicorn.Config(
            app, lifespan="off", log_level="error", ws_ping_interval=None, timeout_graceful_shutdown=2
        )
        self._server = uvicorn.Server(cfg)
        self._task = asyncio.create_task(self._server.serve(sockets=[sock]))
        for _ in range(200):
            if self._server.started:
                return
            await asyncio.sleep(0.02)
        raise RuntimeError("test server did not start")

    async def stop(self) -> None:
        if self._server and self._task:
            self._server.should_exit = True
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._task, 8)

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/ws/voice"

    async def connect(self, token: str | None = None, origin: str | None = ORIGIN, cookie: str | None = None) -> "Client":
        """Open a client. `token` is a session cookie token; `cookie` a raw cookie value."""
        headers = {}
        value = cookie if cookie is not None else token
        if value is not None:
            headers["Cookie"] = f"kit_session={value}"
        kwargs: dict[str, Any] = {}
        if origin is not None:
            kwargs["origin"] = origin
        ws = await websockets.connect(
            self.url, additional_headers=headers, max_size=None, ping_interval=None, open_timeout=5, **kwargs
        )
        return Client(ws)


class Client:
    """A WebSocket client that remembers every frame it received."""

    def __init__(self, ws) -> None:
        self.ws = ws
        self.frames: list[dict | bytes] = []

    async def send_json(self, obj: Any) -> None:
        await self.ws.send(json.dumps(obj))

    async def send_text(self, text: str) -> None:
        await self.ws.send(text)

    async def send_bytes(self, data: bytes) -> None:
        await self.ws.send(data)

    async def mic(self, script: bytes) -> None:
        await self.ws.send(frame(script))

    async def recv(self, timeout: float = 5.0) -> dict | bytes:
        raw = await asyncio.wait_for(self.ws.recv(), timeout)
        item = raw if isinstance(raw, bytes) else json.loads(raw)
        self.frames.append(item)
        return item

    async def until(self, kind: str, timeout: float = 8.0) -> list:
        """Receive frames up to and including the first JSON event of this type."""
        out: list = []
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise AssertionError(f"no {kind!r} event; got {types(out)}")
            item = await self.recv(left)
            out.append(item)
            if isinstance(item, dict) and item.get("type") == kind:
                return out

    async def drain(self, quiet: float = 0.4, limit: float = 8.0) -> list:
        """Receive until nothing arrives for `quiet` seconds (or the socket closes)."""
        out: list = []
        end = time.monotonic() + limit
        while time.monotonic() < end:
            try:
                out.append(await self.recv(quiet))
            except (TimeoutError, ConnectionClosed):
                break
        return out

    async def start(self, cid: str | None = None) -> dict:
        await self.send_json({"type": "start", "conversation_id": cid})
        ready = await self.recv()
        assert isinstance(ready, dict) and ready["type"] == "ready", ready
        return ready

    async def close(self) -> None:
        with contextlib.suppress(Exception):
            await self.ws.close()

    def abort(self) -> None:
        """Drop the TCP connection without a closing handshake."""
        self.ws.transport.abort()

    async def __aenter__(self) -> "Client":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()


async def expect_refused(server: VoiceServer, **kw) -> None:
    """Assert a connection yields no data at all: the handshake fails or the socket closes
    without a single frame, even when the client pushes a `start`."""
    try:
        c = await server.connect(**kw)
    except (InvalidHandshake, ConnectionClosed, OSError):
        return
    try:
        with contextlib.suppress(ConnectionClosed):
            await c.send_json({"type": "start", "conversation_id": None})
            await c.send_json({"type": "text", "text": "hello"})
        try:
            frame_ = await c.recv(3)
        except ConnectionClosed:
            return
        raise AssertionError(f"refused connection still received data: {frame_!r}")
    finally:
        await c.close()


async def close_code(c: Client, timeout: float = 5.0) -> int:
    """Wait for the server to close the socket; return the close code."""
    try:
        while True:
            item = await c.recv(timeout)
            raise AssertionError(f"expected close, got a frame: {item!r}")
    except ConnectionClosed as exc:
        assert exc.rcvd is not None, "no close frame received"
        return exc.rcvd.code


# --------------------------------------------------------------------------- frame checks


def types(frames: list) -> list[str]:
    """Event type names (binary frames as 'audio')."""
    return [f["type"] if isinstance(f, dict) else "audio" for f in frames]


def events_of(frames: list, kind: str) -> list[dict]:
    return [f for f in frames if isinstance(f, dict) and f.get("type") == kind]


def audio_bytes(frames: list) -> bytes:
    return b"".join(f for f in frames if isinstance(f, bytes))


def check_audio_protocol(frames: list) -> None:
    """Binary frames only between audio_start and audio_end (or an `interrupted`), never nested."""
    open_ = False
    for f in frames:
        if isinstance(f, bytes):
            assert open_, "binary audio outside audio_start/audio_end"
            continue
        t = f["type"]
        if t == "audio_start":
            assert not open_, "audio_start while audio already open"
            open_ = True
        elif t == "audio_end":
            assert open_, "audio_end without audio_start"
            open_ = False
        elif t == "interrupted":
            open_ = False


def all_text(frames: list) -> str:
    """Every text frame, concatenated (for secret scanning)."""
    return json.dumps([f for f in frames if isinstance(f, dict)])


async def wait_for(cond: Callable[[], bool], timeout: float = 5.0, what: str = "condition") -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


async def settle_tasks(before: set[asyncio.Task], timeout: float = 5.0) -> set[asyncio.Task]:
    """Wait until every task created since `before` is done; return the leftovers."""
    end = time.monotonic() + timeout
    leftover: set[asyncio.Task] = set()
    while time.monotonic() < end:
        leftover = {t for t in asyncio.all_tasks() - before if not t.done()}
        if not leftover:
            return leftover
        await asyncio.sleep(0.05)
    return leftover


# --------------------------------------------------------------------------- fixture

import pytest_asyncio  # noqa: E402


@pytest_asyncio.fixture(loop_scope="session")
async def make_server():
    """Factory: ``await make_server(agent, stt=..., tts=...)`` starts a real server."""
    from app import limits

    limits.reset_all()
    servers: list[VoiceServer] = []

    async def make(run_turn=None, stt=None, tts=None) -> VoiceServer:
        s = VoiceServer(run_turn or Agent(), stt, tts)
        await s.start()
        servers.append(s)
        return s

    yield make
    for s in servers:
        await s.stop()
    limits.reset_all()


@pytest_asyncio.fixture(loop_scope="session")
async def relaxed_limits(monkeypatch):
    """Lift the socket caps (all test sockets share 127.0.0.1 and some tests reuse one login)."""
    from app import limits

    for name in ("limit_ws_per_ip", "limit_ws_per_user", "limit_ws_total"):
        monkeypatch.setitem(limits.DEFAULTS, name, 1000)
    yield
