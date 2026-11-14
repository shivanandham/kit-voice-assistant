"""Deepgram clients: streaming speech-to-text (Nova-3) and low-latency text-to-speech (Aura-2).

Both talk to Deepgram over plain `websockets`/`httpx`; the API key never leaves the server.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlencode

import httpx
import websockets
from websockets.exceptions import ConnectionClosed

log = logging.getLogger(__name__)

# --- speech-to-text tuning -------------------------------------------------------------------
STT_URL = "wss://api.deepgram.com/v1/listen"
STT_SAMPLE_RATE = 16000
# Silence (ms) after speech before Deepgram marks a segment speech_final. 300 ms ends a turn
# quickly without cutting people off at short pauses; UtteranceEnd is the safety net when
# background noise keeps endpointing from firing.
STT_ENDPOINTING_MS = 300
STT_UTTERANCE_END_MS = 1000  # Deepgram advises >= 1000 (interim results come about every second)
STT_KEEPALIVE_S = 4.0  # Deepgram closes a stream after ~10 s without audio
STT_IDLE_CLOSE_S = 60.0  # no audio for this long: close; the next frame reconnects lazily
STT_KEYTERMS = (
    "Knotopian",
    "KIT",
    "Knotopian Institute of Technology",
    "on-duty",
    "on duty",
    "CGPA",
    "arrears",
    "DBMS",
    "B.Tech",
    "M.Tech",
    "CSE",
    "ECE",
    "MECH",
    "BTECH-CSE",
    "BTECH-ECE",
    "BTECH-MECH",
    "MTECH-CSE",
    "Zoho",
    "Freshworks",
    "TCS",
    "placement drive",
    "leave request",
    "timetable",
    "attendance",
    "hostel",
)

# --- text-to-speech tuning -------------------------------------------------------------------
TTS_WS_URL = "wss://api.deepgram.com/v1/speak"
TTS_REST_URL = "https://api.deepgram.com/v1/speak"
TTS_SAMPLE_RATE = 24000
# The speak WebSocket allows 20 Flush messages per 60 s; beyond our budget we use REST.
TTS_FLUSH_BUDGET = 18
TTS_FLUSH_WINDOW_S = 60.0
TTS_MAX_CHARS = 1900  # Aura-2 limit is 2000 characters per request


def stt_url(model: str, keyterms: tuple[str, ...] = STT_KEYTERMS) -> str:
    """Build the live listen URL with our endpointing settings and keyterm prompts."""
    params: list[tuple[str, str]] = [
        ("model", model),
        ("encoding", "linear16"),
        ("sample_rate", str(STT_SAMPLE_RATE)),
        ("channels", "1"),
        ("interim_results", "true"),
        ("endpointing", str(STT_ENDPOINTING_MS)),
        ("utterance_end_ms", str(STT_UTTERANCE_END_MS)),
        ("vad_events", "true"),
        ("smart_format", "true"),
    ]
    params += [("keyterm", k) for k in keyterms]
    return f"{STT_URL}?{urlencode(params)}"


@dataclass(frozen=True)
class SttEvent:
    """One speech-to-text event.

    kind: speech_started (VAD), interim (text so far, not final), utterance (a complete user
    turn), error (connection lost for good). `text` is the whole utterance so far.
    """

    kind: Literal["speech_started", "interim", "utterance", "error"]
    text: str = ""


Connector = Callable[..., Any]


class DeepgramSTT:
    """A lazily connected live-transcription stream that assembles utterances from segments."""

    def __init__(self, api_key: str, model: str, connect: Connector = websockets.connect) -> None:
        """Prepare the client; nothing connects until the first audio frame."""
        self._api_key = api_key
        self._url = stt_url(model)
        self._connect = connect
        self._ws: Any = None
        self._lock = asyncio.Lock()
        self._events: asyncio.Queue[SttEvent] = asyncio.Queue()
        self._segments: list[str] = []
        self._reader: asyncio.Task | None = None
        self._keeper: asyncio.Task | None = None
        self._last_audio = 0.0
        self._closing = False

    async def _open(self) -> None:
        self._ws = await self._connect(
            self._url,
            additional_headers={"Authorization": f"Token {self._api_key}"},
            max_size=2**20,
            open_timeout=5,
        )
        self._reader = asyncio.create_task(self._read(self._ws))
        if self._keeper is None or self._keeper.done():
            self._keeper = asyncio.create_task(self._keepalive())

    async def _ensure(self) -> Any:
        async with self._lock:
            if self._ws is None:
                await self._open()
            return self._ws

    async def send_audio(self, pcm: bytes) -> None:
        """Send one frame of 16 kHz mono PCM16, connecting (or reconnecting once) as needed."""
        if self._closing:
            return
        self._last_audio = time.monotonic()
        ws = await self._ensure()
        try:
            await ws.send(pcm)
        except ConnectionClosed:
            await self._drop(ws)
            ws = await self._ensure()
            await ws.send(pcm)

    async def events(self) -> AsyncIterator[SttEvent]:
        """Yield transcription events until the client is closed."""
        while True:
            event = await self._events.get()
            if event.kind == "error" and self._closing:
                return
            yield event

    async def _drop(self, ws: Any) -> None:
        async with self._lock:
            if self._ws is ws:
                self._ws = None
        with contextlib.suppress(Exception):
            await ws.close()

    async def _read(self, ws: Any) -> None:
        try:
            async for raw in ws:
                if isinstance(raw, str):
                    self._handle(json.loads(raw))
        except ConnectionClosed:
            pass
        except Exception:  # malformed message: drop the connection, reconnect below
            log.exception("deepgram stt reader failed")
        if self._closing or self._ws is not ws:
            return
        await self._drop(ws)
        if time.monotonic() - self._last_audio < STT_IDLE_CLOSE_S:
            try:  # reconnect once while the user is still talking
                await self._ensure()
            except Exception:
                log.warning("deepgram stt reconnect failed", exc_info=True)
                self._events.put_nowait(SttEvent("error", "Speech recognition disconnected."))

    def _emit_utterance(self) -> None:
        text = " ".join(self._segments).strip()
        self._segments.clear()
        if text:
            self._events.put_nowait(SttEvent("utterance", text))

    def _handle(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "SpeechStarted":
            self._events.put_nowait(SttEvent("speech_started"))
        elif kind == "UtteranceEnd":
            self._emit_utterance()
        elif kind == "Results":
            alts = (msg.get("channel") or {}).get("alternatives") or [{}]
            text = (alts[0].get("transcript") or "").strip()
            if msg.get("is_final"):
                if text:
                    self._segments.append(text)
                if self._segments:
                    self._events.put_nowait(SttEvent("interim", " ".join(self._segments)))
                if msg.get("speech_final"):
                    self._emit_utterance()
            elif text:
                self._events.put_nowait(SttEvent("interim", " ".join([*self._segments, text])))

    async def _keepalive(self) -> None:
        while not self._closing:
            await asyncio.sleep(STT_KEEPALIVE_S)
            ws = self._ws
            if ws is None:
                continue
            idle = time.monotonic() - self._last_audio
            try:
                if idle > STT_IDLE_CLOSE_S:
                    self._segments.clear()
                    await ws.send(json.dumps({"type": "CloseStream"}))
                    await self._drop(ws)
                elif idle >= STT_KEEPALIVE_S:
                    await ws.send(json.dumps({"type": "KeepAlive"}))
            except ConnectionClosed:
                await self._drop(ws)

    async def close(self) -> None:
        """Close the stream cleanly and stop background tasks."""
        self._closing = True
        self._events.put_nowait(SttEvent("error"))
        ws, self._ws = self._ws, None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.send(json.dumps({"type": "CloseStream"}))
            with contextlib.suppress(Exception):
                await ws.close()
        for task in (self._reader, self._keeper):
            if task is not None:
                task.cancel()


@dataclass
class _Pending:
    text: str
    queue: asyncio.Queue
    got_audio: bool = False


_END = object()


class DeepgramTTS:
    """Aura-2 synthesis over one persistent speak WebSocket, with REST as the fallback.

    Measured from this account: first audio ~270 ms over a warm WebSocket versus ~750-830 ms
    for streamed REST even on a reused HTTP connection. Requests are pipelined: calling
    `synthesize` for the next sentence while the previous one is still streaming queues it on
    the same socket, and Deepgram returns the audio in order.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        connect: Connector = websockets.connect,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        """Prepare the client; `warm()` or the first `synthesize()` opens the socket."""
        self._api_key = api_key
        self._model = model
        self._connect = connect
        self._http = http
        self._ws: Any = None
        self._reader: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._pending: deque[_Pending] = deque()
        self._flushes: deque[float] = deque()
        self._tasks: set[asyncio.Task] = set()

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Token {self._api_key}"}

    @property
    def _query(self) -> str:
        return urlencode(
            {"model": self._model, "encoding": "linear16", "sample_rate": TTS_SAMPLE_RATE}
        )

    async def _ensure(self) -> Any:
        if self._ws is None:
            self._ws = await self._connect(
                f"{TTS_WS_URL}?{self._query}",
                additional_headers=self._headers,
                max_size=2**22,
                open_timeout=5,
            )
            self._reader = asyncio.create_task(self._read(self._ws))
        return self._ws

    async def warm(self) -> None:
        """Open the speak socket ahead of the first sentence; failures are left to synthesize."""
        try:
            async with self._lock:
                await self._ensure()
        except Exception:
            log.warning("deepgram tts warm-up failed", exc_info=True)

    def _flush_allowed(self) -> bool:
        now = time.monotonic()
        while self._flushes and now - self._flushes[0] > TTS_FLUSH_WINDOW_S:
            self._flushes.popleft()
        if len(self._flushes) >= TTS_FLUSH_BUDGET:
            return False
        self._flushes.append(now)
        return True

    def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Start synthesising `text` now; return an iterator of raw 24 kHz PCM16 chunks.

        Synthesis begins immediately (not on first iteration), so calling this for sentence
        N+1 while sentence N is playing removes the gap between them.
        """
        item = _Pending(text[:TTS_MAX_CHARS], asyncio.Queue())
        runner = self._via_ws(item) if self._flush_allowed() else self._via_rest(item)
        task = asyncio.create_task(runner)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return self._drain(item.queue)

    async def _via_ws(self, item: _Pending) -> None:
        try:
            async with self._lock:
                ws = await self._ensure()
                self._pending.append(item)
                await ws.send(json.dumps({"type": "Speak", "text": item.text}))
                await ws.send(json.dumps({"type": "Flush"}))
        except Exception:
            log.warning("deepgram tts websocket send failed; using REST", exc_info=True)
            with contextlib.suppress(ValueError):
                self._pending.remove(item)
            await self._via_rest(item)

    async def _via_rest(self, item: _Pending) -> None:
        own = self._http is None
        client = self._http or httpx.AsyncClient(timeout=httpx.Timeout(15, connect=5))
        try:
            async with client.stream(
                "POST",
                f"{TTS_REST_URL}?{self._query}&container=none",
                headers=self._headers,
                json={"text": item.text},
            ) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_raw():
                    item.queue.put_nowait(chunk)
            item.queue.put_nowait(_END)
        except Exception as exc:  # noqa: BLE001 - handed to the consumer, which raises it
            item.queue.put_nowait(exc)
        finally:
            if own:
                await client.aclose()

    async def _read(self, ws: Any) -> None:
        try:
            async for msg in ws:
                if isinstance(msg, bytes):
                    if self._pending:
                        self._pending[0].got_audio = True
                        self._pending[0].queue.put_nowait(msg)
                    continue
                data = json.loads(msg)
                if data.get("type") == "Flushed" and self._pending:
                    self._pending.popleft().queue.put_nowait(_END)
                elif data.get("type") in ("Warning", "Error"):
                    log.warning("deepgram tts %s: %s", data.get("type"), data.get("description"))
        except ConnectionClosed:
            pass
        except Exception:
            log.exception("deepgram tts reader failed")
        if self._ws is ws:
            self._ws = None
        self._fail_pending()

    def _fail_pending(self) -> None:
        """Socket gone: re-run unstarted sentences over REST, end the half-spoken one."""
        while self._pending:
            item = self._pending.popleft()
            if item.got_audio:
                item.queue.put_nowait(_END)
            else:
                task = asyncio.create_task(self._via_rest(item))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)

    @staticmethod
    async def _drain(queue: asyncio.Queue) -> AsyncIterator[bytes]:
        carry = b""
        while True:
            chunk = await queue.get()
            if chunk is _END:
                return
            if isinstance(chunk, Exception):
                raise chunk
            chunk = carry + chunk
            cut = len(chunk) & ~1  # keep frames aligned to whole 16-bit samples
            carry = chunk[cut:]
            if cut:
                yield chunk[:cut]

    async def reset(self) -> None:
        """Abandon all queued speech (barge-in); the socket is dropped and reopened lazily."""
        for task in list(self._tasks):
            task.cancel()
        ws, self._ws = self._ws, None
        for item in self._pending:
            item.queue.put_nowait(_END)
        self._pending.clear()
        if self._reader is not None:
            self._reader.cancel()
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()

    async def close(self) -> None:
        """Close the socket and stop background work."""
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.send(json.dumps({"type": "Close"}))
        await self.reset()
