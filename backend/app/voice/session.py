"""WebSocket /ws/voice: mic audio to Deepgram, turns through the agent, sentences back as speech."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import re
import time
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app import auth, history, limits
from app.config import get_settings
from app.voice.deepgram import TTS_SAMPLE_RATE, DeepgramSTT, DeepgramTTS, SttEvent
from app.voice.gate import EnergyGate

log = logging.getLogger(__name__)

# Input sizes (ABU10) and socket limits (ABU3) come from app.limits settings.
PCM_BYTES_PER_SECOND = 32_000  # 16 kHz mono PCM16
WATCHDOG_INTERVAL_S = 1.0  # how often idle time and socket lifetime are checked
SESSION_RECHECK_S = 30.0  # an idle socket notices a logout within this time
SESSION_ENDED = "Your session has ended. Please sign in again to continue."
TTS_FAILED = "Sorry, I couldn't say that reply out loud. It's shown as text instead."
STT_FAILED = "Speech recognition stopped for a moment. Please say that again, or type it."
IDLE_MESSAGE = (
    "I haven't heard anything for a few minutes, so I'm closing this connection for a while."
    " Just start again when you're ready."
)
LIFETIME_MESSAGE = (
    "This voice connection has reached its time limit. Please reconnect to carry on where you"
    " left off."
)
BARGE_IN_MIN_WORDS = 1  # one real word ("stop", "wait") interrupts a reply
FILLERS = frozenset({"uh", "um", "umm", "hmm", "hm", "ah", "er", "erm", "mm", "mhm"})
FIRST_CLAUSE_CHARS = 30  # flush the first clause at a comma once it is this long
LONG_CLAUSE_CHARS = 160  # later clauses flush at a comma past this length
HARD_SPLIT_CHARS = 300  # no punctuation at all: split at a space past this length

# Never end a sentence: a name or number always follows these.
_TITLES = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "rs",
    "no",
    "vs",
    "e.g",
    "i.e",
    "approx",
    "dept",
    "govt",
    "fig",
    "sec",
    "vol",
}
# May end a sentence: split only when the next word starts with a capital letter.
_MAY_END = {
    "b.tech",
    "m.tech",
    "b.e",
    "m.e",
    "b.sc",
    "m.sc",
    "ph.d",
    "b.com",
    "m.com",
    "etc",
    "a.m",
    "p.m",
    "inc",
    "ltd",
}
_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*\s+|\n+")
_CLAUSE_END = re.compile(r"[,;:—]\s+|\s[-–]\s+")


class SentenceSplitter:
    """Cut a streamed reply into speakable chunks without breaking on abbreviations or decimals."""

    def __init__(self) -> None:
        """Start with an empty buffer; the first chunk is flushed early for fast first audio."""
        self._buf = ""
        self._emitted = 0

    def _period_ends_sentence(self, dot: int, after: int) -> bool | None:
        """Decide whether the '.' at `dot` ends a sentence; None means wait for more text."""
        word = re.split(r"\s", self._buf[:dot])[-1].lstrip("(\"'").lower()
        if word in _TITLES or (len(word) == 1 and word.isalpha()):
            return False
        if word in _MAY_END:
            if after >= len(self._buf):
                return None
            return self._buf[after].isupper()
        return True

    def _take(self, end: int) -> str:
        piece, self._buf = self._buf[:end].strip(), self._buf[end:]
        self._emitted += 1 if piece else 0
        return piece

    def _sentence_end(self) -> int | None:
        """End of the first complete sentence; -1 while a '.' is still ambiguous."""
        for m in _SENTENCE_END.finditer(self._buf):
            if self._buf[m.start()] == ".":
                ends = self._period_ends_sentence(m.start(), m.end())
                if ends is None:
                    return -1
                if not ends:
                    continue
            return m.end()
        return None

    def _next(self) -> str | None:
        # The result never depends on how the text was chunked: a long sentence is cut at its
        # first clause boundary past half the limit, whether or not its end has arrived yet.
        end = self._sentence_end()
        if end == -1:
            return None
        limit = FIRST_CLAUSE_CHARS if self._emitted == 0 else LONG_CLAUSE_CHARS
        if end is not None and end <= limit:
            return self._take(end)
        if end is not None or len(self._buf) >= limit:
            stop = end if end is not None else len(self._buf)
            for c in _CLAUSE_END.finditer(self._buf, 0, stop):
                if c.start() >= limit // 2:
                    return self._take(c.end())
        if end is not None:
            return self._take(end)
        if len(self._buf) >= HARD_SPLIT_CHARS:
            cut = self._buf.rfind(" ", 0, HARD_SPLIT_CHARS)
            return self._take(cut if cut > 0 else HARD_SPLIT_CHARS)
        return None

    def feed(self, text: str) -> list[str]:
        """Add a text delta; return any chunks that are now complete."""
        self._buf += text
        out: list[str] = []
        while (piece := self._next()) is not None:
            if piece:
                out.append(piece)
        return out

    def flush(self) -> list[str]:
        """Return whatever remains at the end of the reply."""
        rest = self._buf.strip()
        self._buf = ""
        return [rest] if rest else []


class Stt(Protocol):
    """What the session needs from a speech-to-text stream."""

    async def send_audio(self, pcm: bytes) -> None: ...
    def events(self) -> AsyncIterator[SttEvent]: ...
    async def close(self) -> None: ...


class Tts(Protocol):
    """What the session needs from a speech synthesiser."""

    async def warm(self) -> None: ...
    def synthesize(self, text: str) -> AsyncIterator[bytes]: ...
    async def reset(self) -> None: ...
    async def close(self) -> None: ...


RunTurn = Callable[[auth.SessionCtx, str, str], AsyncIterator[dict]]


def _default_run_turn(ctx: auth.SessionCtx, conversation_id: str, text: str) -> AsyncIterator[dict]:
    from app.agent.loop import run_turn  # imported late: the agent module is built separately

    return run_turn(ctx, conversation_id, text)


def _default_stt() -> Stt:
    s = get_settings()
    return DeepgramSTT(s.deepgram_api_key, s.dg_stt_model)


def _default_tts() -> Tts:
    s = get_settings()
    return DeepgramTTS(s.deepgram_api_key, s.dg_tts_model)


ECHO_MIN_WORDS = 3  # shorter transcripts are never judged to be echo
ECHO_COVERAGE = 0.8  # share of the transcript that must be one contiguous run of the reply


def is_echo(heard: list[str], reply: list[str]) -> bool:
    """True only when the transcript is essentially a verbatim stretch of the reply.

    The longest run of consecutive words shared with the reply must have at least
    ECHO_MIN_WORDS words and cover ECHO_COVERAGE of what was heard. A question that merely
    reuses the reply's words in another order is the user, not echo.
    """
    n = len(heard)
    if n < ECHO_MIN_WORDS or not reply:
        return False
    best = 0
    prev = [0] * (len(reply) + 1)
    for w in heard:
        row = [0] * (len(reply) + 1)
        for j, r in enumerate(reply, 1):
            if w == r:
                row[j] = prev[j - 1] + 1
                best = max(best, row[j])
        prev = row
    return best >= max(ECHO_MIN_WORDS, math.ceil(ECHO_COVERAGE * n))


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


class VoiceSession:
    """State for one /ws/voice connection: at most one turn runs at a time."""

    def __init__(
        self,
        ws: WebSocket,
        ctx: auth.SessionCtx,
        run_turn: RunTurn,
        stt: Stt,
        tts: Tts,
        ip: str = "unknown",
    ) -> None:
        """Bind the socket, the session and the injected speech and agent dependencies."""
        self.ws = ws
        self.ctx = ctx
        self.ip = ip
        self.connected_at = self.last_activity = limits.now()
        self.utterance_bytes = 0  # audio sent since the current utterance began (ABU3)
        self.in_speech = False
        self.skip_utterance = False  # the capped utterance's own late events are dropped
        self.last_interim = ""
        self.gate = EnergyGate() if get_settings().voice_gate_enabled else None
        self.voice_used_up = False  # ABU11 daily voice time spent: audio is dropped
        self.run_turn = run_turn
        self.stt = stt
        self.tts = tts
        self.conversation_id: str | None = None
        self.conversation_turns = 0  # user turns stored in the conversation (ABU7)
        self.turn: asyncio.Task | None = None
        self.stt_task: asyncio.Task | None = None
        self.reply_text = ""  # what the current reply has said, for the echo check
        self.playback_until = 0.0  # monotonic time the client should finish playing
        self.barged = False  # the current speech already interrupted a reply
        self._send_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()
        self._turn_lock = asyncio.Lock()  # one turn at a time, even with two input sources

    async def send(self, payload: dict | bytes) -> None:
        """Send one frame; the lock keeps frames from concurrent tasks whole and ordered."""
        async with self._send_lock:
            if isinstance(payload, bytes):
                await self.ws.send_bytes(payload)
            else:
                await self.ws.send_text(json.dumps(payload))

    def _background(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def replying(self) -> bool:
        """True while a turn runs or the client is still playing its audio."""
        busy = self.turn is not None and not self.turn.done()
        return busy or time.monotonic() < self.playback_until

    async def interrupt(self) -> None:
        """Cancel the running turn and its speech, and tell the client to drop queued audio."""
        async with self._turn_lock:
            await self._interrupt()

    async def _interrupt(self) -> None:
        was_replying = self.replying()
        task, self.turn = self.turn, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self.playback_until = 0.0
        if was_replying:
            await self.tts.reset()
            self._background(self.tts.warm())  # reopen now so the next reply starts fast
            await self.send({"type": "interrupted"})

    async def start_turn(self, text: str) -> None:
        """Interrupt anything in progress and run a new turn for `text`, within the limits."""
        self.last_activity = limits.now()
        refusal = limits.admit_turn(self.ctx, self.ip, self.conversation_turns)
        if refusal:  # ABU2/ABU7/ABU8: no model, speech or college call; the reply goes on
            await self.send({"type": "error", "message": refusal})
            return
        self.conversation_turns += 1
        async with self._turn_lock:
            await self._interrupt()
            if not await self.session_valid():  # AC6: logged out or expired since connect
                await self.end_session()
                return
            self.reply_text = ""
            self.turn = asyncio.create_task(self._turn(text))

    async def _turn(self, text: str) -> None:
        assert self.conversation_id is not None
        warm = asyncio.create_task(self.tts.warm())  # hide the TTS handshake behind the LLM
        sentences: asyncio.Queue[AsyncIterator[bytes] | None] = asyncio.Queue()
        speaker = asyncio.create_task(self._speak(sentences))
        splitter = SentenceSplitter()

        def queue_sentences(pieces: list[str]) -> None:
            for piece in pieces:
                sentences.put_nowait(self.tts.synthesize(piece))

        try:
            stream = contextlib.aclosing(self.run_turn(self.ctx, self.conversation_id, text))
            async with stream as events:
                async for event in events:
                    kind = event.get("type")
                    if kind == "text_delta":
                        delta = event.get("text") or ""
                        self.reply_text += delta
                        await self.send({"type": "assistant_delta", "text": delta})
                        queue_sentences(splitter.feed(delta))
                    elif kind == "tool":
                        await self.send(
                            {"type": "tool", "name": event.get("name"), "state": event.get("state")}
                        )
                    elif kind == "citations":  # shown, never spoken
                        await self.send(
                            {"type": "citations", "sources": event.get("sources") or []}
                        )
                    elif kind == "done":
                        queue_sentences(splitter.flush())
                        await self.send({"type": "assistant_done", "text": event.get("text") or ""})
                    elif kind == "error":
                        await self.send(
                            {"type": "error", "message": event.get("message") or "Error"}
                        )
            queue_sentences(splitter.flush())
            sentences.put_nowait(None)
            await speaker
            self.last_activity = limits.now()
        except asyncio.CancelledError:
            speaker.cancel()
            warm.cancel()
            raise
        except Exception:
            log.exception("voice turn failed")
            speaker.cancel()
            await self.send({"type": "error", "message": "Sorry, something went wrong."})

    async def _speak(self, sentences: asyncio.Queue[AsyncIterator[bytes] | None]) -> None:
        started = tts_failed = False
        samples = 0
        t0 = 0.0
        while (audio := await sentences.get()) is not None:
            try:
                async for chunk in audio:
                    if not started:
                        await self.send(
                            {
                                "type": "audio_start",
                                "sample_rate": TTS_SAMPLE_RATE,
                                "format": "pcm_s16le",
                            }
                        )
                        started, t0 = True, time.monotonic()
                    await self.send(chunk)
                    samples += len(chunk) // 2
                    self.playback_until = t0 + samples / TTS_SAMPLE_RATE + 0.3
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning("tts failed for one sentence", exc_info=True)
                if not tts_failed:
                    tts_failed = True
                    await self.send({"type": "error", "message": TTS_FAILED})
        if started:
            await self.send({"type": "audio_end"})

    def _is_real_speech(
        self, text: str, min_words: int = BARGE_IN_MIN_WORDS, skip_fillers: bool = False
    ) -> bool:
        """Enough words, and not just the assistant's own words echoing back through the mic."""
        words = _words(text)
        if skip_fillers:  # a bare "uh" or "hmm" does not interrupt on its own
            words = [w for w in words if w not in FILLERS]
        if len(words) < min_words:
            return False
        if is_echo(words, _words(self.reply_text)):
            log.info("transcript discarded as echo of the reply (%d words)", len(words))
            return False
        return True

    async def on_stt(self, event: SttEvent) -> None:
        """Handle one STT event: forward transcripts, barge in, start turns."""
        if event.kind == "error":  # the provider's text may hold details; never forward it
            await self.send({"type": "error", "message": STT_FAILED})
        elif event.kind == "speech_started":
            self.in_speech = True
        elif self.skip_utterance:
            if event.kind == "utterance":  # the tail of an utterance cut off at the cap
                self.skip_utterance = False
        elif event.kind == "interim":
            self.last_activity = limits.now()
            self.in_speech = True
            self.last_interim = event.text
            if self.replying() and not self.barged:
                if not self._is_real_speech(event.text, skip_fillers=True):
                    return  # likely echo or a blip; the client barges in on any transcript
                self.barged = True
                await self.interrupt()
            await self.send({"type": "transcript", "final": False, "text": event.text})
        elif event.kind == "utterance" and event.text.strip():
            await self._end_utterance(event.text.strip())

    async def _end_utterance(self, text: str) -> None:
        """A complete user utterance: forward it as final and start a turn."""
        self.last_activity = limits.now()
        self.in_speech, self.utterance_bytes, self.last_interim = False, 0, ""
        barged, self.barged = self.barged, False
        if self.replying() and not barged and not self._is_real_speech(text, 1):
            return
        await self.send({"type": "transcript", "final": True, "text": text})
        await self.start_turn(text)

    async def _pump_stt(self) -> None:
        try:
            async for event in self.stt.events():
                try:
                    await self.on_stt(event)
                except (WebSocketDisconnect, RuntimeError):
                    return
                except Exception:
                    log.exception("stt event handling failed")
        except Exception:
            log.exception("stt stream failed")
            self.stt_task = None  # the next mic frame starts a fresh pump
            self.in_speech, self.utterance_bytes, self.last_interim = False, 0, ""
            with contextlib.suppress(Exception):
                await self.send({"type": "error", "message": STT_FAILED})

    async def on_audio(self, pcm: bytes) -> None:
        """Relay one mic frame to Deepgram (ignored until a conversation has started)."""
        if self.conversation_id is None:
            return
        if len(pcm) > limits.limit("limit_audio_frame_bytes"):  # ABU10
            await self.send({"type": "error", "message": "That audio frame was too large."})
            return
        pcm = pcm[: len(pcm) & ~1]  # whole 16-bit samples only
        if not pcm or self.voice_used_up:
            return
        # ABU11: with the gate on, silence and steady noise never reach Deepgram
        was_open = self.gate is not None and self.gate.open
        for chunk in self.gate.feed(pcm) if self.gate is not None else [pcm]:
            if not limits.take_voice_seconds(self.ctx, self.ip, len(chunk) / PCM_BYTES_PER_SECOND):
                self.voice_used_up = True
                await self.send({"type": "error", "message": limits.VOICE_USED_UP})
                return
            await self._forward(chunk)
        if was_open and not self.gate.open:  # no more audio follows: let Deepgram finish now
            finalize = getattr(self.stt, "finalize", None)
            if finalize is not None:
                with contextlib.suppress(Exception):
                    await finalize()

    async def _forward(self, pcm: bytes) -> None:
        """Send audio to Deepgram and enforce the utterance cap (ABU3)."""
        if self.stt_task is None:
            self.stt_task = asyncio.create_task(self._pump_stt())
        try:
            await self.stt.send_audio(pcm)
        except Exception:
            log.warning("deepgram stt send failed", exc_info=True)
            await self.send({"type": "error", "message": "Speech recognition is unavailable."})
            return
        if self.in_speech and not self.skip_utterance:
            self.utterance_bytes += len(pcm)
            cap = limits.limit("limit_utterance_seconds") * PCM_BYTES_PER_SECOND
            if self.utterance_bytes >= cap and self.last_interim:  # ABU3: end it here
                self.skip_utterance = True
                finalize = getattr(self.stt, "finalize", None)
                if finalize is not None:
                    await finalize()
                await self._end_utterance(self.last_interim)

    async def on_message(self, msg: dict) -> None:
        """Handle one JSON control message from the client."""
        kind = msg.get("type")
        if kind == "start":
            await self._start(msg.get("conversation_id"))
        elif self.conversation_id is None:
            await self.send({"type": "error", "message": "Send start first."})
        elif kind == "text":
            raw = msg.get("text")
            if not isinstance(raw, str):
                if raw is not None:
                    await self.send({"type": "error", "message": "Typed text must be a string."})
                return
            text = raw.strip()
            cap = int(limits.limit("limit_typed_chars"))
            if len(text) > cap:  # ABU10: refused, never truncated and sent on
                await self.send(
                    {
                        "type": "error",
                        "message": f"Please keep typed messages under {cap} characters.",
                    }
                )
            elif text:
                await self.start_turn(text)  # not echoed: the client shows typed text itself
        elif kind == "interrupt":
            await self.interrupt()
        else:
            await self.send({"type": "error", "message": "Unknown message type."})

    async def _start(self, requested: Any) -> None:
        await self.interrupt()
        cid = requested if isinstance(requested, str) and requested else None
        if cid is None or not await history.conversation_owned(self.ctx, cid):
            cid = await history.create_conversation(self.ctx)
        self.conversation_id = cid
        self.conversation_turns = await limits.conversation_turns(cid)
        self._background(self.tts.warm())  # open the speak socket before the first reply
        user = self.ctx.user or {}
        await self.send(
            {
                "type": "ready",
                "conversation_id": cid,
                "role": self.ctx.role,
                "user_name": user.get("full_name"),
            }
        )

    async def session_valid(self) -> bool:
        """Re-check the cookie's session with the same rules as at connect time."""
        ctx = await auth.session_from_websocket(self.ws)
        return ctx is not None and ctx.session_id == self.ctx.session_id

    async def end_session(self) -> None:
        """The session is gone: stop everything, tell the client, close with 4401."""
        await self._interrupt()
        with contextlib.suppress(Exception):
            await self.send({"type": "error", "message": SESSION_ENDED})
        with contextlib.suppress(Exception):
            await self.ws.close(code=4401)

    async def watchdog(self) -> None:
        """ABU3: close an idle socket, or one past its lifetime, with an explanation."""
        lifetime = limits.limit(
            "limit_visitor_socket_seconds"
            if self.ctx.role == "visitor"
            else "limit_user_socket_seconds"
        )
        last_check = limits.now()
        while True:
            await asyncio.sleep(WATCHDOG_INTERVAL_S)
            now = limits.now()
            if now - last_check >= SESSION_RECHECK_S:
                last_check = now
                if not await self.session_valid():
                    async with self._turn_lock:
                        await self.end_session()
                    return
            if now - self.connected_at >= lifetime:
                message = LIFETIME_MESSAGE
            elif not self.replying() and now - self.last_activity >= limits.limit(
                "limit_idle_seconds"
            ):
                message = IDLE_MESSAGE
            else:
                continue
            await self.interrupt()
            with contextlib.suppress(Exception):
                await self.send({"type": "error", "message": message})
            with contextlib.suppress(Exception):
                await self.ws.close(code=1000)
            return

    async def close(self) -> None:
        """Cancel tasks and close Deepgram connections."""
        task, self.turn = self.turn, None
        for t in (task, self.stt_task, *self._tasks):
            if t is not None:
                t.cancel()
        for t in (task, self.stt_task):
            if t is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await t
        with contextlib.suppress(Exception):
            await self.stt.close()
        with contextlib.suppress(Exception):
            await self.tts.close()


def _parse(raw: str) -> dict | str:
    """Parse a control frame; return the message dict or an error string."""
    if len(raw.encode()) > limits.limit("limit_ws_message_bytes"):  # ABU10
        return "Message too large."
    try:
        msg = json.loads(raw)
    except ValueError:
        return "Invalid JSON."
    return msg if isinstance(msg, dict) else "Expected a JSON object."


def build_router(
    run_turn: RunTurn = _default_run_turn,
    stt_factory: Callable[[], Stt] = _default_stt,
    tts_factory: Callable[[], Tts] = _default_tts,
) -> APIRouter:
    """Create the /ws/voice router; tests inject fake agent and speech dependencies."""
    r = APIRouter()

    @r.websocket("/ws/voice")
    async def voice(ws: WebSocket) -> None:
        """Run one voice connection following the CONTRACT.md protocol."""
        await ws.accept()
        if not auth.origin_allowed(ws.headers.get("origin")):
            await ws.close(code=4403)
            return
        ctx = await auth.session_from_websocket(ws)
        if ctx is None:
            await ws.close(code=4401)
            return
        ip = limits.client_ip(ws)
        slot, per_slot = limits.socket_slot(ctx, ip)
        if not limits.CONNECTIONS.acquire(slot, per_slot):  # ABU1
            await ws.close(code=4429)
            return
        limits.client_ip_var.set(ip)  # tools see the caller's IP (ABU6)
        try:
            session = VoiceSession(ws, ctx, run_turn, stt_factory(), tts_factory(), ip)
        except Exception:
            limits.CONNECTIONS.release(slot)
            raise
        session._background(session.watchdog())
        try:
            while True:
                frame = await ws.receive()
                if frame["type"] == "websocket.disconnect":
                    break
                if (data := frame.get("bytes")) is not None:
                    await session.on_audio(data)
                    continue
                msg = _parse(frame.get("text") or "")
                if isinstance(msg, str):
                    await session.send({"type": "error", "message": msg})
                    continue
                try:
                    await session.on_message(msg)
                except (WebSocketDisconnect, RuntimeError):
                    raise
                except Exception:
                    log.exception("voice message failed")
                    await session.send({"type": "error", "message": "Sorry, something went wrong."})
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            limits.CONNECTIONS.release(slot)
            await session.close()

    return r


router = build_router()
