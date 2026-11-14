"""End-to-end voice probe: speak sentences into /ws/voice as a fake microphone.

    uv run python scripts/fake_mic.py "What courses do you offer?" ["And the fees?" ...]
        [--login arjun] [--base http://localhost:8000] [--barge-in] [--out reply]

Each sentence is synthesised at 16 kHz with Deepgram TTS, streamed in real-time 20 ms frames
followed by silence, and every server event is printed with a timestamp. Reply audio is saved
as WAV files and the latencies (end of speech to final transcript, first text, first audio)
are reported per turn.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import wave
from pathlib import Path

import httpx
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import get_settings

FRAME_BYTES = 640  # 20 ms of 16 kHz PCM16
SILENCE_S = 1.5
BARGE_IN_TEXT = "Sorry, actually, tell me about the hostel instead."


async def speech(client: httpx.AsyncClient, text: str) -> bytes:
    """Synthesise `text` as raw 16 kHz mono PCM16."""
    s = get_settings()
    resp = await client.post(
        "https://api.deepgram.com/v1/speak",
        params={
            "model": s.dg_tts_model,
            "encoding": "linear16",
            "sample_rate": 16000,
            "container": "none",
        },
        headers={"Authorization": f"Token {s.deepgram_api_key}"},
        json={"text": text},
    )
    resp.raise_for_status()
    return resp.content


async def session_cookie(base: str, login: str | None) -> str:
    """Get a kit_session cookie: a visitor session, or a logged-in one with --login."""
    async with httpx.AsyncClient(base_url=base, timeout=20) as c:
        await c.get("/")
        if login:
            password = os.environ.get("KIT_TEST_PASSWORD")
            if not password:
                raise SystemExit("--login needs the test password in the KIT_TEST_PASSWORD environment variable")
            r = await c.post("/api/login", json={"login_id": login, "password": password})
            r.raise_for_status()
        cookie = c.cookies.get("kit_session")
    if not cookie:
        raise SystemExit("server did not set a kit_session cookie")
    return cookie


class Probe:
    """Drives one WebSocket conversation and records timings."""

    def __init__(self, ws, t0: float) -> None:
        """Hold the socket and a reference start time."""
        self.ws = ws
        self.t0 = t0
        self.marks: dict[str, float] = {}
        self.audio = bytearray()
        self.replies: list[bytes] = []
        self.turn_done = asyncio.Event()

    def stamp(self) -> str:
        """Seconds since start, for log lines."""
        return f"{time.perf_counter() - self.t0:7.3f}s"

    async def read(self) -> None:
        """Print every event; collect audio; note first-event times."""
        frames = 0
        muted = False  # after `interrupted`, drop audio until the next audio_start
        async for msg in self.ws:
            now = time.perf_counter()
            if isinstance(msg, bytes):
                if muted:
                    continue
                self.audio += msg
                frames += 1
                self.marks.setdefault("audio", now)
                continue
            ev = json.loads(msg)
            kind = ev.get("type")
            muted = (muted or kind == "interrupted") and kind != "audio_start"
            if kind == "transcript" and ev.get("final"):
                self.marks.setdefault("final", now)
            if kind == "interrupted":
                self.marks.setdefault("interrupted", now)
            if kind == "assistant_delta":
                self.marks.setdefault("text", now)
            if kind == "audio_end":
                print(f"{self.stamp()}  [{frames} audio frames, {len(self.audio)} bytes]")
                frames = 0
            print(f"{self.stamp()}  {json.dumps(ev)}")
            if kind in ("audio_end", "interrupted", "error"):
                self.replies.append(bytes(self.audio))
                self.audio.clear()
            if kind in ("audio_end", "error"):
                self.turn_done.set()

    async def speak(self, pcm: bytes, silence_s: float = SILENCE_S) -> float:
        """Stream PCM in real time followed by silence; return the end-of-speech time."""
        start = time.perf_counter()
        sent = 0
        for i in range(0, len(pcm), FRAME_BYTES):
            await self.ws.send(pcm[i : i + FRAME_BYTES])
            sent += FRAME_BYTES
            delay = start + sent / 32000 - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
        end = time.perf_counter()
        silence = b"\0" * FRAME_BYTES
        for k in range(int(silence_s / 0.02)):
            await self.ws.send(silence)
            delay = end + (k + 1) * 0.02 - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
        return end


def report(label: str, eos: float, marks: dict[str, float]) -> None:
    """Print latencies from end of speech."""

    def ms(key: str) -> str:
        return f"{1000 * (marks[key] - eos):6.0f} ms" if key in marks else "   n/a"

    print(
        f"--- {label}: end of speech -> final transcript {ms('final')}, "
        f"first text {ms('text')}, first audio {ms('audio')}"
    )


def save_wav(path: Path, pcm: bytes) -> None:
    """Write 24 kHz mono PCM16 as WAV."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm)


async def main() -> None:
    """Run the scripted conversation."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("sentences", nargs="+")
    ap.add_argument("--login")
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--barge-in", action="store_true", help="talk over the first reply")
    ap.add_argument("--barge-in-text", default=BARGE_IN_TEXT, help="what to say when barging in")
    ap.add_argument("--out", default="reply", help="WAV file prefix")
    ap.add_argument("--turn-timeout", type=float, default=30.0)
    args = ap.parse_args()

    base = args.base.rstrip("/")
    cookie = await session_cookie(base, args.login)
    async with httpx.AsyncClient(timeout=20) as c:
        clips = [await speech(c, s) for s in args.sentences]
        barge = await speech(c, args.barge_in_text) if args.barge_in else b""

    ws_url = base.replace("http", "ws", 1) + "/ws/voice"
    headers = {"Cookie": f"kit_session={cookie}", "Origin": base}
    async with websockets.connect(ws_url, additional_headers=headers, max_size=2**22) as ws:
        probe = Probe(ws, time.perf_counter())
        reader = asyncio.create_task(probe.read())
        await ws.send(json.dumps({"type": "start", "conversation_id": None}))
        for n, (text, pcm) in enumerate(zip(args.sentences, clips, strict=True), 1):
            print(f"{probe.stamp()}  >>> speaking: {text!r}")
            probe.marks.clear()
            probe.turn_done.clear()
            if args.barge_in and n == 1:
                eos = await probe.speak(pcm, silence_s=0.2)
                deadline = time.perf_counter() + args.turn_timeout
                while "audio" not in probe.marks and time.perf_counter() < deadline:
                    await ws.send(b"\0" * FRAME_BYTES)
                    await asyncio.sleep(0.02)
                report(f"turn {n}", eos, dict(probe.marks))
                await asyncio.sleep(1.0)  # let some reply audio play, then talk over it
                print(f"{probe.stamp()}  >>> barge-in: {args.barge_in_text!r}")
                barge_start = time.perf_counter()
                probe.marks.clear()
                probe.turn_done.clear()
                eos = await probe.speak(barge)
                label = "barge-in turn"
                if "interrupted" in probe.marks:
                    ms = 1000 * (probe.marks["interrupted"] - barge_start)
                    print(f"--- barge-in: start of speech -> interrupted {ms:.0f} ms")
            else:
                eos = await probe.speak(pcm)
                label = f"turn {n}"
            deadline = time.perf_counter() + args.turn_timeout
            while not probe.turn_done.is_set() and time.perf_counter() < deadline:
                await ws.send(b"\0" * FRAME_BYTES)  # keep the mic "open" like a browser
                await asyncio.sleep(0.02)
            report(label, eos, dict(probe.marks))
        await ws.close()
        await reader

    for i, pcm in enumerate(r for r in probe.replies if r):
        path = Path(f"{args.out}-{i + 1}.wav")
        save_wav(path, pcm)
        print(f"saved {path} ({len(pcm) / 48000:.1f} s)")


if __name__ == "__main__":
    asyncio.run(main())
