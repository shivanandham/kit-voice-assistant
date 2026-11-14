"""VOX1/VOX5/REL1/AC5 for app.voice.deepgram: URL and auth, utterance assembly, keep-alive,
reconnect, clean close, and TTS ordering/failures. Only fake transports are used (REL2)."""

import asyncio
import json
import logging
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from app.voice import deepgram
from app.voice.deepgram import DeepgramSTT, DeepgramTTS, SttEvent, stt_url
from websockets.exceptions import ConnectionClosed

KEY = "dg-secret-KEY-0123456789"
CLOSE = object()


# ------------------------------------------------------------------------- fakes


class FakeWS:
    """A fake websockets connection: async-iterable incoming, recorded outgoing."""

    def __init__(self, on_send=None) -> None:
        self.sent: list = []
        self.incoming: asyncio.Queue = asyncio.Queue()
        self.closed = False
        self.on_send = on_send

    async def send(self, data) -> None:
        if self.closed:
            raise ConnectionClosed(None, None)
        self.sent.append(data)
        if self.on_send:
            self.on_send(self, data)

    async def close(self) -> None:
        self.closed = True
        self.incoming.put_nowait(CLOSE)

    def drop(self) -> None:
        """The server went away."""
        self.closed = True
        self.incoming.put_nowait(CLOSE)

    def feed(self, msg) -> None:
        self.incoming.put_nowait(msg if isinstance(msg, (bytes, str)) else json.dumps(msg))

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.incoming.get()
        if item is CLOSE:
            raise StopAsyncIteration
        return item

    @property
    def texts(self) -> list[dict]:
        return [json.loads(s) for s in self.sent if isinstance(s, str)]

    @property
    def audio(self) -> list[bytes]:
        return [s for s in self.sent if isinstance(s, bytes)]


class Connector:
    """Records every connection attempt; `plan` items are a FakeWS factory or an Exception."""

    def __init__(self, plan=None, limit: int = 200, on_send=None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.sockets: list[FakeWS] = []
        self.plan = list(plan or [])
        self.limit = limit
        self.on_send = on_send

    async def __call__(self, url, **kw):
        self.calls.append((url, kw))
        if len(self.calls) > self.limit:
            raise OSError("test limit")
        item = self.plan.pop(0) if self.plan else None
        if isinstance(item, Exception):
            raise item
        ws = item() if callable(item) else FakeWS(self.on_send)
        self.sockets.append(ws)
        return ws


async def take(stt: DeepgramSTT, n: int = 1, timeout: float = 2.0) -> list[SttEvent]:
    it = stt.events()
    out = []
    for _ in range(n):
        out.append(await asyncio.wait_for(it.__anext__(), timeout))
    return out


async def no_event(stt: DeepgramSTT, wait: float = 0.15) -> None:
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(stt.events().__anext__(), wait)


def results(text: str, final: bool = False, speech_final: bool = False) -> dict:
    return {"type": "Results", "is_final": final, "speech_final": speech_final,
            "channel": {"alternatives": [{"transcript": text}]}}


async def started_stt(**kw):
    con = Connector()
    stt = DeepgramSTT(KEY, "nova-3", connect=con)
    await stt.send_audio(b"\x00\x01" * 160)
    return stt, con, con.sockets[0]


# ------------------------------------------------------------------------- STT URL and auth


def _q(url: str) -> dict:
    return parse_qs(urlparse(url).query)


def test_vox1_stt_url_has_model_rate_encoding_and_keyterms():
    url = stt_url("nova-3-test")
    q = _q(url)
    assert urlparse(url).scheme == "wss" and urlparse(url).netloc == "api.deepgram.com"
    assert urlparse(url).path == "/v1/listen"
    assert q["model"] == ["nova-3-test"]
    assert q["sample_rate"] == ["16000"] and q["encoding"] == ["linear16"] and q["channels"] == ["1"]
    assert q["interim_results"] == ["true"]
    assert "Knotopian" in q["keyterm"] and "B.Tech" in q["keyterm"]
    assert set(q["keyterm"]) == set(deepgram.STT_KEYTERMS)


def test_vox1_stt_url_custom_keyterms_and_special_characters_roundtrip():
    terms = ("R&D", "a b", "100%", "C#", "x=y", "naïve", "வணக்கம்")
    q = _q(stt_url("m", terms))
    assert q["keyterm"] == list(terms)


def test_vox1_stt_url_without_keyterms_has_none():
    assert "keyterm" not in _q(stt_url("m", ()))


@pytest.mark.parametrize("model", ["nova-3", "nova-2-general", "a&b=c", "m d"])
def test_vox1_stt_model_is_one_query_value(model):
    assert _q(stt_url(model))["model"] == [model]


def test_ac5_stt_url_does_not_contain_the_key():
    assert KEY not in stt_url("nova-3")
    assert "token" not in stt_url("nova-3").lower().replace("keyterm", "")


async def test_ac5_stt_key_only_in_authorization_header():
    con = Connector()
    stt = DeepgramSTT(KEY, "nova-X", connect=con)
    await stt.send_audio(b"\x00\x01" * 100)
    url, kw = con.calls[0]
    assert KEY not in url
    assert _q(url)["model"] == ["nova-X"]
    assert kw["additional_headers"] == {"Authorization": f"Token {KEY}"}
    assert KEY not in repr({k: v for k, v in kw.items() if k != "additional_headers"})
    await stt.close()


async def test_vox1_stt_is_lazy_until_first_audio():
    con = Connector()
    stt = DeepgramSTT(KEY, "nova-3", connect=con)
    await asyncio.sleep(0.05)
    assert con.calls == []
    await stt.close()
    assert con.calls == []


async def test_vox1_stt_audio_frames_are_sent_in_order_on_one_connection():
    stt, con, ws = await started_stt()
    frames = [bytes([i, 0]) * 50 for i in range(1, 20)]
    for f in frames:
        await stt.send_audio(f)
    assert ws.audio[1:] == frames and len(con.calls) == 1
    await stt.close()


# ------------------------------------------------------------------------- assembly


async def test_vox1_stt_interim_then_final_assemble_one_utterance():
    stt, _, ws = await started_stt()
    ws.feed(results("hello"))
    ws.feed(results("hello there", final=True))
    ws.feed(results("how are"))
    ws.feed(results("how are you", final=True, speech_final=True))
    ev = await take(stt, 5)
    assert ev == [
        SttEvent("interim", "hello"),
        SttEvent("interim", "hello there"),
        SttEvent("interim", "hello there how are"),
        SttEvent("interim", "hello there how are you"),
        SttEvent("utterance", "hello there how are you"),
    ]
    await stt.close()


async def test_vox1_stt_utterance_text_is_not_repeated_in_the_next_utterance():
    stt, _, ws = await started_stt()
    ws.feed(results("first one", final=True, speech_final=True))
    ws.feed(results("second one", final=True, speech_final=True))
    ev = [e for e in await take(stt, 4) if e.kind == "utterance"]
    assert [e.text for e in ev] == ["first one", "second one"]
    await stt.close()


async def test_vox1_stt_final_without_speech_final_does_not_end_utterance():
    stt, _, ws = await started_stt()
    ws.feed(results("part one", final=True))
    assert (await take(stt))[0] == SttEvent("interim", "part one")
    await no_event(stt)
    await stt.close()


async def test_vox1_stt_utterance_end_flushes_pending_segments():
    stt, _, ws = await started_stt()
    ws.feed(results("noisy room words", final=True))
    ws.feed({"type": "UtteranceEnd"})
    ev = await take(stt, 2)
    assert ev[-1] == SttEvent("utterance", "noisy room words")
    await stt.close()


async def test_vox1_stt_utterance_end_with_nothing_pending_emits_nothing():
    stt, _, ws = await started_stt()
    ws.feed({"type": "UtteranceEnd"})
    await no_event(stt)
    await stt.close()


async def test_vox1_stt_utterance_end_after_speech_final_does_not_duplicate():
    stt, _, ws = await started_stt()
    ws.feed(results("only once", final=True, speech_final=True))
    ws.feed({"type": "UtteranceEnd"})
    ev = await take(stt, 2)
    assert [e.kind for e in ev] == ["interim", "utterance"]
    await no_event(stt)
    await stt.close()


async def test_vox1_stt_empty_transcripts_emit_nothing():
    stt, _, ws = await started_stt()
    ws.feed(results(""))
    ws.feed(results("   ", final=True))
    ws.feed(results("", final=True, speech_final=True))
    await no_event(stt)
    await stt.close()


async def test_vox1_stt_speech_final_with_empty_text_still_ends_pending_utterance():
    stt, _, ws = await started_stt()
    ws.feed(results("kept words", final=True))
    ws.feed(results("", final=True, speech_final=True))
    ev = await take(stt, 3)
    assert ev[-1] == SttEvent("utterance", "kept words")
    assert [e.kind for e in ev].count("utterance") == 1
    await stt.close()


async def test_vox1_stt_speech_started_event():
    stt, _, ws = await started_stt()
    ws.feed({"type": "SpeechStarted"})
    assert await take(stt) == [SttEvent("speech_started")]
    await stt.close()


@pytest.mark.parametrize(
    "msg",
    [
        {"type": "Metadata", "request_id": "x"},
        {"type": "SomethingNew"},
        {"no": "type"},
        {"type": "Results"},
        {"type": "Results", "channel": None},
        {"type": "Results", "channel": {"alternatives": []}},
        {"type": "Results", "channel": {"alternatives": [{}]}, "is_final": True},
    ],
)
async def test_vox1_stt_unknown_or_odd_messages_are_ignored_without_dropping_connection(msg):
    stt, con, ws = await started_stt()
    ws.feed(msg)
    ws.feed(results("still works", final=True, speech_final=True))
    ev = await take(stt, 2)
    assert ev[-1] == SttEvent("utterance", "still works")
    assert len(con.calls) == 1
    await stt.close()


async def test_vox1_stt_binary_server_messages_are_ignored():
    stt, con, ws = await started_stt()
    ws.feed(b"\x00\x01binary")
    ws.feed(results("after binary", final=True, speech_final=True))
    assert (await take(stt, 2))[-1].text == "after binary"
    await stt.close()


async def test_vox1_stt_whitespace_in_transcript_is_trimmed():
    stt, _, ws = await started_stt()
    ws.feed(results("  padded  ", final=True, speech_final=True))
    assert (await take(stt, 2))[-1] == SttEvent("utterance", "padded")
    await stt.close()


async def test_rel1_stt_malformed_json_from_server_triggers_reconnect_not_crash():
    stt, con, ws = await started_stt()
    ws.feed("this is not json")
    await asyncio.sleep(0.3)
    assert len(con.calls) >= 2
    await stt.send_audio(b"\x00\x01" * 10)
    con.sockets[-1].feed(results("recovered", final=True, speech_final=True))
    assert (await take(stt, 2))[-1].text == "recovered"
    await stt.close()


# ------------------------------------------------------------------------- keep-alive / idle


async def test_vox1_stt_sends_keepalive_while_idle(monkeypatch):
    monkeypatch.setattr(deepgram, "STT_KEEPALIVE_S", 0.05)
    stt, _, ws = await started_stt()
    await asyncio.sleep(0.4)
    kinds = [m["type"] for m in ws.texts]
    assert kinds.count("KeepAlive") >= 2
    await stt.close()


async def test_vox1_stt_no_keepalive_flood_while_audio_flows(monkeypatch):
    monkeypatch.setattr(deepgram, "STT_KEEPALIVE_S", 0.1)
    stt, _, ws = await started_stt()
    for _ in range(15):
        await stt.send_audio(b"\x00\x01" * 160)
        await asyncio.sleep(0.02)
    assert [m["type"] for m in ws.texts].count("KeepAlive") == 0
    await stt.close()


async def test_rel1_stt_idle_for_long_closes_stream_and_next_audio_reconnects(monkeypatch):
    monkeypatch.setattr(deepgram, "STT_KEEPALIVE_S", 0.05)
    monkeypatch.setattr(deepgram, "STT_IDLE_CLOSE_S", 0.2)
    stt, con, ws = await started_stt()
    await asyncio.sleep(0.6)
    assert {"type": "CloseStream"} in ws.texts and ws.closed
    n = len(con.calls)
    await stt.send_audio(b"\x00\x01" * 10)
    assert len(con.calls) == n + 1 and con.sockets[-1].audio
    await stt.close()


async def test_rel1_stt_idle_close_discards_half_finished_utterance(monkeypatch):
    monkeypatch.setattr(deepgram, "STT_KEEPALIVE_S", 0.05)
    monkeypatch.setattr(deepgram, "STT_IDLE_CLOSE_S", 0.2)
    stt, con, ws = await started_stt()
    ws.feed(results("stale words", final=True))
    await take(stt)
    await asyncio.sleep(0.6)
    await stt.send_audio(b"\x00\x01" * 10)
    con.sockets[-1].feed(results("fresh words", final=True, speech_final=True))
    ev = await take(stt, 2)
    assert ev[-1] == SttEvent("utterance", "fresh words")
    await stt.close()


# ------------------------------------------------------------------------- reconnect


async def test_rel1_stt_reconnects_once_when_server_drops_during_speech():
    stt, con, ws = await started_stt()
    ws.drop()
    await asyncio.sleep(0.2)
    assert len(con.calls) == 2
    await stt.send_audio(b"\x00\x01" * 10)
    assert con.sockets[1].audio
    con.sockets[1].feed(results("after drop", final=True, speech_final=True))
    assert (await take(stt, 2))[-1].text == "after drop"
    await stt.close()


async def test_rel1_stt_send_on_closed_socket_reconnects_and_resends_the_frame():
    stt, con, ws = await started_stt()
    ws.closed = True  # server closed but the reader has not noticed yet
    frame = b"\x07\x00" * 80
    await stt.send_audio(frame)
    assert len(con.calls) == 2 and con.sockets[1].audio[-1] == frame
    await stt.close()


async def test_rel1_stt_failed_reconnect_surfaces_an_error_event():
    con = Connector([None, OSError("down")])
    stt = DeepgramSTT(KEY, "nova-3", connect=con)
    await stt.send_audio(b"\x00\x01" * 10)
    con.sockets[0].drop()
    ev = await take(stt, 1)
    assert ev[0].kind == "error" and ev[0].text
    assert KEY not in ev[0].text and "down" not in ev[0].text
    await stt.close()


async def test_rel1_stt_connect_failure_raises_to_caller_and_recovers():
    con = Connector([OSError(f"cannot connect {KEY}")])
    stt = DeepgramSTT(KEY, "nova-3", connect=con)
    with pytest.raises(OSError):
        await stt.send_audio(b"\x00\x01" * 10)
    await stt.send_audio(b"\x00\x01" * 10)
    assert con.sockets[0].audio
    await stt.close()


async def test_rel1_stt_reconnect_is_bounded_when_server_keeps_dropping():
    plan = [lambda: _dropped() for _ in range(150)]
    con = Connector(plan, limit=150)
    stt = DeepgramSTT(KEY, "nova-3", connect=con)
    try:
        await stt.send_audio(b"\x00\x01" * 10)
    except Exception:
        pass
    await asyncio.sleep(0.5)
    assert len(con.calls) <= 4, f"reconnect storm: {len(con.calls)} connections in 0.5 s"
    await stt.close()


def _dropped() -> FakeWS:
    ws = FakeWS()
    ws.drop()
    return ws


async def test_rel1_stt_after_failed_reconnect_next_audio_connects_again():
    con = Connector([None, OSError("down"), None])
    stt = DeepgramSTT(KEY, "nova-3", connect=con)
    await stt.send_audio(b"\x00\x01" * 10)
    con.sockets[0].drop()
    await take(stt, 1)
    await stt.send_audio(b"\x00\x01" * 10)
    assert len(con.sockets) == 2 and con.sockets[1].audio
    await stt.close()


# ------------------------------------------------------------------------- close


async def test_rel1_stt_close_sends_closestream_closes_socket_and_ends_events():
    stt, con, ws = await started_stt()
    consumer = asyncio.create_task(_collect(stt))
    await asyncio.sleep(0.05)
    await stt.close()
    events = await asyncio.wait_for(consumer, 2)
    assert {"type": "CloseStream"} in ws.texts and ws.closed
    assert all(e.kind != "error" for e in events)


async def _collect(stt):
    return [e async for e in stt.events()]


async def test_rel1_stt_close_without_connection_is_fine_and_idempotent():
    stt = DeepgramSTT(KEY, "nova-3", connect=Connector())
    await stt.close()
    await stt.close()
    assert await asyncio.wait_for(_collect(stt), 2) == []


async def test_rel1_stt_audio_after_close_is_dropped_without_connecting():
    con = Connector()
    stt = DeepgramSTT(KEY, "nova-3", connect=con)
    await stt.close()
    await stt.send_audio(b"\x00\x01" * 10)
    assert con.calls == []


async def test_rel1_stt_close_cancels_background_tasks(monkeypatch):
    monkeypatch.setattr(deepgram, "STT_KEEPALIVE_S", 0.05)
    before = set(asyncio.all_tasks())
    stt, con, ws = await started_stt()
    await stt.close()
    await asyncio.sleep(0.3)
    assert not {t for t in asyncio.all_tasks() - before if not t.done()}


async def test_rel1_stt_close_when_socket_already_broken_does_not_raise():
    stt, con, ws = await started_stt()
    ws.closed = True
    await stt.close()


async def test_ac5_stt_logs_never_contain_the_key(caplog):
    caplog.set_level(logging.DEBUG)
    con = Connector([None, OSError("down")])
    stt = DeepgramSTT(KEY, "nova-3", connect=con)
    await stt.send_audio(b"\x00\x01" * 10)
    con.sockets[0].feed("not json")
    await asyncio.sleep(0.3)
    await stt.close()
    assert KEY not in caplog.text


# ------------------------------------------------------------------------- TTS fakes


def audio_for(text: str) -> bytes:
    data = f"<{text}>".encode()
    return data + b" " * (len(data) % 2)


def tts_server(mode="ok"):
    """on_send handler: Speak remembers text; Flush answers with odd-sized audio chunks then Flushed."""
    texts: dict[int, list[str]] = {}

    def on_send(ws, raw):
        msg = json.loads(raw) if isinstance(raw, str) else {}
        queue = texts.setdefault(id(ws), [])
        if msg.get("type") == "Speak":
            queue.append(msg["text"])
        elif msg.get("type") == "Flush":
            text = queue.pop(0)
            data = audio_for(text)
            if mode == "error":
                ws.feed({"type": "Error", "description": "bad request"})
                return
            if mode == "drop_before_audio":
                ws.drop()
                return
            pieces = [data[i : i + 3] for i in range(0, len(data), 3)]
            if mode == "drop_mid_audio":
                ws.feed(pieces[0])
                ws.drop()
                return
            for p in pieces:
                ws.feed(p)
            ws.feed({"type": "Flushed"})

    return on_send


class _Body(httpx.AsyncByteStream):
    def __init__(self, data: bytes) -> None:
        self.data = data

    async def __aiter__(self):
        yield self.data


def rest_client(status=200, fail: Exception | None = None):
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if fail:
            raise fail
        text = json.loads(req.content)["text"]
        body = audio_for(text) if status == 200 else b"nope"
        return httpx.Response(status, stream=_Body(body))

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), seen


async def collect(it, timeout=3.0) -> list[bytes]:
    async def run():
        return [c async for c in it]

    return await asyncio.wait_for(run(), timeout)


def make_tts(mode="ok", plan=None, http=None):
    con = Connector(plan, on_send=tts_server(mode))
    return DeepgramTTS(KEY, "aura-test", connect=con, http=http or rest_client()[0]), con


# ------------------------------------------------------------------------- TTS


async def test_vox3_tts_yields_pcm_in_order_for_one_sentence():
    tts, con = make_tts()
    chunks = await collect(tts.synthesize("Hello there."))
    assert b"".join(chunks) == audio_for("Hello there.")
    await tts.close()


async def test_vox3_tts_chunks_are_whole_samples():
    tts, _ = make_tts()
    chunks = await collect(tts.synthesize("Odd sized pieces arrive"))
    assert chunks and all(len(c) % 2 == 0 and len(c) > 0 for c in chunks)
    await tts.close()


async def test_vox3_tts_pipelined_sentences_each_get_their_own_audio_in_order():
    tts, con = make_tts()
    its = [tts.synthesize(t) for t in ("First one.", "Second one.", "Third one.", "Fourth.")]
    out = await asyncio.gather(*(collect(i) for i in its))
    assert [b"".join(o) for o in out] == [audio_for(t) for t in ("First one.", "Second one.", "Third one.", "Fourth.")]
    assert len(con.calls) == 1, "one persistent socket"
    await tts.close()


async def test_vox3_tts_consumed_in_reverse_order_still_correct():
    tts, _ = make_tts()
    a, b = tts.synthesize("Alpha."), tts.synthesize("Bravo.")
    assert b"".join(await collect(b)) == audio_for("Bravo.")
    assert b"".join(await collect(a)) == audio_for("Alpha.")
    await tts.close()


async def test_ac5_tts_url_has_model_and_format_and_key_only_in_header():
    tts, con = make_tts()
    await collect(tts.synthesize("Hi."))
    url, kw = con.calls[0]
    q = _q(url)
    assert q["model"] == ["aura-test"] and q["encoding"] == ["linear16"] and q["sample_rate"] == ["24000"]
    assert KEY not in url
    assert kw["additional_headers"] == {"Authorization": f"Token {KEY}"}
    await tts.close()


async def test_ac5_tts_rest_fallback_key_only_in_header():
    client, seen = rest_client()
    tts = DeepgramTTS(KEY, "aura-test", connect=Connector([OSError("ws down")]), http=client)
    chunks = await collect(tts.synthesize("Via rest."))
    assert b"".join(chunks) == audio_for("Via rest.")
    (req,) = seen
    assert req.headers["authorization"] == f"Token {KEY}"
    assert KEY not in str(req.url) and _q(str(req.url))["model"] == ["aura-test"]
    assert json.loads(req.content) == {"text": "Via rest."}
    await tts.close()


async def test_rel1_tts_warm_failure_is_swallowed_and_synthesis_falls_back():
    client, seen = rest_client()
    tts = DeepgramTTS(KEY, "aura-test", connect=Connector([OSError("down"), OSError("down")]), http=client)
    await tts.warm()
    chunks = await collect(tts.synthesize("After failed warm."))
    assert b"".join(chunks) == audio_for("After failed warm.")


async def test_vox3_tts_warm_opens_the_socket_once():
    tts, con = make_tts()
    await tts.warm()
    await tts.warm()
    assert len(con.calls) == 1
    await collect(tts.synthesize("Go."))
    assert len(con.calls) == 1
    await tts.close()


async def test_rel1_tts_rest_http_error_surfaces_to_consumer():
    client, _ = rest_client(status=500)
    tts = DeepgramTTS(KEY, "aura-test", connect=Connector([OSError("down")]), http=client)
    with pytest.raises(Exception):
        await collect(tts.synthesize("Will fail."))


async def test_rel1_tts_rest_timeout_surfaces_to_consumer():
    client, _ = rest_client(fail=httpx.ReadTimeout("slow"))
    tts = DeepgramTTS(KEY, "aura-test", connect=Connector([OSError("down")]), http=client)
    with pytest.raises(httpx.ReadTimeout):
        await collect(tts.synthesize("Will time out."))


async def test_rel1_tts_websocket_and_rest_both_down_raises_not_hangs():
    client, _ = rest_client(status=503)
    tts = DeepgramTTS(KEY, "aura-test", connect=Connector([OSError("down")]), http=client)
    with pytest.raises(Exception):
        await collect(tts.synthesize("Everything is down."))


async def test_rel1_tts_websocket_drop_before_any_audio_falls_back_to_rest():
    client, seen = rest_client()
    tts = DeepgramTTS(KEY, "aura-test", connect=Connector(on_send=tts_server("drop_before_audio")), http=client)
    chunks = await collect(tts.synthesize("Rescued by rest."))
    assert b"".join(chunks) == audio_for("Rescued by rest.") and len(seen) == 1


async def test_rel1_tts_websocket_drop_mid_audio_terminates_the_stream():
    client, _ = rest_client()
    tts = DeepgramTTS(KEY, "aura-test", connect=Connector(on_send=tts_server("drop_mid_audio")), http=client)
    try:
        chunks = await collect(tts.synthesize("Cut off in the middle."), timeout=2)
    except Exception:
        return  # surfacing the failure is also acceptable
    assert len(b"".join(chunks)) <= len(audio_for("Cut off in the middle."))


async def test_rel1_tts_server_error_message_surfaces_instead_of_hanging():
    client, _ = rest_client()
    tts = DeepgramTTS(KEY, "aura-test", connect=Connector(on_send=tts_server("error")), http=client)
    try:
        await collect(tts.synthesize("Rejected text."), timeout=1.5)
    except asyncio.TimeoutError:
        pytest.fail("synthesis hung after Deepgram reported an error")
    except Exception:
        pass


async def test_vox3_tts_flush_budget_respected_and_every_sentence_still_spoken():
    tts, con = make_tts()
    words = [f"Sentence number {i}." for i in range(25)]
    its = [tts.synthesize(w) for w in words]
    out = await asyncio.gather(*(collect(i, 6) for i in its))
    assert [b"".join(o) for o in out] == [audio_for(w) for w in words]
    flushes = sum(m["type"] == "Flush" for ws in con.sockets for m in ws.texts)
    assert flushes <= 20, "Deepgram allows 20 Flush messages per 60 s on one socket"
    await tts.close()


async def test_vox3_tts_long_text_is_cut_to_the_provider_limit():
    tts, con = make_tts()
    await collect(tts.synthesize("x" * 5000))
    speaks = [m for ws in con.sockets for m in ws.texts if m["type"] == "Speak"]
    assert speaks and all(len(m["text"]) <= 2000 for m in speaks)
    await tts.close()


async def test_vox3_tts_empty_audio_chunks_are_not_yielded():
    def on_send(ws, raw):
        msg = json.loads(raw)
        if msg["type"] == "Flush":
            for p in (b"", b"\x01\x00\x02\x00", b"", b"\x03\x00"):
                ws.feed(p)
            ws.feed({"type": "Flushed"})

    tts = DeepgramTTS(KEY, "m", connect=Connector(on_send=on_send), http=rest_client()[0])
    chunks = await collect(tts.synthesize("x"))
    assert chunks == [b"\x01\x00\x02\x00", b"\x03\x00"]


async def test_vox2_tts_reset_ends_pending_streams_closes_socket_and_reopens_lazily():
    def on_send(ws, raw):  # never answers: sentence stays pending
        pass

    con = Connector(on_send=on_send)
    tts = DeepgramTTS(KEY, "m", connect=con, http=rest_client()[0])
    it = tts.synthesize("Pending forever.")
    await asyncio.sleep(0.1)
    await tts.reset()
    assert await collect(it, 2) == []
    assert con.sockets[0].closed
    await tts.warm()
    assert len(con.calls) == 2
    await tts.close()


async def test_vox2_tts_reset_with_nothing_pending_and_double_reset():
    tts, con = make_tts()
    await tts.reset()
    await tts.reset()
    assert (b"".join(await collect(tts.synthesize("Still works.")))) == audio_for("Still works.")
    await tts.close()


async def test_vox2_tts_reset_during_streaming_discards_unsent_audio():
    def on_send(ws, raw):
        pass

    tts = DeepgramTTS(KEY, "m", connect=Connector(on_send=on_send), http=rest_client()[0])
    its = [tts.synthesize(f"S{i}.") for i in range(5)]
    await asyncio.sleep(0.1)
    await tts.reset()
    for it in its:
        assert await collect(it, 2) == []


async def test_rel1_tts_close_sends_close_and_closes_socket():
    tts, con = make_tts()
    await collect(tts.synthesize("Bye."))
    await tts.close()
    ws = con.sockets[0]
    assert {"type": "Close"} in ws.texts and ws.closed


async def test_rel1_tts_close_is_idempotent_and_safe_without_connection():
    tts, con = make_tts()
    await tts.close()
    await tts.close()
    assert con.calls == []


async def test_rel1_tts_close_cancels_background_tasks():
    before = set(asyncio.all_tasks())
    tts, con = make_tts()
    its = [tts.synthesize(f"S{i}.") for i in range(3)]
    await asyncio.gather(*(collect(i) for i in its))
    await tts.close()
    await asyncio.sleep(0.2)
    assert not {t for t in asyncio.all_tasks() - before if not t.done()}


async def test_ac5_tts_logs_never_contain_the_key(caplog):
    caplog.set_level(logging.DEBUG)
    client, _ = rest_client(status=500)
    tts = DeepgramTTS(KEY, "m", connect=Connector([OSError("down")]), http=client)
    with pytest.raises(Exception):
        await collect(tts.synthesize("log check"))
    await tts.close()
    assert KEY not in caplog.text


async def test_vox3_tts_consumer_abandoning_iteration_does_not_break_next_sentence():
    tts, _ = make_tts()
    it = tts.synthesize("Abandoned sentence here.")
    await it.__anext__()
    await it.aclose()
    assert b"".join(await collect(tts.synthesize("Next one."))) == audio_for("Next one.")
    await tts.close()


async def test_vox3_tts_default_construction_opens_nothing():
    con = Connector()
    DeepgramTTS(KEY, "m", connect=con)
    DeepgramSTT(KEY, "m", connect=con)
    await asyncio.sleep(0.05)
    assert con.calls == []
