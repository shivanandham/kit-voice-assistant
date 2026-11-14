"""VOX1 and VOX4: exact message shapes, malformed input, typed text."""

import json

import pytest
from app.voice.deepgram import SttEvent

from tests.acc_voice_helpers import (
    Agent,
    FakeTts,
    audio_bytes,
    check_audio_protocol,
    events_of,
    make_server,  # noqa: F401
    make_session,
    types,
)


async def _open(make_server, agent=None, **kw):
    agent = agent or Agent()
    srv = await make_server(agent, **kw)
    token, ctx = await make_session("student", None, "Pat Student")
    c = await srv.connect(token)
    ready = await c.start()
    return srv, agent, c, ready


# ------------------------------------------------------------------------- exact events


async def test_vox1_full_spoken_turn_event_names_and_fields(make_server):
    agent = Agent(["Admissions are open. ", "Apply online."])
    srv, _, c, ready = await _open(make_server, agent)
    async with c:
        await c.mic(b"S")
        await c.mic(b"Iwhat about")
        await c.mic(b"Uwhat about admissions")
        frames = await c.until("audio_end")
    interim, final = events_of(frames, "transcript")
    assert interim == {"type": "transcript", "final": False, "text": "what about"}
    assert final == {"type": "transcript", "final": True, "text": "what about admissions"}
    assert events_of(frames, "tool") and all(set(t) == {"type", "name", "state"} for t in events_of(frames, "tool"))
    assert [t["state"] for t in events_of(frames, "tool")] == ["start", "end"]
    for d in events_of(frames, "assistant_delta"):
        assert set(d) == {"type", "text"}
    assert [d["text"] for d in events_of(frames, "assistant_delta")] == ["Admissions are open. ", "Apply online."]
    (done,) = events_of(frames, "assistant_done")
    assert done == {"type": "assistant_done", "text": "Admissions are open. Apply online."}
    (start,) = events_of(frames, "audio_start")
    assert start == {"type": "audio_start", "sample_rate": 24000, "format": "pcm_s16le"}
    (end,) = events_of(frames, "audio_end")
    assert end == {"type": "audio_end"}
    check_audio_protocol(frames)
    assert audio_bytes(frames)
    allowed = {"transcript", "assistant_delta", "assistant_done", "tool", "audio_start", "audio_end"}
    assert set(types(frames)) - {"audio"} <= allowed
    assert ready["type"] == "ready"


async def test_vox1_spoken_turn_ordering(make_server):
    agent = Agent(["One. ", "Two."])
    _, _, c, _ = await _open(make_server, agent)
    async with c:
        await c.mic(b"Uhello there friend")
        frames = await c.until("audio_end")
    t = types(frames)
    assert t[0] == "transcript"
    assert t.index("transcript") < t.index("assistant_delta") < t.index("assistant_done")
    assert t.index("assistant_delta") < t.index("audio_start") < t.index("audio") < t.index("audio_end")
    assert t.count("audio_start") == 1 and t.count("audio_end") == 1


async def test_vox1_typed_turn_event_names_and_fields(make_server):
    agent = Agent(["Typed answer."])
    _, _, c, _ = await _open(make_server, agent)
    async with c:
        await c.send_json({"type": "text", "text": "hello"})
        frames = await c.until("audio_end")
    assert set(types(frames)) - {"audio"} == {"tool", "assistant_delta", "assistant_done", "audio_start", "audio_end"}
    assert events_of(frames, "assistant_done") == [{"type": "assistant_done", "text": "Typed answer."}]


async def test_vox1_interim_transcripts_are_not_final(make_server):
    _, agent, c, _ = await _open(make_server)
    async with c:
        await c.mic(b"Ione two")
        assert await c.recv() == {"type": "transcript", "final": False, "text": "one two"}
        assert agent.calls == []


async def test_vox1_speech_started_event_is_not_leaked_as_unknown(make_server):
    _, _, c, _ = await _open(make_server)
    async with c:
        await c.mic(b"S")
        got = await c.drain(0.4)
    assert all(isinstance(f, dict) and f["type"] in {"transcript"} for f in got)


async def test_vox1_tool_event_forwards_only_contract_fields(make_server):
    agent = Agent(script=lambda t: [
        {"type": "tool", "name": "get_attendance", "state": "start", "is_error": True, "input": {"x": 1}, "token": "t"},
        {"type": "done", "text": ""},
    ])
    _, _, c, _ = await _open(make_server, agent)
    async with c:
        await c.send_json({"type": "text", "text": "attendance"})
        frames = await c.until("assistant_done")
    assert events_of(frames, "tool") == [{"type": "tool", "name": "get_attendance", "state": "start"}]


async def test_vox1_unknown_agent_events_are_not_forwarded(make_server):
    agent = Agent(script=lambda t: [
        {"type": "debug", "secret": "x"},
        {"type": "text_delta", "text": "Hi. "},
        {"type": "done", "text": "Hi. "},
    ])
    _, _, c, _ = await _open(make_server, agent)
    async with c:
        await c.send_json({"type": "text", "text": "x"})
        frames = await c.until("audio_end")
    assert "debug" not in types(frames)


# ------------------------------------------------------------------------- malformed input


@pytest.mark.parametrize(
    "raw",
    ["not json", "", "{", "{'type': 'start'}", "[1, 2]", "42", "\"start\"", "null", "true", "\x00\x01", "{\"type\":"],
)
async def test_vox1_non_object_or_invalid_json_gets_error_and_socket_survives(make_server, raw):
    _, agent, c, _ = await _open(make_server)
    async with c:
        await c.send_text(raw)
        msg = await c.recv()
        assert msg["type"] == "error" and set(msg) == {"type", "message"}
        assert isinstance(msg["message"], str) and msg["message"]
        await c.send_json({"type": "text", "text": "still alive"})
        await c.until("assistant_done")
    assert agent.texts == ["still alive"]


@pytest.mark.parametrize(
    "msg",
    [
        {"type": "bogus"},
        {"type": "START"},
        {"type": "Text", "text": "x"},
        {"type": None},
        {"type": 5},
        {"type": ["start"]},
        {"type": {"a": 1}},
        {},
        {"text": "no type"},
        {"type": ""},
    ],
)
async def test_vox1_unknown_message_type_gets_error(make_server, msg):
    _, agent, c, _ = await _open(make_server)
    async with c:
        await c.send_json(msg)
        got = await c.recv()
        assert got["type"] == "error"
        await c.send_json({"type": "text", "text": "ok"})
        await c.until("assistant_done")
    assert agent.texts == ["ok"]


@pytest.mark.parametrize("payload", [["a"], {"a": 1}, True, 12345, 1.5, None, False, [], {}])
async def test_vox1_text_with_non_string_payload_starts_no_turn(make_server, payload):
    _, agent, c, _ = await _open(make_server)
    async with c:
        await c.send_json({"type": "text", "text": payload})
        await c.drain(0.4)
        await c.send_json({"type": "text", "text": "real"})
        await c.until("assistant_done")
    assert agent.texts == ["real"]


async def test_vox1_text_without_text_key_is_ignored(make_server):
    _, agent, c, _ = await _open(make_server)
    async with c:
        await c.send_json({"type": "text"})
        got = await c.drain(0.4)
    assert agent.calls == [] and "assistant_delta" not in types(got)


async def test_vox1_oversized_control_message_rejected(make_server):
    _, agent, c, _ = await _open(make_server)
    async with c:
        await c.send_text(json.dumps({"type": "text", "text": "a" * 50_000}))
        msg = await c.recv()
        assert msg["type"] == "error"
        await c.send_json({"type": "text", "text": "after"})
        await c.until("assistant_done")
    assert agent.texts == ["after"]


async def test_vox1_huge_control_message_one_megabyte(make_server):
    _, agent, c, _ = await _open(make_server)
    async with c:
        await c.send_text(json.dumps({"type": "interrupt", "pad": "x" * 1_000_000}))
        msg = await c.recv()
        assert msg["type"] == "error"
        await c.send_json({"type": "text", "text": "after"})
        await c.until("assistant_done")


async def test_vox1_interrupt_with_extra_fields_ok(make_server):
    _, agent, c, _ = await _open(make_server)
    async with c:
        await c.send_json({"type": "interrupt", "reason": "x"})
        got = await c.drain(0.3)
    assert "error" not in types(got)


async def test_vox1_many_garbage_messages_in_a_row_do_not_kill_socket(make_server):
    _, agent, c, _ = await _open(make_server)
    async with c:
        for i in range(50):
            await c.send_text("garbage %d" % i)
        got = await c.drain(0.6)
        assert types(got) == ["error"] * 50
        await c.send_json({"type": "text", "text": "fine"})
        await c.until("assistant_done")


# ------------------------------------------------------------------------- binary frames


async def test_vox1_zero_length_binary_frame_is_harmless(make_server):
    srv, _, c, _ = await _open(make_server)
    async with c:
        await c.send_bytes(b"")
        got = await c.drain(0.3)
        assert "error" not in types(got)
        await c.send_json({"type": "text", "text": "still fine"})
        await c.until("assistant_done")
    assert all(f != b"" for s in srv.stts for f in s.frames)


async def test_vox1_odd_length_binary_frame_reaches_stt_as_whole_samples(make_server):
    srv, _, c, _ = await _open(make_server)
    async with c:
        await c.send_bytes(b"\x00" + b"\x01\x02" * 400 + b"\x03")
        await c.send_bytes(b"\x00")
        await c.send_bytes(b"\x00\x05\x06")
        await c.drain(0.3)
        await c.send_json({"type": "text", "text": "ok"})
        await c.until("assistant_done")
    sent = [f for s in srv.stts for f in s.frames]
    assert sent and all(len(f) % 2 == 0 for f in sent)


@pytest.mark.parametrize("size", [3_201, 64_000, 5_000_000])
async def test_vox1_oversized_audio_frame_is_not_relayed_whole(make_server, size):
    srv, _, c, _ = await _open(make_server)
    async with c:
        await c.send_bytes(b"\x00" * size)
        await c.drain(0.3)
        await c.send_json({"type": "text", "text": "ok"})
        await c.until("assistant_done")
    # contract: frames are 20-100 ms (<= 3200 bytes); a frame of `size` bytes must not be passed on unbounded
    assert all(len(f) <= 100_000 for s in srv.stts for f in s.frames)


async def test_vox1_valid_audio_frames_are_relayed_unchanged_in_order(make_server):
    srv, _, c, _ = await _open(make_server)
    frames = [bytes([0, i]) * 160 for i in range(1, 30)]  # 20 ms each, starting with NUL (not a script)
    async with c:
        for f in frames:
            await c.send_bytes(f)
        await c.drain(0.5)
    assert [f for s in srv.stts for f in s.frames] == frames


# ------------------------------------------------------------------------- VOX4 typed text


async def test_vox4_typed_text_works_without_any_audio(make_server):
    srv, agent, c, _ = await _open(make_server, Agent(["Fees are listed online."]))
    async with c:
        await c.send_json({"type": "text", "text": "what are the fees"})
        frames = await c.until("audio_end")
    assert agent.texts == ["what are the fees"]
    assert all(not s.frames for s in srv.stts)
    assert "assistant_done" in types(frames)


async def test_vox4_typed_text_is_never_echoed_as_transcript(make_server):
    _, _, c, _ = await _open(make_server)
    async with c:
        await c.send_json({"type": "text", "text": "echo check"})
        frames = await c.until("audio_end") + await c.drain(0.3)
    assert "transcript" not in types(frames)
    assert "echo check" not in [f.get("text") for f in frames if isinstance(f, dict) and f["type"] != "assistant_done"]


@pytest.mark.parametrize("text", ["", " ", "   ", "\n", "\t \n ", "\r\n\r\n", " ", " "])
async def test_vox4_blank_text_is_ignored(make_server, text):
    _, agent, c, _ = await _open(make_server)
    async with c:
        await c.send_json({"type": "text", "text": text})
        got = await c.drain(0.4)
    assert agent.calls == [] and got == []


async def test_vox4_text_is_trimmed_and_otherwise_unchanged(make_server):
    _, agent, c, _ = await _open(make_server)
    async with c:
        await c.send_json({"type": "text", "text": "  line one\nline two  "})
        await c.until("assistant_done")
    assert agent.texts == ["line one\nline two"]


@pytest.mark.parametrize("text", ["வணக்கம் எப்படி இருக்கீங்க", "नमस्ते", "Привет", "emoji \U0001f600 ok", "<script>alert(1)</script>", "{\"type\":\"interrupt\"}"])
async def test_vox4_unicode_and_markup_text_is_passed_through_verbatim(make_server, text):
    _, agent, c, _ = await _open(make_server)
    async with c:
        await c.send_json({"type": "text", "text": text})
        await c.until("assistant_done")
    assert agent.texts == [text]


@pytest.mark.parametrize("n", [2_001, 5_000, 8_000])
async def test_vox4_very_long_text_is_bounded(make_server, n):
    _, agent, c, _ = await _open(make_server)
    async with c:
        await c.send_json({"type": "text", "text": "a" * n})
        await c.drain(0.5)
        await c.send_json({"type": "text", "text": "short"})
        await c.until("assistant_done")
    assert all(len(t) <= 4_000 for t in agent.texts)
    assert "short" in agent.texts


async def test_vox4_typed_text_during_idle_stt_does_not_need_stt_events(make_server):
    srv, agent, c, _ = await _open(make_server)
    async with c:
        for i in range(3):
            await c.send_json({"type": "text", "text": f"q{i}"})
            await c.until("assistant_done")
    assert agent.texts == ["q0", "q1", "q2"]


async def test_vox4_typed_turn_after_voice_turn_has_no_transcript(make_server):
    _, agent, c, _ = await _open(make_server)
    async with c:
        await c.mic(b"Uspoken question here")
        await c.until("audio_end")
        await c.send_json({"type": "text", "text": "typed question"})
        frames = await c.until("audio_end")
    assert "transcript" not in types(frames)
    assert agent.texts == ["spoken question here", "typed question"]


async def test_vox1_final_transcript_empty_text_starts_nothing(make_server):
    srv, agent, c, _ = await _open(make_server)
    async with c:
        await c.mic(b"S")
        await c.drain(0.2)
        srv.stts[0].push(SttEvent("utterance", ""))
        got = await c.drain(0.4)
    assert agent.calls == [], "an empty utterance must not start a turn"


async def test_vox1_citations_event_is_forwarded_unchanged_before_assistant_done(make_server):
    sources = [{"kind": "document", "ref": "D2:6.4.3", "document_title": "Student Handbook", "version": "2026.1",
                "section": "6.4.3", "heading": "Leave", "page_start": 14, "page_end": 14},
               {"kind": "college_record", "label": "College system: attendance"}]
    agent = Agent(script=lambda t: [
        {"type": "text_delta", "text": "Leave needs a note. "},
        {"type": "citations", "sources": sources},
        {"type": "done", "text": "Leave needs a note. ", "citations": sources},
    ])
    _, _, c, _ = await _open(make_server, agent)
    async with c:
        await c.send_json({"type": "text", "text": "leave rule"})
        frames = await c.until("audio_end")
    cites = events_of(frames, "citations")
    assert cites == [{"type": "citations", "sources": sources}]
    t = types(frames)
    assert t.index("citations") < t.index("assistant_done")
