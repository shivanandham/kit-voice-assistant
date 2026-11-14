"""REL1: when the primary model fails before any text, a fallback model answers (fakes only)."""

import json
from types import SimpleNamespace

import anthropic
import httpx
import pytest
from app import db, history
from app.agent import fallback, loop

from tests.test_agent_loop import KNOWLEDGE, VISITOR, Reply, collect, conv, make_setup

REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.fixture
def setup(monkeypatch):
    return make_setup(monkeypatch)


@pytest.fixture
def fb(monkeypatch):
    calls = []

    def install(reply="Fees are fifty thousand rupees.\nSOURCES: D3:2", enabled=True):
        async def answer(system, messages):
            calls.append({"system": system, "messages": messages})
            return reply
        monkeypatch.setattr(loop.fallback, "answer", answer)
        values = {**loop.get_settings().model_dump(), "llm_fallback_enabled": enabled,
                  "moderation_llm": False}
        monkeypatch.setattr(loop, "get_settings", lambda: SimpleNamespace(**values))
        return calls
    return install


async def test_rel1_primary_failure_is_answered_by_fallback_and_saved(setup, fb):
    calls = fb()
    setup([Reply([], error=anthropic.APIConnectionError(request=REQ))])
    cid = await conv()
    events = await collect(VISITOR, cid, "what are the fees")
    assert events[-1]["type"] == "done" and events[-1]["text"] == "Fees are fifty thousand rupees."
    assert any(e["type"] == "text_delta" for e in events)
    assert "SOURCES" not in json.dumps(events)
    saved = await history.recent_messages(cid)
    assert saved[-1] == {"role": "assistant", "content": "Fees are fifty thousand rupees."}
    rows = await db.pool().fetch("SELECT name FROM tool_calls WHERE conversation_id = $1::uuid", cid)
    assert [r["name"] for r in rows] == ["fallback"]
    sent = json.dumps(calls[0], default=str)
    assert KNOWLEDGE in calls[0]["system"][0]["text"]
    assert "what are the fees" in sent and "tools" not in calls[0]


async def test_rel1_fallback_request_carries_no_tools_or_secrets(monkeypatch):
    seen = {}

    class Models:
        async def generate_content(self, **kw):
            seen.update(kw)
            return SimpleNamespace(text="Hello.")

    monkeypatch.setattr(fallback, "_get_client", lambda: SimpleNamespace(aio=SimpleNamespace(
        models=Models())))
    out = await fallback.answer([{"type": "text", "text": "INSTR"}],
                                [{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
    assert out == "Hello."
    cfg = seen["config"]
    assert not getattr(cfg, "tools", None)
    assert "INSTR" in cfg.system_instruction and "temporarily unavailable" in cfg.system_instruction
    dumped = json.dumps(str(seen))
    key = loop.get_settings().anthropic_api_key
    assert not key or key not in dumped


async def test_rel1_fallback_failure_never_raises(monkeypatch):
    class Models:
        async def generate_content(self, **kw):
            raise RuntimeError("down")

    monkeypatch.setattr(fallback, "_get_client", lambda: SimpleNamespace(aio=SimpleNamespace(
        models=Models())))
    assert await fallback.answer([{"text": "x"}], [{"role": "user", "content": "hi"}]) is None


async def test_rel1_no_fallback_after_partial_text(setup, fb):
    calls = fb()
    setup([Reply(["Half an answer. ", "more"], delay=0.0)])
    client = loop._client
    orig = client.stream

    def failing_stream(**kw):
        s = orig(**kw)

        async def events():
            yield SimpleNamespace(type="text", text="Half an answer. ")
            raise anthropic.APIConnectionError(request=REQ)
        s._events = events
        return s
    client.stream = failing_stream
    cid = await conv()
    events = await collect(VISITOR, cid, "q")
    assert events[-1]["type"] == "error" and calls == []


async def test_rel1_both_fail_gives_error(setup, fb):
    fb(reply=None)
    setup([Reply([], error=anthropic.APIConnectionError(request=REQ))])
    cid = await conv()
    events = await collect(VISITOR, cid, "q")
    assert events == [{"type": "error", "message": loop.ERROR_TEXT}]


async def test_rel1_flag_off_gives_error_as_before(setup, fb):
    calls = fb(enabled=False)
    setup([Reply([], error=anthropic.APIConnectionError(request=REQ))])
    cid = await conv()
    events = await collect(VISITOR, cid, "q")
    assert events[-1]["type"] == "error" and calls == []
