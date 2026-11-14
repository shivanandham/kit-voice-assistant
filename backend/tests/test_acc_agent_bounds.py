"""Acceptance: long conversations, rebuild from stored history, save failures, warm-up variants.

Criteria: AGT2 (valid history however long), AGT3, AGT6, AGT5 (per-department prefixes), REL2.
"""

import asyncio
import json

import anthropic
import httpx
import pytest
from app import history
from app.agent import loop
from app.agent.tools import ToolResult
from app.config import get_settings
from tests.acc_agent_helpers import (  # noqa: F401
    ARJUN, PRIYA, VIS1, Reply, assert_valid, cancel_when, collect, db_messages, dump, env_factory,
    evict_memory, new_conv, text, tool_use,
)

REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


async def test_AGT2_long_conversation_with_tools_stays_valid_and_bounded(env_factory):
    def script(kw, n):
        if "tool_result" not in dump(kw["messages"][-1]):
            return Reply(["Checking. "], [text("Checking. "), tool_use(f"t{n}", "get_my_profile")],
                         "tool_use")
        return Reply([f"Answer {n}."])
    env = env_factory(script)
    cid = await new_conv(ARJUN)
    for i in range(40):
        await collect(ARJUN, cid, f"question {i}")
    for call in env.client.calls:
        assert_valid(call["messages"])
    last = env.client.calls[-1]["messages"]
    assert len(last) <= loop.MAX_CACHED_MESSAGES + 3
    assert "question 39" in dump(last) and "question 0" not in dump(last)
    assert len(await db_messages(cid)) == 80  # nothing lost from storage


async def test_AGT2_rebuild_after_many_turns_is_valid_and_recent(env_factory):
    env = env_factory(lambda kw, n: Reply([f"A{n}."]))
    cid = await new_conv(VIS1)
    for i in range(30):
        await collect(VIS1, cid, f"q{i}")
    evict_memory()
    await collect(VIS1, cid, "after rebuild")
    msgs = env.client.calls[-1]["messages"]
    assert_valid(msgs)
    assert "q29" in dump(msgs) and "after rebuild" in dump(msgs)


async def test_AGT2_rebuild_skips_leading_assistant_rows_and_empty_rows(env_factory):
    env = env_factory([Reply(["Fine."])])
    cid = await new_conv(VIS1)
    await history.add_message(cid, "assistant", "orphan greeting")
    await history.add_message(cid, "user", "real question")
    await history.add_message(cid, "assistant", "   ")
    await history.add_message(cid, "assistant", "real answer")
    await collect(VIS1, cid, "next")
    msgs = env.client.calls[0]["messages"]
    assert_valid(msgs)
    assert "orphan greeting" not in dump(msgs)


async def test_AGT2_rebuild_merges_consecutive_same_role_rows(env_factory):
    env = env_factory([Reply(["Fine."])])
    cid = await new_conv(VIS1)
    for t in ("one", "two", "three"):
        await history.add_message(cid, "user", t)  # e.g. turns whose model call failed
    await collect(VIS1, cid, "four")
    msgs = env.client.calls[0]["messages"]
    assert_valid(msgs)
    flat = dump(msgs)
    for t in ("one", "two", "three", "four"):
        assert flat.count(t) == 1


async def test_AGT3_rebuilt_interrupted_reply_is_marked_for_the_model_not_shown_raw(env_factory):
    env = env_factory([Reply(["Fine."])])
    cid = await new_conv(VIS1)
    await history.add_message(cid, "user", "tell me")
    await history.add_message(cid, "assistant", "I was saying" + loop.INTERRUPTED_MARK)
    await collect(VIS1, cid, "go on")
    msgs = env.client.calls[0]["messages"]
    assert_valid(msgs)
    assert loop.INTERRUPTED_MARK not in dump(msgs)
    assert "I was saying" in dump(msgs)


async def test_AGT3_cancel_survives_a_failing_save(env_factory, monkeypatch):
    r = Reply(["Said this. "], hang="after_chunks")
    env = env_factory([r, Reply(["Back."])])
    cid = await new_conv(VIS1)
    real = history.add_message

    async def flaky(c, role, content):
        if role == "assistant":
            raise RuntimeError("db gone")
        return await real(c, role, content)
    monkeypatch.setattr(history, "add_message", flaky)
    await cancel_when(VIS1, cid, "hi", lambda g: any(e["type"] == "text_delta" for e in g))
    monkeypatch.setattr(history, "add_message", real)
    events = await collect(VIS1, cid, "again")
    assert events[-1]["type"] == "done"
    assert_valid(env.client.calls[-1]["messages"])


async def test_AGT6_error_path_survives_a_failing_save_and_stays_safe(env_factory, monkeypatch):
    env_factory([Reply(["Half. "], mid_error=anthropic.APIConnectionError(request=REQ))])
    cid = await new_conv(VIS1)
    real = history.add_message

    async def flaky(c, role, content):
        if role == "assistant":
            raise RuntimeError("db gone")
        return await real(c, role, content)
    monkeypatch.setattr(history, "add_message", flaky)
    events = await collect(VIS1, cid, "hi")
    assert events[-1]["type"] == "error"
    assert "db gone" not in json.dumps(events)


async def test_AGT6_user_message_save_failure_is_safe(env_factory, monkeypatch):
    # DEFECT: when saving the user message fails, run_turn raises a raw exception instead of a safe error event.
    env_factory([Reply(["x"])])
    cid = await new_conv(VIS1)

    async def boom(*a, **k):
        raise RuntimeError("db gone")
    monkeypatch.setattr(history, "add_message", boom)
    events = []
    try:
        async for e in loop.run_turn(VIS1, cid, "hi"):
            events.append(e)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"run_turn crashed with {type(exc).__name__}")
    assert events[-1]["type"] == "error"


async def test_AGT1_model_is_not_called_if_the_user_message_could_not_be_saved(env_factory, monkeypatch):
    env = env_factory([Reply(["x"])])
    cid = await new_conv(VIS1)

    async def boom(*a, **k):
        raise RuntimeError("db gone")
    monkeypatch.setattr(history, "add_message", boom)
    try:
        [e async for e in loop.run_turn(VIS1, cid, "hi")]
    except Exception:  # noqa: BLE001
        pass
    assert env.client.calls == []


def test_REL2_default_client_is_built_without_any_network_and_with_bounded_retries(monkeypatch):
    monkeypatch.setattr(loop, "_client", None)
    c = loop._get_client()
    assert isinstance(c, anthropic.AsyncAnthropic)
    assert loop._get_client() is c
    assert c.max_retries <= 3
    assert c.timeout is not None


async def test_AGT5_warm_up_sends_one_request_per_distinct_prefix(env_factory, monkeypatch):
    from app.knowledge import store

    async def docs():
        return [{"type": "lesson_notes", "applies_to": {"departments": ["CSE"]}},
                {"type": "lesson_notes", "applies_to": {"departments": ["ECE"]}},
                {"type": "lesson_notes", "applies_to": {"departments": ["ECE"]}},
                {"type": "handbook", "applies_to": None}]
    monkeypatch.setattr(store, "active_documents", docs)
    env = env_factory([Reply(["x"])],
                      knowledge=lambda ctx: "KB " + str((ctx.user or {}).get("department")))
    await loop.warm_cache()
    keys = [json.dumps([c["tools"], c["system"]], sort_keys=True) for c in env.client.create_calls]
    assert len(keys) == len(set(keys)), "identical prefixes are warmed once"
    texts = [c["system"][0]["text"] for c in env.client.create_calls]
    assert any("KB CSE" in t for t in texts) and any("KB ECE" in t for t in texts)
    for c in env.client.create_calls:
        assert "Quintday" not in dump(c) and "2031" not in dump(c)


async def test_AGT6_warm_up_logs_and_survives_per_variant_failure(env_factory, monkeypatch):
    env = env_factory([Reply(["x"])])
    n = {"i": 0}

    async def flaky(**kw):
        n["i"] += 1
        if n["i"] == 1:
            raise anthropic.APIConnectionError(request=REQ)
        return await type(env.client).create(env.client, **kw)
    monkeypatch.setattr(env.client, "create", flaky)
    await loop.warm_cache()
    assert n["i"] >= 2


async def test_AGT6_stream_closed_even_when_model_errors_midway(env_factory):
    env = env_factory([Reply(["Part."], mid_error=anthropic.APIConnectionError(request=REQ))])
    cid = await new_conv(VIS1)
    await collect(VIS1, cid, "x")
    assert env.client.streams[0].closed


@pytest.mark.parametrize("extra", [0, 1, 2, 3])
def test_AGT2_trimmed_history_starts_at_a_real_user_turn_and_stays_valid(extra):
    msgs = []
    for i in range(loop.MAX_CACHED_MESSAGES // 2 + extra + 10):
        msgs.append({"role": "user", "content": [{"type": "text", "text": f"q{i}"}]})
        msgs.append({"role": "assistant", "content": [tool_use(f"t{i}", "x")]})
        msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}",
                                                 "content": "{}"}]})
        msgs.append({"role": "assistant", "content": [text("ok")]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": "last"}]})
    trimmed = loop._trim(msgs)
    assert len(trimmed) <= loop.MAX_CACHED_MESSAGES + 1
    assert_valid(trimmed)
    assert trimmed[-1] == msgs[-1]
    assert trimmed[0]["content"][0]["type"] == "text"


def test_AGT2_remember_trims_and_evicts_oldest_first(monkeypatch):
    loop._memory.clear()
    monkeypatch.setattr(loop, "CACHE_CONVERSATIONS", 3)
    for i in range(5):
        loop._remember(f"c{i}", [{"role": "user", "content": [{"type": "text", "text": "x"}]}])
    assert list(loop._memory) == ["c2", "c3", "c4"]
