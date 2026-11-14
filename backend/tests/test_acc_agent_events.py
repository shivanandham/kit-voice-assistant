"""Acceptance: event stream, persistence, validity, bounds and failures of the assistant loop.

Criteria: AGT1, AGT2, AGT4, AGT6, HIS1, AC4 (as seen from the loop), REL2.
"""

import asyncio
import json
import re

import anthropic
import httpx
import pytest
from app import history
from app.agent import loop
from app.agent.tools import ToolResult
from tests.acc_agent_helpers import (  # noqa: F401  (env_factory is a fixture)
    ARJUN, KB_TEXT, LAKSHMI, PRIYA, VIS1, Reply, assert_valid, blocks, bget, btype, collect,
    db_messages, db_tool_calls, dump, env_factory, new_conv, text, tool_use,
)

REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
SECRET_DETAIL = "internal-detail-xyzzy-9981"

TYPES = {
    "text_delta": {"type", "text"},
    "tool": {"type", "name", "state", "is_error"},
    "done": {"type", "text"},
    "citations": {"type", "sources"},
    "error": {"type", "message"},
}


def check_event_shapes(events):
    """Every event has exactly the fields CONTRACT.md lists for its type."""
    assert events
    for e in events:
        assert e["type"] in TYPES, e
        allowed = TYPES[e["type"]] | ({"citations"} if e["type"] == "done" else set())
        assert TYPES[e["type"]] <= set(e) <= allowed, e
    cites = [e for e in events if e["type"] == "citations"]
    assert len(cites) <= 1
    if cites:
        assert events[-2] is cites[0], "citations come just before done"
        assert cites[0]["sources"], "a citations event is only sent when non-empty"
        assert events[-1].get("citations") == cites[0]["sources"]
    else:
        assert "citations" not in events[-1] or not events[-1]["citations"]
    for e in events:
        if e["type"] == "text_delta":
            assert isinstance(e["text"], str) and e["text"]
        if e["type"] == "tool":
            assert e["state"] in ("start", "end") and isinstance(e["is_error"], bool)
            assert isinstance(e["name"], str)
        if e["type"] == "error":
            assert isinstance(e["message"], str) and e["message"].strip()
    if events[-1]["type"] == "done":
        assert "".join(e["text"] for e in events if e["type"] == "text_delta").strip() == \
            events[-1]["text"]
    terminal = [e for e in events if e["type"] in ("done", "error")]
    assert len(terminal) == 1 and events[-1] is terminal[0], "exactly one terminal event, last"


def deltas(events):
    return "".join(e["text"] for e in events if e["type"] == "text_delta")


def assert_safe(message: str):
    """No stack trace, exception class or detail in user-facing text."""
    low = message.lower()
    for bad in ("traceback", "exception", "error:", "apistatuserror", "ratelimit", "httpx",
                "anthropic", SECRET_DETAIL, "line ", ".py", "429", "500", "529"):
        assert bad.lower() not in low, f"unsafe text {message!r} contains {bad!r}"


# ------------------------------------------------------------------ event stream (AGT1/AGT2)

async def test_AGT1_plain_reply_events_and_storage(env_factory):
    env = env_factory([Reply(["Hello ", "there."])])
    cid = await new_conv()
    events = await collect(VIS1, cid, "hi")
    check_event_shapes(events)
    assert events[-1] == {"type": "done", "text": "Hello there."}
    assert deltas(events) == "Hello there."
    assert await db_messages(cid) == [{"role": "user", "content": "hi"},
                                      {"role": "assistant", "content": "Hello there."}]
    assert len(env.client.calls) == 1


async def test_AGT1_done_text_is_already_saved_when_event_arrives(env_factory):
    env_factory([Reply(["All good."])])
    cid = await new_conv()
    seen = None
    from app.agent import loop as lp
    async for e in lp.run_turn(VIS1, cid, "q"):
        if e["type"] == "done":
            seen = await db_messages(cid)
    assert seen[-1] == {"role": "assistant", "content": "All good."}


async def test_AGT1_user_message_saved_before_model_is_called(env_factory):
    snapshots = []
    cid = await new_conv()

    async def inspect():
        snapshots.append(await db_messages(cid))

    env_factory([Reply(["Ok."], on_open=inspect)])
    await collect(VIS1, cid, "please save me first")
    assert snapshots == [[{"role": "user", "content": "please save me first"}]]


async def test_AGT1_user_message_saved_before_second_model_call_and_reply_after(env_factory):
    cid = await new_conv(ARJUN)
    snaps = []

    async def inspect():
        snaps.append(await db_messages(cid))

    env_factory([Reply(["Checking."], [text("Checking."), tool_use("t1", "get_attendance_summary")],
                       "tool_use", on_open=inspect),
                 Reply(["Eighty."], on_open=inspect)])
    await collect(ARJUN, cid, "attendance")
    assert all(s == [{"role": "user", "content": "attendance"}] for s in snaps)
    assert len(snaps) == 2


async def test_AGT1_user_message_saved_even_when_model_fails_immediately(env_factory):
    err = anthropic.APIConnectionError(request=REQ)
    env_factory([Reply(error=err)])
    cid = await new_conv()
    await collect(VIS1, cid, "will fail")
    msgs = await db_messages(cid)
    assert msgs[0] == {"role": "user", "content": "will fail"}
    assert [m["role"] for m in msgs].count("user") == 1


async def test_AGT1_assistant_reply_stored_exactly_once_with_tools(env_factory):
    env_factory([Reply(["Let me check."], [text("Let me check."), tool_use("a", "get_my_profile")],
                       "tool_use"),
                 Reply(["You are in C S E."])])
    cid = await new_conv(ARJUN)
    events = await collect(ARJUN, cid, "who am i")
    msgs = await db_messages(cid)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["content"] == events[-1]["text"]
    assert "Let me check." in msgs[1]["content"] and "You are in C S E." in msgs[1]["content"]


async def test_AGT1_one_tool_event_stream(env_factory):
    env = env_factory([Reply(["Checking."], [text("Checking."), tool_use("t1", "get_attendance_summary")],
                             "tool_use"),
                       Reply(["It is eighty percent."])])
    cid = await new_conv(ARJUN)
    events = await collect(ARJUN, cid, "attendance?")
    check_event_shapes(events)
    kinds = [(e["type"], e.get("state")) for e in events]
    assert kinds.index(("tool", "start")) < kinds.index(("tool", "end"))
    tool_events = [e for e in events if e["type"] == "tool"]
    assert tool_events == [
        {"type": "tool", "name": "get_attendance_summary", "state": "start", "is_error": False},
        {"type": "tool", "name": "get_attendance_summary", "state": "end", "is_error": False}]
    # the text before the tool call is spoken before the tool starts
    first_tool = kinds.index(("tool", "start"))
    assert "Checking." in deltas(events[:first_tool])
    assert "It is eighty percent." in deltas(events[first_tool:])
    full = events[-1]["text"]
    assert full == deltas(events).strip()
    assert "Checking.It" not in full  # sentences from separate model calls stay separated
    assert env.tools.ran[0][0] == "get_attendance_summary"
    assert env.tools.ran[0][2] == "user:arjun"


async def test_AGT1_several_sequential_tools(env_factory):
    env = env_factory([
        Reply([], [tool_use("a", "get_my_profile")], "tool_use"),
        Reply([], [tool_use("b", "get_attendance_summary")], "tool_use"),
        Reply(["Done."]),
    ])
    cid = await new_conv(ARJUN)
    events = await collect(ARJUN, cid, "go")
    check_event_shapes(events)
    tool_events = [(e["name"], e["state"]) for e in events if e["type"] == "tool"]
    assert tool_events == [("get_my_profile", "start"), ("get_my_profile", "end"),
                           ("get_attendance_summary", "start"), ("get_attendance_summary", "end")]
    assert len(env.client.calls) == 3
    for call in env.client.calls:
        assert_valid(call["messages"])
    assert events[-1]["type"] == "done" and events[-1]["text"] == "Done."


async def test_AGT1_parallel_tools_run_concurrently_and_all_answered_in_one_message(env_factory):
    content = [tool_use("a", "get_my_profile"), tool_use("b", "get_attendance_summary"),
               tool_use("c", "submit_leave_request")]
    env = env_factory(
        [Reply([], content, "tool_use"), Reply(["Done."])],
        behaviour={n: (0.2, ToolResult('{"ok": true}')) for n in
                   ("get_my_profile", "get_attendance_summary", "submit_leave_request")})
    cid = await new_conv(ARJUN)
    t0 = asyncio.get_running_loop().time()
    events = await collect(ARJUN, cid, "all three")
    assert asyncio.get_running_loop().time() - t0 < 0.5
    check_event_shapes(events)
    starts = [e for e in events if e["type"] == "tool" and e["state"] == "start"]
    ends = [e for e in events if e["type"] == "tool" and e["state"] == "end"]
    assert sorted(e["name"] for e in starts) == sorted(e["name"] for e in ends)
    assert len(starts) == len(ends) == 3
    second = env.client.calls[1]["messages"]
    assert_valid(second)
    results = blocks(second[-1])
    assert [r["tool_use_id"] for r in results] == ["a", "b", "c"]  # in the model's order


async def test_AGT2_failing_tool_is_reported_and_exception_text_hidden(env_factory):
    env = env_factory(
        [Reply([], [tool_use("a", "get_my_profile")], "tool_use"), Reply(["Sorry, that failed."])],
        behaviour={"get_my_profile": (0, RuntimeError(SECRET_DETAIL))})
    cid = await new_conv(ARJUN)
    events = await collect(ARJUN, cid, "profile")
    check_event_shapes(events)
    end = [e for e in events if e["type"] == "tool" and e["state"] == "end"][0]
    assert end["is_error"] is True
    start = [e for e in events if e["type"] == "tool" and e["state"] == "start"][0]
    assert start["is_error"] is False
    second = env.client.calls[1]["messages"]
    assert_valid(second)
    result = blocks(second[-1])[0]
    assert result["is_error"] is True
    assert SECRET_DETAIL not in dump(second)
    assert "Traceback" not in dump(second)
    assert events[-1]["type"] == "done"
    assert SECRET_DETAIL not in json.dumps(events)


async def test_AGT2_tool_returning_error_result_marks_event_and_result(env_factory):
    env = env_factory(
        [Reply([], [tool_use("a", "get_my_profile")], "tool_use"), Reply(["No."])],
        behaviour={"get_my_profile": (0, ToolResult('{"error": "nope"}', True))})
    cid = await new_conv(ARJUN)
    events = await collect(ARJUN, cid, "x")
    assert [e["is_error"] for e in events if e["type"] == "tool"] == [False, True]
    result = blocks(env.client.calls[1]["messages"][-1])[0]
    assert result["is_error"] is True and result["content"] == '{"error": "nope"}'


async def test_AGT2_successful_result_is_not_flagged_as_error(env_factory):
    env = env_factory(
        [Reply([], [tool_use("a", "get_my_profile")], "tool_use"), Reply(["Ok."])],
        behaviour={"get_my_profile": (0, ToolResult('{"name": "x"}'))})
    cid = await new_conv(ARJUN)
    await collect(ARJUN, cid, "x")
    result = blocks(env.client.calls[1]["messages"][-1])[0]
    assert not result.get("is_error")
    assert result["tool_use_id"] == "a" and result["content"] == '{"name": "x"}'


async def test_AGT2_unknown_tool_requested_by_model_is_answered_with_error(env_factory):
    env = env_factory([Reply([], [tool_use("u1", "format_the_disk", {"path": "/"})], "tool_use"),
                       Reply(["I cannot do that."])])
    cid = await new_conv(ARJUN)
    events = await collect(ARJUN, cid, "do it")
    check_event_shapes(events)
    assert [e["is_error"] for e in events if e["type"] == "tool" and e["state"] == "end"] == [True]
    second = env.client.calls[1]["messages"]
    assert_valid(second)
    assert blocks(second[-1])[0]["is_error"] is True
    assert events[-1]["type"] == "done"


# DEFECT: the loop executes any tool name the model sends; only run_tool gates it (no defence in depth).
@pytest.mark.parametrize("role_ctx,forbidden", [
    (VIS1, "get_attendance_summary"), (VIS1, "approve_request"),
    (ARJUN, "approve_request"), (ARJUN, "reject_request"),
    (LAKSHMI, "submit_leave_request"), (LAKSHMI, "get_attendance_summary"),
])
async def test_AC4_tool_outside_role_is_refused_even_if_gate_in_run_tool_is_absent(
        env_factory, role_ctx, forbidden):
    env = env_factory([Reply([], [tool_use("x", forbidden)], "tool_use"), Reply(["Refused."])],
                      gate=False)
    cid = await new_conv(role_ctx)
    events = await collect(role_ctx, cid, "do the forbidden thing")
    assert all(name != forbidden for name, *_ in env.tools.ran), "forbidden tool was executed"
    end = [e for e in events if e["type"] == "tool" and e["state"] == "end"]
    assert end and end[0]["is_error"] is True
    second = env.client.calls[1]["messages"]
    assert_valid(second)
    assert blocks(second[-1])[0]["is_error"] is True


@pytest.mark.parametrize("role_ctx,forbidden", [
    (VIS1, "get_attendance_summary"), (ARJUN, "approve_request"), (LAKSHMI, "submit_leave_request"),
])
async def test_AC4_tool_outside_role_refused_with_real_run_tool_and_logged(
        monkeypatch, env_factory, fake_college, role_ctx, forbidden):
    env_factory([Reply([], [tool_use("x", forbidden, {})], "tool_use"), Reply(["Refused."])],
                real_tools=True)
    cid = await new_conv(role_ctx)
    events = await collect(role_ctx, cid, "do the forbidden thing")
    end = [e for e in events if e["type"] == "tool" and e["state"] == "end"][0]
    assert end["is_error"] is True
    assert fake_college.writes() == []
    assert not [c for c in fake_college.calls() if "attendance" in c.path or "requests" in c.path]


async def test_AGT2_malformed_tool_input_still_gets_one_safe_result(env_factory):
    env = env_factory([Reply([], [tool_use("m", "get_my_profile", "not-an-object")], "tool_use"),
                       Reply(["Sorry."])],
                      behaviour={"get_my_profile": (0, TypeError("args must be dict"))})
    cid = await new_conv(ARJUN)
    events = await collect(ARJUN, cid, "x")
    check_event_shapes(events)
    assert_valid(env.client.calls[1]["messages"])
    assert events[-1]["type"] == "done"


@pytest.mark.parametrize("bad", ["garbage", None, ["a"], 42])
async def test_AGT6_malformed_tool_input_with_real_tools_is_safe(env_factory, fake_college, bad):
    env = env_factory([Reply([], [tool_use("m", "get_attendance_summary", bad)], "tool_use"),
                       Reply(["Sorry."])], real_tools=True)
    ctx = fake_college.ctx("arjun")
    cid = await new_conv(ctx)
    events = await collect(ctx, cid, "x")
    check_event_shapes(events)
    assert events[-1]["type"] == "done"
    assert_valid(env.client.calls[1]["messages"])
    assert "Traceback" not in dump(env.client.calls[1]["messages"])


async def test_AGT2_two_tool_uses_one_fails_one_succeeds(env_factory):
    env = env_factory(
        [Reply([], [tool_use("a", "get_my_profile"), tool_use("b", "get_attendance_summary")],
               "tool_use"), Reply(["Partly."])],
        behaviour={"get_my_profile": (0, RuntimeError("x")),
                   "get_attendance_summary": (0.05, ToolResult("{}"))})
    cid = await new_conv(ARJUN)
    events = await collect(ARJUN, cid, "x")
    ends = {e["name"]: e["is_error"] for e in events if e["type"] == "tool" and e["state"] == "end"}
    assert ends == {"get_my_profile": True, "get_attendance_summary": False}
    results = blocks(env.client.calls[1]["messages"][-1])
    assert [bool(r.get("is_error")) for r in results] == [True, False]


async def test_AGT2_unexpected_blocks_in_model_content_are_passed_back_unchanged(env_factory):
    thinking = type("T", (), {"type": "thinking", "thinking": "hmm", "signature": "sig"})()
    env = env_factory([Reply(["Ok."], [thinking, text("Ok."), tool_use("a", "get_my_profile")],
                             "tool_use"), Reply(["Done."])])
    cid = await new_conv(ARJUN)
    await collect(ARJUN, cid, "x")
    second = env.client.calls[1]["messages"]
    assert_valid(second)
    assert bget(blocks(second[1])[0], "signature") == "sig"


# ------------------------------------------------------------------ conversation validity matrix

def _scenario_plain():
    return [Reply(["Hi."])], {}


def _scenario_tool():
    return [Reply(["A."], [text("A."), tool_use("t1", "get_my_profile")], "tool_use"),
            Reply(["B."])], {}


def _scenario_sequential():
    return [Reply([], [tool_use("t1", "get_my_profile")], "tool_use"),
            Reply([], [tool_use("t2", "get_attendance_summary")], "tool_use"),
            Reply(["C."])], {}


def _scenario_parallel():
    return [Reply([], [tool_use("t1", "get_my_profile"), tool_use("t2", "get_attendance_summary")],
                  "tool_use"), Reply(["D."])], {}


def _scenario_failing():
    return [Reply([], [tool_use("t1", "get_my_profile")], "tool_use"), Reply(["E."])], \
        {"get_my_profile": (0, RuntimeError("boom"))}


def _scenario_unknown():
    return [Reply([], [tool_use("t1", "nonsense")], "tool_use"), Reply(["F."])], {}


def _scenario_outside():
    return [Reply([], [tool_use("t1", "approve_request")], "tool_use"), Reply(["G."])], {}


def _scenario_refusal():
    return [Reply(["I"], [text("I"), tool_use("t1", "get_my_profile")], "refusal")], {}


def _scenario_max_tokens_tool():
    return [Reply([], [tool_use("t1", "get_my_profile", {"cut": "of"})], "max_tokens"),
            Reply(["H."])], {}


def _scenario_max_tokens_text():
    return [Reply(["Long long"], stop_reason="max_tokens")], {}


def _scenario_api_error():
    return [Reply(error=anthropic.APIConnectionError(request=REQ))], {}


def _scenario_mid_stream_error():
    return [Reply(["Partial."], mid_error=anthropic.APIConnectionError(request=REQ))], {}


def _scenario_error_after_tool():
    return [Reply([], [tool_use("t1", "get_my_profile")], "tool_use"),
            Reply(error=anthropic.APIConnectionError(request=REQ))], {}


def _scenario_cap():
    return (lambda kw, n: Reply([], [tool_use(f"t{n}", "get_my_profile")], "tool_use")), {}


SCENARIOS = [_scenario_plain, _scenario_tool, _scenario_sequential, _scenario_parallel,
             _scenario_failing, _scenario_unknown, _scenario_outside, _scenario_refusal,
             _scenario_max_tokens_tool, _scenario_max_tokens_text, _scenario_api_error,
             _scenario_mid_stream_error, _scenario_error_after_tool, _scenario_cap]


@pytest.mark.parametrize("rebuild", [False, True], ids=["from-memory", "rebuilt-from-db"])
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda f: f.__name__[10:])
async def test_AGT2_next_request_is_valid_after_every_scenario(env_factory, scenario, rebuild):
    script, behaviour = scenario()
    if isinstance(script, list):
        script = script + [Reply(["Next answer."])]
    else:
        base = script
        script = lambda kw, n: base(kw, n) if n < loop.MAX_MODEL_CALLS else Reply(["Next answer."])  # noqa: E731
    env = env_factory(script, behaviour=behaviour)
    cid = await new_conv(ARJUN)
    first = await collect(ARJUN, cid, "first question")
    check_event_shapes(first)
    from tests.acc_agent_helpers import evict_memory
    if rebuild:
        evict_memory()
    before = len(env.client.calls)
    second = await collect(ARJUN, cid, "second question")
    check_event_shapes(second)
    assert len(env.client.calls) > before
    for call in env.client.calls:
        assert_valid(call["messages"])
    last_user = blocks(env.client.calls[before]["messages"][-1])
    assert last_user[-1]["text"] == "second question"
    # the second question appears exactly once in the request
    assert dump(env.client.calls[before]["messages"]).count("second question") == 1


# ------------------------------------------------------------------ persistence ordering

async def test_HIS1_ordering_across_several_turns(env_factory):
    env_factory([Reply([f"Answer {i}."]) for i in range(5)])
    cid = await new_conv()
    for i in range(5):
        await collect(VIS1, cid, f"Question {i}")
    msgs = await db_messages(cid)
    assert msgs == [m for i in range(5) for m in (
        {"role": "user", "content": f"Question {i}"},
        {"role": "assistant", "content": f"Answer {i}."})]


async def test_HIS1_title_comes_from_first_user_message(env_factory):
    env_factory([Reply(["A."]), Reply(["B."])])
    cid = await new_conv()
    await collect(VIS1, cid, "What are the fees")
    await collect(VIS1, cid, "And the hostel")
    from app.db import pool
    import uuid
    title = await pool().fetchval("SELECT title FROM conversations WHERE id=$1", uuid.UUID(cid))
    assert title == "What are the fees"


async def test_HIS1_model_sees_whole_history_in_order(env_factory):
    env = env_factory([Reply(["A1."]), Reply(["A2."]), Reply(["A3."])])
    cid = await new_conv()
    for q in ("Q1", "Q2", "Q3"):
        await collect(VIS1, cid, q)
    msgs = env.client.calls[2]["messages"]
    assert_valid(msgs)
    flat = [(m["role"], "".join(bget(b, "text") or "" for b in blocks(m))) for m in msgs]
    assert flat == [("user", "Q1"), ("assistant", "A1."), ("user", "Q2"), ("assistant", "A2."),
                    ("user", "Q3")]


async def test_HIS1_concurrent_turns_on_one_conversation_do_not_interleave(env_factory):
    env_factory(lambda kw, n: Reply([f"Reply {n}."], delay=0.05))
    cid = await new_conv()
    await asyncio.gather(collect(VIS1, cid, "first"), collect(VIS1, cid, "second"))
    msgs = await db_messages(cid)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    for call in loop_calls(env_factory):
        assert_valid(call["messages"])


def loop_calls(env_factory):  # pragma: no cover - helper
    return loop._client.calls


async def test_AGT1_tool_calls_are_logged_by_real_run_tool(env_factory, fake_college):
    env_factory([Reply([], [tool_use("t1", "get_today", {})], "tool_use"), Reply(["Ok."])],
                real_tools=True)
    ctx = fake_college.ctx("arjun")
    cid = await new_conv(ctx)
    await collect(ctx, cid, "date?")
    rows = await db_tool_calls(cid)
    assert [r["name"] for r in rows] == ["get_today"]
    assert rows[0]["is_error"] is False


async def test_AGT1_refused_tool_calls_are_logged_too(env_factory, fake_college):
    env_factory([Reply([], [tool_use("t1", "approve_request", {})], "tool_use"), Reply(["No."])],
                real_tools=True)
    ctx = fake_college.ctx("arjun")
    cid = await new_conv(ctx)
    await collect(ctx, cid, "approve")
    rows = await db_tool_calls(cid)
    assert [(r["name"], r["is_error"]) for r in rows] == [("approve_request", True)]


async def test_AGT1_internal_retry_does_not_store_anything_twice(env_factory):
    env_factory([Reply([], [tool_use("t", "get_my_profile", {"cut": 1})], "max_tokens"),
                 Reply(["Retried fine."])])
    cid = await new_conv(ARJUN)
    await collect(ARJUN, cid, "once please")
    msgs = await db_messages(cid)
    assert msgs == [{"role": "user", "content": "once please"},
                    {"role": "assistant", "content": "Retried fine."}]


async def test_AGT1_failed_turn_stores_user_once_and_no_phantom_reply(env_factory):
    env_factory([Reply(error=anthropic.APIConnectionError(request=REQ)), Reply(["Now fine."])])
    cid = await new_conv()
    first = await collect(VIS1, cid, "same words")
    assert first[-1]["type"] == "error"
    msgs = await db_messages(cid)
    assert msgs == [{"role": "user", "content": "same words"}]
    await collect(VIS1, cid, "same words")
    msgs = await db_messages(cid)
    assert [m["role"] for m in msgs] == ["user", "user", "assistant"]
    assert msgs[-1]["content"] == "Now fine."


async def test_AGT1_error_text_is_not_stored_as_assistant_reply(env_factory):
    env_factory([Reply(error=anthropic.APIConnectionError(request=REQ))])
    cid = await new_conv()
    events = await collect(VIS1, cid, "x")
    stored = " ".join(m["content"] for m in await db_messages(cid) if m["role"] == "assistant")
    assert events[-1]["message"] not in stored


async def test_AGT1_partial_text_before_error_is_saved_once(env_factory):
    env_factory([Reply(["Half an answer."], mid_error=anthropic.APIConnectionError(request=REQ))])
    cid = await new_conv()
    events = await collect(VIS1, cid, "x")
    check_event_shapes(events)
    assert events[-1]["type"] == "error"
    saved = [m["content"] for m in await db_messages(cid) if m["role"] == "assistant"]
    assert len(saved) <= 1
    if saved:
        assert "Half an answer." in saved[0]


# DEFECT: a database failure while saving the final reply escapes run_turn as a raw exception.
async def test_AGT6_reply_save_failure_does_not_crash_turn(env_factory, monkeypatch):
    env_factory([Reply(["Answer."])])
    cid = await new_conv()
    real = history.add_message

    async def flaky(conv, role, content):
        if role == "assistant":
            raise RuntimeError(SECRET_DETAIL)
        return await real(conv, role, content)

    monkeypatch.setattr(history, "add_message", flaky)
    events = []
    try:
        async for e in loop.run_turn(VIS1, cid, "q"):
            events.append(e)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"run_turn crashed: {type(exc).__name__}")
    assert events and events[-1]["type"] in ("done", "error")
    assert SECRET_DETAIL not in json.dumps(events)


# ------------------------------------------------------------------ bounds (AGT4)

async def test_AGT4_endless_tool_requests_stop_with_spoken_message(env_factory):
    env = env_factory(lambda kw, n: Reply([], [tool_use(f"t{n}", "get_my_profile")], "tool_use")
                      if n < 200 else Reply(["runaway"]))
    cid = await new_conv(ARJUN)
    events = await collect(ARJUN, cid, "loop forever")
    check_event_shapes(events)
    assert 1 < len(env.client.calls) <= 25, "model calls must be bounded"
    assert len(env.tools.ran) < len(env.client.calls) + 1
    final = events[-1]
    assert final["type"] == "done" and final["text"].strip()
    assert "runaway" not in final["text"]
    spoken_tail = deltas(events)
    assert spoken_tail.strip() == final["text"]
    assert not re.search(r"[*#_`\[\]{}<>|\\/]", final["text"]), "spoken style: no markup"
    assert_safe(final["text"])
    for call in env.client.calls:
        assert_valid(call["messages"])
    saved = await db_messages(cid)
    assert saved[-1] == {"role": "assistant", "content": final["text"]}
    assert [m["role"] for m in saved] == ["user", "assistant"]


async def test_AGT4_conversation_is_usable_after_hitting_cap(env_factory):
    def script(kw, n):
        if n < loop.MAX_MODEL_CALLS:
            return Reply([], [tool_use(f"t{n}", "get_my_profile")], "tool_use")
        return Reply(["Fresh start."])
    env = env_factory(script)
    cid = await new_conv(ARJUN)
    await collect(ARJUN, cid, "loop")
    events = await collect(ARJUN, cid, "simple question")
    assert events[-1]["text"].endswith("Fresh start.")
    assert_valid(env.client.calls[-1]["messages"])


async def test_AGT4_tools_requested_at_the_cap_are_not_run_but_answered(env_factory):
    env = env_factory(lambda kw, n: Reply([], [tool_use(f"t{n}", "get_my_profile")], "tool_use"))
    cid = await new_conv(ARJUN)
    await collect(ARJUN, cid, "x")
    assert len(env.tools.ran) == len(env.client.calls) - 1


# ------------------------------------------------------------------ stop reasons and failures (AGT6)

async def test_AGT6_refusal_gives_fallback_and_never_runs_partial_tool(env_factory):
    env = env_factory([Reply(["I"], [text("I"), tool_use("x", "get_my_profile")], "refusal")])
    cid = await new_conv(ARJUN)
    events = await collect(ARJUN, cid, "bad")
    check_event_shapes(events)
    assert env.tools.ran == []
    assert events[-1]["type"] == "done" and len(events[-1]["text"]) > 3
    assert "refusal" not in events[-1]["text"].lower()
    assert_safe(events[-1]["text"])
    saved = await db_messages(cid)
    assert saved[-1]["role"] == "assistant" and saved[-1]["content"] == events[-1]["text"]


async def test_AGT6_refusal_with_no_text_still_says_something(env_factory):
    env_factory([Reply([], [], "refusal")])
    cid = await new_conv()
    events = await collect(VIS1, cid, "bad")
    assert events[-1]["type"] == "done" and events[-1]["text"].strip()
    assert deltas(events).strip() == events[-1]["text"]


async def test_AGT6_max_tokens_text_is_delivered_and_saved(env_factory):
    env_factory([Reply(["A long answer that was cut"], stop_reason="max_tokens")])
    cid = await new_conv()
    events = await collect(VIS1, cid, "q")
    assert events[-1] == {"type": "done", "text": "A long answer that was cut"}
    assert (await db_messages(cid))[-1]["content"] == "A long answer that was cut"


async def test_AGT6_max_tokens_on_tool_call_does_not_run_the_tool(env_factory):
    env = env_factory([Reply([], [tool_use("t", "submit_leave_request", {"from_date": "2031-0"})],
                             "max_tokens"), Reply(["Let me retry."])])
    cid = await new_conv(ARJUN)
    events = await collect(ARJUN, cid, "leave")
    assert env.tools.ran == []
    second = env.client.calls[1]["messages"]
    assert_valid(second)
    assert blocks(second[-1])[0]["is_error"] is True
    assert events[-1]["type"] == "done" and events[-1]["text"] == "Let me retry."
    assert not [e for e in events if e["type"] == "tool"] or all(
        e["is_error"] for e in events if e["type"] == "tool" and e["state"] == "end")


API_ERRORS = {
    "connection": lambda: anthropic.APIConnectionError(request=REQ),
    "timeout": lambda: anthropic.APITimeoutError(request=REQ),
    "rate_limit": lambda: anthropic.RateLimitError(
        SECRET_DETAIL, response=httpx.Response(429, request=REQ), body=None),
    "server_500": lambda: anthropic.InternalServerError(
        SECRET_DETAIL, response=httpx.Response(500, request=REQ), body=None),
    "overloaded_529": lambda: anthropic.APIStatusError(
        SECRET_DETAIL, response=httpx.Response(529, request=REQ), body=None),
    "bad_request_400": lambda: anthropic.BadRequestError(
        SECRET_DETAIL, response=httpx.Response(400, request=REQ), body=None),
    "auth_401": lambda: anthropic.AuthenticationError(
        SECRET_DETAIL, response=httpx.Response(401, request=REQ), body=None),
    "plain_exception": lambda: RuntimeError(SECRET_DETAIL),
    "value_error": lambda: ValueError(SECRET_DETAIL),
}


@pytest.mark.parametrize("when", ["open", "mid_stream"])
@pytest.mark.parametrize("name", API_ERRORS)
async def test_AGT6_model_failures_give_safe_error_event(env_factory, name, when):
    exc = API_ERRORS[name]()
    reply = Reply(error=exc) if when == "open" else Reply(["Part."], mid_error=exc)
    env_factory([reply])
    cid = await new_conv()
    events = await collect(VIS1, cid, "hi")
    check_event_shapes(events)
    assert events[-1]["type"] == "error"
    assert_safe(events[-1]["message"])
    assert SECRET_DETAIL not in json.dumps(events)
    assert len(events[-1]["message"]) < 200


async def test_AGT6_error_after_tools_ran_keeps_conversation_valid(env_factory):
    env = env_factory([Reply([], [tool_use("t1", "get_my_profile")], "tool_use"),
                       Reply(error=anthropic.APIConnectionError(request=REQ)),
                       Reply(["Back again."])])
    cid = await new_conv(ARJUN)
    events = await collect(ARJUN, cid, "x")
    assert events[-1]["type"] == "error"
    events2 = await collect(ARJUN, cid, "again")
    assert events2[-1]["text"] == "Back again."
    for call in env.client.calls:
        assert_valid(call["messages"])


@pytest.mark.parametrize("failing", ["knowledge", "today", "pending"])
async def test_AGT6_college_or_knowledge_failure_still_answers(env_factory, monkeypatch, failing):
    env = env_factory([Reply(["Still here."])], pending="x")

    async def boom(*a, **k):
        raise httpx.ConnectError(SECRET_DETAIL)

    if failing == "knowledge":
        monkeypatch.setattr(loop, "knowledge_block", boom)
    elif failing == "today":
        monkeypatch.setattr(loop, "college_today", boom)
    else:
        monkeypatch.setattr(loop, "confirm", type("C", (), {"pending_summary": staticmethod(boom)}))
    cid = await new_conv(ARJUN)
    events = await collect(ARJUN, cid, "hello")
    check_event_shapes(events)
    assert events[-1] == {"type": "done", "text": "Still here."}
    sent = dump(env.client.calls[0]["system"])
    assert SECRET_DETAIL not in sent
    if failing == "today":
        # without the college date the model must not be handed the server clock
        import datetime
        assert str(datetime.date.today()) not in sent


async def test_AGT6_tool_specs_failure_is_safe(env_factory, monkeypatch):
    env = env_factory([Reply(["x"])])

    def boom(role, include_search):
        raise RuntimeError(SECRET_DETAIL)

    monkeypatch.setattr(env.tools, "tool_specs", boom)
    cid = await new_conv()
    events = await collect(VIS1, cid, "hi")
    assert events[-1]["type"] == "error"
    assert SECRET_DETAIL not in json.dumps(events)


# DEFECT: a model reply with no text ends the turn with an empty done and the user hears silence.
async def test_AGT6_empty_model_reply_is_not_silence(env_factory):
    env_factory([Reply([], [], "end_turn")])
    cid = await new_conv()
    events = await collect(VIS1, cid, "hi")
    # a voice assistant that says nothing is a failure the user cannot see
    assert events[-1]["type"] in ("done", "error")
    if events[-1]["type"] == "done":
        assert events[-1]["text"].strip()


async def test_AGT6_unknown_stop_reason_does_not_crash(env_factory):
    env_factory([Reply(["Some text."], stop_reason="pause_turn")])
    cid = await new_conv()
    events = await collect(VIS1, cid, "hi")
    check_event_shapes(events)


async def test_AGT6_no_stack_trace_in_any_event_for_every_failure(env_factory):
    env_factory([Reply(error=RuntimeError("Traceback (most recent call last): boom"))])
    cid = await new_conv()
    events = await collect(VIS1, cid, "hi")
    assert "Traceback" not in json.dumps(events)


async def test_REL2_fake_model_means_no_real_network_used(env_factory):
    env = env_factory([Reply(["Offline ok."])])
    cid = await new_conv()
    events = await collect(VIS1, cid, "hi")
    assert events[-1]["text"] == "Offline ok."
    assert len(env.client.calls) == 1


async def test_REL2_real_client_not_created_when_fake_installed(env_factory):
    env_factory([Reply(["ok"])])
    assert isinstance(loop._client, type(loop._client)) and not isinstance(
        loop._client, anthropic.AsyncAnthropic)


async def test_AGT1_request_parameters_for_a_normal_call(env_factory):
    env = env_factory([Reply(["Hi."])])
    cid = await new_conv()
    await collect(VIS1, cid, "hi")
    kw = env.client.calls[0]
    assert kw["model"] and isinstance(kw["max_tokens"], int) and kw["max_tokens"] > 0
    assert kw["tools"] and kw["system"]
    assert "temperature" not in kw


async def test_AGT1_unicode_and_long_user_text_round_trip(env_factory):
    env_factory([Reply(["Fine."])])
    cid = await new_conv()
    msg = "வணக்கம் 😀 " + "x" * 5000
    await collect(VIS1, cid, msg)
    assert (await db_messages(cid))[0]["content"] == msg


async def test_AGT1_blank_reply_chunks_are_not_emitted(env_factory):
    env_factory([Reply(["", "Hello", ""])])
    cid = await new_conv()
    events = await collect(VIS1, cid, "hi")
    assert all(e["text"] for e in events if e["type"] == "text_delta")
    assert events[-1]["text"] == "Hello"
