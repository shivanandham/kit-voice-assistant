"""Acceptance: cancelling a turn (barge-in) at every possible point.

Criteria: AGT3 (partial reply saved, model stream closed, next turn works), AGT2 (the conversation
sent next is always valid), AC3 (a cancelled turn leaks nothing into another conversation).
"""

import asyncio

import pytest
from app.agent import loop
from app.agent.tools import ToolResult
from tests.acc_agent_helpers import (  # noqa: F401
    ARJUN, PRIYA, VIS1, Reply, assert_valid, blocks, btype, bget, cancel_when, collect,
    db_messages, dump, env_factory, evict_memory, new_conv, text, tool_use,
)

FOLLOW = Reply(["Sure, go on."])


@pytest.fixture(autouse=True)
def short_grace(monkeypatch):
    monkeypatch.setattr(loop, "TOOL_GRACE_SECONDS", 0.05)


def case_open():
    """Cancelled while the request is still being opened (before the stream exists)."""
    r = Reply(["never"], hang="open")
    return [r, FOLLOW], {}, lambda got: r.opened.is_set(), ""


def case_before_first_token():
    r = Reply(["never"], hang="before_first")
    return [r, FOLLOW], {}, lambda got: r.opened.is_set(), ""


def case_mid_text():
    r = Reply(["One. ", "Two. "], hang="after_chunks")
    return [r, FOLLOW], {}, lambda got: any(e["type"] == "text_delta" for e in got), "One."


def case_text_then_tool_call_pending():
    r = Reply(["Let me look. "], [text("Let me look. "), tool_use("t1", "get_my_profile")],
              "tool_use", hang="final")
    return [r, FOLLOW], {}, lambda got: any(e["type"] == "text_delta" for e in got), "Let me look."


def case_while_tools_run():
    r = Reply(["Checking. "], [text("Checking. "), tool_use("t1", "get_my_profile"),
                               tool_use("t2", "get_attendance_summary")], "tool_use")
    return ([r, FOLLOW],
            {"get_my_profile": (30, ToolResult('"slow"')),
             "get_attendance_summary": (0, ToolResult('"fast"'))},
            lambda got: any(e["type"] == "tool" and e["state"] == "end" for e in got),
            "Checking.")


def case_all_tools_hanging():
    r = Reply([], [tool_use("t1", "get_my_profile")], "tool_use")
    return ([r, FOLLOW], {"get_my_profile": (30, ToolResult('"slow"'))},
            lambda got: any(e["type"] == "tool" and e["state"] == "start" for e in got), "")


def case_after_tool_results_before_second_call_text():
    r1 = Reply(["Checking. "], [text("Checking. "), tool_use("t1", "get_my_profile")], "tool_use")
    r2 = Reply(["never"], hang="before_first")
    return ([r1, r2, FOLLOW], {}, lambda got: r2.opened.is_set(), "Checking.")


def case_second_call_mid_text():
    r1 = Reply(["Checking. "], [text("Checking. "), tool_use("t1", "get_my_profile")], "tool_use")
    r2 = Reply(["You have ", "eighty. "], hang="after_chunks")
    return ([r1, r2, FOLLOW], {},
            lambda got: sum(e["type"] == "text_delta" for e in got) >= 2, "Checking. You have")


def case_second_call_final_hanging():
    r1 = Reply([], [tool_use("t1", "get_my_profile")], "tool_use")
    r2 = Reply(["Done. "], hang="final")
    return ([r1, r2, FOLLOW], {}, lambda got: any(e["type"] == "text_delta" for e in got), "Done.")


def case_parallel_tools_all_finished():
    r1 = Reply([], [tool_use("t1", "get_my_profile"), tool_use("t2", "get_attendance_summary")],
               "tool_use")
    r2 = Reply(["x"], hang="before_first")
    return ([r1, r2, FOLLOW], {}, lambda got: r2.opened.is_set(), "")


CASES = [case_open, case_before_first_token, case_mid_text, case_text_then_tool_call_pending,
         case_while_tools_run, case_all_tools_hanging, case_after_tool_results_before_second_call_text,
         case_second_call_mid_text, case_second_call_final_hanging, case_parallel_tools_all_finished]


@pytest.mark.parametrize("evict", [False, True], ids=["cache", "rebuilt"])
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.__name__[5:])
async def test_AGT3_cancel_saves_said_text_closes_stream_and_next_turn_is_valid(
        env_factory, case, evict):
    script, behaviour, ready, said = case()
    env = env_factory(script, behaviour=behaviour)
    cid = await new_conv(ARJUN)
    await cancel_when(ARJUN, cid, "first question", ready)
    await asyncio.sleep(0.1)

    # the model stream that was entered has been closed
    for s in env.client.streams:
        assert s.closed or not s.entered
    assert env.client.streams[0].closed or not env.client.streams[0].entered

    # the user message is saved; what was already said is saved, nothing else is invented
    saved = await db_messages(cid)
    assert saved[0] == {"role": "user", "content": "first question"}
    assistant = [m["content"] for m in saved if m["role"] == "assistant"]
    if said:
        assert len(assistant) == 1 and said in assistant[0]
    else:
        assert assistant == [] or all(a.strip() for a in assistant)
    assert [m["role"] for m in saved].count("assistant") <= 1

    if evict:
        evict_memory()
    before = len(env.client.calls)
    events = await collect(ARJUN, cid, "and now?")
    assert events[-1] == {"type": "done", "text": "Sure, go on."}
    nxt = env.client.calls[before]["messages"]
    assert_valid(nxt)
    assert dump(nxt).count("and now?") == 1
    assert dump(nxt).count("first question") == 1
    if said:
        assert all(part in dump(nxt) for part in said.split(". "))
    # never-said text must not appear as something the assistant said
    assert "never" not in dump(nxt)
    final = await db_messages(cid)
    assert final[-1] == {"role": "assistant", "content": "Sure, go on."}
    assert [m["role"] for m in final].count("user") == 2


@pytest.mark.parametrize("case", [case_while_tools_run, case_all_tools_hanging],
                         ids=lambda c: c.__name__[5:])
async def test_AGT3_cancel_does_not_leave_tool_tasks_running(env_factory, case):
    script, behaviour, ready, _ = case()
    env_factory(script, behaviour=behaviour)
    cid = await new_conv(ARJUN)
    baseline = len(asyncio.all_tasks())
    await cancel_when(ARJUN, cid, "go", ready)
    await asyncio.sleep(0.3)
    assert len(asyncio.all_tasks()) <= baseline


async def test_AGT3_finished_tool_result_is_kept_unfinished_one_is_marked_error(env_factory):
    script, behaviour, ready, _ = case_while_tools_run()
    env = env_factory(script, behaviour=behaviour)
    cid = await new_conv(ARJUN)
    await cancel_when(ARJUN, cid, "go", ready)
    await collect(ARJUN, cid, "again")
    msgs = env.client.calls[-1]["messages"]
    assert_valid(msgs)
    results = [b for m in msgs for b in blocks(m) if btype(b) == "tool_result"]
    by_id = {r["tool_use_id"]: r for r in results}
    assert not by_id["t2"].get("is_error") and "fast" in by_id["t2"]["content"]
    assert by_id["t1"]["is_error"] is True


async def test_AGT3_aclose_while_streaming_saves_partial(env_factory):
    r = Reply(["Part one. ", "Part two."], hang="after_chunks")
    env = env_factory([r, FOLLOW])
    cid = await new_conv()
    agen = loop.run_turn(VIS1, cid, "hi")
    first = await agen.__anext__()
    assert first["type"] == "text_delta"
    await agen.aclose()
    assert env.client.streams[0].closed
    saved = await db_messages(cid)
    assert saved[-1]["role"] == "assistant" and "Part one." in saved[-1]["content"]
    await collect(VIS1, cid, "next")
    assert_valid(env.client.calls[-1]["messages"])


async def test_AGT3_aclose_during_tool_events_keeps_conversation_valid(env_factory):
    r1 = Reply(["Hold on. "], [text("Hold on. "), tool_use("t1", "get_my_profile")], "tool_use")
    env = env_factory([r1, FOLLOW], behaviour={"get_my_profile": (30, ToolResult("{}"))})
    cid = await new_conv(ARJUN)
    agen = loop.run_turn(ARJUN, cid, "hi")
    seen = []
    async for e in agen:
        seen.append(e)
        if e["type"] == "tool":
            break
    await agen.aclose()
    await collect(ARJUN, cid, "next")
    assert_valid(env.client.calls[-1]["messages"])


async def test_AGT3_barge_in_new_turn_started_right_after_cancel(env_factory):
    r = Reply(["Long answer. ", "More. "], delay=0.05, hang="after_chunks")
    env = env_factory([r, FOLLOW])
    cid = await new_conv()
    got: list[dict] = []

    async def consume():
        async for e in loop.run_turn(VIS1, cid, "tell me everything"):
            got.append(e)

    t1 = asyncio.create_task(consume())
    while not got:
        await asyncio.sleep(0.005)
    t1.cancel()
    # the new utterance starts immediately, without waiting for the old task to finish
    events = await collect(VIS1, cid, "never mind, fees?")
    with pytest.raises(asyncio.CancelledError):
        await t1
    assert events[-1]["type"] == "done"
    assert_valid(env.client.calls[-1]["messages"])
    msgs = await db_messages(cid)
    # the interrupted turn's user message precedes the new one
    users = [m["content"] for m in msgs if m["role"] == "user"]
    assert users == ["tell me everything", "never mind, fees?"]


async def test_AGT3_cancelled_turn_of_one_conversation_does_not_touch_another(env_factory):
    slow = Reply(["Arjun part. "], hang="after_chunks")

    def script(kw, n):
        flat = dump(kw["messages"])
        if "arjun question" in flat:
            return slow
        return Reply(["Priya answer."], delay=0.02)

    env = env_factory(script)
    c1, c2 = await new_conv(ARJUN), await new_conv(PRIYA)

    async def priya():
        return await collect(PRIYA, c2, "priya question")

    p = asyncio.create_task(priya())
    await cancel_when(ARJUN, c1, "arjun question", lambda g: slow.chunk_seen.is_set())
    events = await p
    assert events[-1] == {"type": "done", "text": "Priya answer."}
    assert [m["content"] for m in await db_messages(c2)] == ["priya question", "Priya answer."]
    for call in env.client.calls:
        flat = dump(call["messages"])
        assert not ("arjun question" in flat and "priya question" in flat)


async def test_AGT3_repeated_cancel_and_resume_keeps_history_valid(env_factory):
    replies = []
    for i in range(4):
        replies.append(Reply([f"Said {i}. "], hang="after_chunks"))
    replies.append(Reply(["Finally."]))
    env = env_factory(replies)
    cid = await new_conv(VIS1)
    for i in range(4):
        await cancel_when(VIS1, cid, f"q{i}", lambda got: any(e["type"] == "text_delta" for e in got))
    events = await collect(VIS1, cid, "last")
    assert events[-1]["text"] == "Finally."
    msgs = env.client.calls[-1]["messages"]
    assert_valid(msgs)
    for i in range(4):
        assert dump(msgs).count(f"q{i}") == 1
        assert f"Said {i}." in dump(msgs)
    evict_memory()
    # rebuilt from the database it is valid as well
    env.client.script = [Reply(["Again."])]
    env.client.calls.clear()
    await collect(VIS1, cid, "once more")
    assert_valid(env.client.calls[0]["messages"])
