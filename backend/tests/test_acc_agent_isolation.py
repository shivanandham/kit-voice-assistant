"""Acceptance: one conversation or user never sees another's messages, tool results or prompts.

Criteria: AC3 (conversation memory, in-memory cache, eviction and rebuild), AC2 (tool identity),
AC5 (no cross-user leakage through the loop), AGT2 under concurrency.
"""

import asyncio
import re

import pytest
from app import history
from app.agent import loop
from app.agent.tools import ToolResult
from tests.acc_agent_helpers import (  # noqa: F401
    ARJUN, LAKSHMI, PRIYA, SENTHIL, VIS1, VIS2, Reply, assert_valid, blocks, btype, collect,
    db_messages, db_tool_calls, dump, env_factory, evict_memory, new_conv, request_text, text,
    tool_use,
)

USERS = [ARJUN, PRIYA, LAKSHMI, SENTHIL, VIS1, VIS2]


def tag(ctx) -> str:
    return re.sub(r"\W", "", ctx.owner_key).upper()


def markers_in(flat: str) -> set[str]:
    return set(re.findall(r"MARK[A-Z0-9]+", flat))


def echo_script(kw, n):
    """A model that, when asked about MARKxxx, calls a tool then repeats what the tool returned."""
    flat = dump(kw["messages"])
    mine = re.findall(r"ask-(MARK[A-Z0-9]+)", flat)[-1]
    if "tool_result" not in flat.split(f"ask-{mine}")[-1]:
        return Reply(["Looking. "], [text("Looking. "), tool_use(f"use-{mine}", kw["tools"][0]["name"])],
                     "tool_use", delay=0.01)
    return Reply([f"Result for {mine}. "], delay=0.01)


def tool_behaviour():
    async def run(args, ctx, cid):
        return (0.02 + (hash(ctx.owner_key) % 5) / 100, ToolResult(f'{{"secret": "RESULT-MARK{tag(ctx)}"}}'))
    from tests.acc_agent_helpers import FakeTools
    return {n: run for names in FakeTools.SPECS.values() for n in names}


@pytest.mark.parametrize("rebuild_between", [False, True], ids=["cache", "evicted"])
async def test_AC3_concurrent_conversations_never_see_each_other(env_factory, rebuild_between):
    env = env_factory(echo_script, behaviour=tool_behaviour(), gate=False,
                      pending=lambda ctx: f"Pending for MARK{tag(ctx)}")
    cids = {u.owner_key: await new_conv(u) for u in USERS}

    async def run_user(u, round_):
        return await collect(u, cids[u.owner_key], f"ask-MARK{tag(u)}-r{round_}".replace("-r", "R"))

    # script keys on "ask-MARKxxx": make the marker unique per user (round suffix is part of it)
    for round_ in range(3):
        await asyncio.gather(*(run_user(u, round_) for u in USERS))
        if rebuild_between:
            evict_memory()

    for call in env.client.calls:
        flat = request_text(call)
        owners = {m for m in markers_in(dump(call["messages"]))}
        users_in = {re.match(r"MARK([A-Z0-9]+?)(?:R\d)?$", m).group(1) for m in owners}
        users_in = {u for u in users_in}
        assert len(users_in) == 1, f"request mixes conversations: {users_in}"
        # the system prompt carries only this requester's facts
        who = next(iter(users_in))
        sys_flat = dump(call["system"])
        for other in USERS:
            if tag(other) != who and other.user:
                assert other.user["full_name"] not in flat
                assert other.user["login_id"] not in flat or other.user["login_id"] in who.lower()
            if tag(other) != who:
                assert f"Pending for MARK{tag(other)}" not in sys_flat
        assert_valid(call["messages"])

    for u in USERS:
        stored = await db_messages(cids[u.owner_key])
        for m in stored:
            for other in USERS:
                if other is not u:
                    assert f"MARK{tag(other)}" not in m["content"]
        assert len([m for m in stored if m["role"] == "user"]) == 3
        assert len([m for m in stored if m["role"] == "assistant"]) == 3


async def test_AC3_tool_results_stay_with_their_own_conversation(env_factory):
    env = env_factory(echo_script, behaviour=tool_behaviour(), gate=False)
    pairs = [(ARJUN, PRIYA), (LAKSHMI, SENTHIL), (VIS1, VIS2)]
    cids = {}
    for a, b in pairs:
        cids[a.owner_key], cids[b.owner_key] = await new_conv(a), await new_conv(b)
    await asyncio.gather(*(collect(u, cids[u.owner_key], f"ask-MARK{tag(u)}") for u in USERS))
    for call in env.client.calls:
        flat = dump(call["messages"])
        results = set(re.findall(r"RESULT-MARK([A-Z0-9]+)", flat))
        asked = set(re.findall(r"ask-MARK([A-Z0-9]+)", flat))
        assert results <= asked, (results, asked)
    ran = {(owner, cid) for _n, _a, owner, cid in env.tools.ran}
    for u in USERS:
        assert (u.owner_key, cids[u.owner_key]) in ran
    assert len(ran) == len(USERS)


async def test_AC3_rebuilt_conversation_contains_only_its_own_history(env_factory):
    env = env_factory(lambda kw, n: Reply([f"Reply {n}."]))
    c_a, c_b, c_c = await new_conv(ARJUN), await new_conv(PRIYA), await new_conv(ARJUN)
    for cid, ctx, who in ((c_a, ARJUN, "A1"), (c_b, PRIYA, "B1"), (c_c, ARJUN, "C1")):
        await history.add_message(cid, "user", f"question {who}")
        await history.add_message(cid, "assistant", f"answer {who}")
    evict_memory()
    await collect(PRIYA, c_b, "question B2")
    flat = dump(env.client.calls[-1]["messages"])
    assert "question B1" in flat and "answer B1" in flat and "question B2" in flat
    for foreign in ("A1", "C1"):
        assert foreign not in flat
    evict_memory()
    await collect(ARJUN, c_c, "question C2")
    flat = dump(env.client.calls[-1]["messages"])
    assert "C1" in flat and "A1" not in flat and "B1" not in flat
    assert_valid(env.client.calls[-1]["messages"])


async def test_AC3_two_conversations_of_the_same_user_are_separate(env_factory):
    env = env_factory(lambda kw, n: Reply([f"Reply {n}."]))
    c1, c2 = await new_conv(ARJUN), await new_conv(ARJUN)
    await collect(ARJUN, c1, "alpha-topic")
    await collect(ARJUN, c2, "beta-topic")
    await collect(ARJUN, c1, "alpha-more")
    await collect(ARJUN, c2, "beta-more")
    last_c1, last_c2 = env.client.calls[2]["messages"], env.client.calls[3]["messages"]
    assert "beta" not in dump(last_c1) and "alpha" not in dump(last_c2)
    assert "alpha-topic" in dump(last_c1) and "beta-topic" in dump(last_c2)


async def test_AC3_cache_eviction_then_resume_rebuilds_only_own_history(env_factory, monkeypatch):
    monkeypatch.setattr(loop, "CACHE_CONVERSATIONS", 2)
    env = env_factory(lambda kw, n: Reply([f"Reply {n}."]))
    ctxs = [ARJUN, PRIYA, LAKSHMI, SENTHIL, VIS1]
    cids = [await new_conv(c) for c in ctxs]
    for c, cid in zip(ctxs, cids):
        await collect(c, cid, f"first-{tag(c)}")
    assert len(loop._memory) <= 2
    start = len(env.client.calls)
    for c, cid in zip(ctxs, cids):
        await collect(c, cid, f"second-{tag(c)}")
    for i, c in enumerate(ctxs):
        call = env.client.calls[start + i]
        flat = dump(call["messages"])
        assert f"first-{tag(c)}" in flat and f"second-{tag(c)}" in flat
        for other in ctxs:
            if other is not c:
                assert tag(other) not in flat
        assert_valid(call["messages"])


# DEFECT: evicting a conversation from the cache discards its lock, so a barge-in turn overlaps the running one.
async def test_AC3_eviction_never_drops_the_lock_of_a_running_turn(env_factory, monkeypatch):
    monkeypatch.setattr(loop, "CACHE_CONVERSATIONS", 1)
    env = env_factory(lambda kw, n: Reply([f"R{n}."], delay=0.1) if "slow" in dump(kw["messages"])
                      else Reply([f"R{n}."]))
    c_a, c_b = await new_conv(ARJUN), await new_conv(PRIYA)
    await collect(ARJUN, c_a, "warmup A")  # A is in memory
    slow = asyncio.create_task(collect(ARJUN, c_a, "slow second"))
    await asyncio.sleep(0.03)
    await collect(PRIYA, c_b, "B finishes and evicts A")  # evicts A while A's turn runs
    follow = asyncio.create_task(collect(ARJUN, c_a, "third on A"))
    await asyncio.gather(slow, follow)
    msgs = await db_messages(c_a)
    assert [m["role"] for m in msgs] == ["user", "assistant"] * 3, "turns on one conversation overlapped"
    for call in env.client.calls:
        assert_valid(call["messages"])


# DEFECT: run_turn takes any conversation id and replays its owner's history to a different user.
async def test_AC3_user_cannot_read_another_users_conversation_through_the_loop(env_factory):
    env = env_factory(lambda kw, n: Reply(["Here you go."]))
    victim = await new_conv(ARJUN)
    await history.add_message(victim, "user", "my private question about my leave")
    await history.add_message(victim, "assistant", "my private answer, about arjun")
    events = [e async for e in loop.run_turn(PRIYA, victim, "what did we discuss?")]
    leaked = any("private" in dump(c["messages"]) for c in env.client.calls)
    assert not leaked, "another user's history was sent to the model for a different user"
    stored = [m["content"] for m in await db_messages(victim)]
    assert "what did we discuss?" not in stored, "a foreign message was written into the victim's conversation"
    assert events == [] or events[-1]["type"] in ("error", "done")


# DEFECT: the in-memory conversation cache is keyed by conversation id only, not by owner.
async def test_AC3_user_cannot_resume_through_the_memory_cache_either(env_factory):
    env = env_factory(lambda kw, n: Reply([f"Reply {n}."]))
    victim = await new_conv(ARJUN)
    await collect(ARJUN, victim, "my private question")  # now cached
    before_calls = len(env.client.calls)
    before_rows = await db_messages(victim)
    try:
        events = [e async for e in loop.run_turn(PRIYA, victim, "what did he ask?")]
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"refusal must be a safe event, not {type(exc).__name__}")
    assert len(env.client.calls) == before_calls, "no model call for a non-owner"
    assert events and events[-1]["type"] == "error"
    assert "private" not in dump(events) and "Reply" not in dump(events)
    assert await db_messages(victim) == before_rows


async def test_AC3_stored_tool_calls_belong_to_the_right_conversation(env_factory, fake_college):
    def script(kw, n):
        if "tool_result" in dump(kw["messages"][-1]):
            return Reply(["ok"])
        return Reply([], [tool_use("t", "get_today", {})], "tool_use")
    env_factory(script, real_tools=True)
    a, p = fake_college.ctx("arjun"), fake_college.ctx("priya")
    ca, cp = await new_conv(a), await new_conv(p)
    await asyncio.gather(collect(a, ca, "date a"), collect(p, cp, "date p"), collect(a, ca, "date a2"))
    assert len(await db_tool_calls(ca)) == 2
    assert len(await db_tool_calls(cp)) == 1


async def test_AC2_each_tool_call_runs_with_the_ctx_of_its_own_turn(env_factory):
    env = env_factory(echo_script, behaviour=tool_behaviour(), gate=False)
    cids = {u.owner_key: await new_conv(u) for u in USERS}
    await asyncio.gather(*(collect(u, cids[u.owner_key], f"ask-MARK{tag(u)}") for u in USERS))
    for name, _args, owner, cid in env.tools.ran:
        assert cids[owner] == cid, "tool ran for one user inside another user's conversation"


async def test_AC3_many_concurrent_users_stress(env_factory):
    env = env_factory(echo_script, behaviour=tool_behaviour(), gate=False)
    users = [ARJUN, PRIYA, LAKSHMI, SENTHIL] * 3
    # same user twice at the same time in different conversations
    cids = [await new_conv(u) for u in users]
    await asyncio.gather(*(collect(u, cid, f"ask-MARK{tag(u)}N{i}")
                           for i, (u, cid) in enumerate(zip(users, cids))))
    for call in env.client.calls:
        asked = set(re.findall(r"ask-(MARK[A-Z0-9]+)", dump(call["messages"])))
        assert len(asked) == 1
        assert_valid(call["messages"])


async def test_AC3_memory_list_is_not_mutated_through_request_arguments(env_factory):
    env = env_factory([Reply(["One."]), Reply(["Two."])])
    cid = await new_conv(ARJUN)
    await collect(ARJUN, cid, "first")
    snapshot = dump(loop._memory[cid])
    await collect(ARJUN, cid, "second")
    assert dump(env.client.calls[0]["messages"]) .count("second") == 0
    assert "second" not in snapshot


async def test_AC3_visitors_in_different_sessions_do_not_share_memory(env_factory):
    env = env_factory(lambda kw, n: Reply([f"R{n}."]))
    c1, c2 = await new_conv(VIS1), await new_conv(VIS2)
    await collect(VIS1, c1, "visitor one secret-plan")
    await collect(VIS2, c2, "visitor two hello")
    assert "secret-plan" not in dump(env.client.calls[1])
