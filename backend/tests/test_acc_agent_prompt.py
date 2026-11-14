"""Acceptance: prompt layout, tool list per role, secrets and injection, document mode.

Criteria: AGT5, AC2, AC4, AC5, DOC5, REL2 (as seen from the assistant loop and prompts module).
"""

import json
import re

import pytest
from app import history
from app.agent import loop, prompts
from app.agent import tools as real_tools_mod
from app.auth import SessionCtx
from app.config import get_settings
from tests.acc_agent_helpers import (  # noqa: F401
    ARJUN, KB_TEXT, LAKSHMI, NOTES_INJECTION, PRIYA, ROLE_PAIRS, SENTHIL, TODAY, TOKEN_ARJUN,
    TOKEN_LAKSHMI, TOKEN_PRIYA, VIS1, VIS2, Reply, collect, db_messages, dump, env_factory,
    new_conv, request_text, text, tool_use,
)

ROLES = ["visitor", "student", "teacher"]
DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")


def per_user_pending(ctx):
    return f"Action 77aa-{(ctx.user or {}).get('login_id', 'visitor')}: leave request pending" \
        if ctx.user else None


def specific_strings(ctx: SessionCtx) -> list[str]:
    """Everything that identifies this user or this moment."""
    out = [ctx.owner_key, ctx.session_id, TODAY["date"], TODAY["weekday_name"]]
    if ctx.user:
        out += [ctx.user["full_name"], ctx.user["login_id"], ctx.user["class_code"],
                ctx.user["full_name"].split()[0], ctx.user["full_name"].split()[-1]]
    pend = per_user_pending(ctx)
    if pend:
        out.append(pend)
    return out


def split_system(system: list[dict]):
    """(blocks up to and including the cache breakpoint, blocks after it)."""
    idx = [i for i, b in enumerate(system) if b.get("cache_control")]
    assert len(idx) == 1, "exactly one cache breakpoint in the system prompt"
    return system[: idx[0] + 1], system[idx[0] + 1:]


async def one_turn(env_factory_fn, ctx, **kw):
    env = env_factory_fn([Reply(["Hello."])], mode=kw.pop("mode", "context"),
                         pending=per_user_pending, **kw)
    cid = await new_conv(ctx)
    await collect(ctx, cid, "hello there")
    return env.client.calls[0], env


# ------------------------------------------------------------------ AGT5

@pytest.mark.parametrize("role", ROLES)
async def test_AGT5_cached_block_identical_for_two_users_of_the_same_role(env_factory, role):
    a, b = ROLE_PAIRS[role]
    call_a, _ = await one_turn(env_factory, a)
    call_b, _ = await one_turn(env_factory, b)
    head_a, tail_a = split_system(call_a["system"])
    head_b, tail_b = split_system(call_b["system"])
    assert json.dumps(head_a, sort_keys=True) == json.dumps(head_b, sort_keys=True)
    assert head_a[-1]["text"].encode() == head_b[-1]["text"].encode()
    assert tail_a and tail_b


@pytest.mark.parametrize("role", ROLES)
async def test_AGT5_nothing_user_specific_or_timed_before_the_breakpoint(env_factory, role):
    a, b = ROLE_PAIRS[role]
    for ctx in (a, b):
        call, _ = await one_turn(env_factory, ctx)
        head, _tail = split_system(call["system"])
        flat = dump(head)
        for s in specific_strings(ctx):
            assert s not in flat, f"{s!r} found before the cache breakpoint"
        for s in specific_strings(a if ctx is b else b):
            assert s not in flat
        assert not DATE_RE.search(flat)
        assert not TIME_RE.search(flat)
        assert "2031" not in flat and "Quintday" not in flat
        # the tools sit before the breakpoint in the cache order, so they must be clean too
        tools_flat = dump(call["tools"])
        for s in specific_strings(ctx):
            assert s not in tools_flat


@pytest.mark.parametrize("role", ROLES)
async def test_AGT5_user_facts_block_carries_the_per_user_and_time_facts(env_factory, role):
    ctx = ROLE_PAIRS[role][0]
    call, _ = await one_turn(env_factory, ctx)
    _head, tail = split_system(call["system"])
    flat = dump(tail)
    assert TODAY["date"] in flat, "college date belongs in the per-user block"
    assert role in flat.lower() or {"visitor": "visitor", "student": "student",
                                    "teacher": "teacher"}[role] in flat.lower()
    if ctx.user:
        assert ctx.user["full_name"] in flat
        assert ctx.user["class_code"] in flat
        assert per_user_pending(ctx) in flat
    assert "cache_control" not in flat


@pytest.mark.parametrize("role", ROLES)
async def test_AGT5_system_text_never_uses_the_server_clock(env_factory, role):
    import datetime
    call, _ = await one_turn(env_factory, ROLE_PAIRS[role][0])
    flat = dump(call["system"])
    today = datetime.date.today()
    assert today.isoformat() not in flat
    assert today.strftime("%A") + " " + today.isoformat() not in flat


@pytest.mark.parametrize("role", ROLES)
async def test_AGT5_date_change_between_turns_only_changes_the_per_user_block(
        monkeypatch, env_factory, role):
    ctx = ROLE_PAIRS[role][0]
    env = env_factory([Reply(["A."]), Reply(["B."])], pending=per_user_pending)
    cid = await new_conv(ctx)
    await collect(ctx, cid, "one")

    async def tomorrow():
        return {"date": "2031-07-10", "weekday": 3, "weekday_name": "Sexday", "timezone": "x"}
    monkeypatch.setattr(loop, "college_today", tomorrow)
    await collect(ctx, cid, "two")
    h1, t1 = split_system(env.client.calls[0]["system"])
    h2, t2 = split_system(env.client.calls[1]["system"])
    assert h1 == h2 and env.client.calls[0]["tools"] == env.client.calls[1]["tools"]
    assert dump(t1) != dump(t2) and "2031-07-10" in dump(t2)


async def test_AGT5_pending_actions_only_in_per_user_block(env_factory):
    env = env_factory([Reply(["ok"])], pending=lambda ctx: "Action 4f1c9: submit leave for tomorrow")
    cid = await new_conv(ARJUN)
    await collect(ARJUN, cid, "hi")
    head, tail = split_system(env.client.calls[0]["system"])
    assert "4f1c9" not in dump(head) and "4f1c9" in dump(tail)
    assert "4f1c9" not in dump(env.client.calls[0]["tools"])


@pytest.mark.parametrize("role", ROLES)
async def test_AGT5_tools_identical_across_users_of_a_role_and_match_role_set(env_factory, role):
    a, b = ROLE_PAIRS[role]
    ca, _ = await one_turn(env_factory, a, real_tools=True)
    cb, _ = await one_turn(env_factory, b, real_tools=True)
    assert json.dumps(ca["tools"], sort_keys=True) == json.dumps(cb["tools"], sort_keys=True)
    expected = real_tools_mod.tool_specs(role, include_search=False)
    assert [t["name"] for t in ca["tools"]] == [t["name"] for t in expected]


@pytest.mark.parametrize("role", ROLES)
def test_AGT5_prompts_system_blocks_layout(role):
    a, b = ROLE_PAIRS[role]
    blocks_a = prompts.system_blocks(a, KB_TEXT, TODAY, "pending-A")
    blocks_b = prompts.system_blocks(b, KB_TEXT, {**TODAY, "date": "2031-07-11"}, "pending-B")
    assert blocks_a[0] == blocks_b[0]
    assert blocks_a[0]["cache_control"] == {"type": "ephemeral"}
    assert all("cache_control" not in blk for blk in blocks_a[1:])
    assert len(blocks_a) >= 2 and all(blk["type"] == "text" for blk in blocks_a)
    assert KB_TEXT in blocks_a[0]["text"]
    assert KB_TEXT not in "".join(blk["text"] for blk in blocks_a[1:])
    assert "pending-A" in blocks_a[1]["text"] and "pending-A" not in blocks_a[0]["text"]
    assert "2031-07-09" in blocks_a[1]["text"] and "2031-07-11" in blocks_b[1]["text"]
    assert blocks_a[0]["text"].encode() == blocks_b[0]["text"].encode()


def test_AGT5_prompts_stable_block_changes_only_with_the_knowledge():
    a = prompts.system_blocks(ARJUN, KB_TEXT, TODAY, None)[0]
    b = prompts.system_blocks(PRIYA, KB_TEXT, None, "x")[0]
    c = prompts.system_blocks(PRIYA, KB_TEXT + "\nAnother circular.", None, "x")[0]
    assert a == b and a != c and "Another circular." in c["text"]


def test_AGT5_prompts_unknown_date_is_said_not_guessed():
    blocks = prompts.system_blocks(VIS1, KB_TEXT, None, None)
    assert "unknown" in blocks[1]["text"].lower()
    assert not DATE_RE.search(blocks[1]["text"])


@pytest.mark.parametrize("ctx", [VIS1, ARJUN, LAKSHMI])
def test_AGT5_prompts_user_facts_shape(ctx):
    facts = prompts.user_facts(ctx, TODAY, None)
    assert isinstance(facts, str) and facts.strip()
    assert ctx.role in facts.lower() or "visitor" in facts.lower()
    assert "none" in facts.lower()  # no pending actions said plainly
    if ctx.user:
        assert ctx.user["full_name"] in facts


def test_AGT5_instructions_ask_for_speech_friendly_short_replies():
    flat = prompts.INSTRUCTIONS.lower()
    assert "markdown" in flat and "short" in flat
    assert "college date" in flat
    assert "never" in flat and "guess" in flat or "invent" in flat


async def test_AGT5_instructions_do_not_embed_dates_or_times(env_factory):
    assert not DATE_RE.search(prompts.INSTRUCTIONS)
    assert not TIME_RE.search(prompts.INSTRUCTIONS)


async def test_AGT5_cache_breakpoints_stay_within_the_api_limit(env_factory):
    env = env_factory([Reply([], [tool_use("t", "get_my_profile")], "tool_use"), Reply(["Ok."])])
    cid = await new_conv(ARJUN)
    await collect(ARJUN, cid, "x")
    for call in env.client.calls:
        n = dump(call).count('"cache_control"')
        assert 1 <= n <= 4


# ------------------------------------------------------------------ AC4 / AC2 as seen from the loop

@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("mode", ["context", "rag"])
async def test_AC4_tool_list_sent_to_model_is_exactly_the_roles_set(env_factory, role, mode):
    ctx = ROLE_PAIRS[role][0]
    call, _ = await one_turn(env_factory, ctx, mode=mode, real_tools=True)
    names = [t["name"] for t in call["tools"]]
    assert names == [t["name"] for t in real_tools_mod.tool_specs(role, include_search=mode == "rag")]
    assert len(names) == len(set(names))
    if role != "teacher":
        assert "approve_request" not in names and "reject_request" not in names
    if role != "student":
        for n in ("submit_leave_request", "register_for_placement_drive", "cancel_request",
                  "get_attendance_summary", "get_timetable", "list_placement_drives"):
            assert n not in names
    if role == "teacher":
        assert "list_my_classes" in names
    if role == "visitor":
        assert "get_my_profile" not in names and "submit_admission_application" in names
        assert not [n for n in names if n.startswith(("get_my", "list_my"))]


@pytest.mark.parametrize("role", ROLES)
async def test_AC2_no_tool_sent_to_the_model_takes_an_identity_parameter(env_factory, role):
    call, _ = await one_turn(env_factory, ROLE_PAIRS[role][0], mode="rag", real_tools=True)
    banned = {"user_id", "login_id", "student_id", "teacher_id", "token", "kit_token", "owner_key",
              "session_id", "participant_key", "conversation_id", "password"}
    for t in call["tools"]:
        props = set(t["input_schema"].get("properties", {}))
        assert not props & banned, (t["name"], props & banned)


async def test_AC2_identity_arguments_from_the_model_never_reach_another_users_data(
        env_factory, fake_college):
    env_factory([Reply([], [tool_use("t", "get_my_profile",
                                     {"user_id": 29, "login_id": "priya", "student_id": 29})],
                       "tool_use"), Reply(["Ok."])], real_tools=True)
    ctx = fake_college.ctx("arjun")
    cid = await new_conv(ctx)
    await collect(ctx, cid, "show priya's profile")
    assert {c.login for c in fake_college.calls() if c.path != "/v1/auth/login"} <= {"arjun"}
    from tests.fakes.college import user_summary
    assert user_summary("priya")["full_name"] not in dump(loop._client.calls[1]["messages"])


async def test_AC2_every_college_call_carries_that_sessions_token(env_factory, fake_college):
    def script(kw, n):
        flat = dump(kw["messages"])
        who = "arjun" if "as arjun" in flat else "priya"
        if "tool_result" in flat:
            return Reply([f"Done for {who}."])
        return Reply([], [tool_use(f"t-{who}", "get_my_profile" if who == "arjun" else "get_my_academics")],
                     "tool_use")
    env_factory(script, real_tools=True)
    a, p = fake_college.ctx("arjun"), fake_college.ctx("priya")
    ca, cp = await new_conv(a), await new_conv(p)
    import asyncio
    await asyncio.gather(collect(a, ca, "as arjun"), collect(p, cp, "as priya"))
    me = fake_college.calls(path="/v1/me")
    acad = fake_college.calls(path="/v1/academics") or [c for c in fake_college.calls() if "academ" in c.path]
    assert me and {c.login for c in me} == {"arjun"}
    assert acad and {c.login for c in acad} == {"priya"}


# ------------------------------------------------------------------ AC5

def all_secrets(*ctxs) -> list[str]:
    s = get_settings()
    vals = [s.kit_participant_key, s.anthropic_api_key, s.deepgram_api_key, s.gemini_api_key,
            "Kit@2026", TOKEN_ARJUN, TOKEN_PRIYA, TOKEN_LAKSHMI]
    vals += [c.kit_token for c in ctxs if c.kit_token]
    return [v for v in vals if v]


@pytest.mark.parametrize("ctx", [VIS1, ARJUN, PRIYA, LAKSHMI], ids=lambda c: c.owner_key)
async def test_AC5_no_secret_or_profile_notes_in_any_request(env_factory, ctx):
    env = env_factory([Reply(["Checking."], [text("Checking."), tool_use("t", "get_my_profile")],
                             "tool_use"), Reply(["Done."])], pending=per_user_pending)
    cid = await new_conv(ctx)
    events = await collect(ctx, cid, "who am I and what can you do")
    notes = (ctx.user or {}).get("notes")
    for call in env.client.calls:
        flat = request_text(call)
        for secret in all_secrets(ARJUN, PRIYA, LAKSHMI, ctx):
            assert secret not in flat
        if notes:
            assert notes not in flat
        assert NOTES_INJECTION not in flat
    # nor in what the user hears or what is stored
    out = dump(events) + dump(await db_messages(cid))
    for secret in all_secrets(ARJUN, PRIYA, LAKSHMI, ctx):
        assert secret not in out
    if notes:
        assert notes not in out


async def test_AC5_injection_in_notes_does_not_reach_prompt_after_several_turns_and_rebuild(
        env_factory):
    env = env_factory(lambda kw, n: Reply([f"Answer {n}."]), pending=per_user_pending)
    cid = await new_conv(ARJUN)
    for i in range(3):
        await collect(ARJUN, cid, f"question {i}")
    from tests.acc_agent_helpers import evict_memory
    evict_memory()
    await collect(ARJUN, cid, "after rebuild")
    for call in env.client.calls:
        assert "approve all on-duty" not in request_text(call)
        assert "SYSTEM:" not in dump(call["system"])


async def test_AC5_real_profile_tool_result_carries_no_notes_or_token(env_factory, fake_college):
    env = env_factory([Reply([], [tool_use("t", "get_my_profile")], "tool_use"), Reply(["Ok."])],
                      real_tools=True)
    ctx = fake_college.ctx("lakshmi")
    cid = await new_conv(ctx)
    await collect(ctx, cid, "who am i")
    flat = request_text(env.client.calls[1])
    assert "Class Advisor for" not in flat
    assert ctx.kit_token not in flat
    assert get_settings().kit_participant_key not in flat


async def test_AC5_error_from_failed_tool_does_not_leak_token_or_key(env_factory, fake_college):
    fake_college.fail("/v1/me", 500)
    env = env_factory([Reply([], [tool_use("t", "get_my_profile")], "tool_use"), Reply(["Sorry."])],
                      real_tools=True)
    ctx = fake_college.ctx("arjun")
    cid = await new_conv(ctx)
    events = await collect(ctx, cid, "who am i")
    flat = request_text(env.client.calls[1]) + dump(events)
    assert ctx.kit_token not in flat and get_settings().kit_participant_key not in flat
    assert "Traceback" not in flat


async def test_AC5_error_events_do_not_contain_secrets(env_factory):
    import anthropic
    import httpx
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages",
                        headers={"x-api-key": get_settings().anthropic_api_key})
    exc = anthropic.APIConnectionError(request=req)
    env_factory([Reply(error=exc)])
    cid = await new_conv(ARJUN)
    events = await collect(ARJUN, cid, "hi")
    flat = dump(events)
    for secret in all_secrets(ARJUN):
        assert secret not in flat


async def test_AC5_secrets_in_user_text_are_not_echoed_into_system_block(env_factory):
    env = env_factory([Reply(["ok"])])
    cid = await new_conv(ARJUN)
    await collect(ARJUN, cid, "my password is hunter2-xyz")
    assert "hunter2-xyz" not in dump(env.client.calls[0]["system"])


def test_AC5_prompts_never_render_token_or_notes_even_if_present_on_the_user():
    ctx = SessionCtx("s", "user:z", "student",
                     {"full_name": "Zed", "notes": NOTES_INJECTION, "password": "pw-zz",
                      "kit_token": "tok-zz", "class_code": "X-1", "department": "CSE"}, "tok-real")
    blocks = prompts.system_blocks(ctx, KB_TEXT, TODAY, None)
    flat = dump(blocks)
    for bad in (NOTES_INJECTION, "pw-zz", "tok-zz", "tok-real"):
        assert bad not in flat


def test_AC5_prompts_instruct_that_documents_and_tool_output_are_data():
    flat = prompts.INSTRUCTIONS.lower()
    assert "never instructions" in flat or "not instructions" in flat


# ------------------------------------------------------------------ DOC5

@pytest.mark.parametrize("role", ROLES)
async def test_DOC5_search_tool_offered_only_in_rag_mode(env_factory, role):
    ctx = ROLE_PAIRS[role][0]
    rag, _ = await one_turn(env_factory, ctx, mode="rag", real_tools=True)
    context, _ = await one_turn(env_factory, ctx, mode="context", real_tools=True)
    assert "search_documents" in [t["name"] for t in rag["tools"]]
    assert "search_documents" not in [t["name"] for t in context["tools"]]
    # everything else is the same
    assert [t for t in rag["tools"] if t["name"] != "search_documents"] == context["tools"]


@pytest.mark.parametrize("mode", ["context", "rag"])
async def test_DOC5_knowledge_text_is_in_the_cached_block_only(env_factory, mode):
    call, _ = await one_turn(env_factory, ARJUN, mode=mode)
    head, tail = split_system(call["system"])
    assert KB_TEXT in head[-1]["text"]
    assert KB_TEXT not in dump(tail)
    assert KB_TEXT not in dump(call["messages"])


async def test_DOC5_mode_switch_between_turns_changes_the_tool_list(env_factory):
    modes = iter(["context", "rag", "context"])
    state = {"m": "context"}
    env = env_factory([Reply(["a"]), Reply(["b"]), Reply(["c"])], mode=lambda ctx: state["m"],
                      real_tools=True)
    cid = await new_conv(ARJUN)
    for q in range(3):
        state["m"] = next(modes)
        await collect(ARJUN, cid, f"q{q}")
    has = ["search_documents" in [t["name"] for t in c["tools"]] for c in env.client.calls]
    assert has == [False, True, False]


async def test_DOC5_without_knowledge_text_prompt_still_valid_and_no_search(env_factory):
    env = env_factory([Reply(["ok"])], knowledge="   ")
    cid = await new_conv(VIS1)
    await collect(VIS1, cid, "hi")
    head, _ = split_system(env.client.calls[0]["system"])
    assert head[-1]["text"].strip()


async def test_DOC5_unavailable_knowledge_says_so_and_offers_no_search(env_factory, monkeypatch):
    env = env_factory([Reply(["ok"])], real_tools=True)

    async def boom(ctx):
        raise RuntimeError("db down")
    monkeypatch.setattr(loop, "knowledge_block", boom)
    cid = await new_conv(VIS1)
    events = await collect(VIS1, cid, "hi")
    assert events[-1]["type"] == "done"
    call = env.client.calls[0]
    assert "search_documents" not in [t["name"] for t in call["tools"]]
    assert "unavailable" in dump(call["system"]).lower()


# ------------------------------------------------------------------ warm_cache (prefix reuse)

async def test_AGT5_warm_cache_sends_only_shared_prefix_and_never_raises(env_factory, monkeypatch):
    from app.knowledge import store

    async def docs():
        return []
    monkeypatch.setattr(store, "active_documents", docs, raising=False)
    env = env_factory([Reply(["x"])], pending=per_user_pending)
    await loop.warm_cache()
    flat = dump(env.client.create_calls)
    assert env.client.create_calls, "one warm request per shared prefix"
    for ctx in (ARJUN, PRIYA, LAKSHMI, VIS1):
        for s in specific_strings(ctx):
            assert s not in flat
    for secret in all_secrets(ARJUN):
        assert secret not in flat
    for c in env.client.create_calls:
        assert c["system"][0].get("cache_control")


@pytest.mark.parametrize("role", ROLES)
async def test_AGT5_warm_prefix_matches_a_real_turns_prefix(env_factory, monkeypatch, role):
    from app.knowledge import store

    async def docs():
        return []
    monkeypatch.setattr(store, "active_documents", docs, raising=False)
    ctx = ROLE_PAIRS[role][0]
    env = env_factory([Reply(["x"])], real_tools=True)
    await loop.warm_cache()
    cid = await new_conv(ctx)
    await collect(ctx, cid, "hi")
    real = env.client.calls[0]
    head, _ = split_system(real["system"])
    assert any(c["system"][0] == head[-1] and c["tools"] == real["tools"]
               for c in env.client.create_calls), "warm-up must write the prefix real turns read"


async def test_AGT6_warm_cache_swallows_every_failure(env_factory, monkeypatch):
    env = env_factory([Reply(["x"])])

    async def boom(**kw):
        raise RuntimeError("api down")
    monkeypatch.setattr(env.client, "create", boom)
    await loop.warm_cache()  # must not raise


async def test_AGT6_warm_cache_survives_knowledge_failure(env_factory, monkeypatch):
    env_factory([Reply(["x"])])

    async def boom(ctx):
        raise RuntimeError("db down")
    monkeypatch.setattr(loop, "knowledge_block", boom)
    await loop.warm_cache()  # must not raise
