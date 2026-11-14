"""AC1, AC4: role gating of the tool layer and the shape of tool_specs.

Expected tool sets come from PLAN.md "Tools by role" plus the infrastructure tools every role needs
(today's college date, public college info and courses, confirm/cancel of pending actions).
"""

import json

import pytest
from app.agent import tools as T
from app.db import pool

from tests.acc_tools_helpers import (
    ALL_TOOLS,
    ALLOWED,
    IDENTITY_WORDS,
    PRIVATE_TOOLS,
    ROLE_LOGIN,
    ROLES,
    SAMPLE_ARGS,
    SEARCH,
    STUDENT_ONLY,
    TEACHER_ONLY,
    VISITOR_ONLY,
    WRITE_TOOLS,
    body,
    new_conv,
    run,
)


@pytest.fixture
def stub_search(monkeypatch):
    """search_documents must call app.knowledge.rag.search(query, ctx); record the calls."""
    from app.knowledge import rag

    seen = []

    async def fake_search(query, ctx, k=6):
        seen.append((query, ctx))
        return [{"document_title": "KIT Student Handbook 2026", "version": "2026.1",
                 "heading": "4.1 Attendance", "content": "75% per course.", "score": 0.9}]

    monkeypatch.setattr(rag, "search", fake_search)
    return seen


# ------------------------------------------------------------------ tool_specs


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("include_search", [False, True])
def test_ac4_tool_specs_exact_set_per_role(role, include_search):
    names = [s["name"] for s in T.tool_specs(role, include_search)]
    expected = set(ALLOWED[role]) | ({SEARCH} if include_search else set())
    assert set(names) == expected
    assert len(names) == len(set(names)), "duplicate tool names"


@pytest.mark.parametrize("role", ROLES)
def test_ac4_search_only_when_included(role):
    assert SEARCH not in [s["name"] for s in T.tool_specs(role, False)]
    assert SEARCH in [s["name"] for s in T.tool_specs(role, True)]


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("include_search", [False, True])
def test_ac4_tool_specs_stable_order(role, include_search):
    a = T.tool_specs(role, include_search)
    b = T.tool_specs(role, include_search)
    assert json.dumps(a, sort_keys=False) == json.dumps(b, sort_keys=False)


@pytest.mark.parametrize("role", ROLES)
def test_ac4_tool_specs_order_independent_of_search(role):
    without = [s["name"] for s in T.tool_specs(role, False)]
    with_search = [s["name"] for s in T.tool_specs(role, True) if s["name"] != SEARCH]
    assert without == with_search


@pytest.mark.parametrize("role", ["", "admin", "STUDENT", "Teacher", "guest", "root", "visitor "])
def test_ac4_tool_specs_unknown_role_gets_nothing(role):
    assert T.tool_specs(role, True) == []


def _all_specs():
    seen = {}
    for role in ROLES:
        for s in T.tool_specs(role, True):
            seen[s["name"]] = s
    return sorted(seen.values(), key=lambda s: s["name"])


@pytest.mark.parametrize("spec", _all_specs(), ids=lambda s: s["name"])
def test_ac2_no_tool_parameter_can_carry_identity(spec):
    props = spec["input_schema"].get("properties", {})
    for pname in props:
        assert not any(w in pname.lower() for w in IDENTITY_WORDS), (spec["name"], pname)


@pytest.mark.parametrize("spec", _all_specs(), ids=lambda s: s["name"])
def test_ac4_tool_schema_is_strict(spec):
    schema = spec["input_schema"]
    assert schema["type"] == "object"
    assert schema.get("additionalProperties") is False
    assert set(schema.get("required", [])) <= set(schema.get("properties", {}))
    assert spec["description"].strip()


@pytest.mark.parametrize("spec", _all_specs(), ids=lambda s: s["name"])
def test_ac5_tool_spec_contains_no_secret(spec, fake_college):
    text = json.dumps(spec)
    assert fake_college.participant_key not in text
    assert "Kit@2026" not in text


# ------------------------------------------------------------------ run_tool gating matrix


def _matrix():
    for role in ROLES:
        for name in ALL_TOOLS:
            yield role, name


@pytest.mark.parametrize("role,name", [p for p in _matrix()
                                       if p[1] not in ALLOWED[p[0]] and p[1] != SEARCH])
async def test_ac4_tool_outside_role_refused_without_college_call(role, name, fake_college):
    ctx = fake_college.ctx(ROLE_LOGIN[role])
    cid = await new_conv(ctx)
    fake_college.log.clear()
    r, b = await run(name, SAMPLE_ARGS[name], ctx, cid)
    assert r.is_error
    assert fake_college.log == [], [c.path for c in fake_college.log]
    assert await pool().fetchval("SELECT count(*) FROM pending_actions") == 0


@pytest.mark.parametrize("role,name", [p for p in _matrix() if p[1] in ALLOWED[p[0]]])
async def test_ac4_tool_inside_role_not_refused_by_gate(role, name, fake_college):
    ctx = fake_college.ctx(ROLE_LOGIN[role])
    cid = await new_conv(ctx)
    r, b = await run(name, SAMPLE_ARGS[name], ctx, cid)
    if name in ("confirm_pending_action", "cancel_pending_action"):
        # unknown action id: an error, but about the action, not the role
        assert r.is_error
        assert "not available" not in r.content
        return
    assert not r.is_error, b


@pytest.mark.parametrize("role", ROLES)
async def test_ac4_search_runs_rag_with_session_ctx(role, fake_college,
                                                                         stub_search):
    """search_documents is a known tool for every role; with search on it runs rag.search(ctx)."""
    ctx = fake_college.ctx(ROLE_LOGIN[role])
    cid = await new_conv(ctx)
    r, b = await run(SEARCH, {"query": "attendance rule"}, ctx, cid)
    assert not r.is_error, b
    assert stub_search and stub_search[-1][0] == "attendance rule"
    assert stub_search[-1][1] is ctx
    assert "75% per course" in r.content


@pytest.mark.parametrize("name", ["", "delete_everything", "GET_TODAY", "get_today ",
                                  "get-today", "run_sql", "__init__", "approve_all_requests",
                                  "admin_override", "../get_today", "get_today;drop",
                                  "submit_admission_application\x00"])
@pytest.mark.parametrize("role", ROLES)
async def test_ac4_unknown_tool_refused(name, role, fake_college):
    ctx = fake_college.ctx(ROLE_LOGIN[role])
    cid = await new_conv(ctx)
    fake_college.log.clear()
    r, b = await run(name, {}, ctx, cid)
    assert r.is_error
    assert isinstance(b, dict)
    assert fake_college.log == []


@pytest.mark.parametrize("role", ["admin", "", "STUDENT", "superuser"])
@pytest.mark.parametrize("name", ["get_today", "get_my_profile", "approve_request",
                                  "submit_leave_request"])
async def test_ac4_unknown_role_refused_everything(role, name, fake_college):
    from app.auth import SessionCtx

    base = fake_college.ctx("arjun")
    ctx = SessionCtx(base.session_id, base.owner_key, role, base.user, base.kit_token)
    cid = await new_conv(fake_college.ctx("arjun"))
    fake_college.log.clear()
    r, _ = await run(name, SAMPLE_ARGS[name], ctx, cid)
    assert r.is_error
    assert fake_college.log == []


# ------------------------------------------------------------------ visitors (AC1)


@pytest.mark.parametrize("name", PRIVATE_TOOLS)
async def test_ac1_visitor_cannot_run_private_tool(name, fake_college):
    ctx = fake_college.ctx(None)
    cid = await new_conv(ctx)
    r, b = await run(name, SAMPLE_ARGS[name], ctx, cid)
    assert r.is_error
    assert fake_college.log == []
    for marker in ("Arjun", "Priya", "Lakshmi", "CS304", "CSE-2A", "cgpa"):
        assert marker not in r.content


@pytest.mark.parametrize("name", sorted(ALLOWED["visitor"]))
async def test_ac1_visitor_tools_never_send_a_token(name, fake_college):
    ctx = fake_college.ctx(None)
    cid = await new_conv(ctx)
    await run(name, SAMPLE_ARGS[name], ctx, cid)
    assert all(c.token is None for c in fake_college.log), [(c.path, c.token) for c in
                                                            fake_college.log]
    assert all(c.login is None for c in fake_college.log)


async def test_ac1_visitor_with_stray_token_still_cannot_reach_private(fake_college):
    """Even a visitor ctx that somehow carries a token is gated by role."""
    from app.auth import SessionCtx

    tok = fake_college.issue_token("arjun")
    ctx = SessionCtx("sid-v", "visitor:abc", "visitor", None, tok)
    cid = await new_conv(ctx)
    for name in PRIVATE_TOOLS:
        r, _ = await run(name, SAMPLE_ARGS[name], ctx, cid)
        assert r.is_error, name
    assert [c.path for c in fake_college.log] == []


@pytest.mark.parametrize("login,name", [("arjun", n) for n in STUDENT_ONLY + ["get_my_profile"]]
                         + [("lakshmi", n) for n in TEACHER_ONLY + ["get_my_profile"]])
async def test_ac6_logged_in_role_without_token_is_no_session(name, login, fake_college):
    """A logged-in role whose college token is gone gets no data and no college call."""
    from app.auth import SessionCtx

    base = fake_college.ctx(login)
    ctx = SessionCtx(base.session_id, base.owner_key, base.role, base.user, None)
    cid = await new_conv(ctx)
    fake_college.log.clear()
    r, _ = await run(name, SAMPLE_ARGS[name], ctx, cid)
    assert r.is_error
    assert all(c.token is None for c in fake_college.log)
    assert not [c for c in fake_college.log if c.path.startswith(("/v1/me", "/v1/teacher"))]


# ------------------------------------------------------------------ arguments


@pytest.mark.parametrize("args", [None, [], "get_today", 42, ["action_id"], True])
@pytest.mark.parametrize("name", ["get_today", "get_request", "submit_leave_request"])
async def test_agt6_non_object_args_are_an_error_not_a_crash(args, name, fake_college):
    ctx = fake_college.ctx("arjun")
    cid = await new_conv(ctx)
    r = await T.run_tool(name, args, ctx, cid)
    assert r.is_error
    assert isinstance(json.loads(r.content), dict)
    assert "Traceback" not in r.content


@pytest.mark.parametrize("name", sorted(set(ALL_TOOLS) - {SEARCH}))
async def test_ac4_every_call_is_logged_even_refused(name, fake_college):
    ctx = fake_college.ctx(None)
    cid = await new_conv(ctx)
    await run(name, SAMPLE_ARGS[name], ctx, cid)
    rows = await pool().fetch("SELECT name, is_error FROM tool_calls WHERE conversation_id = $1",
                              __import__("uuid").UUID(cid))
    assert [r["name"] for r in rows] == [name]


@pytest.mark.parametrize("name", WRITE_TOOLS)
def test_ac8_write_tools_are_known(name):
    assert name in {s["name"] for r in ROLES for s in T.tool_specs(r, False)}


@pytest.mark.parametrize("name", VISITOR_ONLY)
@pytest.mark.parametrize("role", ["student", "teacher"])
async def test_ac4_admission_application_is_visitor_only(name, role, fake_college):
    ctx = fake_college.ctx(ROLE_LOGIN[role])
    cid = await new_conv(ctx)
    r, _ = await run(name, SAMPLE_ARGS[name], ctx, cid)
    assert r.is_error
    assert fake_college.log == []
    assert body(r)
