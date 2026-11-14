"""AC1 (with AC4, DOC1) end to end: a visitor never obtains any student or teacher data.

The visitor talks to the real app (typed and scripted-voice turns), the scripted model asks for
every student and teacher tool by name, and every exit is checked: HTTP, WebSocket frames, model
input, college calls and the database. Expectations come from ACCEPTANCE.md, never the code.
"""

import pytest
from app.agent import tools as app_tools

from tests.e2e_helpers import (
    DOC_MARK,
    NON_BROCHURE_MARKS,
    PRIVATE_PATH,
    TURN_MODES,
    A,
    e2e,  # noqa: F401
    echo,
    events,
    forbidden_for,
    hits,
    pending_rows,
    stored_tool_calls,
)

# Every tool CONTRACT.md / the tool layer reserves for students or teachers, with valid arguments.
PRIVATE_TOOL_ARGS = {
    "get_my_profile": {},
    "get_my_academics": {},
    "get_my_subjects": {},
    "get_attendance_summary": {},
    "get_attendance_sessions": {},
    "get_timetable": {},
    "get_calendar": {},
    "list_my_requests": {},
    "get_request": {"request_id": 12},
    "submit_leave_request": {"type": "personal_leave", "from_date": "2026-11-25",
                             "to_date": "2026-11-25", "half_day": False, "reason": "Trip"},
    "cancel_request": {"request_id": 12},
    "list_placement_drives": {},
    "register_for_placement_drive": {"drive_id": 9},
    "list_my_placements": {},
    "list_my_classes": {},
    "get_class_attendance": {"class_code": "CSE-2A"},
    "list_class_requests": {},
    "approve_request": {"request_id": 12},
    "reject_request": {"request_id": 12, "reason": "no proof"},
}


def private_tools() -> dict:
    """The contract's private tools plus any other tool the registry does not give visitors."""
    out = dict(PRIVATE_TOOL_ARGS)
    for t in app_tools.TOOLS:
        if "visitor" not in t.roles and t.name not in out:
            out[t.name] = {}
    return out


QUESTIONS = [
    "What is Arjun's attendance percentage?",
    "Tell me the timetable of CSE-2A for Monday.",
    "Show Arjun's leave requests.",
    "Which placement drives is Arjun eligible for?",
    "What does the student handbook say about on-duty proof?",
]


@TURN_MODES
@pytest.mark.parametrize("doc_mode", ["context", "rag"])
async def test_ac1_visitor_gets_no_private_data_through_any_tool(e2e, how, doc_mode, monkeypatch):
    from app.config import get_settings
    from app.knowledge import mode

    monkeypatch.setattr(get_settings(), "doc_mode", doc_mode)
    mode._cache.clear()
    # A logged-in student exists with live data and a conversation, to be sure there is
    # something to leak.
    a = await e2e.actor("A", A, ip="10.0.0.2")
    sa = await a.socket()
    await sa.start()
    e2e.model.plan("my attendance please", [("get_attendance_summary", {})], echo)
    await sa.turn("my attendance please")

    v = await e2e.actor("V1", ip="10.0.0.9")
    assert (await v.get("/api/me")).json() == {"role": "visitor", "user": None}
    s = await v.socket()
    ready = await s.start()
    assert ready["type"] == "ready" and ready["role"] == "visitor"
    assert ready.get("user_name") in (None, "")
    calls = [(n, args) for n, args in private_tools().items()]
    calls.append(("search_documents", {"query": "handbook on-duty proof attendance rules"}))
    for q in QUESTIONS:
        e2e.model.plan(q, calls, echo)
        frames = await s.turn(q, how)
        assert events(frames, "assistant_done") or events(frames, "error"), frames

    cid = s.conversation_id
    # Tools: every private tool was refused, and the visitor was never offered one.
    rows = await stored_tool_calls(cid)
    for name in private_tools():
        mine = [r for r in rows if r["name"] == name]
        assert mine, f"{name} call was not answered/logged"
        assert all(r["is_error"] for r in mine), f"visitor ran {name}: {mine}"
    for call in e2e.model.by("V1"):
        offered = set(call.tool_names)
        assert not offered & set(private_tools()), offered
    # College: no authenticated call, no private endpoint, nothing attributed to A's token.
    vcalls = e2e.log.by("V1")
    assert all(c.token is None and c.login is None for c in vcalls), \
        [(c.path, c.login) for c in vcalls if c.token]
    assert not [c.path for c in vcalls if PRIVATE_PATH.match(c.path)]
    assert not [c for c in e2e.college.writes() if c in vcalls]
    # Nothing private and no document other than the brochure in model input, frames or storage.
    forbidden = forbidden_for(None) + NON_BROCHURE_MARKS
    model_in = "\n".join(c.text() for c in e2e.model.by("V1"))
    assert DOC_MARK[3] in model_in or doc_mode == "rag"
    assert hits(model_in, forbidden) == []
    assert hits(s.text(), forbidden) == []
    from tests.e2e_helpers import db_text

    assert hits(await db_text(cid), forbidden) == []
    assert hits(v.http_text(), forbidden) == []
    # No pending action was created for the visitor.
    assert [p for p in await pending_rows() if str(p["conversation_id"]) == cid] == []


async def test_ac1_visitor_rest_routes_expose_nothing_of_a_student(e2e):
    a = await e2e.actor("A", A)
    sa = await a.socket()
    await sa.start()
    e2e.model.plan("hello from arjun", [("list_my_requests", {})], echo)
    await sa.turn("hello from arjun")
    a_cid = sa.conversation_id

    v = await e2e.actor("V1", ip=a.ip)  # same network as the student
    for url in ["/api/me", "/api/history", f"/api/history/{a_cid}", "/healthz",
                "/api/public/college", "/api/public/courses", "/", "/history", "/login"]:
        r = await v.get(url)
        assert hits(r.text, forbidden_for(None)) == [], url
        assert a_cid not in r.text, url
    assert (await v.get(f"/api/history/{a_cid}")).status_code == 404
    assert (await v.get("/api/history")).json() == []
    r = await v.get("/assistant", follow_redirects=False)
    assert r.status_code in (302, 303, 307) and "/login" in r.headers["location"]
    assert all(c.token is None for c in e2e.log.by("V1"))


@TURN_MODES
async def test_ac1_ac4_visitor_with_forged_role_login_answer_stays_visitor(e2e, how):
    """DOC9/AC4: a login answer with a role other than student/teacher yields no privileges."""
    e2e.college.login_overrides["arjun"] = {"user": {"login_id": "arjun", "role": "admin",
                                                     "full_name": "Arjun K"}}
    v = await e2e.actor("V1")
    r = await v.post("/api/login", json={"login_id": "arjun", "password": "Kit@2026"})
    assert r.status_code != 200 or r.json().get("user", {}).get("role") in ("student", "teacher")
    assert (await v.get("/api/me")).json()["role"] == "visitor"
    s = await v.socket()
    await s.start()
    e2e.model.plan("show my attendance", [("get_attendance_summary", {})], echo)
    await s.turn("show my attendance", how)
    assert all(r["is_error"] for r in await stored_tool_calls(s.conversation_id))
    assert all(c.login is None for c in e2e.log.by("V1") if c.path != "/v1/auth/login")
