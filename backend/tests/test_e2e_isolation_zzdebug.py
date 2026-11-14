from tests.e2e_helpers import e2e, echo, A, stored_tool_calls, stored_messages  # noqa

async def test_debug(e2e):
    print("REFS", e2e.refs)
    a = await e2e.actor("A", A)
    s = await a.socket()
    print("READY", await s.start())
    e2e.model.plan("my attendance please", [("get_attendance_summary", {}), ("list_my_requests", {})], echo)
    fr = await s.turn("my attendance please", "voice")
    for f in fr: print("FRAME", f if isinstance(f, dict) else ("BYTES", f[:60]))
    print("LOG", [(a_, c.method, c.path, c.login) for a_, c in e2e.log.pairs()])
    print("MODEL", len(e2e.model.calls), [c.actor for c in e2e.model.calls], e2e.model.calls[0].tool_names)
    print("SYS", e2e.model.calls[0].system_text[:3000])
    print("TC", await stored_tool_calls(s.conversation_id))
    print("MSG", await stored_messages(s.conversation_id))
