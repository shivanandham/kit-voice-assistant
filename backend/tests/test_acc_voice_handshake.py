"""AC7, AC1, AC3, VOX1: handshake, origin, session and conversation resumption."""

import re
import uuid

import pytest
from app.config import get_settings
from app.db import pool

from tests.acc_voice_helpers import (
    DG_KEY,
    SECRET_KIT_TOKEN,
    Agent,
    close_code,
    echo_history_agent,
    expect_refused,
    expire_kit_token,
    expire_session,
    history_rows,
    make_server,  # noqa: F401
    make_session,
    types,
)

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


async def _no_upstream_used(srv):
    assert all(not s.frames and s.closed >= 0 for s in srv.stts)
    assert all(not t.requested and not t.warms for t in srv.ttss)


# ------------------------------------------------------------------------- AC7: session


async def test_ac7_no_cookie_closed_4401(make_server):
    srv = await make_server(Agent())
    c = await srv.connect()
    assert await close_code(c) == 4401
    await _no_upstream_used(srv)


async def test_ac1_no_cookie_gets_nothing_even_if_client_talks(make_server):
    agent = Agent()
    srv = await make_server(agent)
    c = await srv.connect()
    try:
        await c.send_json({"type": "start", "conversation_id": None})
        await c.send_json({"type": "text", "text": "show me every student"})
    except Exception:
        pass
    assert await close_code(c) == 4401
    assert agent.calls == []
    assert await pool().fetchval("SELECT count(*) FROM conversations") == 0
    await _no_upstream_used(srv)


@pytest.mark.parametrize(
    "cookie",
    ["garbage", " ", "a" * 5000, "../../etc/passwd", "'; DROP TABLE sessions;--", "null", "éè"],
)
async def test_ac7_garbage_cookie_closed_4401(make_server, cookie):
    srv = await make_server(Agent())
    try:
        c = await srv.connect(cookie=cookie)
    except Exception:
        return  # a header the client cannot even encode never reaches the server
    assert await close_code(c) == 4401


async def test_ac7_stored_hash_is_not_a_credential(make_server):
    """Only a hash of the token is stored, so presenting the stored id must not log in."""
    srv = await make_server(Agent())
    _, ctx = await make_session("student", "hash1")
    c = await srv.connect(cookie=ctx.session_id)
    assert await close_code(c) == 4401


async def test_ac7_empty_cookie_value_closed_4401(make_server):
    srv = await make_server(Agent())
    c = await srv.connect(cookie="")
    assert await close_code(c) == 4401


async def test_ac7_expired_session_closed_4401(make_server):
    srv = await make_server(Agent())
    token, ctx = await make_session("student", "exp1")
    async with await srv.connect(token) as ok:
        assert (await ok.start())["role"] == "student"
    await expire_session(ctx)
    c = await srv.connect(token)
    assert await close_code(c) == 4401


async def test_ac7_expired_college_token_closed_4401(make_server):
    srv = await make_server(Agent())
    token, ctx = await make_session("teacher", "exp2")
    await expire_kit_token(ctx)
    c = await srv.connect(token)
    assert await close_code(c) == 4401


async def test_ac7_college_token_inside_safety_margin_is_expired(make_server):
    """A college token with seconds left is as good as expired: treated as no session."""
    srv = await make_server(Agent())
    token, _ = await make_session("student", "exp3", kit_expires_in=5)
    c = await srv.connect(token)
    assert await close_code(c) == 4401


async def test_ac7_deleted_session_cannot_reconnect(make_server):
    srv = await make_server(Agent())
    token, ctx = await make_session("student", "gone1")
    await pool().execute("DELETE FROM sessions WHERE id=$1", ctx.session_id)
    c = await srv.connect(token)
    assert await close_code(c) == 4401


async def test_ac7_session_of_a_deleted_user_does_not_resume_into_another(make_server):
    """Token of A, after A's session was removed, never becomes B's."""
    srv = await make_server(Agent())
    ta, ca = await make_session("student", "a1")
    tb, _ = await make_session("student", "b1")
    await pool().execute("DELETE FROM sessions WHERE id=$1", ca.session_id)
    c = await srv.connect(ta)
    assert await close_code(c) == 4401
    async with await srv.connect(tb) as ok:
        assert (await ok.start())["role"] == "student"


# ------------------------------------------------------------------------- AC7: origin


@pytest.mark.parametrize(
    "origin",
    [
        None,
        "",
        "null",
        "https://evil.example",
        "http://evil.example:8000",
        "http://localhost.evil.com",
        "http://localhost.evil.com:8000",
        "http://evil.com/localhost",
        "http://localhost@evil.com",
        "http://127.0.0.1.evil.com",
        "http://localhost:8000.evil.com",
        "https://kit.bixops.in",
        "file://",
        "chrome-extension://abcdef",
        "ftp://localhost",
        "localhost",
    ],
)
async def test_ac7_bad_or_missing_origin_refused_even_with_valid_session(make_server, origin):
    srv = await make_server(Agent())
    token, _ = await make_session("student", "org1")
    await expect_refused(srv, token=token, origin=origin)
    assert await pool().fetchval("SELECT count(*) FROM conversations") == 0


async def test_ac7_bad_origin_does_not_create_conversation_or_run_turn(make_server):
    agent = Agent()
    srv = await make_server(agent)
    token, _ = await make_session("student", "org2")
    await expect_refused(srv, token=token, origin="https://evil.example")
    assert agent.calls == []


async def test_ac7_bad_origin_without_session_gets_nothing(make_server):
    srv = await make_server(Agent())
    await expect_refused(srv, origin="https://evil.example")


@pytest.mark.parametrize(
    "origin", ["http://localhost:8000", "http://localhost:8000/", "http://localhost:3000", "http://127.0.0.1:8000"]
)
async def test_ac7_allowed_local_origins_connect_when_not_secure(make_server, origin):
    srv = await make_server(Agent())
    token, _ = await make_session("student", "org3")
    async with await srv.connect(token, origin=origin) as c:
        assert (await c.start())["type"] == "ready"


@pytest.fixture
def secure_settings(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "cookie_secure", True)
    monkeypatch.setattr(s, "public_origin", "https://kit.example.org")


async def test_ac7_production_origin_allowed(make_server, secure_settings):
    srv = await make_server(Agent())
    token, _ = await make_session("student", "org4")
    async with await srv.connect(token, origin="https://kit.example.org") as c:
        assert (await c.start())["role"] == "student"


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://kit.example.org",
        "https://kit.example.org.evil.com",
        "https://evilkit.example.org",
        "https://kit.example.org:8443",
        "https://evil.example",
        None,
    ],
)
async def test_ac7_production_rejects_everything_but_public_origin(make_server, secure_settings, origin):
    srv = await make_server(Agent())
    token, _ = await make_session("student", "org5")
    await expect_refused(srv, token=token, origin=origin)


# ------------------------------------------------------------------------- VOX1: start


@pytest.mark.parametrize(
    "first",
    [
        {"type": "text", "text": "hello"},
        {"type": "interrupt"},
        {"type": "mystery"},
        {"text": "no type"},
    ],
)
async def test_vox1_message_before_start_runs_nothing(make_server, first):
    agent = Agent()
    srv = await make_server(agent)
    token, _ = await make_session("student", "pre1")
    async with await srv.connect(token) as c:
        await c.send_json(first)
        got = await c.drain(0.4)
        assert "assistant_delta" not in types(got) and "audio_start" not in types(got)
        assert agent.calls == []
        assert (await c.start())["type"] == "ready"


async def test_vox1_audio_before_start_is_not_forwarded(make_server):
    srv = await make_server(Agent())
    token, _ = await make_session("student", "pre2")
    async with await srv.connect(token) as c:
        await c.send_bytes(b"\x00\x01" * 800)
        await c.send_bytes(b"U\x00" + b"\x00" * 10)
        got = await c.drain(0.3)
        assert "transcript" not in types(got) and "assistant_delta" not in types(got)
        assert all(not s.frames for s in srv.stts)


async def test_vox1_second_start_is_handled_and_keeps_ownership(make_server):
    agent = Agent()
    srv = await make_server(agent)
    token, _ = await make_session("student", "two1")
    async with await srv.connect(token) as c:
        first = await c.start()
        await c.send_json({"type": "start", "conversation_id": None})
        got = await c.until("ready")
        assert types(got)[-1] == "ready"
        second = got[-1]
        assert UUID_RE.match(second["conversation_id"])
        await c.send_json({"type": "text", "text": "after restart"})
        await c.until("assistant_done")
    # the turn after the second start goes to the conversation announced by the second ready
    assert agent.calls[-1][1] == second["conversation_id"]
    assert first["conversation_id"]


async def test_vox1_second_start_resuming_first_conversation_returns_it(make_server):
    srv = await make_server(Agent())
    token, _ = await make_session("student", "two2")
    async with await srv.connect(token) as c:
        first = await c.start()
        second = await c.start(first["conversation_id"])
    assert second["conversation_id"] == first["conversation_id"]


async def test_vox1_second_start_during_turn_stops_old_turn(make_server):
    agent = Agent(["Slow words. "] * 30, delay=0.05)
    srv = await make_server(agent)
    token, _ = await make_session("student", "two3")
    async with await srv.connect(token) as c:
        first = await c.start()
        await c.send_json({"type": "text", "text": "long answer please"})
        await c.until("assistant_delta")
        await c.send_json({"type": "start", "conversation_id": None})
        got = await c.until("ready")
        new_cid = got[-1]["conversation_id"]
        assert new_cid != first["conversation_id"]
        tail = await c.drain(0.5)
        # nothing of the old turn is delivered after the new ready
        assert "assistant_delta" not in types(tail)
        assert agent.stopped_early == ["long answer please"]


async def test_vox1_start_with_extra_fields_is_fine(make_server):
    srv = await make_server(Agent())
    token, _ = await make_session("student", "extra1")
    async with await srv.connect(token) as c:
        await c.send_json({"type": "start", "conversation_id": None, "role": "teacher", "user_id": "x"})
        ready = await c.recv()
    assert ready["role"] == "student"


async def test_vox1_start_without_conversation_id_key_creates_conversation(make_server):
    srv = await make_server(Agent())
    token, _ = await make_session("student", "nokey1")
    async with await srv.connect(token) as c:
        await c.send_json({"type": "start"})
        ready = await c.recv()
    assert ready["type"] == "ready" and UUID_RE.match(ready["conversation_id"])


# ------------------------------------------------------------------------- ready fields


@pytest.mark.parametrize(
    "role,name",
    [("visitor", None), ("student", "Arjun Kumar"), ("teacher", "Lakshmi Devi")],
)
async def test_vox1_ready_fields_per_contract(make_server, role, name):
    srv = await make_server(Agent())
    token, ctx = await make_session(role, name=name) if role != "visitor" else await make_session("visitor")
    async with await srv.connect(token) as c:
        ready = await c.start()
    assert set(ready) == {"type", "conversation_id", "role", "user_name"}
    assert ready["role"] == role
    assert ready["user_name"] == name
    assert UUID_RE.match(ready["conversation_id"])
    row = await pool().fetchrow("SELECT owner_key, role FROM conversations WHERE id=$1", uuid.UUID(ready["conversation_id"]))
    assert row["owner_key"] == ctx.owner_key and row["role"] == role


@pytest.mark.parametrize("role", ["visitor", "student", "teacher"])
async def test_ac5_ready_contains_no_secret(make_server, role):
    srv = await make_server(Agent())
    token, ctx = await make_session(role)
    async with await srv.connect(token) as c:
        ready = await c.start()
    blob = str(ready)
    for secret in (SECRET_KIT_TOKEN, token, ctx.session_id, DG_KEY):
        assert secret not in blob


async def test_vox1_role_comes_from_session_not_client(make_server):
    srv = await make_server(Agent())
    token, _ = await make_session("student", "role1")
    async with await srv.connect(token) as c:
        await c.send_json({"type": "start", "conversation_id": None, "role": "teacher"})
        assert (await c.recv())["role"] == "student"


async def test_ac7_ready_names_the_session_user_only(make_server):
    srv = await make_server(Agent())
    ta, _ = await make_session("student", "nm1", "Alice A")
    tb, _ = await make_session("student", "nm2", "Bob B")
    async with await srv.connect(ta) as a, await srv.connect(tb) as b:
        ra, rb = await a.start(), await b.start()
    assert ra["user_name"] == "Alice A" and rb["user_name"] == "Bob B"
    assert ra["conversation_id"] != rb["conversation_id"]


# ------------------------------------------------------------------------- resume / ownership


async def test_ac7_resume_own_conversation(make_server):
    srv = await make_server(Agent())
    token, _ = await make_session("student", "own1")
    async with await srv.connect(token) as c:
        first = await c.start()
    async with await srv.connect(token) as c:
        again = await c.start(first["conversation_id"])
    assert again["conversation_id"] == first["conversation_id"]


async def test_ac7_resume_own_conversation_from_a_new_session_of_same_user(make_server):
    srv = await make_server(Agent())
    t1, _ = await make_session("student", "same1")
    t2, _ = await make_session("student", "same1")
    async with await srv.connect(t1) as c:
        first = await c.start()
    async with await srv.connect(t2) as c:
        again = await c.start(first["conversation_id"])
    assert again["conversation_id"] == first["conversation_id"]


async def test_ac7_null_conversation_creates_new_each_time(make_server):
    srv = await make_server(Agent())
    token, _ = await make_session("student", "new1")
    ids = set()
    for _ in range(3):
        async with await srv.connect(token) as c:
            ids.add((await c.start(None))["conversation_id"])
    assert len(ids) == 3


async def _foreign_start(srv, cid, token):
    """Start with a conversation id; returns (first frame, client-open flag handled)."""
    async with await srv.connect(token) as c:
        await c.send_json({"type": "start", "conversation_id": cid})
        return await c.recv()


def _refused_or_fresh(frame, victim_cid):
    if frame["type"] == "error":
        return
    assert frame["type"] == "ready"
    assert frame["conversation_id"] != victim_cid


@pytest.mark.parametrize("attacker_role", ["visitor", "student", "teacher"])
async def test_ac7_other_users_conversation_cannot_be_resumed(make_server, attacker_role):
    srv = await make_server(Agent())
    victim_tok, victim = await make_session("student", "victim1")
    atk_tok, _ = await make_session(attacker_role, "atk1")
    async with await srv.connect(victim_tok) as v:
        vcid = (await v.start())["conversation_id"]
    frame = await _foreign_start(srv, vcid, atk_tok)
    _refused_or_fresh(frame, vcid)
    owner = await pool().fetchval("SELECT owner_key FROM conversations WHERE id=$1", uuid.UUID(vcid))
    assert owner == victim.owner_key


@pytest.mark.parametrize(
    "variant",
    [
        lambda c: c.upper(),
        lambda c: c.replace("-", ""),
        lambda c: "{" + c + "}",
        lambda c: "urn:uuid:" + c,
    ],
    ids=["upper", "nodash", "braces", "urn"],
)
async def test_ac3_spelling_variants_of_someone_elses_id_do_not_resume(make_server, variant):
    srv = await make_server(Agent())
    vt, _ = await make_session("student", "victim2")
    at, _ = await make_session("student", "atk2")
    async with await srv.connect(vt) as v:
        vcid = (await v.start())["conversation_id"]
    frame = await _foreign_start(srv, variant(vcid), at)
    if frame["type"] == "ready":
        assert uuid.UUID(frame["conversation_id"]) != uuid.UUID(vcid)


async def test_ac3_visitor_cannot_resume_a_students_conversation(make_server):
    srv = await make_server(echo_history_agent())
    st, _ = await make_session("student", "stu1")
    vt, _ = await make_session("visitor")
    async with await srv.connect(st) as s:
        cid = (await s.start())["conversation_id"]
        await s.send_json({"type": "text", "text": "My attendance is VICTIMSECRET42"})
        await s.until("audio_end")
    before = await history_rows(cid)
    async with await srv.connect(vt) as v:
        await v.send_json({"type": "start", "conversation_id": cid})
        first = await v.recv()
        frames = [first]
        if first["type"] == "ready":
            await v.send_json({"type": "text", "text": "what did we talk about"})
            frames += await v.until("assistant_done")
            frames += await v.drain(0.3)
    assert "VICTIMSECRET42" not in str(frames)
    assert await history_rows(cid) == before  # nothing was added to the victim's conversation


async def test_ac3_resume_attempt_never_sends_or_stores_anything_for_owner(make_server):
    """Even after a failed resume the attacker's turns land in their own conversation."""
    agent = Agent()
    srv = await make_server(agent)
    vt, victim = await make_session("student", "victim3")
    at, attacker = await make_session("student", "atk3")
    async with await srv.connect(vt) as v:
        vcid = (await v.start())["conversation_id"]
        await v.send_json({"type": "text", "text": "private question"})
        await v.until("audio_end")
    async with await srv.connect(at) as a:
        await a.send_json({"type": "start", "conversation_id": vcid})
        first = await a.recv()
        if first["type"] == "ready":
            await a.send_json({"type": "text", "text": "attacker text"})
            await a.until("assistant_done")
    for ctx, cid, text in agent.calls:
        if text == "attacker text":
            assert cid != vcid
            assert ctx.owner_key == attacker.owner_key
    assert [t for r, t in await history_rows(vcid) if r == "user"] == ["private question"]


@pytest.mark.parametrize(
    "bad",
    [
        "not-a-uuid",
        "",
        " ",
        "0" * 36,
        str(uuid.uuid4()),  # valid but does not exist
        "00000000-0000-0000-0000-000000000000",
        "x" * 10_000,
        "' OR 1=1 --",
        "1; DROP TABLE conversations",
        123,
        1.5,
        True,
        ["a"],
        {"id": "x"},
        {},
    ],
)
async def test_ac7_malformed_or_unknown_conversation_ids_are_safe(make_server, bad):
    srv = await make_server(Agent())
    token, ctx = await make_session("student", "bad1")
    async with await srv.connect(token) as c:
        await c.send_json({"type": "start", "conversation_id": bad})
        first = await c.recv()
        if first["type"] == "ready":
            cid = first["conversation_id"]
            assert UUID_RE.match(cid)
            owner = await pool().fetchval("SELECT owner_key FROM conversations WHERE id=$1", uuid.UUID(cid))
            assert owner == ctx.owner_key
        else:
            assert first["type"] == "error"
        # the socket is still usable
        ready = await c.start(None)
        assert ready["type"] == "ready"
    assert await pool().fetchval("SELECT count(*) FROM conversations WHERE owner_key <> $1", ctx.owner_key) == 0


async def test_ac7_nonexistent_id_is_not_adopted_as_a_new_conversation(make_server):
    """A client cannot choose the id of a conversation it creates."""
    srv = await make_server(Agent())
    token, _ = await make_session("student", "adopt1")
    wanted = str(uuid.uuid4())
    async with await srv.connect(token) as c:
        first = await c.start(wanted)
    assert first["conversation_id"] != wanted
    assert await pool().fetchval("SELECT count(*) FROM conversations WHERE id=$1", uuid.UUID(wanted)) == 0


async def test_ac3_two_users_cannot_swap_conversations_via_second_start(make_server):
    srv = await make_server(echo_history_agent())
    at, _ = await make_session("student", "swapA")
    bt, _ = await make_session("student", "swapB")
    async with await srv.connect(at) as a:
        acid = (await a.start())["conversation_id"]
        await a.send_json({"type": "text", "text": "ALPHA-ONLY-MARKER"})
        await a.until("audio_end")
    async with await srv.connect(bt) as b:
        await b.start()
        await b.send_json({"type": "start", "conversation_id": acid})
        got = await b.until("ready")
        assert got[-1]["conversation_id"] != acid
        await b.send_json({"type": "text", "text": "hi"})
        frames = got + await b.until("audio_end") + await b.drain(0.3)
    assert "ALPHA-ONLY-MARKER" not in str(frames)
