"""AC3 / HIS2: conversation history is visible only to its owner.

Five actors in separate browsers: visitors V1 and V2, students A (arjun) and B (priya), teacher T
(lakshmi). Each owns a conversation holding a unique secret; every ordered pair is checked.
"""

import itertools
import uuid

import pytest
from app import auth, history
from app.db import pool

from tests.fakes.college import login_as

ACTORS = {"V1": None, "V2": None, "A": "arjun", "B": "priya", "T": "lakshmi"}
PAIRS = [(x, y) for x, y in itertools.permutations(ACTORS, 2)]


class Actor:
    def __init__(self, name, client, ctx, cid, secret):
        self.name, self.client, self.ctx, self.cid, self.secret = name, client, ctx, cid, secret


async def make_actor(browser, name: str) -> Actor:
    client = await browser()
    login = ACTORS[name]
    if login:
        await login_as(client, login)
    else:
        await client.get("/")
    ctx = await auth._lookup(client.cookies["kit_session"])
    cid = await history.create_conversation(ctx)
    secret = f"secret-of-{name}-{uuid.uuid4().hex[:8]}"
    await history.add_message(cid, "user", f"{secret} question")
    await history.add_message(cid, "assistant", f"{secret} answer")
    await history.log_tool_call(cid, "get_attendance", {"q": secret}, {"tool": secret}, False)
    return Actor(name, client, ctx, cid, secret)


@pytest.fixture
async def actors(browser, fake_college):
    return {name: await make_actor(browser, name) for name in ACTORS}


@pytest.mark.parametrize("name", list(ACTORS))
async def test_ac3_owner_sees_own_conversation(actors, name):
    me = actors[name]
    listing = (await me.client.get("/api/history")).json()
    assert [c["id"] for c in listing] == [me.cid]
    assert listing[0]["message_count"] == 2
    detail = await me.client.get(f"/api/history/{me.cid}")
    assert detail.status_code == 200
    assert [m["content"] for m in detail.json()["messages"]] == [
        f"{me.secret} question", f"{me.secret} answer"]


@pytest.mark.parametrize("owner,other", PAIRS)
async def test_ac3_other_owner_cannot_list(actors, owner, other):
    r = await actors[other].client.get("/api/history")
    assert actors[owner].cid not in r.text and actors[owner].secret not in r.text


@pytest.mark.parametrize("owner,other", PAIRS)
async def test_ac3_other_owner_conversation_returns_404(actors, owner, other):
    r = await actors[other].client.get(f"/api/history/{actors[owner].cid}")
    assert r.status_code == 404
    assert actors[owner].secret not in r.text
    assert r.json() == {"detail": "Conversation not found"}


@pytest.mark.parametrize("owner,other", PAIRS)
async def test_ac3_conversation_owned_false_across_owners(actors, owner, other):
    assert not await history.conversation_owned(actors[other].ctx, actors[owner].cid)
    assert await history.conversation_owned(actors[owner].ctx, actors[owner].cid)


@pytest.mark.parametrize("variant", [
    str.upper, lambda s: "{" + s + "}", lambda s: "urn:uuid:" + s, lambda s: s.replace("-", ""),
])
@pytest.mark.parametrize("owner,other", [("A", "B"), ("A", "V1"), ("T", "A"), ("V1", "V2")])
async def test_ac3_alternate_uuid_spellings_do_not_cross_owners(actors, owner, other, variant):
    spelled = variant(actors[owner].cid)
    r = await actors[other].client.get(f"/api/history/{spelled}")
    assert r.status_code == 404 and actors[owner].secret not in r.text
    assert not await history.conversation_owned(actors[other].ctx, spelled)


async def test_ac3_not_found_and_foreign_look_identical(actors):
    foreign = await actors["B"].client.get(f"/api/history/{actors['A'].cid}")
    missing = await actors["B"].client.get(f"/api/history/{uuid.uuid4()}")
    assert foreign.status_code == missing.status_code == 404
    assert foreign.json() == missing.json()


@pytest.mark.parametrize("bad", [
    "garbage", "1", "0", "-1", "null", "undefined", "%27%20OR%201%3D1--",
    "00000000-0000-0000-0000-000000000000", "ffffffff-ffff-ffff-ffff-ffffffffffff",
    "a" * 300, "%00", "..%2f..%2fetc%2fpasswd", "%E2%80%AE",
])
async def test_ac3_malformed_or_guessed_ids_are_404(actors, bad):
    r = await actors["A"].client.get(f"/api/history/{bad}")
    assert r.status_code == 404
    for a in actors.values():
        assert a.secret not in r.text


async def test_ac3_random_guessing_finds_nothing(actors):
    for _ in range(50):
        r = await actors["V1"].client.get(f"/api/history/{uuid.uuid4()}")
        assert r.status_code == 404


async def test_ac3_no_cookie_and_bad_cookie_see_nothing(actors, browser):
    anon = await browser()
    assert (await anon.get("/api/history")).json() == []
    for a in actors.values():
        assert (await anon.get(f"/api/history/{a.cid}")).status_code == 404
        r = await anon.get(f"/api/history/{a.cid}", headers={"cookie": "kit_session=forged"})
        assert r.status_code == 404


async def test_ac3_tool_calls_not_exposed_through_history(actors):
    me = actors["A"]
    listing = await me.client.get("/api/history")
    detail = await me.client.get(f"/api/history/{me.cid}")
    for r in (listing, detail):
        assert "get_attendance" not in r.text and '"tool"' not in r.text
    assert set(listing.json()[0]) == {"id", "title", "started_at", "updated_at", "message_count"}
    assert set(detail.json()) == {"id", "title", "messages"}
    # CIT6: each message carries its citations; nothing else.
    assert all(set(m) <= {"role", "content", "created_at", "citations"}
               for m in detail.json()["messages"])


async def test_ac3_history_contains_no_owner_key_or_session_data(actors):
    me = actors["A"]
    for r in (await me.client.get("/api/history"), await me.client.get(f"/api/history/{me.cid}")):
        assert "user:arjun" not in r.text and me.ctx.session_id not in r.text
        assert me.ctx.kit_token not in r.text


async def test_ac6_after_logout_old_conversations_unreachable_from_browser(actors):
    me = actors["A"]
    await me.client.post("/api/logout")
    assert (await me.client.get("/api/history")).json() == []
    assert (await me.client.get(f"/api/history/{me.cid}")).status_code == 404


async def test_ac3_conversations_belong_to_the_account_across_devices(actors, browser):
    # Owner is the account: the same user on another device sees their own history.
    second = await browser()
    await login_as(second, "arjun")
    assert [c["id"] for c in (await second.get("/api/history")).json()] == [actors["A"].cid]


async def test_ac3_history_returns_after_logging_back_in(actors):
    me = actors["A"]
    await me.client.post("/api/logout")
    await login_as(me.client, "arjun")
    assert (await me.client.get(f"/api/history/{me.cid}")).status_code == 200


async def test_his2_visitor_history_tied_to_browser_session(browser, fake_college):
    v = await make_actor(browser, "V1")
    assert [c["id"] for c in (await v.client.get("/api/history")).json()] == [v.cid]
    other = await browser()
    assert (await other.get("/api/history")).json() == []


async def test_his2_visitor_history_not_merged_into_account_on_login(browser, fake_college):
    """Stated behaviour: login rotates the session, so a visitor's pre-login conversations stay
    with that (now ended) visitor session; they are not shown under the account."""
    v = await make_actor(browser, "V1")
    await login_as(v.client, "arjun")
    r = await v.client.get("/api/history")
    assert r.json() == [] and v.secret not in r.text
    assert (await v.client.get(f"/api/history/{v.cid}")).status_code == 404
    priya = await browser()
    await login_as(priya, "priya")
    assert (await priya.get(f"/api/history/{v.cid}")).status_code == 404


async def test_ac6_login_as_b_after_a_hides_a_history(actors):
    a = actors["A"]
    await login_as(a.client, "priya")
    assert a.cid not in (await a.client.get("/api/history")).text
    assert actors["B"].cid in (await a.client.get("/api/history")).text  # priya's own


async def test_ac3_recent_messages_only_for_that_conversation(actors):
    for a in actors.values():
        msgs = await history.recent_messages(a.cid)
        assert all(a.secret in m["content"] for m in msgs)
        for b in actors.values():
            if b is not a:
                assert all(b.secret not in m["content"] for m in msgs)


async def test_ac3_owner_key_is_exact_not_prefix(fake_college):
    cid = await history.create_conversation(auth.SessionCtx("s", "user:arjun", "student", None,
                                                            None))
    for owner in ("user:arj", "user:arjun ", "USER:ARJUN", "user:arjun%", "user:%", "%", "",
                  "visitor:arjun", "user:arjun' OR '1'='1"):
        ctx = auth.SessionCtx("s", owner, "student", None, None)
        assert not await history.conversation_owned(ctx, cid)


async def test_ac3_listing_scoped_even_with_many_owners(browser, fake_college):
    people = [await make_actor(browser, n) for n in ("A", "B", "V1")]
    for p in people:
        extra = await history.create_conversation(p.ctx)
        await history.add_message(extra, "user", p.secret + " second")
    for p in people:
        listing = (await p.client.get("/api/history")).json()
        assert len(listing) == 2
        owners = await pool().fetch(
            "SELECT DISTINCT owner_key FROM conversations WHERE id = ANY($1::uuid[])",
            [c["id"] for c in listing])
        assert [o["owner_key"] for o in owners] == [p.ctx.owner_key]
