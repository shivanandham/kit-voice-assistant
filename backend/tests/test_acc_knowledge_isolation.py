"""Audience isolation (DOC1, DOC4, AC1): who gets which document through every door."""

import pytest
from app.auth import SessionCtx
from app.knowledge import rag, store
from app.knowledge.mode import knowledge_block

from tests.acc_knowledge_helpers import (
    BRO, CIR_ALL, CIR_UG, HB25, HB26, LN_CSE, LN_ECE, LN_MECH, MARK, NODEPT, PG_CSE, PG_MECH, TITLE,
    UNKNOWN, VISITOR, VISITOR_WITH_USER, CSE, CSE2, ECE, MECH, TEACHER_CSE, TEACHER_ECE,
    TEACHER_NODEPT, base_texts, contains_in_order, ctx, env, expected_ids, ids, seed_and_index,  # noqa: F401
)

ALL_IDS = [HB25, HB26, BRO, LN_CSE, LN_ECE, LN_MECH, CIR_UG, CIR_ALL, UNKNOWN]
AUDIENCES = [
    pytest.param(VISITOR, id="visitor"),
    pytest.param(VISITOR_WITH_USER, id="visitor-carrying-user-dict"),
    pytest.param(CSE, id="ug-cse"), pytest.param(ECE, id="ug-ece"), pytest.param(MECH, id="ug-mech"),
    pytest.param(PG_CSE, id="pg-cse"), pytest.param(PG_MECH, id="pg-mech"),
    pytest.param(NODEPT, id="student-no-department"),
    pytest.param(TEACHER_CSE, id="teacher-cse"), pytest.param(TEACHER_ECE, id="teacher-ece"),
    pytest.param(TEACHER_NODEPT, id="teacher-no-department"),
]
MODES = ["context", "rag", "auto"]


def strict(c) -> set[int]:
    """What the criteria say the audience gets (a document of an unlisted type reaches nobody)."""
    return expected_ids(c.role, (c.user or {}).get("department") if c.role != "visitor" else None)


def allowed(c) -> set[int]:
    """Brochure for visitors; logged-in users also get a document of a type no criterion names (DOC4/DOC9 wording)."""
    return strict(c) | ({UNKNOWN} if c.role != "visitor" else set())


@pytest.fixture
async def loaded(env):
    """The standard nine documents stored, chunked and embedded; college date inside 2026."""
    await seed_and_index()
    return env


@pytest.mark.parametrize("c", AUDIENCES)
async def test_DOC1_DOC4_applicable_documents_matrix(loaded, c):
    assert set(ids(await store.applicable_documents(c))) == allowed(c)


@pytest.mark.parametrize("c", AUDIENCES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("budget", [1, 10**9], ids=["tiny-budget", "huge-budget"])
async def test_AC1_DOC1_knowledge_block_never_contains_other_audiences_documents(loaded, c, mode, budget):
    loaded.settings.doc_mode = mode
    loaded.settings.doc_context_token_budget = budget
    block = await knowledge_block(c)
    low = block.text.lower()
    for doc_id in ALL_IDS:
        if doc_id not in allowed(c):
            assert MARK[doc_id] not in low, f"{MARK[doc_id]} leaked to {c.role}"
            assert TITLE[doc_id].lower() not in low


@pytest.mark.parametrize("c", AUDIENCES)
async def test_DOC4_context_block_contains_every_allowed_document_in_full(loaded, c):
    loaded.settings.doc_mode = "context"
    block = await knowledge_block(c)
    texts = base_texts()
    assert block.mode == "context"
    for doc_id in allowed(c):
        assert contains_in_order(block.text, texts[doc_id]), doc_id


@pytest.mark.parametrize("c", AUDIENCES)
async def test_DOC3_DOC1_rag_block_has_circulars_in_full_for_users_and_no_other_bodies(loaded, c):
    loaded.settings.doc_mode = "rag"
    block = await knowledge_block(c)
    texts = base_texts()
    assert block.mode == "rag"
    for doc_id in ALL_IDS:
        in_full = contains_in_order(block.text, texts[doc_id])
        circular_for_user = doc_id in (CIR_UG, CIR_ALL) and c.role != "visitor"
        assert in_full == circular_for_user, f"doc {doc_id} in_full={in_full}"


@pytest.mark.parametrize("c", AUDIENCES)
@pytest.mark.parametrize("doc_id", ALL_IDS)
@pytest.mark.parametrize("embeddings", [True, False], ids=["vector", "fulltext"])
async def test_AC1_DOC7_search_only_returns_applicable_documents(loaded, c, doc_id, embeddings):
    loaded.emb.fail = not embeddings
    res = await rag.search(f"{MARK[doc_id]} rule fee circular lesson", c, k=20)
    got = {r["document_title"] for r in res}
    assert got <= {TITLE[i] for i in allowed(c)}
    for r in res:
        for other in ALL_IDS:
            if other not in allowed(c):
                assert MARK[other] not in r["content"].lower()


@pytest.mark.parametrize("c", AUDIENCES)
@pytest.mark.parametrize("doc_id", ALL_IDS)
async def test_DOC7_best_match_is_the_allowed_document_itself(loaded, c, doc_id):
    if doc_id not in allowed(c):
        pytest.skip("not allowed for this audience; covered by the isolation test")
    res = await rag.search(f"{MARK[doc_id]} {MARK[doc_id]} {MARK[doc_id]}", c, k=3)
    assert res and res[0]["document_title"] == TITLE[doc_id]


@pytest.mark.parametrize("doc_id", [HB25, HB26, LN_CSE, LN_ECE, LN_MECH, CIR_UG, CIR_ALL, UNKNOWN])
async def test_AC1_visitor_search_for_other_audiences_best_match_returns_nothing_of_it(loaded, doc_id):
    res = await rag.search(base_texts()[doc_id], VISITOR, k=20)
    assert all(r["document_title"] == TITLE[BRO] for r in res)


@pytest.mark.parametrize("c", AUDIENCES)
async def test_DOC1_DOC4_new_unknown_document_type_goes_to_logged_in_users_never_visitors(loaded, c):
    got = ids(await store.applicable_documents(c))
    assert (UNKNOWN in got) == (c.role != "visitor")


@pytest.mark.parametrize("c", [CSE, ECE, PG_MECH, NODEPT, TEACHER_CSE, TEACHER_NODEPT])
async def test_DOC4_unknown_type_reaches_logged_in_users_in_context_and_search(loaded, c):
    loaded.settings.doc_mode = "context"
    assert contains_in_order((await knowledge_block(c)).text, base_texts()[UNKNOWN])
    res = await rag.search(f"{MARK[UNKNOWN]} {MARK[UNKNOWN]}", c, k=3)
    assert res and res[0]["document_title"] == TITLE[UNKNOWN]


@pytest.mark.parametrize("c", [VISITOR, VISITOR_WITH_USER])
@pytest.mark.parametrize("m", ["context", "rag"])
async def test_AC1_unknown_type_never_reaches_visitors_in_block_or_search(loaded, c, m):
    loaded.settings.doc_mode = m
    assert MARK[UNKNOWN] not in (await knowledge_block(c)).text.lower()
    assert all(r["document_title"] != TITLE[UNKNOWN] for r in await rag.search(MARK[UNKNOWN], c, k=20))


# DEFECT: any role string other than 'visitor' (admin, '', None, 'Student') is treated as logged in (DOC9).
@pytest.mark.parametrize("role", ["admin", "guest", "", "Student", "TEACHER", None])
async def test_DOC9_AC1_only_student_and_teacher_count_as_logged_in_other_roles_are_visitors(loaded, role):
    odd = SessionCtx("so", "user:odd", role, {"login_id": "odd", "department": "CSE"}, "tok")
    assert set(ids(await store.applicable_documents(odd))) <= {BRO}


@pytest.mark.parametrize("order", [("visitor", "student", "visitor"), ("student", "visitor", "student"),
                                   ("teacher", "visitor", "student", "visitor")])
async def test_AC1_DOC8_interleaved_audiences_never_share_cached_text(loaded, order):
    pick = {"visitor": VISITOR, "student": CSE, "teacher": TEACHER_CSE}
    seen = {}
    for who in order:
        text = (await knowledge_block(pick[who])).text
        if who in seen:
            assert text == seen[who]
        seen[who] = text
    assert "hbtwentysix" not in seen["visitor"].lower()


@pytest.mark.parametrize("dept_value", ["cse", "Cse", " CSE", "CSE ", "C", "", "CSE,ECE"])
async def test_DOC4_department_must_match_exactly_not_loosely(loaded, dept_value):
    c = ctx("student", dept_value, login="loose")
    got = set(ids(await store.applicable_documents(c)))
    assert LN_CSE not in got or dept_value == "CSE"


@pytest.mark.parametrize("applies", [{}, {"departments": []}, {"departments": None}, {"semester": 3}])
async def test_DOC4_lesson_notes_listing_no_department_go_to_every_student_and_teacher_not_visitors(loaded, applies):
    await store._upsert(LN_CSE, store._meta({"id": LN_CSE, "title": TITLE[LN_CSE], "type": "lesson_notes",
                                             "version": "1", "effective_date": "2026-10-22",
                                             "applies_to": applies}), base_texts()[LN_CSE], "h-x")
    for c in (CSE, ECE, MECH, PG_MECH, NODEPT, TEACHER_CSE, TEACHER_ECE):
        assert LN_CSE in ids(await store.applicable_documents(c))
    assert LN_CSE not in ids(await store.applicable_documents(VISITOR))


async def test_DOC4_lesson_notes_for_several_departments_reach_each_listed_department(loaded):
    await store._upsert(LN_CSE, store._meta({"id": LN_CSE, "title": "multi", "type": "lesson_notes",
                                             "version": "1", "effective_date": "2026-10-22",
                                             "applies_to": {"departments": ["CSE", "ECE"]}}),
                        base_texts()[LN_CSE], "h-m")
    assert LN_CSE in ids(await store.applicable_documents(CSE))
    assert LN_CSE in ids(await store.applicable_documents(ECE))
    assert LN_CSE not in ids(await store.applicable_documents(MECH))
    assert LN_CSE not in ids(await store.applicable_documents(VISITOR))


async def test_DOC1_AC1_visitor_gets_surprise_brochure_but_not_surprise_circular(loaded):
    for item in loaded.college.items:
        if item["id"] in (BRO, CIR_ALL):
            item["is_surprise"] = True
            await store._upsert(item["id"], store._meta(item), base_texts()[item["id"]], f"hs-{item['id']}")
    got = set(ids(await store.applicable_documents(VISITOR)))
    assert got == {BRO}


async def test_DOC1_second_brochure_is_given_to_visitors_and_nothing_else(loaded):
    extra = {"id": 20, "title": "Brochure PG", "type": "brochure", "version": "1", "effective_date": None,
             "applies_to": {}, "is_surprise": False}
    await store._upsert(20, store._meta(extra), "3. PG fees\nPG tuition fee is high.", "h20")
    assert set(ids(await store.applicable_documents(VISITOR))) == {BRO, 20}


async def test_AC1_inactive_documents_reach_nobody(loaded):
    from app import db
    await db.pool().execute("UPDATE documents SET active=false")
    for c in (VISITOR, CSE, TEACHER_CSE):
        assert await store.applicable_documents(c) == []
        assert (await rag.search("fee rule circular", c)) == []
