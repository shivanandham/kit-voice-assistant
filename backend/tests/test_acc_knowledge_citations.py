"""Citations in the knowledge module: CIT1 (chunk metadata, refs, tags) and CIT3 (resolve_refs scope)."""

import re

import pytest
from app import db
from app.knowledge import rag, store
from app.knowledge.mode import knowledge_block

from tests.acc_knowledge_helpers import (
    BRO, CIR_ALL, CIR_UG, CSE, CSE2, ECE, HB25, HB26, LN_CSE, LN_ECE, LN_MECH, MECH, NODEPT, PG_MECH,
    TEACHER_CSE, TEACHER_ECE, TEACHER_NODEPT, UNKNOWN, VISITOR, VISITOR_WITH_USER, base_items,
    base_texts, env, expected_ids, seed, seed_and_index,  # noqa: F401
)

AUDIENCES = [
    pytest.param(VISITOR, id="visitor"), pytest.param(VISITOR_WITH_USER, id="visitor-with-user"),
    pytest.param(CSE, id="cse"), pytest.param(ECE, id="ece"), pytest.param(MECH, id="mech"),
    pytest.param(PG_MECH, id="pg-mech"), pytest.param(NODEPT, id="student-no-dept"),
    pytest.param(TEACHER_CSE, id="teacher-cse"), pytest.param(TEACHER_ECE, id="teacher-ece"),
    pytest.param(TEACHER_NODEPT, id="teacher-no-dept"),
]
TAG = re.compile(r"^\[ref ([^\]]+)\]$", re.M)
SHAPE = {"ref", "document_title", "document_type", "version", "effective_date", "section", "heading",
         "page_start", "page_end"}

PAGED = (
    "\f1\nChapter 1 · General rules\n1.1 Scope of rules\n"
    "1.1.1 The first rule says students must carry the identity card at all times.\n"
    "\f2\n1.1.2 The second rule is about library fines of rupees ten per day.\n"
    "1.2 Fees and payments\n1.2.1 Fees are payable before the first of July every year.\n"
    "\f3\n1.2.2 Late payment attracts a fine of five hundred rupees after the due date.\n"
    "Chapter 2 · Conduct\n2.1 Discipline\n2.1.1 Ragging is strictly prohibited on the campus and hostels."
)


def paged_items():
    items = base_items()
    return items


@pytest.fixture
async def loaded(env):
    texts = base_texts()
    texts[HB26] = PAGED
    await seed_and_index(None, texts)
    return env


def allowed_doc_ids(c):
    d = (c.user or {}).get("department") if c.role != "visitor" else None
    return expected_ids(c.role, d) | ({UNKNOWN} if c.role != "visitor" else set())


def doc_of(ref):
    return int(ref[1:].split(":")[0])


async def all_refs(env):
    refs = {r["ref"] for r in await db.pool().fetch("SELECT ref FROM doc_chunks")}
    for today in ("2026-11-21", "2026-06-30"):
        env.today = today
        refs |= set(TAG.findall((await _ctx_block(TEACHER_CSE)).text))
    env.today = "2026-11-21"
    return refs


async def _ctx_block(c):
    from app.config import get_settings
    s = get_settings()
    old = s.doc_mode
    s.doc_mode = "context"
    try:
        return await knowledge_block(c)
    finally:
        s.doc_mode = old


# ---------------------------------------------------------------- CIT1: chunk metadata

async def test_CIT1_every_chunk_has_ref_heading_and_page_range(loaded):
    rows = await db.pool().fetch("SELECT * FROM doc_chunks")
    assert rows
    for r in rows:
        assert r["ref"] and r["ref"].startswith(f"D{r['document_id']}:")
        assert r["heading"] and r["heading"].strip()
        assert r["page_start"] is not None and r["page_end"] is not None
        assert 1 <= r["page_start"] <= r["page_end"]


async def test_CIT1_refs_unique_within_a_document_and_across_the_corpus(loaded):
    rows = await db.pool().fetch("SELECT document_id, ref FROM doc_chunks")
    pairs = [(r["document_id"], r["ref"]) for r in rows]
    assert len(pairs) == len(set(pairs))
    assert len({r for _, r in pairs}) == len(pairs)


async def test_CIT1_chunks_with_a_numbered_section_carry_the_section_number(loaded):
    rows = {r["heading"]: r for r in await db.pool().fetch("SELECT * FROM doc_chunks WHERE document_id=$1", HB26)}
    scope = next(r for h, r in rows.items() if "1.1 Scope" in h)
    fees = next(r for h, r in rows.items() if "1.2 Fees" in h)
    assert scope["section"] == "1.1" and scope["ref"].startswith(f"D{HB26}:")
    assert fees["section"] == "1.2"
    assert scope["ref"] != fees["ref"]


async def test_CIT1_page_range_follows_the_pages_the_text_came_from(loaded):
    rows = {r["section"]: r for r in await db.pool().fetch("SELECT * FROM doc_chunks WHERE document_id=$1", HB26)}
    assert rows["1.1"]["page_start"] == 1 and rows["1.1"]["page_end"] >= 2     # rules 1.1.1 (p1) and 1.1.2 (p2)
    assert rows["1.2"]["page_start"] <= 2 and rows["1.2"]["page_end"] == 3


async def test_CIT1_rule_refs_resolve_to_their_own_page(loaded):
    got = await rag.resolve_refs([f"D{HB26}:1.1.1", f"D{HB26}:1.2.2", f"D{HB26}:2.1.1"], CSE)
    pages = {g["ref"]: (g["page_start"], g["page_end"]) for g in got}
    assert pages[f"D{HB26}:1.1.1"][0] == 1
    assert pages[f"D{HB26}:1.2.2"] == (3, 3)
    assert pages[f"D{HB26}:2.1.1"] == (3, 3)


async def test_CIT1_chunk_without_a_section_number_uses_a_chunk_ref(env):
    await seed_and_index(None, {**base_texts(), UNKNOWN: "Plain paragraph without any numbering at all in it.\nSecond line of text here."})
    rows = await db.pool().fetch("SELECT * FROM doc_chunks WHERE document_id=$1", UNKNOWN)
    assert rows and all(r["ref"].startswith(f"D{UNKNOWN}:") and r["heading"] for r in rows)
    assert len({r["ref"] for r in rows}) == len(rows)


@pytest.mark.parametrize("n_rules", [3, 40, 200])
async def test_CIT1_long_documents_give_unique_refs(env, n_rules):
    text = "Chapter 1 · Long\n1.1 Everything\n" + "\n".join(
        f"1.1.{i} Rule {i} " + "word " * 40 for i in range(1, n_rules + 1))
    await seed_and_index(None, {**base_texts(), HB26: text})
    refs = [r["ref"] for r in await db.pool().fetch("SELECT ref FROM doc_chunks WHERE document_id=$1", HB26)]
    assert len(refs) == len(set(refs))
    got = await rag.resolve_refs([f"D{HB26}:1.1.{i}" for i in (1, n_rules)], CSE)
    assert [g["ref"] for g in got] == [f"D{HB26}:1.1.1", f"D{HB26}:1.1.{n_rules}"]


async def test_CIT1_refs_are_stable_across_reindexing_unchanged_text(loaded):
    async def snap():
        return [(r["document_id"], r["chunk_index"], r["ref"], r["section"], r["page_start"], r["page_end"])
                for r in await db.pool().fetch("SELECT * FROM doc_chunks ORDER BY document_id, chunk_index")]
    before = await snap()
    for doc in (HB26, BRO, CIR_UG):
        await rag.index_document(doc)
    assert await snap() == before


async def test_CIT1_refs_stay_stable_for_unchanged_sections_when_another_document_changes(loaded):
    before = {r["ref"] for r in await db.pool().fetch("SELECT ref FROM doc_chunks WHERE document_id=$1", HB26)}
    await db.pool().execute("UPDATE documents SET text=text||' extra words here for change.' WHERE id=$1", CIR_UG)
    await rag.index_document(CIR_UG)
    assert {r["ref"] for r in await db.pool().fetch("SELECT ref FROM doc_chunks WHERE document_id=$1", HB26)} == before


async def test_CIT1_search_results_carry_document_identity_and_citation_fields(loaded):
    res = await rag.search("identity card at all times", CSE, k=3)
    top = res[0]
    assert top["document_id"] == HB26 and top["document_type"] == "handbook"
    assert top["ref"].startswith(f"D{HB26}:") and top["section"]
    assert str(top["effective_date"]).startswith("2026-07-01")
    assert top["version"] == "2026.1" and top["page_start"] >= 1


@pytest.mark.parametrize("embeddings", [True, False], ids=["vector", "fulltext"])
async def test_CIT1_search_refs_resolve_for_the_same_caller(loaded, embeddings):
    loaded.emb.fail = not embeddings
    res = await rag.search("fee fine rule identity", TEACHER_CSE, k=10)
    assert res
    resolved = await rag.resolve_refs([r["ref"] for r in res], TEACHER_CSE)
    assert {r["ref"] for r in res} == {x["ref"] for x in resolved}


# ---------------------------------------------------------------- CIT1: tags in the rendered block

@pytest.mark.parametrize("c", AUDIENCES)
@pytest.mark.parametrize("m", ["context", "rag"])
async def test_CIT1_every_ref_tag_in_the_block_resolves_for_that_caller(loaded, c, m):
    loaded.settings.doc_mode = m
    text = (await knowledge_block(c)).text
    tags = TAG.findall(text)
    if m == "context":
        assert tags
    resolved = {r["ref"] for r in await rag.resolve_refs(tags, c)}
    assert set(tags) == resolved


@pytest.mark.parametrize("c", AUDIENCES)
async def test_CIT1_AC1_tags_point_only_at_documents_the_caller_may_use(loaded, c):
    loaded.settings.doc_mode = "context"
    for tag in TAG.findall((await knowledge_block(c)).text):
        assert doc_of(tag) in allowed_doc_ids(c)


async def test_CIT1_every_section_and_rule_line_is_preceded_by_its_tag(loaded):
    full = (await _ctx_block(CSE)).text
    start = full.index(f'<document title="TITLEHBTWENTYSIX"')
    text = full[start:full.index("</document>", start)].splitlines()
    for i, line in enumerate(text):
        if re.match(r"^(1\.1|1\.2|2\.1)\s+[A-Z]|^1\.[12]\.\d\s|^2\.1\.1\s", line) and f"D{HB26}" not in line:
            assert TAG.match(text[i - 1]), f"no ref before: {line}"
            assert text[i - 1].startswith(f"[ref D{HB26}:")
    assert f"[ref D{HB26}:1.1.1]" in "\n".join(text)
    assert f"[ref D{HB26}:2.1.1]" in "\n".join(text)


async def test_CIT1_page_marks_never_leak_into_the_block(loaded):
    for c in (CSE, TEACHER_CSE, VISITOR):
        assert "\f" not in (await _ctx_block(c)).text


async def test_CIT1_circulars_in_rag_mode_also_carry_tags(loaded):
    loaded.settings.doc_mode = "rag"
    tags = TAG.findall((await knowledge_block(CSE)).text)
    assert {doc_of(t) for t in tags} >= {CIR_UG, CIR_ALL}


async def test_CIT1_block_with_tags_is_still_byte_stable_and_user_independent(loaded):
    a = (await _ctx_block(CSE)).text
    assert a == (await _ctx_block(CSE)).text == (await _ctx_block(CSE2)).text


# ---------------------------------------------------------------- CIT3: resolve_refs scope

@pytest.mark.parametrize("c", AUDIENCES)
async def test_CIT3_AC1_only_refs_of_applicable_documents_are_returned(loaded, c):
    refs = sorted(await all_refs(loaded))
    got = await rag.resolve_refs(refs, c)
    assert got
    assert {doc_of(g["ref"]) for g in got} <= allowed_doc_ids(c)
    for doc in allowed_doc_ids(c):
        assert any(doc_of(g["ref"]) == doc for g in got), f"no refs for allowed doc {doc}"


@pytest.mark.parametrize("doc", [HB25, HB26, CIR_UG, CIR_ALL, LN_CSE, LN_ECE, LN_MECH, UNKNOWN])
@pytest.mark.parametrize("c", [VISITOR, VISITOR_WITH_USER])
async def test_CIT3_AC1_visitor_gets_nothing_for_handbook_circular_lesson_or_other_refs(loaded, c, doc):
    refs = [r for r in await all_refs(loaded) if doc_of(r) == doc]
    assert refs
    assert await rag.resolve_refs(refs, c) == []


async def test_CIT3_visitor_may_cite_the_brochure(loaded):
    refs = [r for r in await all_refs(loaded) if doc_of(r) == BRO]
    got = await rag.resolve_refs(refs, VISITOR)
    assert {g["ref"] for g in got} == set(refs) and all(g["document_type"] == "brochure" for g in got)


@pytest.mark.parametrize("c,other", [(CSE, LN_ECE), (CSE, LN_MECH), (ECE, LN_CSE), (MECH, LN_CSE)])
async def test_CIT3_student_cannot_cite_other_departments_notes(loaded, c, other):
    refs = [r for r in await all_refs(loaded) if doc_of(r) == other]
    assert await rag.resolve_refs(refs, c) == []


async def test_CIT3_superseded_edition_refs_are_dropped(loaded):
    every = await all_refs(loaded)
    old = [r for r in every if doc_of(r) == HB25]
    new = [r for r in every if doc_of(r) == HB26]
    assert old and new
    assert await rag.resolve_refs(old, CSE) == []
    loaded.today = "2026-06-30"        # the old edition is now the one in force; the new one is not yet
    assert {g["ref"] for g in await rag.resolve_refs(old, CSE)} == set(old)
    assert await rag.resolve_refs(new, CSE) == []


async def test_CIT3_inactive_document_refs_are_dropped(loaded):
    refs = [r for r in await all_refs(loaded) if doc_of(r) in (CIR_UG, BRO)]
    await db.pool().execute("UPDATE documents SET active=false WHERE id IN ($1,$2)", CIR_UG, BRO)
    assert await rag.resolve_refs(refs, CSE) == []
    assert await rag.resolve_refs(refs, VISITOR) == []


@pytest.mark.parametrize("ref", [
    "", " ", "garbage", "D", "D:", "D2", "D2:", ":1.1", "Dx:1.1", "D-2:1.1", "D2:1.1:1", "D2 1.1", "[ref D2:1.1]",
    "ref D2:1.1", "SOURCES: D2:1.1", None, 5, 2.5, [], {}, "D99999:1.1", f"D{HB26}:99.99.99", f"D{HB26}:c999",
    f"D{HB26}:", f"D{HB26}:1.1.1.1", f"D{HB26}:1.1.1#9", f"D0:1", f"D{HB26}:../etc", "D" + "9" * 40 + ":1",
])
async def test_CIT3_malformed_and_unknown_refs_are_dropped_not_errors(loaded, ref):
    good = f"D{HB26}:1.1.1"
    got = await rag.resolve_refs([ref, good], CSE)
    assert [g["ref"] for g in got] == [good]


async def test_CIT3_empty_and_none_input(loaded):
    assert await rag.resolve_refs([], CSE) == []
    assert await rag.resolve_refs(None, CSE) == []


async def test_CIT3_order_follows_input_and_no_duplicates(loaded):
    a, b, c = f"D{HB26}:1.1.1", f"D{HB26}:2.1.1", f"D{BRO}:" + (await db.pool().fetchval(
        "SELECT ref FROM doc_chunks WHERE document_id=$1 ORDER BY chunk_index LIMIT 1", BRO)).split(":", 1)[1]
    got = [g["ref"] for g in await rag.resolve_refs([b, a, b, c, a, a, c], CSE)]
    assert got == [b, a, c]
    got = [g["ref"] for g in await rag.resolve_refs([c, a, b], CSE)]
    assert got == [c, a, b]


async def test_CIT3_whitespace_around_a_valid_ref_does_not_create_a_second_entry(loaded):
    a = f"D{HB26}:1.1.1"
    got = await rag.resolve_refs([a, f" {a} ", f"{a}\n"], CSE)
    assert len(got) == 1


async def test_CIT3_result_shape_and_values(loaded):
    got = await rag.resolve_refs([f"D{HB26}:1.2.2"], CSE)
    assert len(got) == 1
    r = got[0]
    assert set(r) == SHAPE
    assert r["document_title"] == base_items()[1]["title"] and r["document_type"] == "handbook"
    assert r["version"] == "2026.1" and str(r["effective_date"]) == "2026-07-01"
    assert r["section"] == "1.2.2" and r["heading"] and r["page_start"] == 3 and r["page_end"] == 3


async def test_CIT3_resolution_uses_the_callers_college_date_at_call_time(loaded):
    ref = f"D{HB26}:1.1.1"
    assert len(await rag.resolve_refs([ref], CSE)) == 1
    loaded.today = "2026-06-30"
    assert await rag.resolve_refs([ref], CSE) == []


HOSTILE = [
    f"D{HB26}:1.1.1'; DROP TABLE documents; --", "D2:1.1' OR '1'='1", "D2:1.1\"; --", "D2:\x00", "D2:1.1\x001",
    "💥", "D" + "2" * 10000, f"D{HB26}:" + "1." * 5000, "\\", "%", "$1", "D2:1.1\nD3:2", f"D{HB26}:1.1.1\n[ref D3:x]",
    "{{7*7}}", "<script>alert(1)</script>", "../../etc/passwd", "D2:1.1;SELECT pg_sleep(1)", "D٣:1",
]


@pytest.mark.parametrize("ref", HOSTILE)
@pytest.mark.parametrize("c", [VISITOR, CSE])
async def test_CIT3_hostile_ref_strings_do_not_crash_inject_or_widen_access(loaded, c, ref):
    got = await rag.resolve_refs([ref], c)
    assert isinstance(got, list)
    assert {doc_of(g["ref"]) for g in got} <= allowed_doc_ids(c)
    assert await db.pool().fetchval("SELECT count(*) FROM documents") == 9
    assert await db.pool().fetchval("SELECT count(*) FROM doc_chunks") > 0


async def test_CIT3_huge_input_list_is_handled(loaded):
    refs = [f"D{HB26}:1.1.1"] * 20000 + [f"D{i}:x{i}" for i in range(5000)]
    got = await rag.resolve_refs(refs, CSE)
    assert [g["ref"] for g in got] == [f"D{HB26}:1.1.1"]


async def test_CIT3_nothing_stored_returns_nothing(env):
    assert await rag.resolve_refs(["D1:1.1", "D2:6.4.3"], CSE) == []
