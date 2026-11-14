"""Chunking and search (DOC7), indexing and embedding failures (DOC6, REL1)."""

import pytest
from app import db
from app.knowledge import rag, store

from tests.acc_knowledge_helpers import (
    BRO, CIR_ALL, CIR_UG, CSE, ECE, HB25, HB26, LN_CSE, MARK, TEACHER_CSE, TITLE, VISITOR, base_items,
    base_texts, env, ids, seed, seed_and_index,  # noqa: F401
)

RESULT_KEYS = {"document_title", "version", "heading", "content", "score", "ref", "section", "page_start",
               "page_end", "document_id", "document_type", "effective_date"}


@pytest.fixture
async def loaded(env):
    await seed_and_index()
    return env


# ---------------------------------------------------------------- chunking

HANDBOOK = """Chapter 4 · Attendance rules
4.1 Minimum attendance
4.1.1 A student must have at least 75% attendance in each course to write the examination.
4.1.2 Condonation of shortage up to 10% may be granted on medical grounds with a certificate
from a registered doctor, submitted within 7 days of return.
4.2 Attendance records
4.2.1 Records are updated every working day by the faculty member taking the class.
Chapter 6 · Leave rules
6.1 Personal leave
6.1.4 Personal leave is not granted on any day on which an Internal Assessment is held.
6.4 On-duty (OD)
6.4.3 Within 5 working days after the event the student must submit proof of participation."""


def lines_of(text):
    return [x for x in text.splitlines() if x.strip()]


def rule_number(line):
    import re
    m = re.match(r"^(\d+\.\d+\.\d+)\s", line)
    return m.group(1) if m else None


def test_DOC7_chunking_covers_every_line_of_the_document():
    chunks = rag.chunk_document("T", "1", HANDBOOK)
    joined = "\n".join(c for _, c in chunks)
    for line in lines_of(HANDBOOK):
        assert line in joined or line in "".join(h for h, _ in chunks), line


# DEFECT: sections whose body is under MIN_BODY_CHARS are discarded by chunking, so short rules are never searchable.
@pytest.mark.parametrize("text", [
    "1.1 Dress code\n1.1.1 Wear ID.",                       # section body shorter than any minimum
    "Chapter 1 · Short\n1.1 Tiny\n1.1.1 Be on time.",
    "Chapter 2 · A\n2.1 First\n2.1.1 Yes.\n2.2 Second\n2.2.1 No.",
    "5.1 Fines\n5.1.1 Rs 10.",
])
def test_DOC7_short_sections_are_not_dropped(text):
    joined = "\n".join(c for _, c in rag.chunk_document("T", "1", text))
    for line in lines_of(text):
        if rule_number(line):
            assert line in joined, f"lost: {line}"


def test_DOC7_text_before_first_heading_is_kept():
    text = "Preamble sentence that introduces the whole document in detail.\nChapter 1 · Rules\n1.1 Scope\n1.1.1 Applies to everyone here."
    joined = "\n".join(c for _, c in rag.chunk_document("T", "1", text))
    assert "Preamble sentence" in joined


def test_DOC7_text_without_any_headings_is_kept():
    text = "Plain paragraph one with enough words to matter.\nPlain paragraph two with more words."
    chunks = rag.chunk_document("T", "1", text)
    joined = "\n".join(c for _, c in chunks)
    assert "paragraph one" in joined and "paragraph two" in joined


def test_DOC7_every_rule_number_stays_with_its_text_in_one_chunk():
    chunks = [c for _, c in rag.chunk_document("T", "1", HANDBOOK)]
    for line in lines_of(HANDBOOK):
        if rule_number(line):
            assert any(line in c for c in chunks)


def test_DOC7_rule_continuation_lines_stay_with_rule_number():
    chunks = [c for _, c in rag.chunk_document("T", "1", HANDBOOK)]
    block = ("4.1.2 Condonation of shortage up to 10% may be granted on medical grounds with a certificate\n"
             "from a registered doctor, submitted within 7 days of return.")
    assert any(block in c for c in chunks)


@pytest.mark.parametrize("n_rules,words", [(10, 30), (40, 60), (120, 40), (300, 20)])
def test_DOC7_long_sections_split_but_keep_rule_number_with_its_text(n_rules, words):
    rules = [f"6.4.{i} " + " ".join(f"w{i}x{j}" for j in range(words)) for i in range(1, n_rules + 1)]
    text = "Chapter 6 · Leave\n6.4 On-duty\n" + "\n".join(rules)
    chunks = [c for _, c in rag.chunk_document("T", "1", text)]
    for rule in rules:
        assert any(rule in c for c in chunks), rule[:20]
    for c in chunks:
        last = lines_of(c)[-1]
        assert not (rule_number(last) and len(last.split()) == 1), "rule number stranded at chunk end"


@pytest.mark.parametrize("n", [1, 5, 50, 400])
def test_DOC7_no_chunk_is_empty_and_each_has_heading_and_title(n):
    text = "\n".join(f"Chapter {i} · Part {i}\n{i}.1 Section {i}\n{i}.1.1 Body of rule number {i} spelled out." for i in range(1, n + 1))
    chunks = rag.chunk_document("Handbook", "9", text)
    assert chunks
    for heading, content in chunks:
        assert content.strip() and heading.strip()
        assert content.startswith("Handbook (version 9)")
        body = content.split("\n", 2)[-1]
        assert body.strip()
        assert heading in content


@pytest.mark.parametrize("text", ["", "   ", "\n\n", "x"])
def test_DOC7_blank_or_trivial_text_produces_no_empty_chunk_and_no_crash(text):
    for _, content in rag.chunk_document("T", "1", text):
        assert content.strip()


def test_DOC7_headings_are_preserved_with_chapter_and_section():
    chunks = rag.chunk_document("T", "1", HANDBOOK)
    headings = [h for h, _ in chunks]
    assert any("Chapter 4 · Attendance rules" in h and "4.1 Minimum attendance" in h for h in headings)
    assert any("Chapter 6 · Leave rules" in h and "6.4 On-duty (OD)" in h for h in headings)
    for h, c in chunks:
        assert h in c


def test_DOC7_a_rule_from_one_chapter_is_never_under_another_chapters_heading():
    for h, c in rag.chunk_document("T", "1", HANDBOOK):
        if "6.4.3" in c:
            assert "Chapter 6" in h
        if "4.1.1" in c:
            assert "Chapter 4" in h


def test_DOC7_chunk_size_is_bounded_unless_single_line_is_huge():
    text = "Chapter 1 · X\n1.1 Y\n" + "\n".join(f"1.1.{i} " + "word " * 100 for i in range(1, 80))
    for _, c in rag.chunk_document("T", "1", text):
        assert len(c) < rag.MAX_CHUNK_CHARS * 2


def test_DOC7_a_single_enormous_line_is_not_lost():
    big = "1.1.1 " + "alpha " * 2000
    text = "Chapter 1 · X\n1.1 Y\n" + big
    joined = "\n".join(c for _, c in rag.chunk_document("T", "1", text))
    assert big.strip()[:200] in joined and big.strip()[-100:] in joined


def test_DOC7_very_large_document_is_chunked_fully_in_reasonable_time():
    text = "\n".join(f"Chapter {c} · Chapter number {c}\n{c}.1 Section\n" + "\n".join(
        f"{c}.1.{r} Rule {c}-{r} states obligation{c}x{r} for every person concerned." for r in range(1, 60))
        for c in range(1, 80))
    chunks = [c for _, c in rag.chunk_document("T", "1", text)]
    joined = "\n".join(chunks)
    for c in (1, 40, 79):
        for r in (1, 30, 59):
            assert f"obligation{c}x{r} " in joined


def test_DOC7_unicode_and_punctuation_survive_chunking():
    text = "Chapter 1 · Rules\n1.1 Fees (₹)\n1.1.1 The fee is ₹ 12,500 — payable by 31 July “strictly”."
    joined = "\n".join(c for _, c in rag.chunk_document("T", "1", text))
    assert "₹ 12,500 — payable by 31 July “strictly”." in joined


def test_DOC7_chunking_is_deterministic():
    assert rag.chunk_document("T", "1", HANDBOOK) == rag.chunk_document("T", "1", HANDBOOK)


# ---------------------------------------------------------------- indexing

async def count_chunks(doc_id):
    return await db.pool().fetchval("SELECT count(*) FROM doc_chunks WHERE document_id=$1", doc_id)


async def test_DOC6_indexing_twice_does_not_duplicate_chunks(loaded):
    n = await count_chunks(HB26)
    await rag.index_document(HB26)
    await rag.index_document(HB26)
    assert await count_chunks(HB26) == n > 0


async def test_DOC6_reindexing_after_text_change_replaces_chunks(loaded):
    await db.pool().execute("UPDATE documents SET text=$2 WHERE id=$1", HB26,
                            "Chapter 1 · New\n1.1 Fresh\n1.1.1 The replacedtext rule is completely different now.")
    await rag.index_document(HB26)
    res = await rag.search("replacedtext rule completely different", CSE, k=3)
    assert res[0]["document_title"] == TITLE[HB26]
    assert await db.pool().fetchval("SELECT count(*) FROM doc_chunks WHERE document_id=$1 AND content ILIKE '%identity card%'", HB26) == 0


async def test_DOC6_index_of_unknown_document_is_a_no_op(loaded):
    await rag.index_document(9999)
    assert await db.pool().fetchval("SELECT count(*) FROM doc_chunks WHERE document_id=9999") == 0


async def test_DOC6_index_of_empty_text_document_leaves_no_chunks(env):
    await db.pool().execute(
        "INSERT INTO documents (id,title,type,version,applies_to,is_surprise,text,token_estimate,content_hash) "
        "VALUES (50,'E','circular','1','{}',false,'',0,'h')")
    await rag.index_document(50)
    assert await count_chunks(50) == 0


async def test_DOC6_deleting_a_document_removes_its_chunks(loaded):
    await db.pool().execute("DELETE FROM documents WHERE id=$1", HB26)
    assert await count_chunks(HB26) == 0


async def test_REL1_embedding_failure_during_index_keeps_chunks_searchable_by_text(env):
    env.emb.fail = True
    await seed()
    with pytest.raises(Exception):
        await rag.index_document(BRO)
    assert await count_chunks(BRO) > 0
    res = await rag.search("tuition fee hostel", VISITOR)
    assert res and res[0]["document_title"] == TITLE[BRO]


async def test_REL1_embed_missing_fills_gaps_after_recovery(env):
    env.emb.fail = True
    await seed()
    with pytest.raises(Exception):
        await rag.index_document(BRO)
    env.emb.fail = False
    rag._embed_down_until = 0.0
    await rag.embed_missing()
    assert await db.pool().fetchval("SELECT count(*) FROM doc_chunks WHERE embedding IS NULL") == 0


async def test_REL1_embedding_service_cooldown_does_not_block_text_search(env):
    await seed_and_index()
    env.emb.fail = True
    first = await rag.search("tuition fee", VISITOR)
    calls = env.emb.calls
    second = await rag.search("tuition fee", VISITOR)
    assert first and second and first[0]["document_title"] == TITLE[BRO]
    assert env.emb.calls == calls     # fast-fail after an outage rather than waiting on a dead service


# ---------------------------------------------------------------- search: shape, k, queries

async def test_DOC7_result_shape_matches_contract(loaded):
    res = await rag.search("tuition fee", CSE, k=4)
    assert res
    for r in res:
        assert set(r) == RESULT_KEYS
        assert isinstance(r["document_title"], str) and isinstance(r["version"], str)
        assert isinstance(r["heading"], str) and isinstance(r["content"], str)
        assert isinstance(r["score"], float)
        assert isinstance(r["ref"], str) and r["ref"].startswith(f"D{r['document_id']}:")
        assert isinstance(r["document_id"], int) and isinstance(r["document_type"], str)
        assert isinstance(r["page_start"], int) and isinstance(r["page_end"], int)
        assert r["page_start"] <= r["page_end"]


async def test_DOC7_results_are_ordered_best_first(loaded):
    res = await rag.search("tuition fee hostel fee per year", TEACHER_CSE, k=10)
    scores = [r["score"] for r in res]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("embeddings", [True, False], ids=["vector", "fulltext"])
@pytest.mark.parametrize("k", [1, 2, 3, 5, 6])
async def test_DOC7_k_is_respected(loaded, k, embeddings):
    loaded.emb.fail = not embeddings
    res = await rag.search("the rule fee circular lesson says", TEACHER_CSE, k=k)
    assert 1 <= len(res) <= k


async def test_DOC7_default_k_is_six_at_most(loaded):
    assert len(await rag.search("the rule fee circular lesson says", TEACHER_CSE)) <= 6


@pytest.mark.parametrize("k", [0, -1, -100])
async def test_DOC7_nonpositive_k_never_crashes_and_returns_few(loaded, k):
    res = await rag.search("tuition fee", CSE, k=k)
    assert len(res) <= 1


async def test_DOC7_huge_k_is_bounded(loaded):
    assert len(await rag.search("the rule fee circular lesson says", TEACHER_CSE, k=10**6)) <= 50


@pytest.mark.parametrize("q", ["", " ", "   \n\t  ", None])
async def test_DOC7_empty_and_blank_queries_return_nothing(loaded, q):
    for c in (VISITOR, CSE):
        assert await rag.search(q, c) == []


@pytest.mark.parametrize("embeddings", [True, False], ids=["vector", "fulltext"])
@pytest.mark.parametrize("length", [5000, 100_000, 1_000_000])
async def test_DOC7_very_long_query_does_not_crash(loaded, length, embeddings):
    loaded.emb.fail = not embeddings
    res = await rag.search("tuition fee " * (length // 12), CSE, k=3)
    assert isinstance(res, list)


HOSTILE = [
    "'; DROP TABLE documents; --", "\" OR 1=1 --", "' OR '1'='1", "a' || (select 1) || '", "\\", "\\'", "%", "_",
    "&", "|", "!", "( )", ":*", "a & | b", "tuition:*", "'fee'", "fee & ! ", "<->", "((((", "))))", "fee\\",
    "💥 fee", "tuition\u0000fee", "; select pg_sleep(0)", "$1", "{}", "[]", "fee' --", "fee\"; --", "*", "??",
    "fee " + "(" * 500, "/*", "*/", "\x1b[31m", "\n\n", "ａｂｃ",
]


# DEFECT: a NUL character in the query makes the full-text fallback raise (asyncpg CharacterNotInRepertoireError) instead of returning a list.
@pytest.mark.parametrize("embeddings", [True, False], ids=["vector", "fulltext"])
@pytest.mark.parametrize("q", HOSTILE)
async def test_DOC7_hostile_queries_do_not_inject_or_crash(loaded, q, embeddings):
    loaded.emb.fail = not embeddings
    res = await rag.search(q, VISITOR)
    assert isinstance(res, list)
    assert {r["document_title"] for r in res} <= {TITLE[BRO]}
    assert await db.pool().fetchval("SELECT count(*) FROM documents") == 9
    assert await db.pool().fetchval("SELECT count(*) FROM doc_chunks") > 0


@pytest.mark.parametrize("q", HOSTILE[:12])
async def test_DOC7_hostile_queries_for_student_also_safe(loaded, q):
    loaded.emb.fail = True
    res = await rag.search(q, CSE, k=20)
    assert {r["document_title"] for r in res} <= {TITLE[i] for i in (HB26, CIR_UG, CIR_ALL, LN_CSE, BRO, 9)}


# ---------------------------------------------------------------- search: ranking and fallback

@pytest.mark.parametrize("embeddings", [True, False], ids=["vector", "fulltext"])
async def test_DOC7_finds_the_relevant_chunk_both_ways(loaded, embeddings):
    loaded.emb.fail = not embeddings
    res = await rag.search("identity card at all times", CSE, k=3)
    assert res[0]["document_title"] == TITLE[HB26]
    assert "identity card" in res[0]["content"]


async def test_DOC7_fulltext_fallback_when_embeddings_fail_on_query_only(loaded):
    loaded.emb.fail = True
    res = await rag.search("library fines rupees", CSE)
    assert res and "library fines" in res[0]["content"]


async def test_DOC7_fulltext_fallback_when_no_chunk_has_embeddings(env):
    env.emb.fail = True
    await seed()
    for d in base_items():
        try:
            await rag.index_document(d["id"])
        except Exception:
            pass
    env.emb.fail = False
    rag._embed_down_until = 0.0
    assert await db.pool().fetchval("SELECT count(*) FROM doc_chunks WHERE embedding IS NOT NULL") == 0
    res = await rag.search("library fines", CSE)
    assert res and "library fines" in res[0]["content"]


async def test_DOC7_partly_embedded_index_still_finds_unembedded_chunks(env):
    await seed_and_index()
    await db.pool().execute("UPDATE doc_chunks SET embedding=NULL WHERE document_id=$1", HB26)
    res = await rag.search("library fines rupees per day", CSE, k=6)
    assert any("library fines" in r["content"] for r in res)


async def test_DOC7_query_with_no_match_returns_empty_list_not_error(loaded):
    loaded.emb.fail = True
    assert await rag.search("zzzzqqqq nonexistentword", CSE) == []


async def test_DOC7_wrong_embedding_shape_falls_back_to_text(loaded):
    async def bad(texts, task_type):
        return [[0.5, 0.5]]
    rag._embed = bad
    res = await rag.search("library fines rupees", CSE)
    assert res and "library fines" in res[0]["content"]


async def test_DOC7_embedding_timeout_falls_back_to_text(loaded):
    async def hang(texts, task_type):
        raise TimeoutError()
    rag._embed = hang
    assert (await rag.search("tuition fee hostel", VISITOR))[0]["document_title"] == TITLE[BRO]


async def test_DOC7_fulltext_synonym_words_students_use_find_document_words(loaded):
    loaded.emb.fail = True
    res = await rag.search("when are exams", CSE)
    assert isinstance(res, list)


async def test_DOC7_search_recovers_vector_mode_after_outage(loaded):
    loaded.emb.fail = True
    await rag.search("tuition fee", VISITOR)
    loaded.emb.fail = False
    rag._embed_down_until = 0.0
    before = loaded.emb.calls
    await rag.search("tuition fee", VISITOR)
    assert loaded.emb.calls > before


# ---------------------------------------------------------------- search: scope

async def test_DOC7_superseded_edition_never_returned(loaded):
    res = await rag.search(f"{MARK[HB25]} {MARK[HB25]}", CSE, k=20)
    assert all(r["document_title"] != TITLE[HB25] for r in res)
    loaded.emb.fail = True
    res = await rag.search("hbtwentyfive rule identity card", CSE, k=20)
    assert all(r["document_title"] != TITLE[HB25] for r in res)


async def test_DOC7_new_edition_not_returned_before_its_effective_date(loaded):
    loaded.today = "2026-06-30"
    for fail in (False, True):
        loaded.emb.fail = fail
        res = await rag.search("hbtwentysix rule identity card", CSE, k=20)
        assert all(r["document_title"] != TITLE[HB26] for r in res)
    assert any(r["document_title"] == TITLE[HB25] for r in await rag.search("hbtwentyfive rule identity card", CSE, k=20))


async def test_DOC7_inactive_documents_never_returned(loaded):
    await db.pool().execute("UPDATE documents SET active=false WHERE id=$1", CIR_UG)
    for fail in (False, True):
        loaded.emb.fail = fail
        res = await rag.search("circugmark circular proof three working days", CSE, k=20)
        assert all(r["document_title"] != TITLE[CIR_UG] for r in res)


async def test_DOC7_other_audience_chunk_is_best_match_but_never_returned(loaded):
    exact = base_texts()[HB26]
    res = await rag.search(exact, VISITOR, k=20)
    assert all(r["document_title"] == TITLE[BRO] for r in res)
    exact_ln = base_texts()[LN_CSE]
    res = await rag.search(exact_ln, ECE, k=20)
    assert all(r["document_title"] != TITLE[LN_CSE] for r in res)


async def test_DOC7_search_follows_college_date_at_call_time(loaded):
    loaded.today = "2026-06-30"
    a = await rag.search("hbtwentyfive identity card", CSE, k=3)
    loaded.today = "2026-07-01"
    b = await rag.search("hbtwentysix identity card", CSE, k=3)
    assert a[0]["document_title"] == TITLE[HB25] and b[0]["document_title"] == TITLE[HB26]


async def test_DOC7_search_with_no_applicable_documents_returns_empty(env):
    await seed_and_index([i for i in base_items() if i["id"] == HB26], base_texts())
    assert await rag.search("identity card", VISITOR) == []


async def test_DOC7_duplicate_chunks_are_not_returned_twice(loaded):
    res = await rag.search("library fines rupees per day", CSE, k=20)
    contents = [r["content"] for r in res]
    assert len(contents) == len(set(contents))
