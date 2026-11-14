"""Helpers for the knowledge acceptance suite: fake college, fake embedder, PDF builder, document set.

Nothing here encodes what the application currently does; it only supplies fakes and fixtures.
"""

import copy
import io
import re
from dataclasses import dataclass, field
from datetime import date
from types import SimpleNamespace

import pytest
from app.auth import SessionCtx
from app.config import get_settings
from app.knowledge import mode, rag, store

from tests.test_knowledge_fakes import fake_vector

TODAY = date(2026, 11, 21)

# ---------------------------------------------------------------- sessions


def ctx(role: str, department: str | None = None, programme: str = "UG", login: str = "u1",
        user: bool = True) -> SessionCtx:
    """A session context for a role; visitors carry no user and no token."""
    if role == "visitor":
        return SessionCtx("sv-" + login, "visitor:" + login, "visitor", None, None)
    u = {"id": 1, "login_id": login, "full_name": f"Name {login}", "role": role,
         "department": department, "programme": programme} if user else None
    return SessionCtx("s-" + login, "user:" + login, role, u, "tok-" + login)  # type: ignore[arg-type]


VISITOR = ctx("visitor", login="vis")
VISITOR_WITH_USER = SessionCtx("sv2", "visitor:odd", "visitor",  # a visitor ctx that wrongly carries a user
                               {"login_id": "x", "department": "CSE", "programme": "UG"}, "tok")
CSE = ctx("student", "CSE", login="arjun")
CSE2 = ctx("student", "CSE", login="priya")
ECE = ctx("student", "ECE", login="karthik")
MECH = ctx("student", "MECH", login="rahul")
PG_CSE = ctx("student", "CSE", programme="PG", login="pgcse")
PG_MECH = ctx("student", "MECH", programme="PG", login="pgmech")
NODEPT = ctx("student", None, login="nodept")
TEACHER_CSE = ctx("teacher", "CSE", login="lakshmi")
TEACHER_ECE = ctx("teacher", "ECE", login="senthil")
TEACHER_NODEPT = ctx("teacher", None, login="tnone")

# ---------------------------------------------------------------- document set with unique markers

HB25, HB26, BRO, LN_CSE, LN_ECE, LN_MECH, CIR_UG, CIR_ALL, UNKNOWN = range(1, 10)
MARK = {
    HB25: "hbtwentyfive", HB26: "hbtwentysix", BRO: "brochuremark", LN_CSE: "lncsemark",
    LN_ECE: "lnecemark", LN_MECH: "lnmechmark", CIR_UG: "circugmark", CIR_ALL: "circallmark",
    UNKNOWN: "newslettermark",
}
TITLE = {i: f"TITLE{MARK[i].upper()}" for i in MARK}


def _body(doc_id: int, kind: str) -> str:
    m = MARK[doc_id]
    if kind == "handbook":
        return (f"Chapter 1 · General rules {m}\n1.1 Scope of rules {m}\n"
                f"1.1.1 The {m} rule says students must carry the identity card at all times.\n"
                f"1.1.2 Another {m} rule about library fines of rupees ten per day.")
    if kind == "brochure":
        return (f"2. Programmes and fees {m}\n"
                f"The {m} tuition fee is 125000 per year and hostel fee is 60000 per year.")
    if kind == "lesson_notes":
        return (f"Unit 1 · Introduction {m}\n1.1 Basics of the subject {m}\n"
                f"1.1.1 The {m} lesson explains the first topic of the semester in depth.")
    if kind == "circular":
        return (f"Circular {doc_id}\nSubject: Change {m}\n"
                f"The {m} circular says proof must be submitted within three working days.")
    return f"1. Newsletter {m}\nThe {m} newsletter talks about campus events and results."


def _item(doc_id, typ, version, eff, applies, surprise=False, title=None):
    return {"id": doc_id, "title": title or TITLE[doc_id], "type": typ, "version": version,
            "effective_date": eff, "applies_to": applies, "is_surprise": surprise}


def base_items() -> list[dict]:
    """Listing items for the standard set (ids 1..9)."""
    return [
        _item(HB25, "handbook", "2025.1", "2025-07-01", {"programmes": ["UG", "PG"]}),
        _item(HB26, "handbook", "2026.1", "2026-07-01", {"programmes": ["UG", "PG"]}),
        _item(BRO, "brochure", "2026.1", "2026-11-01", {"programmes": ["UG"]}),
        _item(LN_CSE, "lesson_notes", "1", "2026-10-22", {"departments": ["CSE"]}),
        _item(LN_ECE, "lesson_notes", "1", "2026-10-22", {"departments": ["ECE"]}),
        _item(LN_MECH, "lesson_notes", "1", "2026-10-22", {"departments": ["MECH"]}),
        _item(CIR_UG, "circular", "12/2026", "2026-11-01", {"programmes": ["UG"]}),
        _item(CIR_ALL, "circular", "13/2026", "2026-11-05", {}),
        _item(UNKNOWN, "newsletter", "1", "2026-11-05", {}),
    ]


_KIND = {HB25: "handbook", HB26: "handbook", BRO: "brochure", LN_CSE: "lesson_notes",
         LN_ECE: "lesson_notes", LN_MECH: "lesson_notes", CIR_UG: "circular", CIR_ALL: "circular",
         UNKNOWN: "other"}


def base_texts() -> dict[int, str]:
    """Document text per id, each carrying its unique marker."""
    return {i: _body(i, k) for i, k in _KIND.items()}


def type_of(doc_id: int) -> str:
    """The listing type of a standard document."""
    return next(i["type"] for i in base_items() if i["id"] == doc_id)


# ---------------------------------------------------------------- PDF builder


def make_pdf(lines: list[str]) -> bytes:
    """A real single-page PDF whose text layer has one line per string (in memory)."""
    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    stream = "BT /F1 10 Tf 12 TL 40 800 Td\n" + "\n".join(f"({esc(x)}) '" for x in lines) + "\nET"
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream.encode('latin-1'))} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for n, o in enumerate(objs, 1):
        offsets.append(out.tell())
        out.write(f"{n} 0 obj\n{o}\nendobj\n".encode("latin-1"))
    xref = out.tell()
    out.write(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return out.getvalue()


def blank_pdf() -> bytes:
    """A valid PDF with one page and no text."""
    from pypdf import PdfWriter

    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


# ---------------------------------------------------------------- fake college


@dataclass
class FakeCollege:
    """Controllable stand-in for the college listing and file endpoints."""

    items: list[dict] = field(default_factory=list)
    files: dict[int, bytes] = field(default_factory=dict)
    list_fail: bool = False
    download_fail: set[int] = field(default_factory=set)
    downloads: list[int] = field(default_factory=list)
    list_calls: int = 0

    def load_base(self) -> None:
        self.items = base_items()
        self.files = {i: t.encode() for i, t in base_texts().items()}

    def set_item(self, doc_id: int, **changes) -> None:
        for it in self.items:
            if it["id"] == doc_id:
                it.update(changes)
                return
        raise KeyError(doc_id)

    def remove(self, doc_id: int) -> None:
        self.items = [i for i in self.items if i["id"] != doc_id]

    async def list(self) -> list[dict]:
        self.list_calls += 1
        if self.list_fail:
            raise ConnectionError("college API down")
        return copy.deepcopy(self.items)

    async def download(self, doc_id: int) -> bytes:
        self.downloads.append(doc_id)
        if doc_id in self.download_fail:
            raise ConnectionError("download failed")
        return self.files[doc_id]


class FakeEmbedder:
    """Deterministic embeddings; set .fail to simulate a Gemini outage."""

    def __init__(self) -> None:
        self.fail = False
        self.calls = 0

    async def __call__(self, texts, task_type):
        self.calls += 1
        if self.fail:
            raise RuntimeError("gemini down")
        return [fake_vector(t) for t in texts]


def build_env(monkeypatch, pdf: bool = False) -> SimpleNamespace:
    """Install every fake. With pdf=False the extraction step decodes the bytes as UTF-8."""
    college = FakeCollege()
    college.load_base()
    emb = FakeEmbedder()
    settings = get_settings()
    env = SimpleNamespace(college=college, emb=emb, today=TODAY, today_fail=False, settings=settings,
                          server_today=None)

    async def college_today():
        if env.today_fail:
            raise ConnectionError("college date down")
        return {"date": env.today.isoformat() if isinstance(env.today, date) else env.today}

    monkeypatch.setattr("app.kit.client.college_today", college_today)
    monkeypatch.setattr(store, "_last_college_date", None)
    monkeypatch.setattr(store, "_list_remote", college.list)
    monkeypatch.setattr(store, "_download", college.download)
    if not pdf:
        monkeypatch.setattr(store, "extract_text", lambda b: b.decode())
    monkeypatch.setattr(store, "_sync_count", 1)
    monkeypatch.setattr(rag, "_embed", emb)
    monkeypatch.setattr(rag, "_embed_down_until", 0.0)
    monkeypatch.setattr(settings, "doc_mode", "auto")
    monkeypatch.setattr(settings, "doc_context_token_budget", 60000)
    mode._cache.clear()
    return env


def set_server_clock(monkeypatch, day: date) -> None:
    """Make the server's notion of today (date.today in the store module) a fixed, different day."""
    class FakeDate(date):
        @classmethod
        def today(cls):
            return cls(day.year, day.month, day.day)

    monkeypatch.setattr(store, "date", FakeDate)


@pytest.fixture
def env(monkeypatch):
    """Fakes installed, nothing stored yet."""
    return build_env(monkeypatch)


@pytest.fixture
def pdf_env(monkeypatch):
    """Fakes installed, real pypdf extraction."""
    return build_env(monkeypatch, pdf=True)


async def seed(items: list[dict] | None = None, texts: dict[int, str] | None = None) -> None:
    """Store documents directly, as a finished sync would, then chunk them (embeddings per fake)."""
    items = items if items is not None else base_items()
    texts = texts if texts is not None else base_texts()
    for it in items:
        await store._upsert(it["id"], store._meta(it), texts[it["id"]], f"hash-{it['id']}-{len(texts[it['id']])}")


async def seed_and_index(items=None, texts=None) -> None:
    """seed() followed by indexing every stored document."""
    await seed(items, texts)
    for it in items if items is not None else base_items():
        await rag.index_document(it["id"])


def ids(docs: list[dict]) -> list[int]:
    """Document ids in order."""
    return [d["id"] for d in docs]


def expected_ids(role: str, department: str | None, handbook: int | None = HB26) -> set[int]:
    """What the criteria say a user should get from the standard set (unknown type excluded)."""
    if role == "visitor":
        return {BRO}
    out = {BRO, CIR_UG, CIR_ALL}
    if handbook:
        out.add(handbook)
    if role == "teacher":
        out |= {LN_CSE, LN_ECE, LN_MECH}
    else:
        out |= {{"CSE": LN_CSE, "ECE": LN_ECE, "MECH": LN_MECH}[department]} if department in ("CSE", "ECE", "MECH") else set()
    return out


_REF_LINE = re.compile(r"^\[ref [^\]]*\]$")


def strip_refs(text: str) -> str:
    """The block text with the citation tag lines (CIT1) removed."""
    return "\n".join(x for x in text.splitlines() if not _REF_LINE.match(x.strip()))


def contains_in_order(haystack: str, body: str) -> bool:
    """True when every non-empty line of body appears in haystack, in order (tag lines ignored)."""
    pos = 0
    hay = strip_refs(haystack)
    for line in body.splitlines():
        if not line.strip():
            continue
        found = hay.find(line, pos)
        if found < 0:
            return False
        pos = found + len(line)
    return True
