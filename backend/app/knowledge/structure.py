"""Document structure shared by chunking, rendering and citations.

One deterministic pass over a document's stored text finds its sections (chapter and numbered
section headings), its numbered rules and the page each line came from. From that pass come the
search chunks, the "[ref ...]" tags in the rendered prompt text and the reference index used to
resolve citations, so all three always agree.

Refs: a section or chunk is ``D<id>:<number>`` (``D2:6.4``; a circular uses its circular number),
``D<id>:c<chunk_index>`` when it has no number, with ``#2``, ``#3`` ... for further chunks of a
long section; a rule is ``D<id>:<rule number>`` (``D2:6.4.3``).
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache

MAX_CHUNK_CHARS = 1600     # about 400 tokens
OVERLAP_CHARS = 250        # tail of the previous piece repeated when a long section is split
MIN_BODY_CHARS = 20

PAGE_MARK = re.compile(r"^\f(\d+)$")      # a line "\f<page>" opens each PDF page in stored text
_CHAPTER = re.compile(r"^(Chapter \d+|Annexure [A-Z]|Unit \d+)\s*·\s*\S.{0,120}$")
_SECTION = re.compile(r"^\d+\.\d+\s+[A-Z(].{0,90}[^.]$")
_LIST_SECTION = re.compile(r"^\d+\.\s+[A-Z].{0,45}[^.,]$")   # brochure style "2. Programmes and fees"
_NUMBER = re.compile(r"^(\d+(?:\.\d+)*)\.?\s")
RULE = re.compile(r"^(\d+\.\d+\.\d+)\s")

Line = tuple[int, str, int]   # (index in the stored text, text, page)


@dataclass
class Chunk:
    """One search chunk with its citation metadata."""

    chunk_index: int
    ref: str
    section: str | None
    heading: str
    content: str
    page_start: int
    page_end: int


@dataclass
class Structure:
    """Everything derived from one document's text."""

    lines: list[Line]
    chunks: list[Chunk] = field(default_factory=list)
    tags: dict[int, list[str]] = field(default_factory=dict)   # line index -> refs to print before it
    refs: dict[str, dict] = field(default_factory=dict)        # ref -> {section, heading, page_start, page_end}


@dataclass
class _Section:
    path: str
    number: str | None
    own: Line | None
    lines: list[Line]


def strip_page_marks(text: str) -> str:
    """The stored text without its page-marker lines."""
    return "\n".join(line for line in text.split("\n") if not PAGE_MARK.match(line))


def _parse(text: str) -> list[Line]:
    out: list[Line] = []
    page = 1
    for i, line in enumerate(text.split("\n")):
        mark = PAGE_MARK.match(line)
        if mark:
            page = int(mark.group(1))
        else:
            out.append((i, line, page))
    return out


def _sections(lines: list[Line]) -> list[_Section]:
    raw: list[_Section] = []
    chapter, section = "", None
    body: list[Line] = []

    def flush() -> None:
        if any(text.strip() for _, text, _ in body):
            own = section or chapter_line
            path = " › ".join(p for p in (chapter, section[1] if section else "") if p)
            number = None
            if section is not None:
                m = _NUMBER.match(section[1])
                number = m.group(1) if m else None
            raw.append(_Section(path, number, own, list(body)))
        body.clear()

    chapter_line: Line | None = None
    for line in lines:
        text = line[1]
        if _CHAPTER.match(text):
            flush()
            chapter, chapter_line, section = text, line, None
        elif (_SECTION.match(text) or _LIST_SECTION.match(text)) and not RULE.match(text):
            flush()
            section = line
        else:
            body.append(line)
    flush()

    # A section with a very short body joins the next one (or the previous one at the end),
    # heading line included, so short rules stay searchable.
    merged: list[_Section] = []
    carry: list[Line] = []
    for i, sec in enumerate(raw):
        lines_ = carry + sec.lines
        carry = []
        own = [sec.own] if sec.own else []
        if sum(len(t) for _, t, _ in lines_) < MIN_BODY_CHARS and i + 1 < len(raw):
            carry = own + lines_
        elif sum(len(t) for _, t, _ in lines_) < MIN_BODY_CHARS and merged:
            merged[-1].lines.extend(own + lines_)
        else:
            merged.append(_Section(sec.path, sec.number, sec.own, lines_))
    return merged


def _size(lines: list[Line]) -> int:
    return sum(len(t) + 1 for _, t, _ in lines)


def _tail(lines: list[Line]) -> list[Line]:
    tail: list[Line] = []
    used = 0
    for line in reversed(lines):
        if used + len(line[1]) > OVERLAP_CHARS:
            break
        tail.insert(0, line)
        used += len(line[1]) + 1
    return tail


def _split_long(lines: list[Line]) -> list[list[Line]]:
    """Pieces of about MAX_CHUNK_CHARS, broken at rule starts where possible, with a small overlap."""
    if _size(lines) <= MAX_CHUNK_CHARS:
        return [lines]
    units: list[list[Line]] = []
    for line in lines:
        if not units or RULE.match(line[1]):
            units.append([line])
        else:
            units[-1].append(line)
    flat: list[list[Line]] = []
    for unit in units:
        group: list[Line] = []
        for line in unit:
            if group and _size(group) + len(line[1]) + 1 > MAX_CHUNK_CHARS:
                flat.append(group)
                group = []
            group.append(line)
        flat.append(group)
    pieces: list[list[Line]] = []
    current: list[Line] = []
    for unit in flat:
        if current and _size(current) + _size(unit) > MAX_CHUNK_CHARS:
            pieces.append(current)
            current = _tail(current)
        current.extend(unit)
    pieces.append(current)
    return pieces


def _unique(ref: str, used: set[str]) -> str:
    if ref not in used:
        return ref
    n = 2
    while f"{ref}#{n}" in used:
        n += 1
    return f"{ref}#{n}"


@lru_cache(maxsize=64)
def analyse(document_id: int, title: str, version: str, doc_type: str, text: str) -> Structure:
    """Sections, chunks, ref tags and the ref index of one document (deterministic, cached)."""
    lines = _parse(text)
    result = Structure(lines=lines)
    used: set[str] = set()
    prefix_title = f"{title} (version {version})"
    for sec in _sections(lines):
        heading = sec.path or "Introduction"
        number = sec.number or (version if doc_type == "circular" and version else None)
        body = sec.lines
        first_line = sec.own if sec.own else body[0]
        for j, piece in enumerate(_split_long(body)):
            index = len(result.chunks)
            ref = _unique(f"D{document_id}:{number}" if number else f"D{document_id}:c{index}", used)
            used.add(ref)
            span = ([first_line] if j == 0 else []) + piece
            pages = [p for _, _, p in span]
            content = "\n".join([prefix_title, heading, *(t for _, t, _ in piece)])
            chunk = Chunk(index, ref, number, heading, content, min(pages), max(pages))
            result.chunks.append(chunk)
            result.refs[ref] = {"section": number, "heading": heading,
                                "page_start": chunk.page_start, "page_end": chunk.page_end}
            if j == 0:
                result.tags.setdefault(first_line[0], []).append(ref)
        # Rules inside the section get their own refs, tagged where each rule starts.
        rule_starts = [k for k, (_, t, _) in enumerate(body) if RULE.match(t)]
        for n, k in enumerate(rule_starts):
            rule = RULE.match(body[k][1]).group(1)
            ref = f"D{document_id}:{rule}"
            if ref in used:
                continue          # a wrapped line that only looks like a rule number
            used.add(ref)
            end = rule_starts[n + 1] if n + 1 < len(rule_starts) else len(body)
            pages = [p for _, _, p in body[k:end]]
            result.refs[ref] = {"section": rule, "heading": heading,
                                "page_start": min(pages), "page_end": max(pages)}
            result.tags.setdefault(body[k][0], []).append(ref)
    return result


def tagged_text(structure: Structure) -> str:
    """The document text with "[ref ...]" lines before every section and rule, page marks removed."""
    out: list[str] = []
    for index, text, _ in structure.lines:
        out.extend(f"[ref {ref}]" for ref in structure.tags.get(index, []))
        out.append(text)
    return "\n".join(out)
