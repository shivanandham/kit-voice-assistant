"""Sentence splitting for streamed TTS."""

from app.voice.session import SentenceSplitter


def split(text: str, step: int = 3) -> list[str]:
    """Feed `text` in small deltas, as the LLM would, and collect every chunk."""
    s = SentenceSplitter()
    out: list[str] = []
    for i in range(0, len(text), step):
        out += s.feed(text[i : i + step])
    return out + s.flush()


def test_basic_sentences():
    assert split("Hello there. How are you? Great!") == ["Hello there.", "How are you?", "Great!"]


def test_decimals_and_abbreviations_are_not_split():
    text = "Your CGPA is 8.75 in B.Tech CSE. Fees are Rs. 1.2 lakh, e.g. for Mr. Kumar. Done."
    assert split(text) == [
        "Your CGPA is 8.75 in B.Tech CSE.",
        "Fees are Rs. 1.2 lakh, e.g. for Mr. Kumar.",
        "Done.",
    ]


def test_initials_and_degrees():
    assert split("Talk to Dr. A. Rao about the M.Tech. It starts in June.") == [
        "Talk to Dr. A. Rao about the M.Tech.",
        "It starts in June.",
    ]


def test_first_long_clause_flushes_early():
    text = "We offer four undergraduate programmes in engineering, including CSE and ECE, and more"
    out = split(text)
    assert out[0] == "We offer four undergraduate programmes in engineering,"
    assert " ".join(out) == text


def test_later_short_clauses_wait_for_sentence_end():
    text = "First sentence here. Then a clause, and another, done."
    assert split(text) == ["First sentence here.", "Then a clause, and another, done."]


def test_no_punctuation_hard_split():
    text = "word " * 100
    out = split(text.strip(), step=7)
    assert all(len(p) <= 300 for p in out) and len(out) >= 2
    assert " ".join(out).split() == text.split()


def test_newlines_split_and_empty_flush():
    s = SentenceSplitter()
    assert s.feed("Line one\nLine two") == ["Line one"]
    assert s.flush() == ["Line two"]
    assert s.flush() == []
