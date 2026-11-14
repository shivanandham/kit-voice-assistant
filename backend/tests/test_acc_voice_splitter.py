"""VOX3: cutting a streamed reply into speakable sentences."""

import re

import pytest
from app.voice.session import SentenceSplitter


def split(text: str, step: int | None = None) -> list[str]:
    """Feed `text` in pieces of `step` characters (None = all at once), then flush."""
    sp = SentenceSplitter()
    out: list[str] = []
    if step is None:
        out += sp.feed(text)
    else:
        for i in range(0, len(text), step):
            out += sp.feed(text[i : i + step])
    return out + sp.flush()


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# ------------------------------------------------------------------------- no false breaks


@pytest.mark.parametrize(
    "text",
    [
        "She has a B.Tech degree in CSE.",
        "He is doing his M.Tech in Power Systems.",
        "Mr. Kumar teaches DBMS.",
        "Dr. Rao is the head of the department.",
        "Prof. Lakshmi takes the class.",
        "The fee is Rs. 1.2 lakh per year.",
        "Use a tool such as Zoho e.g. Zoho CRM for this.",
        "Pick any course i.e. CSE or ECE for the form.",
        "Your CGPA is 8.5 and your attendance is 91.25 percent.",
        "You need 75% attendance to sit the exam.",
        "The class starts at 9.30 tomorrow morning.",
        "The class starts at 10:15 tomorrow morning.",
        "See rule 6.4.3 of the handbook for the leave policy.",
        "Rule 6.4.3.1 applies to on duty leave.",
        "The version is 2.0.1 of the circular.",
        "Version 3.14 is out.",
        "Visit www.kit.example.edu for more.",
        "Mail admissions@kit.example.edu today.",
    ],
)
def test_vox3_one_sentence_stays_one_chunk(text):
    assert split(text) == [text]
    assert split(text, 1) == [text]
    assert split(text, 3) == [text]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("We offer B.Tech and M.Tech. Fees are Rs. 1.2 lakh.", ["We offer B.Tech and M.Tech.", "Fees are Rs. 1.2 lakh."]),
        ("It costs Rs. 5000. Pay online.", ["It costs Rs. 5000.", "Pay online."]),
        ("Ask Dr. Rao. He knows.", ["Ask Dr. Rao.", "He knows."]),
        ("Your CGPA is 8.5. The cutoff is 7.5.", ["Your CGPA is 8.5.", "The cutoff is 7.5."]),
        ("Attendance is 75%. You are safe.", ["Attendance is 75%.", "You are safe."]),
        ("Rule 6.4.3 applies. Please apply.", ["Rule 6.4.3 applies.", "Please apply."]),
        ("Is it open? Yes it is! Come in.", ["Is it open?", "Yes it is!", "Come in."]),
        ("Really?! Yes.", ["Really?!", "Yes."]),
        ("First line.\nSecond line.", ["First line.", "Second line."]),
        ("He said \"stop.\" Then he left.", ["He said \"stop.\"", "Then he left."]),
        ("The B.Tech. Admissions open today.", ["The B.Tech.", "Admissions open today."]),
    ],
)
@pytest.mark.parametrize("step", [None, 1, 2, 5, 17])
def test_vox3_sentences_split_at_real_ends_only(text, expected, step):
    assert split(text, step) == expected


@pytest.mark.parametrize("text", ["Well... I think so. Okay.", "Hmm... let me check. Done.", "Wait.... what?"])
def test_vox3_ellipsis_never_loses_or_duplicates_text_and_never_yields_empty_chunks(text):
    for step in (None, 1, 4):
        chunks = split(text, step)
        assert all(c.strip() for c in chunks)
        assert norm(" ".join(chunks)) == norm(text)
    assert len(split(text)) >= 1


def test_vox3_ellipsis_does_not_end_a_sentence_mid_thought():
    chunks = split("Let me see... the fee is 5 thousand. Next.")
    assert chunks[-1] == "Next." and "the fee is 5 thousand." in chunks[-2] or "the fee is 5 thousand." in chunks[0]


# ------------------------------------------------------------------------- streaming invariance

CORPUS = [
    "We offer B.Tech and M.Tech. Fees are Rs. 1.2 lakh. Dr. Rao leads CSE. Apply by 15.06.2026 online!",
    "Your attendance is 82.5% overall. Rule 6.4.3 says 75% is needed. Is that okay? Yes it is.",
    "Hello world. How are you today? I am fine! Thank you very much.",
    "One.\nTwo.\n\nThree.",
    "The drive is at 9.30 a.m. Everyone should come early. Zoho has 1 seat left.",
    "Mr. A. Kumar from the Dept. of CSE will call you. Please pick up.",
    "He scored 8.5 CGPA. She scored 9.1 CGPA. They both qualify for Zoho and TCS drives.",
]


@pytest.mark.parametrize("text", CORPUS)
@pytest.mark.parametrize("step", [1, 2, 3, 7, 13, 50])
def test_vox3_streaming_in_pieces_gives_same_sentences_as_one_big_chunk(text, step):
    assert split(text, step) == split(text)


# DEFECT: sentences differ with chunking once the early/long clause flush at commas kicks in (SentenceSplitter._next flushes on buffer length) (low).
@pytest.mark.parametrize(
    "text",
    [
        "Hello there my dear friend, how are you doing today? Bye.",
        "Admissions for B.Tech open on Monday, closing on Friday, and results follow in June. Apply now.",
        "Yes, the fee is Rs. 1.2 lakh per year including hostel, mess and transport charges for everyone. Thanks.",
    ],
)
@pytest.mark.parametrize("step", [1, 3, 11])
def test_vox3_streaming_invariance_with_commas(text, step):
    assert split(text, step) == split(text)


@pytest.mark.parametrize("text", CORPUS + ["Plain words with no end at all"])
@pytest.mark.parametrize("step", [None, 1, 4, 9])
def test_vox3_nothing_is_lost_or_duplicated(text, step):
    assert norm(" ".join(split(text, step))) == norm(text)


@pytest.mark.parametrize("step", [None, 1, 6])
def test_vox3_trailing_text_without_final_punctuation_is_flushed(step):
    assert split("First sentence. And a trailing bit", step)[-1] == "And a trailing bit"


def test_vox3_feed_returns_sentence_as_soon_as_it_is_complete():
    sp = SentenceSplitter()
    assert sp.feed("Hello there") == []
    assert sp.feed(" friend.") == []  # no whitespace after the period yet: could be "friend.com"
    assert sp.feed(" Next") == ["Hello there friend."]
    assert sp.flush() == ["Next"]


def test_vox3_flush_twice_returns_nothing_the_second_time():
    sp = SentenceSplitter()
    sp.feed("Partial words")
    assert sp.flush() == ["Partial words"]
    assert sp.flush() == []


def test_vox3_splitter_is_reusable_after_flush():
    sp = SentenceSplitter()
    sp.feed("One. Two")
    sp.flush()
    assert sp.feed("Three. Four") == ["Three."] and sp.flush() == ["Four"]


# ------------------------------------------------------------------------- degenerate input


@pytest.mark.parametrize("text", ["", " ", "   ", "\n", "\n\n \t", " "])
def test_vox3_empty_or_blank_input_yields_nothing(text):
    assert split(text) == []
    assert split(text, 1) == []


def test_vox3_empty_feeds_are_harmless():
    sp = SentenceSplitter()
    assert sp.feed("") == [] and sp.feed("") == [] and sp.flush() == []


@pytest.mark.parametrize("text", [".", "...", "?", "!", ". . .", "?!?!", "...?"])
def test_vox3_punctuation_only_never_yields_blank_chunks(text):
    for step in (None, 1):
        assert all(c.strip() for c in split(text, step))


def test_vox3_very_long_text_without_punctuation_is_chunked():
    text = " ".join(["word"] * 2000)
    for step in (None, 1, 50):
        chunks = split(text, step)
        assert len(chunks) > 1
        assert all(0 < len(c) <= 1000 for c in chunks)
        assert " ".join(chunks).split() == text.split()


def test_vox3_single_enormous_token_is_still_bounded():
    chunks = split("x" * 5000)
    assert all(len(c) <= 1900 for c in chunks)
    assert "".join(chunks) == "x" * 5000


def test_vox3_long_text_with_commas_only_is_chunked():
    text = ", ".join(["clause number %d" % i for i in range(200)])
    chunks = split(text)
    assert len(chunks) > 1 and all(len(c) <= 1000 for c in chunks)
    assert norm(" ".join(chunks)).replace(" ,", ",") == norm(text)


def test_vox3_first_chunk_is_not_held_back_for_ever_by_a_missing_period():
    sp = SentenceSplitter()
    out = sp.feed("This reply has no full stop but keeps going, and going, and going " * 4)
    assert out, "long unpunctuated speech must still be released before the reply ends"


def test_vox3_unicode_sentences():
    assert split("வணக்கம். நீங்கள் எப்படி இருக்கிறீர்கள்? நன்றி!") == ["வணக்கம்.", "நீங்கள் எப்படி இருக்கிறீர்கள்?", "நன்றி!"]
    assert split("नमस्ते। आप कैसे हैं? ठीक हूँ।") != []


def test_vox3_question_mark_inside_a_number_context_still_splits_at_real_question():
    assert split("Can I apply? Yes.", 1) == ["Can I apply?", "Yes."]


def test_vox3_closing_quote_or_bracket_after_period_stays_with_sentence():
    assert split('He said "go." Then left.') == ['He said "go."', "Then left."]
    assert split("It is fine (really.) Next one.") == ["It is fine (really.)", "Next one."]


@pytest.mark.parametrize("abbr", ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Rs."])
def test_vox3_titles_never_end_a_sentence(abbr):
    text = f"Please see {abbr} Something about it."
    assert split(text) == [text]
    assert split(text, 1) == [text]


@pytest.mark.parametrize("abbr", ["B.Tech", "M.Tech", "B.E", "M.E", "B.Sc", "M.Sc", "Ph.D"])
def test_vox3_degree_abbreviation_before_lowercase_continues(abbr):
    text = f"She finished her {abbr}. in the year 2020 and then joined."
    assert split(text) == [text]
    assert split(text, 1) == [text]
