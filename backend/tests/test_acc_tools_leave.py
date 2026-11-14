"""COR3: leave and on-duty requests.

Hard errors (to_date before from_date, on-duty without an event name, invalid input) are rejected
before anything is stored. Exam days, holidays and past dates in the range are warnings returned
with the summary, before confirmation. College date is 2026-11-21 (Saturday). For arjun (CSE-2A):
exam 2026-11-23 (DBMS, CS304), holidays 2026-11-22 (Sunday Holiday) and 2026-11-27 (Founders Day).
"""

import pytest
from app.kit import client as kc

from tests.acc_tools_helpers import (
    mentions,
    new_conv,
    pending_count,
    propose,
    run,
    user_says,
    writes,
)

BASE = {"type": "medical_leave", "from_date": "2026-11-25", "to_date": "2026-11-25",
        "half_day": False, "reason": "Fever"}
PAST = ("past", "before today", "backdated")
EXAM = ("exam",)
HOLIDAY = ("holiday", "Founders Day")


def leave(**kw):
    return {**BASE, **kw}


async def submit(fake, args, login="arjun"):
    ctx = fake.ctx(login)
    cid = await new_conv(ctx)
    r, b = await run("submit_leave_request", args, ctx, cid)
    return ctx, cid, r, b


def _has(b, words):
    return any(mentions(w, *words) for w in b.get("warnings", []))


def _exam_warning(b):
    return any(mentions(w, "exam") for w in b.get("warnings", []))


def _holiday_warning(b):
    return any(mentions(w, "holiday") for w in b.get("warnings", []))


# ------------------------------------------------------------------ hard errors

HARD = {
    "to_before_from": leave(from_date="2026-11-26", to_date="2026-11-25"),
    "to_before_from_far": leave(from_date="2027-01-01", to_date="2026-11-25"),
    "to_before_from_one_day": leave(from_date="2026-11-30", to_date="2026-11-29"),
    "on_duty_no_event": leave(type="on_duty"),
    "on_duty_empty_event": leave(type="on_duty", event_name=""),
    "on_duty_blank_event": leave(type="on_duty", event_name="   \t  "),
    "on_duty_none_event_and_reverse": leave(type="on_duty", from_date="2026-11-26",
                                            to_date="2026-11-25"),
    "empty_reason": leave(reason=""),
    "blank_reason": leave(reason="    "),
    "type_vacation": leave(type="vacation"),
    "type_empty": leave(type=""),
    "type_upper": leave(type="ON_DUTY", event_name="X"),
    "type_label": leave(type="personal leave"),
    "type_int": leave(type=1),
    "from_month_13": leave(from_date="2026-13-01"),
    "from_feb_30": leave(from_date="2026-02-30", to_date="2026-03-01"),
    "to_nov_31": leave(to_date="2026-11-31"),
    "from_slashes": leave(from_date="25/11/2026"),
    "from_word": leave(from_date="tomorrow"),
    "from_empty": leave(from_date=""),
    "from_short": leave(from_date="2026-1-5", to_date="2026-01-05"),
    "from_datetime": leave(from_date="2026-11-25T09:00:00"),
    "from_compact": leave(from_date="20261125"),
    "from_int": leave(from_date=20261125),
    "to_none_string": leave(to_date="None"),
    "half_day_string": leave(half_day="yes"),
    "half_day_int": leave(half_day=1),
    "reason_int": leave(reason=42),
    "missing_type": {k: v for k, v in BASE.items() if k != "type"},
    "missing_from": {k: v for k, v in BASE.items() if k != "from_date"},
    "missing_to": {k: v for k, v in BASE.items() if k != "to_date"},
    "missing_reason": {k: v for k, v in BASE.items() if k != "reason"},
    "missing_half_day": {k: v for k, v in BASE.items() if k != "half_day"},
    "extra_field": leave(priority="urgent"),
}


@pytest.mark.parametrize("case", list(HARD))
async def test_cor3_hard_error_rejected_before_storing(case, fake_college):
    ctx, cid, r, b = await submit(fake_college, HARD[case])
    assert r.is_error, b
    assert isinstance(b, dict)
    assert await pending_count() == 0
    assert writes(fake_college) == []


@pytest.mark.parametrize("case", ["to_before_from", "on_duty_no_event", "on_duty_blank_event"])
async def test_cor3_hard_error_message_is_specific(case, fake_college):
    ctx, cid, r, b = await submit(fake_college, HARD[case])
    text = r.content.lower()
    if case.startswith("to_before"):
        assert "to_date" in text or "before" in text
    else:
        assert "event" in text


async def test_cor3_hard_errors_in_combination(fake_college):
    args = leave(type="on_duty", from_date="2026-11-26", to_date="2026-11-20", reason="   ")
    ctx, cid, r, b = await submit(fake_college, args)
    assert r.is_error
    assert await pending_count() == 0


@pytest.mark.parametrize("case", ["to_before_from", "on_duty_no_event"])
async def test_cor3_hard_error_makes_no_college_write_even_later(case, fake_college):
    ctx, cid, r, b = await submit(fake_college, HARD[case])
    await user_says(cid)
    rows = await pending_count()
    assert rows == 0
    assert writes(fake_college) == []


# ------------------------------------------------------------------ valid inputs


@pytest.mark.parametrize("kind", ["personal_leave", "medical_leave"])
async def test_cor3_leave_types_accepted_without_event(kind, fake_college):
    ctx, cid, r, b = await submit(fake_college, leave(type=kind))
    assert not r.is_error, b
    assert b["status"] == "needs_confirmation"


async def test_cor3_on_duty_with_event_accepted(fake_college):
    ctx, cid, r, b = await submit(fake_college, leave(type="on_duty", event_name="TechFest"))
    assert not r.is_error, b
    assert "TechFest" in b["summary"]


@pytest.mark.parametrize("kind", ["personal_leave", "medical_leave", "on_duty"])
async def test_cor3_every_type_confirms_to_college(kind, fake_college):
    args = leave(type=kind, event_name="TechFest") if kind == "on_duty" else leave(type=kind)
    ctx, cid, aid = await propose(fake_college, "submit_leave_request", args=args)
    await user_says(cid)
    r, b = await run("confirm_pending_action", {"action_id": aid}, ctx, cid)
    assert not r.is_error, b
    assert writes(fake_college)[0].body["type"] == kind


async def test_cor3_single_day_half_day(fake_college):
    ctx, cid, aid = await propose(fake_college, "submit_leave_request",
                                  args=leave(half_day=True))
    await user_says(cid)
    await run("confirm_pending_action", {"action_id": aid}, ctx, cid)
    assert writes(fake_college)[0].body["half_day"] is True


async def test_cor3_half_day_summary_says_half_day(fake_college):
    ctx, cid, r, b = await submit(fake_college, leave(half_day=True))
    assert "half" in b["summary"].lower()


async def test_cor3_half_day_over_several_days_flagged(fake_college):
    ctx, cid, r, b = await submit(fake_college,
                                  leave(half_day=True, from_date="2026-11-25", to_date="2026-11-26"))
    assert r.is_error or any(mentions(w, "half") for w in b["warnings"])


async def test_cor3_multi_day_range_summary_has_both_ends(fake_college):
    ctx, cid, r, b = await submit(fake_college, leave(from_date="2026-11-25", to_date="2026-11-26"))
    assert not r.is_error
    s = b["summary"]
    assert ("25" in s and "26" in s)


async def test_cor3_multi_day_confirm_sends_range(fake_college):
    ctx, cid, aid = await propose(fake_college, "submit_leave_request",
                                  args=leave(from_date="2026-11-25", to_date="2026-11-26"))
    await user_says(cid)
    await run("confirm_pending_action", {"action_id": aid}, ctx, cid)
    sent = writes(fake_college)[0].body
    assert (sent["from_date"], sent["to_date"]) == ("2026-11-25", "2026-11-26")


async def test_cor3_whitespace_in_reason_normalised_not_rejected(fake_college):
    ctx, cid, r, b = await submit(fake_college, leave(reason="  high   fever  "))
    assert not r.is_error


@pytest.mark.parametrize("n", [500, 2000, 10000])
async def test_cor3_very_long_reason_handled(n, fake_college):
    ctx, cid, r, b = await submit(fake_college, leave(reason="fever " * (n // 6)))
    assert isinstance(b, dict)
    assert "Something went wrong" not in r.content
    if not r.is_error:
        assert b["status"] == "needs_confirmation"


async def test_cor3_unicode_reason(fake_college):
    ctx, cid, r, b = await submit(fake_college, leave(reason="காய்ச்சல் – fever"))
    assert not r.is_error
    assert "காய்ச்சல்" in b["summary"]


# ------------------------------------------------------------------ warnings separately


async def test_cor3_clean_range_has_no_exam_holiday_or_past_warning(fake_college):
    ctx, cid, r, b = await submit(fake_college, leave(from_date="2026-11-25", to_date="2026-11-26"))
    assert not r.is_error
    assert not _exam_warning(b) and not _holiday_warning(b) and not _has(b, PAST)


async def test_cor3_exam_day_single_warns(fake_college):
    ctx, cid, r, b = await submit(fake_college, leave(from_date="2026-11-23", to_date="2026-11-23"))
    assert not r.is_error
    assert _exam_warning(b)
    assert any("DBMS" in w or "23" in w for w in b["warnings"])


async def test_cor3_exam_at_start_edge_warns(fake_college):
    ctx, cid, r, b = await submit(fake_college, leave(from_date="2026-11-23", to_date="2026-11-25"))
    assert _exam_warning(b)


async def test_cor3_exam_at_end_edge_warns(fake_college):
    fake_college.calendar_events.append(
        {"id": 60, "date": "2026-12-03", "type": "internal_exam", "title": "OOP Internal",
         "subject_code": "CS303", "follows_weekday": None, "status": "active"})
    ctx, cid, r, b = await submit(fake_college, leave(from_date="2026-12-01", to_date="2026-12-03"))
    assert _exam_warning(b)
    assert any("OOP" in w for w in b["warnings"])


async def test_cor3_exam_just_outside_range_no_warning(fake_college):
    ctx, cid, r, b = await submit(fake_college, leave(from_date="2026-11-24", to_date="2026-11-25"))
    assert not any("DBMS" in w for w in b["warnings"])


async def test_cor3_other_departments_exam_not_warned(fake_college):
    """EC301's exam on 2026-11-24 is not arjun's (CSE)."""
    ctx, cid, r, b = await submit(fake_college, leave(from_date="2026-11-24", to_date="2026-11-24"))
    assert not _exam_warning(b)


async def test_cor3_ece_student_warned_of_own_exam(fake_college):
    ctx, cid, r, b = await submit(fake_college, leave(from_date="2026-11-24", to_date="2026-11-24"),
                                  login="kavya")
    assert _exam_warning(b)


async def test_cor3_holiday_at_end_edge_warns(fake_college):
    ctx, cid, r, b = await submit(fake_college, leave(from_date="2026-11-25", to_date="2026-11-27"))
    assert _holiday_warning(b)
    assert any("Founders Day" in w for w in b["warnings"])


async def test_cor3_holiday_at_start_edge_warns(fake_college):
    ctx, cid, r, b = await submit(fake_college, leave(from_date="2026-11-27", to_date="2026-11-30"))
    assert _holiday_warning(b)


async def test_cor3_holiday_single_day_warns(fake_college):
    ctx, cid, r, b = await submit(fake_college, leave(from_date="2026-11-27", to_date="2026-11-27"))
    assert _holiday_warning(b)
    assert not r.is_error


async def test_cor3_cancelled_holiday_not_warned(fake_college):
    for e in fake_college.calendar_events:
        if e["title"] == "Founders Day":
            e["status"] = "cancelled"
    ctx, cid, r, b = await submit(fake_college, leave(from_date="2026-11-27", to_date="2026-11-27"))
    assert not any("Founders Day" in w for w in b["warnings"])


@pytest.mark.parametrize("d", ["2026-11-20", "2026-11-01", "2025-11-21"])
async def test_cor3_past_date_warns(d, fake_college):
    ctx, cid, r, b = await submit(fake_college, leave(from_date=d, to_date=max("2026-11-20", d)))
    assert not r.is_error, b
    assert _has(b, PAST)


async def test_cor3_today_is_not_past(fake_college):
    ctx, cid, r, b = await submit(fake_college, leave(from_date="2026-11-21", to_date="2026-11-21"))
    assert not _has(b, ("past", "backdated"))


# ------------------------------------------------------------------ warnings in combination


async def test_cor3_past_and_exam_and_holiday_together(fake_college):
    ctx, cid, r, b = await submit(fake_college, leave(from_date="2026-11-20", to_date="2026-11-27"))
    assert not r.is_error
    assert _has(b, PAST) and _exam_warning(b) and _holiday_warning(b)
    assert any("Founders Day" in w for w in b["warnings"])


async def test_cor3_exam_and_holiday_together(fake_college):
    ctx, cid, r, b = await submit(fake_college, leave(from_date="2026-11-22", to_date="2026-11-23"))
    assert _exam_warning(b) and _holiday_warning(b)
    assert not _has(b, PAST)


async def test_cor3_past_and_exam_at_end_edge(fake_college):
    ctx, cid, r, b = await submit(fake_college, leave(from_date="2026-11-19", to_date="2026-11-23"))
    assert _has(b, PAST) and _exam_warning(b)


async def test_cor3_on_duty_with_warnings_still_pending(fake_college):
    ctx, cid, r, b = await submit(fake_college, leave(type="on_duty", event_name="Hack",
                                                      from_date="2026-11-23",
                                                      to_date="2026-11-27"))
    assert b["status"] == "needs_confirmation"
    assert _exam_warning(b) and _holiday_warning(b)


async def test_cor3_warnings_shown_before_confirmation(fake_college):
    """Warnings come with needs_confirmation and nothing is sent until the confirm turn."""
    ctx, cid, r, b = await submit(fake_college, leave(from_date="2026-11-23", to_date="2026-11-23"))
    assert b["warnings"]
    assert writes(fake_college) == []


async def test_cor3_overlap_with_existing_request_warns(fake_college):
    ctx, cid, r, b = await submit(fake_college, leave(from_date="2026-11-30", to_date="2026-11-30"))
    assert any("12" in w or mentions(w, "overlap") for w in b["warnings"])


async def test_cor3_calendar_unavailable_still_safe(fake_college, monkeypatch):
    monkeypatch.setattr(kc, "RETRY_BACKOFF_SECONDS", 0.0)
    fake_college.fail("/v1/calendar", 503)
    ctx, cid, r, b = await submit(fake_college, leave(from_date="2026-11-23", to_date="2026-11-27"))
    # Cannot check exams or holidays: either refuse, or say so in a warning.
    assert r.is_error or any(mentions(w, "calendar", "could not", "unable") for w in b["warnings"])

