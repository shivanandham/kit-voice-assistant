"""COR1 (college date, never the server clock) and COR2 (timetable resolution).

The server clock is patched far away from the college date in every COR1 test: if any tool read
the system clock, the answers would move with it.
"""

import datetime as _dt
from datetime import date, timedelta

import pytest
from app.agent import tools as T
from app.kit import client as kc

from tests.acc_tools_helpers import mentions, new_conv, run
from tests.fakes import college as fc

FAR_CLOCKS = [date(2019, 1, 1), date(2026, 11, 14), date(2031, 6, 30)]


def _fake_date_cls(fixed: date):
    class FrozenDate(date):
        @classmethod
        def today(cls):
            return cls(fixed.year, fixed.month, fixed.day)

    return FrozenDate


def _fake_datetime_cls(fixed: date):
    class FrozenDateTime(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(fixed.year, fixed.month, fixed.day, 12, 0, tzinfo=tz)

        @classmethod
        def utcnow(cls):
            return cls(fixed.year, fixed.month, fixed.day, 12, 0)

        @classmethod
        def today(cls):
            return cls(fixed.year, fixed.month, fixed.day, 12, 0)

    return FrozenDateTime


@pytest.fixture(params=FAR_CLOCKS, ids=lambda d: f"clock{d.isoformat()}")
def far_clock(request, monkeypatch):
    """Patch every clock the tool layer could read to a date far from the college date."""
    fixed = request.param
    fd = _fake_date_cls(fixed)
    for mod in (T, kc):
        if hasattr(mod, "date"):
            monkeypatch.setattr(mod, "date", fd)
        if hasattr(mod, "datetime"):
            monkeypatch.setattr(mod, "datetime", _fake_datetime_cls(fixed))
    import datetime as dt_mod

    monkeypatch.setattr(dt_mod, "date", fd)
    return fixed


def _set_college_date(fake, d: date):
    fake.today = d
    kc.clear_today_cache()


# ------------------------------------------------------------------ COR1


@pytest.mark.parametrize("college_day", [date(2026, 11, 21), date(2027, 3, 10), date(2025, 2, 28)])
async def test_cor1_get_today_is_college_date(college_day, far_clock, fake_college):
    _set_college_date(fake_college, college_day)
    ctx = fake_college.ctx("arjun")
    r, b = await run("get_today", {}, ctx, await new_conv(ctx))
    assert college_day.isoformat() in r.content
    assert college_day.strftime("%A") in r.content
    assert far_clock.isoformat() not in r.content or far_clock == college_day


async def test_cor1_get_today_for_visitor(far_clock, fake_college):
    ctx = fake_college.ctx(None)
    r, b = await run("get_today", {}, ctx, await new_conv(ctx))
    assert "2026-11-21" in r.content
    assert "Saturday" in r.content


async def test_cor1_week_view_starts_at_college_date(far_clock, fake_college):
    ctx = fake_college.ctx("arjun")
    r, b = await run("get_timetable", {}, ctx, await new_conv(ctx))
    assert not r.is_error, b
    days = _week_days(b)
    assert [d["date"] for d in days] == [(fc.TODAY + timedelta(days=i)).isoformat()
                                         for i in range(7)]
    cal = fake_college.calls("GET", "/v1/calendar")
    assert cal and cal[-1].params.get("from") == "2026-11-21"


@pytest.mark.parametrize("college_day", [date(2026, 11, 23), date(2027, 1, 4)])
async def test_cor1_week_view_follows_college_date_change(college_day, far_clock, fake_college):
    _set_college_date(fake_college, college_day)
    ctx = fake_college.ctx("arjun")
    r, b = await run("get_timetable", {}, ctx, await new_conv(ctx))
    assert _week_days(b)[0]["date"] == college_day.isoformat()


async def test_cor1_calendar_default_range_from_college_date(far_clock, fake_college):
    ctx = fake_college.ctx("arjun")
    r, b = await run("get_calendar", {}, ctx, await new_conv(ctx))
    assert not r.is_error, b
    p = fake_college.calls("GET", "/v1/calendar")[-1].params
    assert p.get("from") == "2026-11-21"
    assert p.get("to") and date.fromisoformat(p["to"]) > fc.TODAY
    assert "Founders Day" in r.content


async def test_cor1_calendar_with_explicit_range(far_clock, fake_college):
    ctx = fake_college.ctx("arjun")
    r, b = await run("get_calendar", {"from_date": "2026-11-22", "to_date": "2026-11-24"}, ctx,
                     await new_conv(ctx))
    assert "DBMS Internal" in r.content
    assert "Founders Day" not in r.content


async def test_cor1_calendar_reverse_range_rejected(far_clock, fake_college):
    ctx = fake_college.ctx("arjun")
    r, _ = await run("get_calendar", {"from_date": "2026-11-24", "to_date": "2026-11-22"}, ctx,
                     await new_conv(ctx))
    assert r.is_error


async def test_cor1_leave_past_warning_uses_college_date(far_clock, fake_college):
    """2026-11-18 is after the real clock (2026-11-14) but before the college date."""
    ctx = fake_college.ctx("arjun")
    args = {"type": "medical_leave", "from_date": "2026-11-18", "to_date": "2026-11-18",
            "half_day": False, "reason": "Fever"}
    r, b = await run("submit_leave_request", args, ctx, await new_conv(ctx))
    assert not r.is_error, b
    assert any(mentions(w, "past", "before today", "backdated") for w in b["warnings"])


async def test_cor1_leave_future_no_past_warning_whatever_the_clock(far_clock, fake_college):
    ctx = fake_college.ctx("arjun")
    args = {"type": "medical_leave", "from_date": "2026-11-25", "to_date": "2026-11-25",
            "half_day": False, "reason": "Fever"}
    r, b = await run("submit_leave_request", args, ctx, await new_conv(ctx))
    assert not any(mentions(w, "past", "before today", "backdated") for w in b["warnings"])


async def test_cor1_college_date_moved_back_makes_leave_future(far_clock, fake_college):
    _set_college_date(fake_college, date(2026, 11, 10))
    ctx = fake_college.ctx("arjun")
    args = {"type": "medical_leave", "from_date": "2026-11-12", "to_date": "2026-11-12",
            "half_day": False, "reason": "Fever"}
    r, b = await run("submit_leave_request", args, ctx, await new_conv(ctx))
    assert not any(mentions(w, "past", "before today", "backdated") for w in b["warnings"])


async def test_cor1_placement_deadline_uses_college_date(far_clock, fake_college):
    ctx = fake_college.ctx("karthik")
    r, b = await run("list_placement_drives", {}, ctx, await new_conv(ctx))
    zoho = next(d for d in b["drives"] if d["company"] == "Zoho")
    assert not any(mentions(x, "deadline") for x in zoho["reasons"])
    _set_college_date(fake_college, date(2026, 11, 27))
    r, b = await run("list_placement_drives", {}, ctx, await new_conv(ctx))
    zoho = next(d for d in b["drives"] if d["company"] == "Zoho")
    assert zoho["eligible"] is False
    assert any(mentions(x, "deadline") for x in zoho["reasons"])


async def test_cor1_delegation_validity_uses_college_date(far_clock, fake_college):
    ctx = fake_college.ctx("senthil")
    r, b = await run("list_my_classes", {}, ctx, await new_conv(ctx))
    assert "MECH-3A" in r.content
    mech = next(k for k in b["classes"] if k["class_code"] == "MECH-3A")
    assert mech.get("delegation_active") is not False


async def test_cor1_cancel_started_warning_uses_college_date(far_clock, fake_college):
    ctx = fake_college.ctx("arjun")
    fake_college.requests[12]["from_date"] = "2026-11-20"
    fake_college.requests[12]["to_date"] = "2026-11-20"
    r, b = await run("cancel_request", {"request_id": 12}, ctx, await new_conv(ctx))
    assert any(mentions(w, "started", "start date", "past") for w in b["warnings"])
    fake_college.requests[12]["from_date"] = "2026-11-30"
    fake_college.requests[12]["to_date"] = "2026-11-30"
    r, b = await run("cancel_request", {"request_id": 12}, ctx, await new_conv(ctx))
    assert not any(mentions(w, "started", "start date") for w in b["warnings"])


async def test_cor1_personal_leave_today_warning_uses_college_date(far_clock, fake_college):
    ctx = fake_college.ctx("arjun")
    args = {"type": "personal_leave", "from_date": "2026-11-21", "to_date": "2026-11-21",
            "half_day": False, "reason": "Family"}
    r, b = await run("submit_leave_request", args, ctx, await new_conv(ctx))
    assert b["warnings"], "leave starting on the college's today should carry a warning"


async def test_cor1_today_is_fetched_from_college(fake_college):
    ctx = fake_college.ctx("arjun")
    await run("get_today", {}, ctx, await new_conv(ctx))
    assert fake_college.calls("GET", "/v1/today")


async def test_cor1_college_today_cached(fake_college):
    kc.clear_today_cache()
    a = await kc.college_today()
    b = await kc.college_today()
    assert a == b and a["date"] == "2026-11-21"
    assert len(fake_college.calls("GET", "/v1/today")) == 1


async def test_cor1_college_today_cache_expires(fake_college, monkeypatch):
    kc.clear_today_cache()
    await kc.college_today()
    real = kc.time.monotonic
    monkeypatch.setattr(kc.time, "monotonic", lambda: real() + kc.TODAY_TTL_SECONDS + 5)
    fake_college.today = date(2026, 11, 22)
    t = await kc.college_today()
    assert t["date"] == "2026-11-22"
    assert len(fake_college.calls("GET", "/v1/today")) == 2


async def test_cor1_today_down_is_an_error_not_server_clock(fake_college):
    kc.RETRY_BACKOFF_SECONDS, saved = 0.0, kc.RETRY_BACKOFF_SECONDS
    fake_college.fail("/v1/today", 503)
    ctx = fake_college.ctx("arjun")
    r, b = await run("get_today", {}, ctx, await new_conv(ctx))
    kc.RETRY_BACKOFF_SECONDS = saved
    assert r.is_error
    assert _dt.date.today().isoformat() not in r.content


# ------------------------------------------------------------------ COR2 timetable


def _expected_codes(class_id: int, weekday: int) -> list[str]:
    """Subject codes of the fake's weekly timetable for (class, ISO weekday), periods 1..4."""
    subs = fc.SUBJECTS[class_id]
    return [subs[(weekday + p) % len(subs)][1] for p in range(1, 5)]


def _day(b) -> dict:
    return b["day"]


def _week_days(b) -> list[dict]:
    for k in ("next_7_days", "days", "week"):
        if k in b:
            return b[k]
    raise AssertionError(f"no week list in {list(b)}")


def _codes(day: dict) -> list[str]:
    return [c["subject_code"] for c in sorted(day.get("classes", []), key=lambda c: c["period"])]


async def _tt(fake, login, d=None):
    ctx = fake.ctx(login)
    args = {"date": d} if d else {}
    r, b = await run("get_timetable", args, ctx, await new_conv(ctx))
    assert not r.is_error, b
    return r, b


@pytest.mark.parametrize("d", ["2026-11-22", "2026-11-27"])
async def test_cor2_holiday_has_no_classes(d, fake_college):
    r, b = await _tt(fake_college, "arjun", d)
    assert _codes(_day(b)) == []
    assert mentions(r.content, "holiday")


@pytest.mark.parametrize("d", ["2026-11-29", "2026-12-06", "2027-01-03"])
async def test_cor2_sunday_without_entry_has_no_classes(d, fake_college):
    r, b = await _tt(fake_college, "arjun", d)
    assert _codes(_day(b)) == []


async def test_cor2_working_saturday_follows_monday(fake_college):
    r, b = await _tt(fake_college, "arjun", "2026-11-28")
    assert _codes(_day(b)) == _expected_codes(11, 1)
    assert _codes(_day(b)) != _expected_codes(11, 6)
    assert "Monday" in r.content


@pytest.mark.parametrize("follows", [1, 2, 3, 4, 5, 6])
async def test_cor2_follows_weekday_substitution_every_weekday(follows, fake_college):
    fake_college.calendar_events.append(
        {"id": 50, "date": "2026-12-05", "type": "working_day", "title": "Compensation day",
         "subject_code": None, "follows_weekday": follows, "status": "active"})
    r, b = await _tt(fake_college, "arjun", "2026-12-05")
    assert _codes(_day(b)) == _expected_codes(11, follows)


@pytest.mark.parametrize("follows", [1, 3, 5])
async def test_cor2_sunday_declared_working_day_follows_weekday(follows, fake_college):
    fake_college.calendar_events.append(
        {"id": 51, "date": "2026-11-29", "type": "working_day", "title": "Working Sunday",
         "subject_code": None, "follows_weekday": follows, "status": "active"})
    r, b = await _tt(fake_college, "arjun", "2026-11-29")
    assert _codes(_day(b)) == _expected_codes(11, follows)


@pytest.mark.parametrize("d,wd", [("2026-11-25", 3), ("2026-11-26", 4), ("2026-11-24", 2),
                                  ("2026-11-21", 6), ("2026-12-04", 5), ("2027-02-10", 3)])
async def test_cor2_ordinary_day_uses_own_weekday(d, wd, fake_college):
    r, b = await _tt(fake_college, "arjun", d)
    assert _codes(_day(b)) == _expected_codes(11, wd)


@pytest.mark.parametrize("login,cls", [("kavya", 12), ("meena", 13), ("karthik", 14)])
async def test_cor2_other_classes_use_their_timetable(login, cls, fake_college):
    r, b = await _tt(fake_college, login, "2026-11-25")
    assert _codes(_day(b)) == _expected_codes(cls, 3)


async def test_cor2_no_calendar_entry_date_is_ordinary(fake_college):
    r, b = await _tt(fake_college, "arjun", "2026-12-15")
    assert _codes(_day(b)) == _expected_codes(11, 2)


async def test_cor2_exam_day_still_has_classes_and_lists_exam(fake_college):
    r, b = await _tt(fake_college, "arjun", "2026-11-23")
    assert _codes(_day(b)) == _expected_codes(11, 1)
    assert "DBMS Internal Assessment Exam" in r.content


async def test_cor2_other_departments_exam_not_shown(fake_college):
    r, b = await _tt(fake_college, "arjun", "2026-11-24")
    assert "Signals" not in r.content


async def test_cor2_cancelled_holiday_does_not_cancel_classes(fake_college):
    fake_college.calendar_events.append(
        {"id": 52, "date": "2026-12-02", "type": "holiday", "title": "Withdrawn holiday",
         "subject_code": None, "follows_weekday": None, "status": "cancelled"})
    r, b = await _tt(fake_college, "arjun", "2026-12-02")
    assert _codes(_day(b)) == _expected_codes(11, 3)


async def test_cor2_holiday_beats_working_day_same_date(fake_college):
    fake_college.calendar_events.append(
        {"id": 53, "date": "2026-11-27", "type": "working_day", "title": "Conflicting",
         "subject_code": None, "follows_weekday": 2, "status": "active"})
    r, b = await _tt(fake_college, "arjun", "2026-11-27")
    assert _codes(_day(b)) == []


async def test_cor2_week_view_default(fake_college):
    r, b = await _tt(fake_college, "arjun")
    days = {d["date"]: d for d in _week_days(b)}
    assert len(days) == 7
    assert _codes(days["2026-11-21"]) == _expected_codes(11, 6)
    assert _codes(days["2026-11-22"]) == []
    assert _codes(days["2026-11-23"]) == _expected_codes(11, 1)
    assert _codes(days["2026-11-25"]) == _expected_codes(11, 3)
    assert _codes(days["2026-11-27"]) == []


async def test_cor2_week_view_includes_substitution_and_sunday(fake_college):
    _set_college_date(fake_college, date(2026, 11, 23))
    r, b = await _tt(fake_college, "arjun")
    days = {d["date"]: d for d in _week_days(b)}
    assert list(days)[0] == "2026-11-23"
    assert _codes(days["2026-11-28"]) == _expected_codes(11, 1)
    assert _codes(days["2026-11-29"]) == []
    assert _codes(days["2026-11-27"]) == []


async def test_cor2_week_days_have_weekday_names(fake_college):
    r, b = await _tt(fake_college, "arjun")
    for d in _week_days(b):
        assert d["weekday"] == date.fromisoformat(d["date"]).strftime("%A")


@pytest.mark.parametrize("bad", ["2026-13-01", "2026-11-31", "21/11/2026", "tomorrow", "2026-1-5",
                                 "20261121", "", "2026-11-21T00:00"])
async def test_cor2_malformed_date_rejected(bad, fake_college):
    ctx = fake_college.ctx("arjun")
    r, b = await run("get_timetable", {"date": bad}, ctx, await new_conv(ctx))
    assert r.is_error
    assert fake_college.log == []
    assert "Traceback" not in r.content


async def test_cor2_calendar_down_is_error_not_wrong_timetable(fake_college, monkeypatch):
    monkeypatch.setattr(kc, "RETRY_BACKOFF_SECONDS", 0.0)
    fake_college.fail("/v1/calendar", 503)
    ctx = fake_college.ctx("arjun")
    r, b = await run("get_timetable", {"date": "2026-11-27"}, ctx, await new_conv(ctx))
    # Without the calendar a holiday cannot be known: an error, never Friday's classes.
    assert r.is_error or _codes(_day(b)) == []
