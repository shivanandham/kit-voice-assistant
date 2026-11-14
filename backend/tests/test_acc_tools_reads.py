"""Read tools in detail: AC2 (own data), COR1/COR2 corners, COR4 consistency, AGT6 edge payloads."""

import pytest
from app.auth import SessionCtx

from tests.acc_tools_helpers import mentions, new_conv, override, run
from tests.fakes import college as fc


async def srun(fake, name, args, login="arjun", ctx=None):
    ctx = ctx or fake.ctx(login)
    r, b = await run(name, args, ctx, await new_conv(ctx))
    return r, b


# ------------------------------------------------------------------ profile and academics


@pytest.mark.parametrize("login", fc.STUDENTS + fc.TEACHERS)
async def test_ac2_profile_is_own(login, fake_college):
    r, b = await srun(fake_college, "get_my_profile", {}, login)
    assert not r.is_error
    assert fc.USERS[login]["full_name"] in r.content
    for other in fc.USERS:
        if other != login and fc.USERS[other]["full_name"] != fc.USERS[login]["full_name"]:
            assert fc.USERS[other]["full_name"] not in r.content


async def test_ac2_profile_omits_internal_ids(fake_college):
    r, b = await srun(fake_college, "get_my_profile", {})
    assert "id" not in b and "class_id" not in b


@pytest.mark.parametrize("programme", ["PG", None, "DIPLOMA"])
async def test_ac2_profile_programme_variants(programme, fake_college, monkeypatch):
    real = fc.user_summary
    monkeypatch.setattr(fc, "user_summary", lambda login: {**real(login), "programme": programme})
    r, b = await srun(fake_college, "get_my_profile", {})
    assert not r.is_error
    if programme == "PG":
        assert "M.Tech" in r.content


@pytest.mark.parametrize("login", fc.STUDENTS)
async def test_ac2_academics_are_own(login, fake_college):
    r, b = await srun(fake_college, "get_my_academics", {}, login)
    cgpa, arrears, *_ = fc.ACADEMICS[login]
    assert b["cgpa"] == cgpa and b["standing_arrears"] == arrears
    assert "user_id" not in b


@pytest.mark.parametrize("login", fc.STUDENTS)
async def test_ac2_subjects_are_own_class(login, fake_college):
    r, b = await srun(fake_college, "get_my_subjects", {}, login)
    codes = {s["code"] for s in b["subjects"]}
    assert codes == {c for _, c, _ in fc.SUBJECTS[fc.USERS[login]["class_id"]]}


# ------------------------------------------------------------------ attendance


@pytest.mark.parametrize("login", fc.STUDENTS)
async def test_ac2_attendance_summary_matches_own_record(login, fake_college):
    r, b = await srun(fake_college, "get_attendance_summary", {}, login)
    expected = fake_college.attendance_summary(login)
    assert b["overall_percentage"] == expected["overall_percentage"]
    assert {s["code"] for s in b["subjects"]} == {s["subject_code"] for s in expected["subjects"]}


async def test_agt6_attendance_with_no_classes(fake_college):
    override(fake_college, "GET", "/v1/me/attendance", lambda call, orig: (200, {
        "as_of_date": "2026-11-21", "overall_percentage": 0, "total_classes_held": 0,
        "total_classes_attended": 0, "subjects": []}))
    r, b = await srun(fake_college, "get_attendance_summary", {})
    assert not r.is_error
    assert mentions(r.content, "no classes")


async def test_ac2_attendance_sessions_for_subject(fake_college):
    r, b = await srun(fake_college, "get_attendance_sessions", {"subject_code": "cs304"})
    assert not r.is_error
    assert b["total"] == 10
    assert all(s["subject"] == "CS304" for s in b["sessions"])
    assert fake_college.calls("GET", "/v1/me/attendance/sessions")[-1].params["subject"] == "CS304"


async def test_ac2_attendance_sessions_none_subject_is_all(fake_college):
    r, b = await srun(fake_college, "get_attendance_sessions", {"subject_code": None})
    assert not r.is_error
    assert b["total"] == 40


async def test_ac2_attendance_sessions_date_range(fake_college):
    r, b = await srun(fake_college, "get_attendance_sessions",
                      {"from_date": "2026-11-15", "to_date": "2026-11-20"})
    assert all("2026-11-15" <= s["date"] <= "2026-11-20" for s in b["sessions"])


async def test_ac2_attendance_sessions_reverse_range_rejected(fake_college):
    r, b = await srun(fake_college, "get_attendance_sessions",
                      {"from_date": "2026-11-20", "to_date": "2026-11-15"})
    assert r.is_error
    assert not fake_college.calls("GET", "/v1/me/attendance/sessions")


async def test_ac2_attendance_sessions_subject_not_mine(fake_college):
    r, b = await srun(fake_college, "get_attendance_sessions", {"subject_code": "EC301"})
    assert not r.is_error
    assert b["total"] == 0
    assert "EC301" in b.get("note", "") and "CS304" in b.get("note", "")


async def test_ac2_attendance_sessions_subject_mine_but_empty_period(fake_college):
    r, b = await srun(fake_college, "get_attendance_sessions",
                      {"subject_code": "CS304", "from_date": "2027-01-01",
                       "to_date": "2027-01-31"})
    assert b["total"] == 0
    assert mentions(b.get("note", ""), "no attendance")


async def test_agt6_attendance_sessions_capped(fake_college):
    rows = [{"session_id": i, "date": "2026-11-20", "period": 1, "subject_code": "CS304",
             "subject_name": "DBMS", "status": "present"} for i in range(250)]
    override(fake_college, "GET", "/v1/me/attendance/sessions", lambda call, orig: (200, rows))
    r, b = await srun(fake_college, "get_attendance_sessions", {})
    assert b["total"] == 250
    assert len(b["sessions"]) <= 100
    assert "note" in b


# ------------------------------------------------------------------ calendar and timetable corners


async def test_cor2_calendar_marks_cancelled_events(fake_college):
    fake_college.calendar_events.append(
        {"id": 80, "date": "2026-11-26", "type": "holiday", "title": "Withdrawn", "subject_code": None,
         "follows_weekday": None, "status": "cancelled"})
    r, b = await srun(fake_college, "get_calendar", {})
    for e in b["events"]:
        if e["title"] == "Withdrawn":
            assert e.get("status") == "cancelled"


async def test_cor2_calendar_shows_follows_weekday(fake_college):
    r, b = await srun(fake_college, "get_calendar", {})
    sat = next(e for e in b["events"] if e["date"] == "2026-11-28")
    assert "Monday" in str(sat)


async def test_cor2_calendar_only_own_subjects_exams(fake_college):
    r, b = await srun(fake_college, "get_calendar", {})
    assert "Signals" not in r.content
    r, b = await srun(fake_college, "get_calendar", {}, "kavya")
    assert "Signals" in r.content and "DBMS" not in r.content


async def test_cor2_declared_working_day_without_follows(fake_college):
    fake_college.calendar_events.append(
        {"id": 81, "date": "2026-12-05", "type": "working_day", "title": "Extra Saturday",
         "subject_code": None, "follows_weekday": None, "status": "active"})
    r, b = await srun(fake_college, "get_timetable", {"date": "2026-12-05"})
    subs = fc.SUBJECTS[11]
    expected = [subs[(6 + p) % 4][1] for p in range(1, 5)]
    assert [c["subject_code"] for c in b["day"]["classes"]] == expected


async def test_cor2_weekday_without_slots(fake_college):
    override(fake_college, "GET", "/v1/me/timetable", lambda call, orig: (200, [
        {"id": 1, "weekday": 1, "period": 1, "start_time": "09:00", "end_time": "09:50",
         "subject_code": "CS301", "subject_name": "DSA"}]))
    r, b = await srun(fake_college, "get_timetable", {"date": "2026-11-24"})
    assert b["day"]["classes"] == []


async def test_cor2_other_event_types_listed(fake_college):
    fake_college.calendar_events.append(
        {"id": 82, "date": "2026-11-25", "type": "event", "title": "Sports Day", "subject_code": None,
         "follows_weekday": None, "status": "active"})
    r, b = await srun(fake_college, "get_timetable", {"date": "2026-11-25"})
    assert "Sports Day" in r.content
    assert b["day"]["classes"]


async def test_cor2_timetable_periods_in_order(fake_college):
    slots = list(reversed([
        {"id": p, "weekday": 3, "period": p, "start_time": f"{8 + p}:00", "end_time": f"{8 + p}:50",
         "subject_code": f"CS30{p}", "subject_name": "x"} for p in range(1, 5)]))
    override(fake_college, "GET", "/v1/me/timetable", lambda call, orig: (200, slots))
    r, b = await srun(fake_college, "get_timetable", {"date": "2026-11-25"})
    assert [c["period"] for c in b["day"]["classes"]] == [1, 2, 3, 4]


# ------------------------------------------------------------------ requests


async def test_ac2_list_my_requests_only_own(fake_college):
    r, b = await srun(fake_college, "list_my_requests", {}, "priya")
    assert {x["id"] for x in b["requests"]} == {15, 19}
    assert "Arjun" not in r.content


async def test_ac2_get_own_request(fake_college):
    r, b = await srun(fake_college, "get_request", {"request_id": 12})
    assert not r.is_error
    assert b["request"]["id"] == 12
    assert "student_id" not in b["request"]


@pytest.mark.parametrize("status", ["rejected", "cancelled"])
async def test_cor6_cancel_already_closed_request_warns(status, fake_college):
    fake_college.requests[12]["status"] = status
    r, b = await srun(fake_college, "cancel_request", {"request_id": 12})
    if not r.is_error:
        assert any(mentions(w, status, "already") for w in b["warnings"])


async def test_cor3_leave_with_no_attendance_yet(fake_college):
    override(fake_college, "GET", "/v1/me/attendance", lambda call, orig: (200, {
        "as_of_date": "2026-11-21", "overall_percentage": 0, "total_classes_held": 0,
        "total_classes_attended": 0, "subjects": []}))
    r, b = await srun(fake_college, "submit_leave_request",
                      {"type": "medical_leave", "from_date": "2026-11-25", "to_date": "2026-11-25",
                       "half_day": False, "reason": "Fever"})
    assert not r.is_error


async def test_cor3_leave_attendance_all_fine(fake_college):
    override(fake_college, "GET", "/v1/me/attendance", lambda call, orig: (200, {
        "as_of_date": "2026-11-21", "overall_percentage": 95, "total_classes_held": 40,
        "total_classes_attended": 38, "subjects": [{"subject_code": "CS304", "percentage": 95}]}))
    r, b = await srun(fake_college, "submit_leave_request",
                      {"type": "medical_leave", "from_date": "2026-11-25", "to_date": "2026-11-25",
                       "half_day": False, "reason": "Fever"})
    assert not r.is_error
    assert not any(mentions(w, "below") for w in b["warnings"])


@pytest.mark.parametrize("programme", ["UG", "PG"])
async def test_cor3_leave_low_attendance_warns(programme, fake_college):
    override(fake_college, "GET", "/v1/me/attendance", lambda call, orig: (200, {
        "as_of_date": "2026-11-21", "overall_percentage": 60, "total_classes_held": 40,
        "total_classes_attended": 24, "subjects": [{"subject_code": "CS304", "percentage": 60}]}))
    base = fake_college.ctx("arjun")
    ctx = SessionCtx(base.session_id, base.owner_key, "student",
                     {**base.user, "programme": programme}, base.kit_token)
    r, b = await srun(fake_college, "submit_leave_request",
                      {"type": "medical_leave", "from_date": "2026-11-25", "to_date": "2026-11-25",
                       "half_day": False, "reason": "Fever"}, ctx=ctx)
    assert any("CS304" in w for w in b["warnings"])


# ------------------------------------------------------------------ placements consistency (COR4)


@pytest.mark.parametrize("login", fc.STUDENTS)
@pytest.mark.parametrize("programme", ["UG", "PG"])
async def test_cor4_eligible_iff_no_reasons(login, programme, fake_college, monkeypatch):
    real = fc.user_summary
    monkeypatch.setattr(fc, "user_summary", lambda lg: {**real(lg), "programme": programme})
    base = fake_college.ctx(login)
    ctx = SessionCtx(base.session_id, base.owner_key, "student",
                     {**base.user, "programme": programme}, base.kit_token)
    for d in fake_college.drives:
        d["allow_standing_arrears"] = True
    monkeypatch.setitem(fc.ACADEMICS, login, (6.8, 1, None, None))
    r, b = await srun(fake_college, "list_placement_drives", {}, ctx=ctx)
    assert not r.is_error
    for d in b["drives"]:
        assert d["eligible"] is (d["reasons"] == [])
        assert all(isinstance(x, str) and x for x in d["reasons"])


async def test_cor4_profile_fetched_when_ctx_lacks_it(fake_college):
    base = fake_college.ctx("arjun")
    ctx = SessionCtx(base.session_id, base.owner_key, "student", None, base.kit_token)
    r, b = await srun(fake_college, "list_placement_drives", {}, ctx=ctx)
    assert not r.is_error
    assert all(any(mentions(x, "year") for x in d["reasons"]) for d in b["drives"])
    assert all(c.login == "arjun" for c in fake_college.log if c.token)


async def test_cor4_drive_reasons_name_their_source_when_from_handbook(fake_college, monkeypatch):
    monkeypatch.setitem(fc.ACADEMICS, "karthik", (8.6, 0, "TCS", "non_dream"))
    r, b = await srun(fake_college, "list_placement_drives", {}, "karthik")
    tcs = next(d for d in b["drives"] if d["company"] == "TCS")
    assert tcs["eligible"] is False
