"""Correctness helpers: timetable resolution, leave pre-checks, placement eligibility, admissions."""

import json
from datetime import date

import pytest
import respx
from app import history
from app.agent import tools as T
from app.auth import SessionCtx
from app.config import get_settings
from app.kit import client as kc

BASE = get_settings().kit_base_url
SLOTS = [
    {"id": 1, "weekday": 1, "period": 1, "start_time": "09:00", "end_time": "09:50",
     "subject_code": "CS301", "subject_name": "DSA"},
    {"id": 2, "weekday": 1, "period": 2, "start_time": "10:00", "end_time": "10:50",
     "subject_code": "CS304", "subject_name": "DBMS"},
    {"id": 3, "weekday": 3, "period": 1, "start_time": "09:00", "end_time": "09:50",
     "subject_code": "CS303", "subject_name": "OOP"},
]
BY_WD = {1: SLOTS[:2], 3: SLOTS[2:]}
USER = {"id": 28, "login_id": "arjun", "department": "CSE", "year_of_study": 4, "programme": "UG"}


def ev(day: str, type_: str, title: str, follows=None, subject=None, status="active") -> dict:
    return {"id": 1, "date": day, "type": type_, "title": title, "subject_code": subject,
            "follows_weekday": follows, "status": status}


def student(user: dict = USER) -> SessionCtx:
    return SessionCtx("sid", "user:arjun", "student", user, "tok")


@pytest.fixture
def college():
    kc.clear_today_cache()
    with respx.mock(base_url=BASE, assert_all_called=False) as m:
        m.get("/v1/today").respond(200, json={"date": "2026-11-21", "weekday": 6,
                                              "weekday_name": "Saturday", "timezone": "IST"})
        m.get("/v1/me/timetable").respond(200, json=SLOTS)
        m.get("/v1/me/requests").respond(200, json=[])
        m.get("/v1/me/attendance").respond(200, json={
            "as_of_date": "2026-11-21", "overall_percentage": 72.0, "total_classes_held": 50,
            "total_classes_attended": 36, "subjects": [{
                "subject_id": 9, "subject_code": "CS304", "subject_name": "DBMS",
                "classes_held": 50, "classes_attended": 36, "classes_od": 0,
                "classes_absent": 14, "percentage": 72.0}]})
        yield m


async def run(name: str, args: dict, ctx: SessionCtx | None = None) -> tuple[dict, bool]:
    ctx = ctx or student()
    cid = await history.create_conversation(ctx)
    r = await T.run_tool(name, args, ctx, cid)
    return json.loads(r.content), r.is_error


# ---------------------------------------------------------------- timetable

def test_follows_weekday_uses_that_days_slots():
    sat = date(2026, 11, 28)
    day = T.resolve_day(sat, BY_WD, [ev("2026-11-28", "working_day", "Compensatory working day",
                                        follows=3)])
    assert [c["subject_code"] for c in day["classes"]] == ["CS303"]
    assert day["follows_timetable_of"] == "Wednesday" and "Wednesday" in day["note"]


def test_holiday_cancels_classes_and_sunday_is_empty():
    mon = date(2026, 11, 23)
    day = T.resolve_day(mon, BY_WD, [ev("2026-11-23", "holiday", "Deepavali")])
    assert day["classes"] == [] and "Deepavali" in day["note"]
    assert T.resolve_day(date(2026, 11, 22), BY_WD, [])["classes"] == []
    normal = T.resolve_day(mon, BY_WD, [ev("2026-11-23", "holiday", "Old", status="cancelled")])
    assert len(normal["classes"]) == 2


def test_exam_listed_with_classes():
    day = T.resolve_day(date(2026, 11, 23), BY_WD,
                        [ev("2026-11-23", "internal_exam", "DBMS IA", subject="CS304")])
    assert len(day["classes"]) == 2 and day["exams"][0]["subject_code"] == "CS304"


async def test_get_timetable_for_date_and_week(college):
    college.get("/v1/calendar").respond(200, json=[
        ev("2026-11-22", "holiday", "Sunday Holiday"),
        ev("2026-11-25", "working_day", "Follows Monday", follows=1)])
    body, err = await run("get_timetable", {"date": "2026-11-25"})
    assert not err and body["college_today"] == {"date": "2026-11-21", "weekday": "Saturday"}
    body, _ = await run("get_timetable", {})
    days = body["next_7_days"]
    assert days[0]["date"] == "2026-11-21" and len(days) == 7
    wed = next(d for d in days if d["date"] == "2026-11-25")
    assert [c["subject_code"] for c in wed["classes"]] == ["CS301", "CS304"]


# ---------------------------------------------------------------- leave pre-checks

async def test_leave_warnings(college):
    college.get("/v1/calendar").respond(200, json=[
        ev("2026-11-22", "holiday", "Sunday Holiday"),
        ev("2026-11-23", "internal_exam", "DBMS Internal Assessment Exam", subject="CS304")])
    college.get("/v1/me/requests").respond(200, json=[{
        "id": 15, "type": "personal_leave", "from_date": "2026-11-23", "to_date": "2026-11-23",
        "status": "approved"}])
    body, err = await run("submit_leave_request", {
        "type": "personal_leave", "from_date": "2026-11-20", "to_date": "2026-11-23",
        "half_day": False, "reason": "Family function"})
    assert not err and body["status"] == "needs_confirmation"
    w = " | ".join(body["warnings"])
    assert "before today's college date Saturday 21 November 2026" in w
    assert "Internal exam on Monday 23 November 2026" in w
    assert "holiday (Sunday Holiday)" in w
    assert "Overlaps your approved personal leave request #15" in w
    assert "72%" in w and "CS304" in w
    assert "Friday 20 November 2026 to Monday 23 November 2026" in body["summary"]


async def test_leave_hard_errors(college):
    body, err = await run("submit_leave_request", {
        "type": "medical_leave", "from_date": "2026-11-25", "to_date": "2026-11-24",
        "half_day": False, "reason": "Fever"})
    assert err and "before from_date" in body["error"]
    body, err = await run("submit_leave_request", {
        "type": "on_duty", "from_date": "2026-11-25", "to_date": "2026-11-25",
        "half_day": False, "reason": "Symposium"})
    assert err and "event_name" in body["error"]


async def test_on_duty_payload_keeps_event(college):
    college.get("/v1/calendar").respond(200, json=[])
    body, err = await run("submit_leave_request", {
        "type": "on_duty", "from_date": "2026-11-30", "to_date": "2026-11-30",
        "half_day": False, "reason": "Paper", "event_name": "Tech Symposium"})
    assert not err and "Event: Tech Symposium" in body["summary"]


# ---------------------------------------------------------------- placements

DRIVE = {"id": 7, "company": "Zoho", "category": "non_dream", "min_cgpa": 7.0,
         "allow_standing_arrears": False, "allowed_departments": ["CSE", "ECE"],
         "eligible_years": [4], "deadline": "2026-11-26", "seats_total": 10, "seats_left": 1}
ACAD = {"cgpa": 8.2, "standing_arrears": 0, "placed_company": None, "placed_category": None}
TODAY = date(2026, 11, 21)


def test_eligible_student():
    assert T.drive_eligibility(DRIVE, USER, ACAD, set(), TODAY) == (True, [])


@pytest.mark.parametrize("change,needle", [
    ({"acad": {"cgpa": 6.9}}, "CGPA 6.9 is below the minimum 7"),
    ({"acad": {"standing_arrears": 1}}, "does not allow arrears"),
    ({"user": {"department": "MECH"}}, "Open to CSE, ECE only"),
    ({"user": {"year_of_study": 2}}, "you are in year 2"),
    ({"drive": {"deadline": "2026-11-20"}}, "deadline"),
    ({"drive": {"seats_left": 0}}, "No seats"),
    ({"regs": {7}}, "already registered"),
    ({"acad": {"placed_company": "Google", "placed_category": "dream"}}, "only register for dream"),
])
def test_ineligible_reasons(change, needle):
    drive = {**DRIVE, **change.get("drive", {})}
    user = {**USER, **change.get("user", {})}
    acad = {**ACAD, **change.get("acad", {})}
    ok, reasons = T.drive_eligibility(drive, user, acad, change.get("regs", set()), TODAY)
    assert not ok and any(needle in r for r in reasons), reasons


def test_deadline_today_still_open_and_dream_allowed_when_placed():
    ok, _ = T.drive_eligibility({**DRIVE, "deadline": "2026-11-21"}, USER, ACAD, set(), TODAY)
    assert ok
    placed = {**ACAD, "placed_company": "TCS", "placed_category": "non_dream"}
    ok, _ = T.drive_eligibility({**DRIVE, "category": "dream"}, USER, placed, set(), TODAY)
    assert ok


def test_mtech_rules():
    pg = {**USER, "programme": "PG"}
    ok, reasons = T.drive_eligibility({**DRIVE, "allow_standing_arrears": True, "min_cgpa": 6.0},
                                      pg, {**ACAD, "cgpa": 6.5, "standing_arrears": 1}, set(), TODAY)
    assert not ok and any("7.0" in r for r in reasons) and any("M.Tech" in r for r in reasons)


async def test_list_and_register_drive(college):
    college.get("/v1/placements/drives").respond(200, json=[DRIVE])
    college.get("/v1/me/academics").respond(200, json={**ACAD, "user_id": 28})
    college.get("/v1/me/placements").respond(200, json=[])
    register = college.post("/v1/placements/drives/7/register")
    body, err = await run("list_placement_drives", {})
    assert not err and body["drives"][0]["eligible"] is True
    body, err = await run("register_for_placement_drive", {"drive_id": 7})
    assert body["status"] == "needs_confirmation" and "Only one seat" in body["warnings"][0]
    assert register.call_count == 0
    body, err = await run("register_for_placement_drive", {"drive_id": 99})
    assert err


# ---------------------------------------------------------------- admissions

COURSES = [{"id": 7, "code": "BTECH-CSE", "name": "B.Tech Computer Science and Engineering",
            "department": "CSE", "min_12th_percent": 75.0, "annual_fee": 125000.0,
            "hostel_fee": 60000.0, "seats": 120}]
VISITOR = SessionCtx("vid", "visitor:abc", "visitor", None, None)
APP = {"full_name": "Asha R", "phone": "98765 43210", "email": "asha@example.com",
       "course_code": "btech-cse", "twelfth_percent": 82.5}


@pytest.mark.parametrize("change,needle", [
    ({"email": "asha at example dot com"}, "not a valid email"),
    ({"phone": "12345"}, "phone"),
    ({"twelfth_percent": 120}, "between 0 and 100"),
    ({"course_code": "BTECH-AERO"}, "Unknown course_code"),
    ({"full_name": " "}, "full_name"),
])
async def test_admission_validation(college, change, needle):
    college.get("/v1/courses").respond(200, json=COURSES)
    body, err = await run("submit_admission_application", {**APP, **change}, VISITOR)
    assert err and needle in body["error"]


async def test_admission_normalises_and_warns(college):
    college.get("/v1/courses").respond(200, json=COURSES)
    submit = college.post("/v1/admissions/applications")
    body, err = await run("submit_admission_application", {**APP, "twelfth_percent": 70},
                          VISITOR)
    assert not err and body["status"] == "needs_confirmation"
    assert "below the minimum 75%" in body["warnings"][0]
    assert "BTECH-CSE" in body["summary"] and "9876543210" in body["summary"]
    assert submit.call_count == 0


# ---------------------------------------------------------------- provenance of handbook rules

def test_cor4_handbook_reasons_cite_their_rule_and_drive_criteria_do_not():
    placed = {**ACAD, "placed_company": "Google", "placed_category": "dream"}
    _, reasons = T.drive_eligibility(DRIVE, USER, placed, set(), TODAY)
    assert any("per Handbook 2026 rules 10.3.2 and 10.3.3" in r for r in reasons)
    pg = {**USER, "programme": "PG", "year_of_study": 4}
    _, reasons = T.drive_eligibility({**DRIVE, "allow_standing_arrears": True, "min_cgpa": 6.0},
                                     pg, {**ACAD, "cgpa": 6.5, "standing_arrears": 1}, set(), TODAY)
    assert any("per Handbook 2026 rule 11.1.2" in r for r in reasons)
    assert any("per Handbook 2026 rule 11.2.1" in r for r in reasons)
    drive_only = {**DRIVE, "min_cgpa": 9.5, "allowed_departments": ["ECE"], "eligible_years": [3],
                  "deadline": "2026-11-01", "seats_left": 0}
    _, reasons = T.drive_eligibility(drive_only, USER, {**ACAD, "standing_arrears": 2}, {7}, TODAY)
    assert len(reasons) == 7 and not any("Handbook" in r for r in reasons)


async def test_cor4_drive_list_tells_model_circulars_override(college):
    college.get("/v1/placements/drives").respond(200, json=[DRIVE])
    college.get("/v1/me/academics").respond(200, json={**ACAD, "user_id": 28})
    college.get("/v1/me/placements").respond(200, json=[])
    body, err = await run("list_placement_drives", {})
    assert not err and "circular" in body["note"] and "out of date" in body["note"]


async def test_cor3_attendance_warning_cites_handbook_and_circular_note(college):
    college.get("/v1/calendar").respond(200, json=[])
    body, _ = await run("submit_leave_request", {
        "type": "medical_leave", "from_date": "2026-11-24", "to_date": "2026-11-24",
        "half_day": False, "reason": "Fever"})
    assert any("Below 75%" in w and "per Handbook 2026 rule 4.1.1" in w for w in body["warnings"])
    assert "circular" in body["note"]
    body, _ = await run("submit_leave_request", {
        "type": "medical_leave", "from_date": "2026-11-25", "to_date": "2026-11-25",
        "half_day": False, "reason": "Fever"}, student({**USER, "programme": "PG"}))
    assert any("Below 80%" in w and "per Handbook 2026 rule 5.1.1" in w for w in body["warnings"])
