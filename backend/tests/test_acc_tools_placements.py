"""COR4: placement drives are marked eligible or not, with reasons.

Reasons: CGPA, standing arrears, department, year, deadline, seats left, already registered,
already placed. The baseline student is karthik (CSE-4A, B.Tech, year 4, CGPA 8.6, no arrears,
not placed); the college date is 2026-11-21.
"""

import itertools

import pytest

from tests.acc_tools_helpers import mentions, new_conv, pending_count, run, user_says, writes
from tests.fakes import college as fc

OPEN = {"id": 50, "company": "Acme", "category": "non_dream", "min_cgpa": 7.0,
        "allow_standing_arrears": True, "allowed_departments": ["CSE", "ECE"],
        "eligible_years": [4], "deadline": "2026-11-30", "seats_total": 10, "seats_left": 5}

KEYWORDS = {
    "cgpa": ("cgpa",),
    "arrears": ("arrear",),
    "department": ("department", "MECH only", "you are CSE"),
    "year": ("year",),
    "deadline": ("deadline",),
    "seats": ("seat",),
    "registered": ("registered",),
    "placed": ("placed",),
}

# reason -> (drive overrides, academics overrides (cgpa, arrears, placed_company), registered?)
REASONS = {
    "cgpa": ({"min_cgpa": 9.0}, None, False),
    "arrears": ({"allow_standing_arrears": False}, (8.6, 1, None), False),
    "department": ({"allowed_departments": ["MECH"]}, None, False),
    "year": ({"eligible_years": [3]}, None, False),
    "deadline": ({"deadline": "2026-11-20"}, None, False),
    "seats": ({"seats_left": 0}, None, False),
    "registered": ({}, None, True),
    "placed": ({}, (8.6, 0, "TCS"), False),
}


def _scenario(fake, monkeypatch, reasons):
    drive = dict(OPEN)
    acad = (8.6, 0, None)
    registered = False
    for r in reasons:
        d_over, a_over, reg = REASONS[r]
        drive.update(d_over)
        if a_over:
            acad = (acad[0], max(acad[1], a_over[1]), a_over[2] or acad[2])
        registered = registered or reg
    fake.drives = [drive]
    monkeypatch.setitem(fc.ACADEMICS, "karthik", (acad[0], acad[1], acad[2],
                                                  "non_dream" if acad[2] else None))
    if registered:
        fake.registrations.append({"id": 999, "drive_id": drive["id"], "company": drive["company"],
                                   "student_id": fc.USERS["karthik"]["id"],
                                   "registered_at": "2026-11-20T10:00:00+05:30"})
    return drive


async def _list(fake, login="karthik"):
    ctx = fake.ctx(login)
    r, b = await run("list_placement_drives", {}, ctx, await new_conv(ctx))
    assert not r.is_error, b
    return b


def _drive(b, company="Acme"):
    return next(d for d in b["drives"] if d["company"] == company)


def _matches(reason_text, category):
    return mentions(reason_text, *KEYWORDS[category])


async def test_cor4_baseline_drive_is_eligible(fake_college, monkeypatch):
    _scenario(fake_college, monkeypatch, [])
    d = _drive(await _list(fake_college))
    assert d["eligible"] is True
    assert d["reasons"] == []


@pytest.mark.parametrize("reason", list(REASONS))
async def test_cor4_single_reason(reason, fake_college, monkeypatch):
    _scenario(fake_college, monkeypatch, [reason])
    d = _drive(await _list(fake_college))
    assert d["eligible"] is False
    assert len(d["reasons"]) == 1, d["reasons"]
    assert _matches(d["reasons"][0], reason), d["reasons"]


@pytest.mark.parametrize("combo", list(itertools.combinations(
    ["cgpa", "department", "year", "deadline", "seats", "registered"], 2)))
async def test_cor4_pairs_of_reasons(combo, fake_college, monkeypatch):
    _scenario(fake_college, monkeypatch, list(combo))
    d = _drive(await _list(fake_college))
    assert d["eligible"] is False
    for c in combo:
        assert any(_matches(x, c) for x in d["reasons"]), (c, d["reasons"])


async def test_cor4_all_reasons_at_once(fake_college, monkeypatch):
    _scenario(fake_college, monkeypatch, list(REASONS))
    d = _drive(await _list(fake_college))
    assert d["eligible"] is False
    for c in REASONS:
        assert any(_matches(x, c) for x in d["reasons"]), (c, d["reasons"])


# ------------------------------------------------------------------ boundaries


@pytest.mark.parametrize("min_cgpa,eligible", [(8.6, True), (8.59, True), (8.61, False),
                                               (0.0, True), (10.0, False)])
async def test_cor4_cgpa_boundary(min_cgpa, eligible, fake_college, monkeypatch):
    _scenario(fake_college, monkeypatch, [])
    fake_college.drives[0]["min_cgpa"] = min_cgpa
    d = _drive(await _list(fake_college))
    assert d["eligible"] is eligible, d["reasons"]


@pytest.mark.parametrize("deadline,eligible", [("2026-11-21", True), ("2026-11-22", True),
                                               ("2026-11-20", False), ("2025-11-21", False)])
async def test_cor4_deadline_boundary(deadline, eligible, fake_college, monkeypatch):
    _scenario(fake_college, monkeypatch, [])
    fake_college.drives[0]["deadline"] = deadline
    d = _drive(await _list(fake_college))
    assert d["eligible"] is eligible, d["reasons"]


@pytest.mark.parametrize("seats,eligible", [(1, True), (0, False), (2, True), (-1, False)])
async def test_cor4_seats_boundary(seats, eligible, fake_college, monkeypatch):
    _scenario(fake_college, monkeypatch, [])
    fake_college.drives[0]["seats_left"] = seats
    d = _drive(await _list(fake_college))
    assert d["eligible"] is eligible, d["reasons"]


@pytest.mark.parametrize("arrears,allowed,eligible", [(0, False, True), (1, False, False),
                                                      (3, False, False), (2, True, True)])
async def test_cor4_arrears(arrears, allowed, eligible, fake_college, monkeypatch):
    _scenario(fake_college, monkeypatch, [])
    fake_college.drives[0]["allow_standing_arrears"] = allowed
    monkeypatch.setitem(fc.ACADEMICS, "karthik", (8.6, arrears, None, None))
    d = _drive(await _list(fake_college))
    assert d["eligible"] is eligible, d["reasons"]


async def test_cor4_placed_student_non_dream_ineligible(fake_college, monkeypatch):
    _scenario(fake_college, monkeypatch, ["placed"])
    d = _drive(await _list(fake_college))
    assert d["eligible"] is False


# ------------------------------------------------------------------ real fake data, per student


async def test_cor4_karthik_seeded_drives(fake_college):
    b = await _list(fake_college)
    by = {d["company"]: d for d in b["drives"]}
    assert by["Freshworks"]["eligible"] is True   # dream, 8.0, CSE, y4
    assert by["TCS"]["eligible"] is True
    assert by["Zoho"]["eligible"] is True         # one seat left is still a seat
    assert by["Zoho"]["seats_left"] == 1


@pytest.mark.parametrize("login", ["arjun", "priya", "rahul", "kavya", "meena"])
async def test_cor4_non_final_years_ineligible_with_year_reason(login, fake_college):
    b = await _list(fake_college, login)
    for d in b["drives"]:
        assert d["eligible"] is False
        assert any(_matches(x, "year") for x in d["reasons"])


async def test_cor4_department_reason_for_mech_on_zoho(fake_college):
    b = await _list(fake_college, "meena")
    z = _drive(b, "Zoho")
    assert any(_matches(x, "department") or "MECH" in x for x in z["reasons"])


async def test_cor4_reasons_use_own_academics_only(fake_college):
    b = await _list(fake_college, "rahul")
    text = str(b)
    assert "6.1" in text and "8.6" not in text.replace("min_cgpa", "")


async def test_cor4_registration_marks_registered_on_next_list(fake_college):
    ctx = fake_college.ctx("karthik")
    cid = await new_conv(ctx)
    await user_says(cid, "register me for TCS")
    _, b = await run("register_for_placement_drive", {"drive_id": 8}, ctx, cid)
    await user_says(cid)
    await run("confirm_pending_action", {"action_id": b["action_id"]}, ctx, cid)
    listed = await _list(fake_college)
    tcs = _drive(listed, "TCS")
    assert tcs["eligible"] is False
    assert any(_matches(x, "registered") for x in tcs["reasons"])


# ------------------------------------------------------------------ register tool


async def test_cor4_register_unknown_drive_rejected(fake_college):
    ctx = fake_college.ctx("karthik")
    r, b = await run("register_for_placement_drive", {"drive_id": 4242}, ctx, await new_conv(ctx))
    assert r.is_error
    assert await pending_count() == 0


@pytest.mark.parametrize("reason", list(REASONS))
async def test_cor4_register_ineligible_warns_before_confirmation(reason, fake_college,
                                                                  monkeypatch):
    drive = _scenario(fake_college, monkeypatch, [reason])
    ctx = fake_college.ctx("karthik")
    r, b = await run("register_for_placement_drive", {"drive_id": drive["id"]}, ctx,
                     await new_conv(ctx))
    if r.is_error:
        assert await pending_count() == 0
    else:
        assert b["warnings"], "an ineligible registration must be flagged before confirmation"
        assert any(_matches(w, reason) for w in b["warnings"])
    assert writes(fake_college) == []


async def test_cor4_register_one_seat_left_warns(fake_college):
    ctx = fake_college.ctx("karthik")
    r, b = await run("register_for_placement_drive", {"drive_id": 7}, ctx, await new_conv(ctx))
    assert not r.is_error
    assert any(mentions(w, "seat") for w in b["warnings"])
    assert writes(fake_college) == []


async def test_cor4_register_eligible_has_no_eligibility_warning(fake_college):
    ctx = fake_college.ctx("karthik")
    r, b = await run("register_for_placement_drive", {"drive_id": 8}, ctx, await new_conv(ctx))
    assert not r.is_error
    assert not any(mentions(w, "not be eligible", "ineligible") for w in b["warnings"])
    assert "TCS" in b["summary"]


@pytest.mark.parametrize("bad", ["8", 8.5, None, True, -1, 0])
async def test_cor4_register_bad_drive_id(bad, fake_college):
    ctx = fake_college.ctx("karthik")
    r, b = await run("register_for_placement_drive", {"drive_id": bad}, ctx, await new_conv(ctx))
    assert r.is_error
    assert await pending_count() == 0


async def test_cor4_list_my_placements_own_only(fake_college):
    fake_college.registrations += [
        {"id": 1, "drive_id": 8, "company": "TCS", "student_id": fc.USERS["karthik"]["id"],
         "registered_at": "2026-11-20T10:00:00+05:30"},
        {"id": 2, "drive_id": 9, "company": "Freshworks", "student_id": fc.USERS["arjun"]["id"],
         "registered_at": "2026-11-20T10:00:00+05:30"},
    ]
    ctx = fake_college.ctx("karthik")
    r, b = await run("list_my_placements", {}, ctx, await new_conv(ctx))
    assert "TCS" in r.content and "Freshworks" not in r.content
