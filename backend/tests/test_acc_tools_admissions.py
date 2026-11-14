"""COR5: admission applications are validated (name, phone, email, course code, percentage) and warn
when the percentage is below the course minimum. Visitors only; the write goes through the gate.
"""

import pytest

from tests.acc_tools_helpers import (
    ADMISSION,
    mentions,
    new_conv,
    pending_count,
    propose,
    run,
    user_says,
    writes,
)


def app_(**kw):
    return {**ADMISSION, **kw}


async def submit(fake, args):
    ctx = fake.ctx(None)
    cid = await new_conv(ctx)
    r, b = await run("submit_admission_application", args, ctx, cid)
    return ctx, cid, r, b


async def _assert_rejected(fake, args):
    ctx, cid, r, b = await submit(fake, args)
    assert r.is_error, b
    assert await pending_count() == 0
    assert writes(fake) == []
    return r, b


async def _assert_accepted(fake, args):
    ctx, cid, r, b = await submit(fake, args)
    assert not r.is_error, b
    assert b["status"] == "needs_confirmation"
    return r, b


@pytest.mark.parametrize("name", ["Ravi Kumar", "A. R. Rahman", "Sneha", "Mary-Jane O'Neil",
                                  "  Ravi   Kumar  ", "முருகன்", "Priya Lakshmi Narayanan"])
async def test_cor5_valid_names(name, fake_college):
    await _assert_accepted(fake_college, app_(full_name=name))


# DEFECT: names made only of digits or symbols ("12345", "@@@") are accepted (tools.py submit_admission_application).
@pytest.mark.parametrize("name", ["", " ", "\t\n", "12345", "@@@", 42, None])
async def test_cor5_invalid_names(name, fake_college):
    await _assert_rejected(fake_college, app_(full_name=name))


@pytest.mark.parametrize("phone", ["9876543210", "+919876543210", "98765 43210",
                                   "+91 98765-43210", "(987) 654-3210"])
async def test_cor5_valid_phones(phone, fake_college):
    await _assert_accepted(fake_college, app_(phone=phone))


@pytest.mark.parametrize("phone", ["", "12345", "abcdefghij", "98765432101234",
                                   "987654321", "+91", "9876543210x", "98765abcde",
                                   "++919876543210", 9876543210])
async def test_cor5_invalid_phones(phone, fake_college):
    await _assert_rejected(fake_college, app_(phone=phone))


@pytest.mark.parametrize("email", ["a@b.co", "first.last@example.com", "x+tag@kit.ac.in",
                                   " ravi@example.com "])
async def test_cor5_valid_emails(email, fake_college):
    await _assert_accepted(fake_college, app_(email=email))


@pytest.mark.parametrize("email", ["", "plainaddress", "a@b", "@example.com", "a b@example.com",
                                   "a@@b.com", "ravi@", "ravi at example dot com",
                                   "ravi@example.", "ravi@.com", 5])
async def test_cor5_invalid_emails(email, fake_college):
    await _assert_rejected(fake_college, app_(email=email))


@pytest.mark.parametrize("code", ["BTECH-CSE", "BTECH-ECE", "BTECH-MECH"])
async def test_cor5_valid_course_codes(code, fake_college):
    r, b = await _assert_accepted(fake_college, app_(course_code=code, twelfth_percent=90))
    assert code in b["summary"]


@pytest.mark.parametrize("code", ["", "MBA", "BTECH CSE", "BTECH-IT", "CSE", "BTECH-CSE; DROP",
                                  "MTECH-CSE", 7])
async def test_cor5_invalid_course_codes(code, fake_college):
    await _assert_rejected(fake_college, app_(course_code=code))


@pytest.mark.parametrize("pct", [-1, -0.01, 100.01, 101, 1000, "eighty", "85", True,
                                 float("nan"), float("inf"), float("-inf"), None])
async def test_cor5_invalid_percentages(pct, fake_college):
    await _assert_rejected(fake_college, app_(twelfth_percent=pct))


@pytest.mark.parametrize("pct", [100, 99.99, 88, 75.5])
async def test_cor5_valid_percentages(pct, fake_college):
    await _assert_accepted(fake_college, app_(twelfth_percent=pct))


@pytest.mark.parametrize("code,minimum", [("BTECH-CSE", 75.0), ("BTECH-ECE", 70.0),
                                          ("BTECH-MECH", 60.0)])
async def test_cor5_exactly_course_minimum_no_warning(code, minimum, fake_college):
    r, b = await _assert_accepted(fake_college, app_(course_code=code, twelfth_percent=minimum))
    assert not any(mentions(w, "minimum", "below") for w in b["warnings"])


@pytest.mark.parametrize("code,pct", [("BTECH-CSE", 74.99), ("BTECH-CSE", 74), ("BTECH-ECE", 69.9),
                                      ("BTECH-MECH", 59), ("BTECH-CSE", 1)])
async def test_cor5_below_minimum_warns_not_rejects(code, pct, fake_college):
    r, b = await _assert_accepted(fake_college, app_(course_code=code, twelfth_percent=pct))
    assert any(mentions(w, "minimum", "below") for w in b["warnings"])


async def test_cor5_above_minimum_no_warning(fake_college):
    r, b = await _assert_accepted(fake_college, app_(twelfth_percent=95))
    assert b["warnings"] == []


async def test_cor5_summary_reads_back_email_and_phone(fake_college):
    r, b = await _assert_accepted(fake_college, ADMISSION)
    assert "ravi@example.com" in b["summary"]
    assert "9876543210" in b["summary"]


async def test_cor5_missing_fields_rejected(fake_college):
    for k in ADMISSION:
        args = {x: v for x, v in ADMISSION.items() if x != k}
        await _assert_rejected(fake_college, args)


async def test_cor5_extra_fields_rejected(fake_college):
    await _assert_rejected(fake_college, app_(status="approved"))


async def test_cor5_confirmed_application_sent_once_with_key(fake_college):
    ctx, cid, aid = await propose(fake_college, "submit_admission_application")
    await user_says(cid)
    r, b = await run("confirm_pending_action", {"action_id": aid}, ctx, cid)
    assert not r.is_error, b
    w = writes(fake_college)
    assert len(w) == 1 and w[0].idempotency_key and w[0].token is None
    assert len(fake_college.applications) == 1


async def test_cor5_courses_listed_for_visitor(fake_college):
    ctx = fake_college.ctx(None)
    r, b = await run("list_courses", {}, ctx, await new_conv(ctx))
    assert not r.is_error
    for code in ("BTECH-CSE", "BTECH-ECE", "BTECH-MECH"):
        assert code in r.content


async def test_cor5_college_info_for_visitor(fake_college):
    ctx = fake_college.ctx(None)
    r, b = await run("get_college_info", {}, ctx, await new_conv(ctx))
    assert "Knotopian" in r.content

