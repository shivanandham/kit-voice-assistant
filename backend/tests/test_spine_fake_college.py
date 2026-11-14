"""The fake college itself: strict enough that other suites can prove AC2 (own token, own data)
and REL2 (nothing reaches a real service). Exercised through the real app.kit.client.KitClient."""

import httpx
import pytest
from app.kit.client import KitClient, KitError

from tests.fakes.college import (
    PASSWORD,
    STUDENTS,
    TEACHERS,
    USERS,
    error_body,
    user_summary,
)

STUDENT_GETS = ["academics", "subjects", "timetable", "calendar", "attendance",
                "attendance_sessions", "my_requests", "my_placements"]
TEACHER_GETS = ["teacher_classes", "teacher_requests"]
PUBLIC_GETS = ["today", "college", "courses", "documents", "placement_drives"]


async def test_rel2_fake_answers_public_endpoints_without_token(fake_college):
    for name in PUBLIC_GETS:
        assert await getattr(KitClient(), name)() is not None
    assert all(c.token is None for c in fake_college.log)
    assert all(c.status == 200 for c in fake_college.log)


async def test_rel2_health_needs_no_key(fake_college):
    assert (await KitClient().health())["status"] == "ok"


@pytest.mark.parametrize("login", list(USERS))
async def test_ac2_login_returns_token_mapping_to_exactly_that_user(fake_college, login):
    data = await KitClient().login(login, PASSWORD)
    assert data["user"] == user_summary(login)
    assert data["token_type"] == "bearer" and data["expires_in"] > 0
    assert fake_college.tokens[data["access_token"]][0] == login
    me = await KitClient(data["access_token"]).me()
    assert me["login_id"] == login
    assert fake_college.log[-1].login == login and fake_college.log[-1].token == data["access_token"]


async def test_ac2_tokens_are_unique_per_login(fake_college):
    a = (await KitClient().login("arjun", PASSWORD))["access_token"]
    b = (await KitClient().login("arjun", PASSWORD))["access_token"]
    assert a != b


@pytest.mark.parametrize("login_id,password", [
    ("arjun", "wrong"), ("nobody", PASSWORD), ("ARJUN", PASSWORD), ("arjun", ""),
    ("arjun' OR '1'='1", PASSWORD),
])
async def test_ac2_bad_credentials_are_401_with_college_message(fake_college, login_id, password):
    with pytest.raises(KitError) as e:
        await KitClient().login(login_id, password)
    assert e.value.status in (401, 422)
    if e.value.status == 401:
        assert e.value.message == "Invalid login ID or password."


@pytest.mark.parametrize("token", [None, "", "garbage", "Bearer", "fake-jwt.arjun.0000"])
async def test_ac2_missing_or_invalid_token_is_401(fake_college, token):
    with pytest.raises(KitError) as e:
        await KitClient(token).me()
    assert e.value.status == 401 and e.value.code == "unauthorized"


async def test_ac6_expired_token_is_401(fake_college):
    ctx = fake_college.ctx("arjun")
    fake_college.expire_token(ctx.kit_token)
    with pytest.raises(KitError) as e:
        await KitClient(ctx.kit_token).me()
    assert e.value.status == 401
    assert fake_college.log[-1].login is None


@pytest.mark.parametrize("key,message", [
    ("", "Missing required header: X-Participant-Key."),
    ("wrong-key", "Invalid X-Participant-Key provided."),
])
async def test_ac5_participant_key_checked(fake_college, monkeypatch, key, message):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "kit_participant_key", key)
    with pytest.raises(KitError) as e:
        await KitClient().college()
    assert e.value.status == 401 and e.value.message == message


@pytest.mark.parametrize("method", STUDENT_GETS)
@pytest.mark.parametrize("login", STUDENTS)
async def test_ac2_student_data_is_per_user(fake_college, login, method):
    ctx = fake_college.ctx(login)
    result = await getattr(KitClient(ctx.kit_token), method)()
    assert result is not None
    assert fake_college.log[-1].login == login
    if method == "academics":
        assert result["user_id"] == USERS[login]["id"]
    if method == "my_requests":
        assert all(r["student_id"] == USERS[login]["id"] for r in result)
    if method == "subjects":
        assert {s["class_id"] for s in result} == {USERS[login]["class_id"]}


@pytest.mark.parametrize("method", STUDENT_GETS)
@pytest.mark.parametrize("login", TEACHERS)
async def test_ac4_teacher_refused_on_student_endpoints(fake_college, login, method):
    with pytest.raises(KitError) as e:
        await getattr(KitClient(fake_college.ctx(login).kit_token), method)()
    assert e.value.status == 403 and e.value.message == "Only students can access this endpoint."


@pytest.mark.parametrize("method", TEACHER_GETS)
@pytest.mark.parametrize("login", STUDENTS)
async def test_ac4_student_refused_on_teacher_endpoints(fake_college, login, method):
    with pytest.raises(KitError) as e:
        await getattr(KitClient(fake_college.ctx(login).kit_token), method)()
    assert e.value.status == 403 and e.value.message == "Only teachers can access this endpoint."


async def test_ac2_attendance_differs_between_students(fake_college):
    a = await KitClient(fake_college.ctx("arjun").kit_token).attendance()
    p = await KitClient(fake_college.ctx("priya").kit_token).attendance()
    assert a != p and a["as_of_date"] == "2026-11-21"


async def test_ac9_teachers_have_different_authorised_classes(fake_college):
    lak = await KitClient(fake_college.ctx("lakshmi").kit_token).teacher_classes()
    sen = await KitClient(fake_college.ctx("senthil").kit_token).teacher_classes()
    assert {c["class_code"] for c in lak} == {"CSE-2A", "CSE-4A"}
    assert {c["class_code"] for c in sen} == {"ECE-2B", "MECH-3A"}
    delegated = [c for c in sen if c["role_type"] == "delegated"]
    assert delegated == [c for c in sen if c["valid_until"] == "2026-11-30"]


async def test_ac9_expired_delegation_not_honoured(fake_college):
    t = KitClient(fake_college.ctx("senthil").kit_token)
    with pytest.raises(KitError) as e:
        await t.class_attendance(11)
    assert e.value.status == 403
    reqs = await t.teacher_requests()
    assert {r["class_code"] for r in reqs} <= {"ECE-2B", "MECH-3A"}
    with pytest.raises(KitError) as e:
        await t.approve_request(12)  # arjun, CSE-2A
    assert e.value.status == 403
    assert fake_college.requests[12]["status"] == "pending"


async def test_ac9_class_attendance_authorised_and_unknown(fake_college):
    t = KitClient(fake_college.ctx("lakshmi").kit_token)
    data = await t.class_attendance(11)
    assert {s["student_login"] for s in data["students"]} == {"arjun", "priya", "rahul"}
    with pytest.raises(KitError) as e:
        await t.class_attendance(999)
    assert e.value.status == 404


async def test_teacher_request_status_filter(fake_college):
    t = KitClient(fake_college.ctx("lakshmi").kit_token)
    assert all(r["status"] == "pending" for r in await t.teacher_requests("pending"))
    assert {r["id"] for r in await t.teacher_requests("approved")} == {15}


async def test_approve_reject_state_machine(fake_college):
    t = KitClient(fake_college.ctx("lakshmi").kit_token)
    assert (await t.approve_request(12, "ok"))["status"] == "approved"
    with pytest.raises(KitError) as e:
        await t.approve_request(12)
    assert e.value.status == 409
    with pytest.raises(KitError) as e:
        await t.reject_request(19, "")
    assert e.value.status == 422 and e.value.field == "reason"
    assert (await t.reject_request(19, "no"))["decision_note"] == "no"
    with pytest.raises(KitError) as e:
        await t.approve_request(424242)
    assert e.value.status == 404


async def test_ac2_student_cannot_read_or_cancel_anothers_request(fake_college):
    a = KitClient(fake_college.ctx("arjun").kit_token)
    for call in (a.get_request(15), a.cancel_request(15)):
        with pytest.raises(KitError) as e:
            await call
        assert e.value.status == 404
    assert fake_college.requests[15]["status"] == "approved"
    assert (await a.get_request(12))["id"] == 12


async def test_cancel_rules(fake_college):
    p = KitClient(fake_college.ctx("priya").kit_token)
    assert (await p.cancel_request(19))["status"] == "cancelled"
    with pytest.raises(KitError) as e:
        await p.cancel_request(19)
    assert e.value.status == 409
    r = KitClient(fake_college.ctx("rahul").kit_token)
    with pytest.raises(KitError):
        await r.cancel_request(16)  # rejected


LEAVE = {"type": "personal_leave", "from_date": "2026-11-30", "to_date": "2026-11-30",
         "reason": "Family"}


async def test_cor7_create_request_idempotent(fake_college):
    c = KitClient(fake_college.ctx("arjun").kit_token)
    first = await c.create_request(LEAVE, "key-1")
    again = await c.create_request(LEAVE, "key-1")
    assert first == again
    assert len([r for r in fake_college.requests.values() if r["reason"] == "Family"]) == 1
    with pytest.raises(KitError) as e:
        await c.create_request({**LEAVE, "reason": "Other"}, "key-1")
    assert e.value.status == 409 and e.value.code == "idempotency_conflict"
    other = await c.create_request(LEAVE, "key-2")
    assert other["id"] != first["id"]
    assert fake_college.calls("POST", "/v1/me/requests")[0].idempotency_key == "key-1"


async def test_cor7_idempotency_keys_scoped_per_user(fake_college):
    a = await KitClient(fake_college.ctx("arjun").kit_token).create_request(LEAVE, "same")
    p = await KitClient(fake_college.ctx("priya").kit_token).create_request(LEAVE, "same")
    assert a["student_id"] != p["student_id"]


@pytest.mark.parametrize("patch,field", [
    ({"type": "vacation"}, "type"),
    ({"to_date": "2026-11-29"}, "to_date"),
    ({"type": "on_duty"}, "event_name"),
    ({"from_date": "30/11/2026"}, "from_date"),
    ({"reason": ""}, "reason"),
])
async def test_cor3_create_request_validation(fake_college, patch, field):
    c = KitClient(fake_college.ctx("arjun").kit_token)
    with pytest.raises(KitError) as e:
        await c.create_request({**LEAVE, **patch}, "k")
    assert e.value.status == 422 and e.value.field == field


APPLICATION = {"full_name": "Asha", "phone": "9876543210", "email": "a@example.com",
               "course_code": "BTECH-CSE", "twelfth_percent": 91.0}


async def test_cor7_admission_idempotent_and_validated(fake_college):
    c = KitClient()
    a = await c.submit_admission(APPLICATION, "adm-1")
    assert a == await c.submit_admission(APPLICATION, "adm-1")
    assert len(fake_college.applications) == 1
    with pytest.raises(KitError) as e:
        await c.submit_admission({**APPLICATION, "course_code": "NOPE"}, "adm-2")
    assert e.value.field == "course_code"


async def test_cor4_placement_registration_rules(fake_college):
    k = KitClient(fake_college.ctx("karthik").kit_token)
    reg = await k.register_drive(7)
    assert reg["company"] == "Zoho"
    assert next(d for d in fake_college.drives if d["id"] == 7)["seats_left"] == 0
    with pytest.raises(KitError) as e:
        await k.register_drive(7)
    assert e.value.code == "already_registered"
    assert [r["drive_id"] for r in await k.my_placements()] == [7]
    with pytest.raises(KitError) as e:
        await KitClient(fake_college.ctx("arjun").kit_token).register_drive(8)
    assert e.value.code == "not_eligible" and "year" in e.value.message
    with pytest.raises(KitError) as e:
        await k.register_drive(1234)
    assert e.value.status == 404


async def test_cor4_no_seats(fake_college):
    fake_college.drives[0]["seats_left"] = 0
    with pytest.raises(KitError) as e:
        await KitClient(fake_college.ctx("karthik").kit_token).register_drive(9)
    assert e.value.code == "no_seats"


async def test_calendar_and_sessions_filters(fake_college):
    a = KitClient(fake_college.ctx("arjun").kit_token)
    cal = await a.calendar("2026-11-23", "2026-11-28")
    assert [e["date"] for e in cal] == ["2026-11-23", "2026-11-27", "2026-11-28"]  # no ECE exam
    sessions = await a.attendance_sessions(subject="CS304", from_date="2026-11-10")
    assert sessions and all(s["subject_code"] == "CS304" and s["date"] >= "2026-11-10"
                            for s in sessions)
    tt = await a.timetable()
    assert {s["weekday"] for s in tt} == set(range(1, 7))


async def test_documents_and_files(fake_college):
    docs = await KitClient().documents()
    assert docs[0]["download_url"] == "/v1/documents/1/file"
    assert (await KitClient().document_file(1)).startswith(b"%PDF")
    with pytest.raises(KitError) as e:
        await KitClient().document_file(99)
    assert e.value.status == 404


async def test_today_is_the_college_date(fake_college):
    t = await KitClient().today()
    assert t == {"date": "2026-11-21", "weekday": 6, "weekday_name": "Saturday",
                 "timezone": "Asia/Kolkata"}


async def test_errors_use_standard_shape(fake_college):
    async with httpx.AsyncClient(base_url=fake_college.base_url) as c:
        r = await c.get("/v1/me", headers={"X-Participant-Key": fake_college.participant_key})
        assert r.json() == error_body("unauthorized", "Missing Authorization header.")
        r = await c.get("/v1/nope", headers={"X-Participant-Key": fake_college.participant_key})
        assert r.status_code == 404 and set(r.json()["error"]) == {"code", "message", "field"}
        r = await c.delete("/v1/me", headers={"X-Participant-Key": fake_college.participant_key})
        assert r.status_code == 405
        r = await c.post("/v1/auth/login", content=b"not json",
                         headers={"X-Participant-Key": fake_college.participant_key})
        assert r.status_code == 422


async def test_rel1_fault_injection(fake_college):
    fake_college.fail("/v1/college", 500, times=1)
    assert (await KitClient().college())["code"] == "KIT"  # GET retried after one 500
    fake_college.fail("/v1/courses", exc=httpx.ConnectError)
    with pytest.raises(KitError) as e:
        await KitClient().courses()
    assert e.value.status == 0
    fake_college.clear_faults()
    fake_college.down = True
    with pytest.raises(KitError):
        await KitClient().college()


async def test_recording_helpers(fake_college):
    ctx = fake_college.ctx("arjun")
    await KitClient(ctx.kit_token).me()
    await KitClient().college()
    assert fake_college.identities() == {"arjun", None}
    assert fake_college.writes() == []
    await KitClient(ctx.kit_token).create_request(LEAVE, "x")
    assert [c.path for c in fake_college.writes()] == ["/v1/me/requests"]
    assert fake_college.calls(path="/v1/me")[0].login == "arjun"
    assert "calls" in repr(fake_college)


async def test_rel2_unmocked_host_is_refused(fake_college):
    with pytest.raises((httpx.HTTPError, AssertionError)):  # respx: AllMockedAssertionError
        async with httpx.AsyncClient() as c:
            await c.get("https://api.anthropic.com/v1/messages")


def test_ctx_factory_and_visitor(fake_college):
    v = fake_college.ctx(None)
    assert v.role == "visitor" and v.user is None and v.kit_token is None
    assert v.owner_key.startswith("visitor:")
    t = fake_college.ctx("senthil")
    assert t.role == "teacher" and t.owner_key == "user:senthil" and t.user["class_code"] == "ECE-2B"


async def test_db_session_helper(fake_college):
    from app import auth

    token, _ = await fake_college.db_session("priya")
    assert (await auth._lookup(token)).owner_key == "user:priya"
    vtoken, _ = await fake_college.db_session(None)
    assert (await auth._lookup(vtoken)).role == "visitor"
