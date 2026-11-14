"""In-memory fake of the KIT college API (https://ht.knowvally.com/openapi.json), served via respx.

Response shapes are copied from live reads of the sandbox. Use it through the fixtures that
``tests/conftest.py`` provides:

``fake_college``
    A :class:`FakeCollege` mounted on ``settings.kit_base_url`` with respx. Every httpx request to
    that host is answered in memory; any other httpx request fails (respx ``assert_all_mocked``),
    and the conftest network guard fails the test on any real non-local socket.

``app_client``
    An ``httpx.AsyncClient`` over the real FastAPI app (lifespan run) with the college faked.
    ``browser`` is a factory for more clients (separate cookie jars) on the same app.

Typical use::

    async def test_x(app_client, fake_college):
        await login_as(app_client, "arjun")              # real /api/login against the fake
        r = await app_client.get("/api/me")
        assert fake_college.calls("POST", "/v1/auth/login")[-1].body["login_id"] == "arjun"

    async def test_tool(fake_college):
        ctx = fake_college.ctx("priya")                   # SessionCtx with a valid fake token
        ...                                               # call code that uses KitClient(ctx.kit_token)
        assert fake_college.identities() == {"priya"}     # every call carried priya's token

    token, ctx = await fake_college.db_session("lakshmi")  # stored session; cookie value `token`

Population (college date ``2026-11-21``, a Saturday; password ``Kit@2026`` for everyone):

* students ``arjun``, ``priya``, ``rahul`` in CSE-2A (class 11); ``kavya`` in ECE-2B (12);
  ``meena`` in MECH-3A (13); ``karthik`` in CSE-4A (14, final year, placement-eligible).
* teacher ``lakshmi``: advisor of CSE-2A and CSE-4A.
* teacher ``senthil``: advisor of ECE-2B, delegated MECH-3A ``valid_until`` 2026-11-30, and an
  expired delegation for CSE-2A (``valid_until`` 2026-11-20) that the college must not honour.
* a visitor is simply a request with no token: public endpoints only.

Behaviour: tokens map to exactly one user and expire (``expire_token``); a missing, unknown or
expired token or participant key is 401; role-restricted endpoints are 403 ``not_allowed``; bad
input is 422 ``invalid_input`` with ``field``; all errors use ``{"error": {code, message, field}}``.
POST /v1/me/requests and /v1/admissions/applications honour ``Idempotency-Key`` (same key and body
replays the first response, same key with another body is 409 ``idempotency_conflict``).

Every request is recorded as a :class:`Call` (method, path, params, token, resolved login,
participant key, idempotency key, body, status) in ``fake_college.log``; ``calls()``,
``identities()`` and ``writes()`` filter it. Faults: ``fail(...)`` answers matching requests with an
error or raises an httpx exception; ``down = True`` makes every request a connection error;
``login_overrides[login] = {...}`` replaces fields of that user's login answer (e.g. a strange role).
State is plain attributes (``requests``, ``drives``, ``grants``, ``today``, ``documents``,
``document_files``...) that a test may edit before acting.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import secrets
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import httpx

PASSWORD = "Kit@2026"
TODAY = date(2026, 11, 21)
DEFAULT_TOKEN_TTL = 28800

STUDENT_ONLY = "Only students can access this endpoint."
TEACHER_ONLY = "Only teachers can access this endpoint."

CLASSES: dict[int, dict] = {
    11: {"class_code": "CSE-2A", "department": "CSE", "programme": "UG", "year": 2},
    12: {"class_code": "ECE-2B", "department": "ECE", "programme": "UG", "year": 2},
    13: {"class_code": "MECH-3A", "department": "MECH", "programme": "UG", "year": 3},
    14: {"class_code": "CSE-4A", "department": "CSE", "programme": "UG", "year": 4},
}

SUBJECTS: dict[int, list[tuple[int, str, str]]] = {
    11: [
        (10, "CS301", "Data Structures and Algorithms"),
        (11, "CS302", "Digital Principles and Computer Organization"),
        (12, "CS303", "Object Oriented Programming"),
        (9, "CS304", "Database Management Systems"),
    ],
    12: [(20, "EC301", "Signals and Systems"), (21, "EC302", "Analog Circuits")],
    13: [(30, "ME401", "Thermodynamics"), (31, "ME402", "Fluid Mechanics")],
    14: [(40, "CS701", "Distributed Systems"), (41, "CS702", "Compiler Design")],
}

# login -> UserSummary (shape of /v1/me), plus private academics for students.
USERS: dict[str, dict] = {
    "arjun": {"id": 28, "full_name": "Arjun K", "role": "student", "class_id": 11, "year": 2},
    "priya": {"id": 29, "full_name": "Priya M", "role": "student", "class_id": 11, "year": 2},
    "rahul": {"id": 31, "full_name": "Rahul R", "role": "student", "class_id": 11, "year": 2},
    "kavya": {"id": 30, "full_name": "Kavya S", "role": "student", "class_id": 12, "year": 2},
    "meena": {"id": 33, "full_name": "Meena P", "role": "student", "class_id": 13, "year": 3},
    "karthik": {"id": 32, "full_name": "Karthik V", "role": "student", "class_id": 14, "year": 4},
    "lakshmi": {"id": 25, "full_name": "Dr. Lakshmi", "role": "teacher", "class_id": 11},
    "senthil": {"id": 26, "full_name": "Prof. Senthil", "role": "teacher", "class_id": 12},
}
STUDENTS = [k for k, v in USERS.items() if v["role"] == "student"]
TEACHERS = [k for k, v in USERS.items() if v["role"] == "teacher"]

ACADEMICS = {
    "arjun": (8.2, 0, None, None),
    "priya": (7.4, 0, None, None),
    "rahul": (6.1, 2, None, None),
    "kavya": (8.9, 0, None, None),
    "meena": (7.0, 1, None, None),
    "karthik": (8.6, 0, None, None),
}

# teacher -> [(class_id, role_type, valid_until)]
TEACHER_GRANTS: dict[str, list[tuple[int, str, str | None]]] = {
    "lakshmi": [(11, "advisor", None), (14, "advisor", None)],
    "senthil": [(12, "advisor", None), (13, "delegated", "2026-11-30"), (11, "delegated", "2026-11-20")],
}

COLLEGE = {
    "name": "Knotopian Institute of Technology",
    "code": "KIT",
    "tagline": "Excellence in Engineering and Applied Technology",
    "description": "Knotopian Institute of Technology (KIT) is an autonomous engineering institution.",
    "contact": {
        "phone": "+91 422 2619000",
        "email": "admissions@kit.ac.in",
        "website": "https://kit.ac.in",
        "address": "Knotopian Knowledge City, Coimbatore, Tamil Nadu 641042",
    },
}

COURSES = [
    {"id": 7, "code": "BTECH-CSE", "name": "B.Tech Computer Science and Engineering",
     "department": "CSE", "min_12th_percent": 75.0, "annual_fee": 125000.0,
     "hostel_fee": 60000.0, "seats": 120},
    {"id": 8, "code": "BTECH-ECE", "name": "B.Tech Electronics and Communication Engineering",
     "department": "ECE", "min_12th_percent": 70.0, "annual_fee": 110000.0,
     "hostel_fee": 60000.0, "seats": 60},
    {"id": 9, "code": "BTECH-MECH", "name": "B.Tech Mechanical Engineering", "department": "MECH",
     "min_12th_percent": 60.0, "annual_fee": 95000.0, "hostel_fee": 55000.0, "seats": 60},
]

DOCUMENTS = [
    {"id": 1, "title": "KIT Student Handbook 2025", "type": "handbook", "version": "2025.1",
     "effective_date": "2025-07-01", "applies_to": {"programmes": ["UG", "PG"], "years": [1, 2, 3, 4]},
     "is_surprise": False},
    {"id": 2, "title": "KIT Student Handbook 2026", "type": "handbook", "version": "2026.1",
     "effective_date": "2026-07-01", "applies_to": {"programmes": ["UG", "PG"], "years": [1, 2, 3, 4]},
     "is_surprise": False},
    {"id": 3, "title": "KIT Admissions Brochure 2026", "type": "brochure", "version": "2026.1",
     "effective_date": "2026-11-01", "applies_to": {"programmes": ["UG"], "intake": "2027-28"},
     "is_surprise": False},
    {"id": 4, "title": "DBMS Lesson Notes - Units 1 to 3", "type": "lesson_notes",
     "version": "2026.1", "effective_date": "2026-10-22",
     "applies_to": {"departments": ["CSE"], "semester": 3}, "is_surprise": False},
    {"id": 5, "title": "Circular 12/2026: On-duty proof within 3 working days (B.Tech)",
     "type": "circular", "version": "12/2026", "effective_date": "2026-11-01",
     "applies_to": {"programmes": ["UG"]}, "is_surprise": False},
]

DRIVES = [
    {"id": 9, "company": "Freshworks", "category": "dream", "min_cgpa": 8.0,
     "allow_standing_arrears": False, "allowed_departments": ["CSE"], "eligible_years": [4],
     "deadline": "2026-11-28", "seats_total": 20, "seats_left": 12},
    {"id": 8, "company": "TCS", "category": "non_dream", "min_cgpa": 6.0,
     "allow_standing_arrears": True, "allowed_departments": ["CSE", "ECE", "MECH"],
     "eligible_years": [4], "deadline": "2026-12-01", "seats_total": 100, "seats_left": 75},
    {"id": 7, "company": "Zoho", "category": "non_dream", "min_cgpa": 7.0,
     "allow_standing_arrears": False, "allowed_departments": ["CSE", "ECE"], "eligible_years": [4],
     "deadline": "2026-11-26", "seats_total": 10, "seats_left": 1},
]

CALENDAR = [
    {"id": 6, "date": "2026-11-22", "type": "holiday", "title": "Sunday Holiday",
     "subject_code": None, "follows_weekday": None, "status": "active"},
    {"id": 5, "date": "2026-11-23", "type": "internal_exam", "title": "DBMS Internal Assessment Exam",
     "subject_code": "CS304", "follows_weekday": None, "status": "active"},
    {"id": 7, "date": "2026-11-24", "type": "internal_exam", "title": "Signals Internal Exam",
     "subject_code": "EC301", "follows_weekday": None, "status": "active"},
    {"id": 8, "date": "2026-11-27", "type": "holiday", "title": "Founders Day",
     "subject_code": None, "follows_weekday": None, "status": "active"},
    {"id": 9, "date": "2026-11-28", "type": "working_day", "title": "Working Saturday (Monday order)",
     "subject_code": None, "follows_weekday": 1, "status": "active"},
]

_SEED_REQUESTS = [
    (12, "arjun", "on_duty", "2026-11-30", "2026-11-30",
     "Participating in State Level Technical Symposium", "National Technical Symposium 2026",
     "pending", None, None),
    (15, "priya", "personal_leave", "2026-11-25", "2026-11-25", "Family function in hometown",
     None, "approved", 25, "Approved."),
    (16, "rahul", "on_duty", "2026-11-19", "2026-11-19", "Attended hackathon event",
     "CodeFest 2026", "rejected", 25, "event proof missing"),
    (17, "kavya", "medical_leave", "2026-11-26", "2026-11-27", "Fever", None, "pending", None, None),
    (18, "meena", "personal_leave", "2026-11-30", "2026-12-01", "Sister's wedding", None,
     "pending", None, None),
    (19, "priya", "medical_leave", "2026-12-02", "2026-12-02", "Dental appointment", None,
     "pending", None, None),
]

REQUEST_TYPES = {"personal_leave", "medical_leave", "on_duty"}
_PERIODS = [("09:00", "09:50"), ("10:00", "10:50"), ("11:00", "11:50"), ("12:00", "12:50")]


def user_summary(login: str) -> dict:
    """The UserSummary the college returns for `login` (as in /v1/auth/login and /v1/me)."""
    u = USERS[login]
    c = CLASSES[u["class_id"]]
    teacher = u["role"] == "teacher"
    return {
        "id": u["id"],
        "login_id": login,
        "full_name": u["full_name"],
        "role": u["role"],
        "department": c["department"],
        "class_id": u["class_id"],
        "class_code": c["class_code"],
        "programme": c["programme"],
        "year_of_study": None if teacher else u["year"],
        "notes": f"Class Advisor for {c['class_code']}" if teacher else None,
    }


@dataclass
class Call:
    """One request the fake received."""

    method: str
    path: str
    params: dict[str, str]
    token: str | None  # bearer token as sent (None when absent)
    login: str | None  # user the token resolved to (None: none, unknown or expired)
    participant_key: str | None
    idempotency_key: str | None
    body: Any
    status: int = 0


class Fail:
    """A scripted fault for matching requests."""

    def __init__(self, method, path, status, code, message, exc, times):
        self.method, self.status, self.code, self.message = method, status, code, message
        self.path = re.compile(path if path.startswith("^") else "^" + re.escape(path) + "$")
        self.exc, self.times = exc, times

    def matches(self, method: str, path: str) -> bool:
        return (self.method in (None, method)) and bool(self.path.match(path)) and self.times != 0


class _Err(Exception):
    def __init__(self, status: int, code: str, message: str, field: str | None = None):
        self.status, self.code, self.message, self.field = status, code, message, field


def error_body(code: str, message: str, field: str | None = None) -> dict:
    """The college's standard error JSON."""
    return {"error": {"code": code, "message": message, "field": field}}


@dataclass
class FakeCollege:
    """Stateful in-memory college. Mutate the public attributes to shape a scenario."""

    base_url: str
    participant_key: str
    today: date = TODAY
    token_ttl: int = DEFAULT_TOKEN_TTL
    down: bool = False
    log: list[Call] = field(default_factory=list)
    tokens: dict[str, tuple[str, float]] = field(default_factory=dict)  # token -> (login, expiry)
    passwords: dict[str, str] = field(default_factory=lambda: {k: PASSWORD for k in USERS})
    college: dict = field(default_factory=lambda: copy.deepcopy(COLLEGE))
    courses: list = field(default_factory=lambda: copy.deepcopy(COURSES))
    documents: list = field(default_factory=lambda: copy.deepcopy(DOCUMENTS))
    document_files: dict[int, bytes] = field(default_factory=dict)
    drives: list = field(default_factory=lambda: copy.deepcopy(DRIVES))
    calendar_events: list = field(default_factory=lambda: copy.deepcopy(CALENDAR))
    grants: dict = field(default_factory=lambda: copy.deepcopy(TEACHER_GRANTS))
    requests: dict[int, dict] = field(default_factory=dict)
    registrations: list[dict] = field(default_factory=list)
    applications: list[dict] = field(default_factory=list)
    login_overrides: dict[str, dict] = field(default_factory=dict)
    _faults: list[Fail] = field(default_factory=list)
    _idem: dict[tuple, tuple[str, int, Any]] = field(default_factory=dict)
    _next_id: int = 100

    def __post_init__(self) -> None:
        for rid, login, kind, f, t, reason, event, status, by, note in _SEED_REQUESTS:
            self.requests[rid] = self._request_row(rid, login, kind, f, t, False, reason, event)
            self.requests[rid].update(status=status, decided_by=by, decision_note=note)
        self.document_files = {d["id"]: f"%PDF-fake {d['title']}".encode() for d in self.documents}

    def __repr__(self) -> str:
        return f"FakeCollege({self.base_url}, {len(self.log)} calls)"

    # ------------------------------------------------------------------ helpers for tests

    def install(self, router) -> None:
        """Route every request for base_url on a respx router to this fake."""
        router.route(url__startswith=self.base_url.rstrip("/")).mock(side_effect=self.handle)

    def issue_token(self, login: str, ttl: int | None = None) -> str:
        """Mint a token for `login` as a successful login would."""
        token = f"fake-jwt.{login}.{secrets.token_hex(12)}"
        self.tokens[token] = (login, time.time() + (self.token_ttl if ttl is None else ttl))
        return token

    def expire_token(self, token: str) -> None:
        """Make the college reject `token` from now on."""
        login, _ = self.tokens[token]
        self.tokens[token] = (login, time.time() - 1)

    def expire_all(self, login: str | None = None) -> None:
        """Expire every token (of one user, or all)."""
        for tok, (who, _) in list(self.tokens.items()):
            if login in (None, who):
                self.expire_token(tok)

    def ctx(self, login: str | None, *, session_id: str | None = None):
        """A SessionCtx for `login` (None: a visitor) carrying a fresh valid fake token."""
        from app.auth import SessionCtx

        if login is None:
            return SessionCtx(session_id or f"sid-visitor-{secrets.token_hex(4)}",
                              f"visitor:{secrets.token_hex(12)}", "visitor", None, None)
        user = user_summary(login)
        return SessionCtx(session_id or f"sid-{login}-{secrets.token_hex(4)}", f"user:{login}",
                          user["role"], user, self.issue_token(login))

    async def db_session(self, login: str | None) -> tuple[str, Any]:
        """Store a real session row; returns (cookie token, SessionCtx)."""
        from app import auth

        if login is None:
            return await auth._create_session(f"visitor:{secrets.token_hex(12)}", "visitor")
        user = user_summary(login)
        return await auth._create_session(f"user:{login}", user["role"], user,
                                          self.issue_token(login), self.token_ttl)

    def fail(self, path: str, status: int = 503, *, method: str | None = None,
             code: str = "unavailable", message: str = "Service temporarily unavailable.",
             exc: type[Exception] | None = None, times: int = -1) -> None:
        """Answer requests whose path equals `path` (or matches it when it starts with ^)
        with an error, or raise `exc` (e.g. httpx.ConnectError); `times=-1` means always."""
        self._faults.append(Fail(method, path, status, code, message, exc, times))

    def clear_faults(self) -> None:
        """Remove all scripted faults."""
        self._faults.clear()

    def calls(self, method: str | None = None, path: str | None = None) -> list[Call]:
        """Recorded calls, optionally filtered by method and exact path."""
        return [c for c in self.log if method in (None, c.method) and path in (None, c.path)]

    def identities(self) -> set[str | None]:
        """Set of users whose tokens were used (None for calls with no or an invalid token)."""
        return {c.login for c in self.log if c.path != "/v1/auth/login"}

    def writes(self) -> list[Call]:
        """Every state-changing call (POST other than login)."""
        return [c for c in self.log if c.method == "POST" and c.path != "/v1/auth/login"]

    # ------------------------------------------------------------------ dispatch

    def handle(self, request: httpx.Request) -> httpx.Response:
        """respx side effect: answer one request."""
        path = request.url.path
        token = None
        auth_header = request.headers.get("authorization")
        if auth_header:
            scheme, _, value = auth_header.partition(" ")
            token = value.strip() if scheme.lower() == "bearer" else auth_header
        try:
            body = json.loads(request.content) if request.content else None
        except ValueError:
            body = request.content
        login = None
        if token in self.tokens and self.tokens[token][1] > time.time():
            login = self.tokens[token][0]
        call = Call(request.method, path, dict(request.url.params), token, login,
                    request.headers.get("x-participant-key"),
                    request.headers.get("idempotency-key"), body)
        self.log.append(call)
        if self.down:
            raise httpx.ConnectError("college down (fake)", request=request)
        for f in self._faults:
            if f.matches(request.method, path):
                f.times -= 1 if f.times > 0 else 0
                if f.exc is not None:
                    raise f.exc("fault (fake)", request=request)
                call.status = f.status
                return httpx.Response(f.status, json=error_body(f.code, f.message))
        try:
            status, payload = self._dispatch(request.method, path, call)
        except _Err as e:
            status, payload = e.status, error_body(e.code, e.message, e.field)
        call.status = status
        if isinstance(payload, bytes):
            return httpx.Response(status, content=payload,
                                  headers={"content-type": "application/pdf"})
        return httpx.Response(status, json=payload)

    def _dispatch(self, method: str, path: str, call: Call) -> tuple[int, Any]:
        if path == "/v1/health" and method == "GET":
            return 200, {"status": "ok", "app": "KIT College System", "version": "1.0.0",
                         "timestamp": f"{self.today}T12:00:00+05:30", "environment": "fake"}
        self._check_key(call)
        for pattern, (verb, handler) in _ROUTES:
            m = pattern.match(path)
            if m and verb == method:
                return 200, handler(self, call, *m.groups())
        if any(p.match(path) for p, _ in _ROUTES):
            raise _Err(405, "method_not_allowed", "Method not allowed.")
        raise _Err(404, "not_found", "Not found.")

    def _check_key(self, call: Call) -> None:
        if not call.participant_key:
            raise _Err(401, "unauthorized", "Missing required header: X-Participant-Key.")
        if call.participant_key != self.participant_key:
            raise _Err(401, "unauthorized", "Invalid X-Participant-Key provided.")

    def _user(self, call: Call, role: str | None = None) -> str:
        if call.token is None:
            raise _Err(401, "unauthorized", "Missing Authorization header.")
        if call.token not in self.tokens:
            raise _Err(401, "unauthorized", "Invalid login token.")
        if call.login is None:
            raise _Err(401, "unauthorized", "Login token has expired.")
        if role and USERS[call.login]["role"] != role:
            raise _Err(403, "not_allowed", STUDENT_ONLY if role == "student" else TEACHER_ONLY)
        return call.login

    @staticmethod
    def _require(body: Any, *names: str) -> dict:
        if not isinstance(body, dict):
            raise _Err(422, "invalid_input", "Field required", names[0] if names else None)
        for n in names:
            if body.get(n) in (None, ""):
                raise _Err(422, "invalid_input", "Field required", n)
        return body

    def _idempotent(self, scope: str, call: Call, create) -> Any:
        if not call.idempotency_key:
            return create()
        key = (scope, call.idempotency_key)
        digest = hashlib.sha256(json.dumps(call.body, sort_keys=True).encode()).hexdigest()
        if key in self._idem:
            seen_digest, _, result = self._idem[key]
            if seen_digest != digest:
                raise _Err(409, "idempotency_conflict",
                           "Idempotency-Key was already used with a different request body.")
            return copy.deepcopy(result)
        result = create()
        self._idem[key] = (digest, 200, copy.deepcopy(result))
        return result

    def _request_row(self, rid, login, kind, f, t, half_day, reason, event) -> dict:
        u = USERS[login]
        return {"id": rid, "student_id": u["id"], "student_name": u["full_name"],
                "class_code": CLASSES[u["class_id"]]["class_code"], "type": kind,
                "from_date": f, "to_date": t, "half_day": bool(half_day), "reason": reason,
                "event_name": event, "status": "pending", "decided_by": None,
                "decision_note": None, "created_at": f"{self.today}T05:20:42+00:00",
                "student_notes": None}

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def authorised_classes(self, teacher: str) -> list[dict]:
        """Classes `teacher` may act on today (expired delegations excluded)."""
        out = []
        for cid, role_type, until in self.grants.get(teacher, []):
            if until is not None and date.fromisoformat(until) < self.today:
                continue
            c = CLASSES[cid]
            out.append({"class_id": cid, "class_code": c["class_code"],
                        "department": c["department"], "programme": c["programme"],
                        "year": c["year"], "role_type": role_type, "valid_until": until})
        return out

    def _sessions(self, login: str) -> list[dict]:
        u = USERS[login]
        out, sid = [], u["id"] * 1000
        for n, (_, code, name) in enumerate(SUBJECTS[u["class_id"]]):
            for k in range(10):
                day = self.today - timedelta(days=1 + k * 3 + n)
                h = int(hashlib.md5(f"{login}{code}{k}".encode()).hexdigest(), 16) % 10
                status = "absent" if h < 2 else "on_duty" if h == 2 else "present"
                sid += 1
                out.append({"session_id": sid, "date": day.isoformat(), "period": 1 + k % 4,
                            "subject_code": code, "subject_name": name, "status": status})
        return sorted(out, key=lambda s: (s["date"], s["period"]), reverse=True)

    def attendance_summary(self, login: str) -> dict:
        """The /v1/me/attendance payload for `login`."""
        sessions, subjects = self._sessions(login), []
        held_t = att_t = 0
        for sub_id, code, name in SUBJECTS[USERS[login]["class_id"]]:
            mine = [s for s in sessions if s["subject_code"] == code]
            att = sum(s["status"] == "present" for s in mine)
            od = sum(s["status"] == "on_duty" for s in mine)
            held = len(mine)
            held_t, att_t = held_t + held, att_t + att + od
            subjects.append({"subject_id": sub_id, "subject_code": code, "subject_name": name,
                             "classes_held": held, "classes_attended": att, "classes_od": od,
                             "classes_absent": held - att - od,
                             "percentage": round(100 * (att + od) / held, 1)})
        return {"as_of_date": self.today.isoformat(),
                "overall_percentage": round(100 * att_t / held_t, 1),
                "total_classes_held": held_t, "total_classes_attended": att_t,
                "subjects": subjects}

    # ------------------------------------------------------------------ endpoints

    def h_today(self, call):
        return {"date": self.today.isoformat(), "weekday": self.today.isoweekday(),
                "weekday_name": self.today.strftime("%A"), "timezone": "Asia/Kolkata"}

    def h_college(self, call):
        return copy.deepcopy(self.college)

    def h_courses(self, call):
        return copy.deepcopy(self.courses)

    def h_documents(self, call):
        return [{**d, "download_url": f"/v1/documents/{d['id']}/file"} for d in self.documents]

    def h_document_file(self, call, doc_id):
        if int(doc_id) not in self.document_files:
            raise _Err(404, "not_found", f"Document with id {doc_id} not found.")
        return self.document_files[int(doc_id)]

    def h_login(self, call):
        body = self._require(call.body, "login_id", "password")
        login = body["login_id"]
        if not isinstance(login, str) or self.passwords.get(login) != body["password"]:
            raise _Err(401, "unauthorized", "Invalid login ID or password.")
        override = copy.deepcopy(self.login_overrides.get(login) or {})
        user = override.get("user", user_summary(login) if login in USERS else None)
        owner = user.get("login_id") if isinstance(user, dict) else None
        token = self.issue_token(owner if owner in USERS else login)
        resp = {"access_token": token, "token_type": "bearer", "expires_in": self.token_ttl,
                "user": user}
        return {**resp, **override}

    def h_admission(self, call):
        body = self._require(call.body, "full_name", "phone", "email", "course_code",
                             "twelfth_percent")
        if body["course_code"] not in {c["code"] for c in self.courses}:
            raise _Err(422, "invalid_input", "Unknown course code.", "course_code")

        def create():
            row = {"id": self._new_id(), **{k: body[k] for k in (
                "full_name", "phone", "email", "course_code", "twelfth_percent")},
                "status": "submitted", "created_at": f"{self.today}T10:00:00+05:30"}
            self.applications.append(row)
            return row

        return self._idempotent("admission", call, create)

    def h_drives(self, call):
        return copy.deepcopy(self.drives)

    def h_me(self, call):
        return user_summary(self._user(call))

    def h_academics(self, call):
        login = self._user(call, "student")
        cgpa, arrears, company, category = ACADEMICS[login]
        return {"user_id": USERS[login]["id"], "cgpa": cgpa, "standing_arrears": arrears,
                "placed_company": company, "placed_category": category}

    def h_subjects(self, call):
        login = self._user(call, "student")
        cid = USERS[login]["class_id"]
        return [{"id": i, "code": c, "name": n, "class_id": cid} for i, c, n in SUBJECTS[cid]]

    def h_timetable(self, call):
        login = self._user(call, "student")
        cid = USERS[login]["class_id"]
        subs, out, sid = SUBJECTS[cid], [], cid * 100
        for wd in range(1, 7):
            for p, (start, end) in enumerate(_PERIODS, 1):
                _, code, name = subs[(wd + p) % len(subs)]
                sid += 1
                out.append({"id": sid, "weekday": wd, "period": p, "start_time": start,
                            "end_time": end, "subject_code": code, "subject_name": name})
        return out

    def h_calendar(self, call):
        login = self._user(call, "student")
        codes = {c for _, c, _ in SUBJECTS[USERS[login]["class_id"]]}
        lo, hi = call.params.get("from"), call.params.get("to")
        out = []
        for e in self.calendar_events:
            if e["subject_code"] and e["subject_code"] not in codes:
                continue
            if lo is None and hi is None and e["date"] < self.today.isoformat():
                continue
            if (lo and e["date"] < lo) or (hi and e["date"] > hi):
                continue
            out.append(copy.deepcopy(e))
        return out

    def h_attendance(self, call):
        return self.attendance_summary(self._user(call, "student"))

    def h_attendance_sessions(self, call):
        login = self._user(call, "student")
        p = call.params
        return [s for s in self._sessions(login)
                if (not p.get("subject") or s["subject_code"] == p["subject"])
                and (not p.get("from") or s["date"] >= p["from"])
                and (not p.get("to") or s["date"] <= p["to"])]

    def _mine(self, login: str) -> list[dict]:
        sid = USERS[login]["id"]
        return [r for r in self.requests.values() if r["student_id"] == sid]

    def h_my_requests(self, call):
        login = self._user(call, "student")
        return sorted(copy.deepcopy(self._mine(login)), key=lambda r: -r["id"])

    def h_create_request(self, call):
        login = self._user(call, "student")
        body = self._require(call.body, "type", "from_date", "to_date", "reason")
        if body["type"] not in REQUEST_TYPES:
            raise _Err(422, "invalid_input", "Invalid request type.", "type")
        try:
            f, t = date.fromisoformat(body["from_date"]), date.fromisoformat(body["to_date"])
        except (TypeError, ValueError):
            raise _Err(422, "invalid_input", "Dates must be YYYY-MM-DD.", "from_date")
        if t < f:
            raise _Err(422, "invalid_input", "to_date cannot be before from_date.", "to_date")
        if body["type"] == "on_duty" and not body.get("event_name"):
            raise _Err(422, "invalid_input", "event_name is required for on-duty.", "event_name")

        def create():
            rid = self._new_id()
            row = self._request_row(rid, login, body["type"], body["from_date"],
                                    body["to_date"], body.get("half_day"), body["reason"],
                                    body.get("event_name"))
            self.requests[rid] = row
            return copy.deepcopy(row)

        return self._idempotent(f"request:{login}", call, create)

    def _own_request(self, login: str, rid: str) -> dict:
        row = self.requests.get(int(rid))
        if row is None or row["student_id"] != USERS[login]["id"]:
            raise _Err(404, "not_found", f"Request with id {rid} not found in this sandbox.")
        return row

    def h_get_request(self, call, rid):
        return copy.deepcopy(self._own_request(self._user(call, "student"), rid))

    def h_cancel_request(self, call, rid):
        row = self._own_request(self._user(call, "student"), rid)
        if row["status"] not in ("pending", "approved"):
            raise _Err(409, "invalid_state", f"Request is already {row['status']}.")
        if date.fromisoformat(row["from_date"]) <= self.today:
            raise _Err(409, "invalid_state", "Request has already started.")
        row["status"] = "cancelled"
        return copy.deepcopy(row)

    def h_register(self, call, drive_id):
        login = self._user(call, "student")
        drive = next((d for d in self.drives if d["id"] == int(drive_id)), None)
        if drive is None:
            raise _Err(404, "not_found", f"Drive with id {drive_id} not found.")
        u, (cgpa, arrears, placed, _) = USERS[login], ACADEMICS[login]
        dept = CLASSES[u["class_id"]]["department"]
        if any(r["drive_id"] == drive["id"] and r["student_id"] == u["id"]
               for r in self.registrations):
            raise _Err(409, "already_registered", "Already registered for this drive.")
        reasons = []
        if dept not in drive["allowed_departments"]:
            reasons.append("department")
        if u["year"] not in drive["eligible_years"]:
            reasons.append("year")
        if cgpa < drive["min_cgpa"]:
            reasons.append("cgpa")
        if arrears and not drive["allow_standing_arrears"]:
            reasons.append("arrears")
        if placed:
            reasons.append("already placed")
        if reasons:
            raise _Err(403, "not_eligible", "Not eligible: " + ", ".join(reasons) + ".")
        if drive["deadline"] < self.today.isoformat():
            raise _Err(409, "deadline_passed", "Registration deadline has passed.")
        if drive["seats_left"] <= 0:
            raise _Err(409, "no_seats", "No seats left.")
        drive["seats_left"] -= 1
        row = {"id": self._new_id(), "drive_id": drive["id"], "company": drive["company"],
               "student_id": u["id"], "registered_at": f"{self.today}T10:00:00+05:30"}
        self.registrations.append(row)
        return row

    def h_my_placements(self, call):
        login = self._user(call, "student")
        return [r for r in self.registrations if r["student_id"] == USERS[login]["id"]]

    def h_teacher_classes(self, call):
        return self.authorised_classes(self._user(call, "teacher"))

    def _class_ids(self, teacher: str) -> set[int]:
        return {c["class_id"] for c in self.authorised_classes(teacher)}

    def h_teacher_requests(self, call):
        teacher = self._user(call, "teacher")
        codes = {CLASSES[c]["class_code"] for c in self._class_ids(teacher)}
        status = call.params.get("status")
        rows = [r for r in self.requests.values() if r["class_code"] in codes
                and (not status or r["status"] == status)]
        return sorted(copy.deepcopy(rows), key=lambda r: -r["id"])

    def h_class_attendance(self, call, class_id):
        teacher = self._user(call, "teacher")
        cid = int(class_id)
        if cid not in CLASSES:
            raise _Err(404, "not_found", f"Class with id {cid} not found.")
        if cid not in self._class_ids(teacher):
            raise _Err(403, "not_allowed", "You are not authorized for this class today.")
        students = []
        for login in STUDENTS:
            if USERS[login]["class_id"] != cid:
                continue
            s = self.attendance_summary(login)
            od = sum(x["classes_od"] for x in s["subjects"])
            students.append({"student_id": USERS[login]["id"], "student_login": login,
                             "student_name": USERS[login]["full_name"],
                             "classes_held": s["total_classes_held"],
                             "classes_attended": s["total_classes_attended"] - od,
                             "classes_od": od,
                             "classes_absent": s["total_classes_held"] - s["total_classes_attended"],
                             "percentage": s["overall_percentage"]})
        return {"class_id": cid, "class_code": CLASSES[cid]["class_code"],
                "as_of_date": self.today.isoformat(), "students": students}

    def _decide(self, call, rid, status, note):
        teacher = self._user(call, "teacher")
        row = self.requests.get(int(rid))
        if row is None:
            raise _Err(404, "not_found", f"Request with id {rid} not found in this sandbox.")
        codes = {CLASSES[c]["class_code"] for c in self._class_ids(teacher)}
        if row["class_code"] not in codes:
            raise _Err(403, "not_allowed", "You are not authorized to act on this request.")
        if row["status"] != "pending":
            raise _Err(409, "invalid_state", f"Request is already {row['status']}.")
        row.update(status=status, decided_by=USERS[teacher]["id"], decision_note=note)
        return copy.deepcopy(row)

    def h_approve(self, call, rid):
        comment = call.body.get("comment") if isinstance(call.body, dict) else None
        return self._decide(call, rid, "approved", comment)

    def h_reject(self, call, rid):
        body = call.body if isinstance(call.body, dict) else {}
        if not str(body.get("reason") or "").strip():
            self._user(call, "teacher")
            raise _Err(422, "invalid_input", "Field required", "reason")
        return self._decide(call, rid, "rejected", body["reason"])


_ROUTES: list[tuple[re.Pattern, tuple[str, Any]]] = [
    (re.compile(p + "$"), (m, h))
    for p, m, h in [
        ("/v1/today", "GET", FakeCollege.h_today),
        ("/v1/college", "GET", FakeCollege.h_college),
        ("/v1/courses", "GET", FakeCollege.h_courses),
        ("/v1/admissions/applications", "POST", FakeCollege.h_admission),
        ("/v1/documents", "GET", FakeCollege.h_documents),
        (r"/v1/documents/(\d+)/file", "GET", FakeCollege.h_document_file),
        ("/v1/auth/login", "POST", FakeCollege.h_login),
        ("/v1/placements/drives", "GET", FakeCollege.h_drives),
        (r"/v1/placements/drives/(\d+)/register", "POST", FakeCollege.h_register),
        ("/v1/me", "GET", FakeCollege.h_me),
        ("/v1/me/academics", "GET", FakeCollege.h_academics),
        ("/v1/me/subjects", "GET", FakeCollege.h_subjects),
        ("/v1/me/timetable", "GET", FakeCollege.h_timetable),
        ("/v1/calendar", "GET", FakeCollege.h_calendar),
        ("/v1/me/attendance", "GET", FakeCollege.h_attendance),
        ("/v1/me/attendance/sessions", "GET", FakeCollege.h_attendance_sessions),
        ("/v1/me/requests", "GET", FakeCollege.h_my_requests),
        ("/v1/me/requests", "POST", FakeCollege.h_create_request),
        (r"/v1/me/requests/(\d+)", "GET", FakeCollege.h_get_request),
        (r"/v1/me/requests/(\d+)/cancel", "POST", FakeCollege.h_cancel_request),
        ("/v1/me/placements", "GET", FakeCollege.h_my_placements),
        ("/v1/teacher/classes", "GET", FakeCollege.h_teacher_classes),
        ("/v1/teacher/requests", "GET", FakeCollege.h_teacher_requests),
        (r"/v1/teacher/classes/(\d+)/attendance", "GET", FakeCollege.h_class_attendance),
        (r"/v1/teacher/requests/(\d+)/approve", "POST", FakeCollege.h_approve),
        (r"/v1/teacher/requests/(\d+)/reject", "POST", FakeCollege.h_reject),
    ]
]


async def login_as(client: httpx.AsyncClient, login: str, password: str = PASSWORD) -> dict:
    """Log `client` in through the real /api/login; returns the user from the response."""
    r = await client.post("/api/login", json={"login_id": login, "password": password})
    assert r.status_code == 200, (r.status_code, r.text)
    return r.json()["user"]
