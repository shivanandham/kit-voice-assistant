"""Shared fakes for the knowledge tests: listing items, sessions and a deterministic embedder."""

import hashlib
import math
import re
from datetime import date

from app.auth import SessionCtx
from app.knowledge import store

HANDBOOK_2025 = """Chapter 6 · Leave and on-duty rules
6.4 On-duty
6.4.3 Proof of participation must be submitted within 7 working days after the event."""

HANDBOOK_2026 = """Chapter 4 · Attendance rules for undergraduate (B.Tech) students
4.1 Minimum attendance for B.Tech students
4.1.1 A B.Tech student must have at least 75% attendance in each course to write the End
Semester Examination in that course.
Chapter 6 · Leave and on-duty rules for undergraduate (B.Tech) students
6.1 Personal leave
6.1.4 Personal leave is not granted on any day on which an Internal Assessment is held.
6.4 On-duty (OD)
6.4.3 Within 5 working days after the event, the student must submit proof of participation."""

BROCHURE = """2. Programmes, eligibility and fees
B.Tech Computer Science and Engineering tuition fee 125000, hostel fee per year 60000."""

LESSON = """Unit 1 · Data models and the relational model
1.3 Keys
A foreign key is an attribute that refers to the primary key of another table."""

CIRCULAR = """Circular 12/2026
Subject: Change to the time allowed for submitting on-duty proof (Rule 6.4.3)
A B.Tech student must submit on-duty proof within 3 working days after the event."""

ITEMS = [
    {"id": 1, "title": "KIT Student Handbook 2025", "type": "handbook", "version": "2025.1",
     "effective_date": "2025-07-01", "applies_to": {"programmes": ["UG", "PG"]}, "is_surprise": False},
    {"id": 2, "title": "KIT Student Handbook 2026", "type": "handbook", "version": "2026.1",
     "effective_date": "2026-07-01", "applies_to": {"programmes": ["UG", "PG"]}, "is_surprise": False},
    {"id": 3, "title": "KIT Admissions Brochure 2026", "type": "brochure", "version": "2026.1",
     "effective_date": "2026-11-01", "applies_to": {"programmes": ["UG"]}, "is_surprise": False},
    {"id": 4, "title": "DBMS Lesson Notes", "type": "lesson_notes", "version": "2026.1",
     "effective_date": "2026-10-22", "applies_to": {"departments": ["CSE"], "semester": 3},
     "is_surprise": False},
    {"id": 5, "title": "Circular 12/2026", "type": "circular", "version": "12/2026",
     "effective_date": "2026-11-01", "applies_to": {"programmes": ["UG"]}, "is_surprise": False},
]
TEXTS = {1: HANDBOOK_2025, 2: HANDBOOK_2026, 3: BROCHURE, 4: LESSON, 5: CIRCULAR}

VISITOR = SessionCtx("s1", "visitor:x", "visitor", None, None)
CSE_STUDENT = SessionCtx("s2", "user:arjun", "student",
                         {"login_id": "arjun", "department": "CSE", "programme": "UG"}, "t")
ECE_STUDENT = SessionCtx("s3", "user:karthik", "student",
                         {"login_id": "karthik", "department": "ECE", "programme": "UG"}, "t")
TEACHER = SessionCtx("s4", "user:lakshmi", "teacher",
                     {"login_id": "lakshmi", "department": "CSE"}, "t")


def fake_vector(text: str, dim: int = 768) -> list[float]:
    """Bag-of-words hashing embedding: texts sharing words get a high cosine similarity."""
    v = [0.0] * dim
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        v[int(hashlib.md5(word.encode()).hexdigest(), 16) % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


async def insert_documents(items=ITEMS) -> None:
    """Store documents directly (no college API), as a sync would."""
    for item in items:
        text = TEXTS[item["id"]]
        await store._upsert(item["id"], store._meta(item), text, f"hash-{item['id']}")


def college_date(d: date):
    """A replacement for store._college_date returning a fixed date."""
    async def fake() -> date:
        return d
    return fake
