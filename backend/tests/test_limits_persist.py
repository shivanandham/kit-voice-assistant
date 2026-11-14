"""Daily limit counters survive a restart: saved to limit_counters, reloaded at startup."""

from types import SimpleNamespace

import pytest
from app import limits
from app.config import get_settings


@pytest.fixture
def persist(monkeypatch):
    """Persistence on, a fake clock, and small limits."""
    monkeypatch.setattr(get_settings(), "limits_persist", True)
    t = [1000.0]
    monkeypatch.setattr(limits, "_clock", lambda: t[0])
    monkeypatch.setitem(limits.DEFAULTS, "limit_visitor_turns_per_ip_per_day", 3)
    monkeypatch.setitem(limits.DEFAULTS, "limit_user_turns_per_day", 2)
    monkeypatch.setitem(limits.DEFAULTS, "limit_daily_turns", 1000)
    monkeypatch.setitem(limits.DEFAULTS, "limit_daily_turns_users", 1000)
    monkeypatch.setitem(limits.DEFAULTS, "limit_turns_per_conversation", 1000)
    limits.reset_all()
    yield t
    limits.reset_all()


VISITOR = SimpleNamespace(role="visitor", user=None, owner_key="anon:x")
USER = SimpleNamespace(role="student", user={"login_id": "Alice"}, owner_key="anon:y")


async def test_visitor_turns_survive_restart(persist):
    for _ in range(2):
        assert limits.admit_turn(VISITOR, "1.2.3.4", 0) is None
    await limits.save_counters()
    limits.reset_all()
    await limits.load_counters()
    assert limits.TURNS_VISITOR_IP.count("1.2.3.4") == 2
    assert limits.admit_turn(VISITOR, "1.2.3.4", 0) is None
    assert limits.admit_turn(VISITOR, "1.2.3.4", 0) == limits.DAILY_LIMIT
    assert limits.admit_turn(VISITOR, "5.6.7.8", 0) is None


async def test_user_turns_survive_restart(persist):
    for _ in range(2):
        assert limits.admit_turn(USER, "1.2.3.4", 0) is None
    await limits.save_counters()
    await limits.save_counters()  # idempotent
    limits.reset_all()
    await limits.load_counters()
    assert limits.TURNS_USER.count("login:alice") == 2
    assert limits.TURNS_GLOBAL.count("all") == 2
    assert limits.admit_turn(USER, "9.9.9.9", 0) == limits.DAILY_LIMIT
