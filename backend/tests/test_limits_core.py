"""Limit primitives: sliding windows, the connection gate, client IP (ABU9), settings."""

import pytest
from app import limits
from app.config import get_settings


@pytest.fixture
def clock(monkeypatch):
    """A fake monotonic clock: ``clock[0] += 60`` moves time forward."""
    t = [1000.0]
    monkeypatch.setattr(limits, "_clock", lambda: t[0])
    return t


def test_window_at_below_and_above_limit(clock):
    w = limits.SlidingWindow(60)
    assert [w.take("a", 3) for _ in range(4)] == [True, True, True, False]
    assert w.count("a") == 3


def test_window_expiry_with_fake_clock(clock):
    w = limits.SlidingWindow(60)
    for _ in range(3):
        w.take("a", 3)
    clock[0] += 59
    assert not w.take("a", 3)
    clock[0] += 2
    assert w.take("a", 3)
    assert w.count("a") == 1


def test_window_keys_are_independent(clock):
    w = limits.SlidingWindow(60)
    for _ in range(3):
        w.take("a", 3)
    assert not w.take("a", 3)
    assert w.take("b", 3)


def test_window_memory_is_bounded(clock):
    w = limits.SlidingWindow(60, max_keys=100)
    for i in range(1000):
        w.take(f"k{i}", 5)
    assert len(w) <= 100
    clock[0] += 61
    w.take("fresh", 5)
    assert len(w) <= 100


def test_window_reads_window_length_from_settings(clock, monkeypatch):
    monkeypatch.setitem(limits.DEFAULTS, "limit_idle_seconds", 10)
    w = limits.SlidingWindow("limit_idle_seconds")
    w.take("a", 1)
    clock[0] += 11
    assert w.take("a", 1)


def test_connection_gate_visitor_ip_user_login_and_total():
    gate = limits.ConnectionGate()
    assert all(gate.acquire("visitor:ip1", 3) for _ in range(3))
    assert not gate.acquire("visitor:ip1", 3)  # ABU1: 3 per visitor IP
    assert gate.acquire("login:arjun", 2) and gate.acquire("login:arjun", 2)
    assert not gate.acquire("login:arjun", 2)  # ABU1: 2 per login id
    assert gate.acquire("login:priya", 2)  # another user is unaffected
    gate.release("visitor:ip1")
    assert gate.acquire("visitor:ip1", 3)
    for i in range(34):
        assert gate.acquire(f"login:u{i}", 2)
    assert gate.total == 40
    assert not gate.acquire("login:late", 2)  # ABU1: 40 overall
    gate.release("login:priya")
    assert gate.acquire("login:late", 2)


def test_connection_gate_release_frees_keys():
    gate = limits.ConnectionGate()
    gate.acquire("k", 2)
    gate.release("k")
    assert gate.total == 0 and gate.open == {}


def test_limits_come_from_settings_or_defaults(monkeypatch):
    assert limits.limit("limit_ws_per_ip") == 3
    monkeypatch.setitem(limits.DEFAULTS, "limit_ws_per_ip", 7)
    settings = get_settings()
    if hasattr(settings, "limit_ws_per_ip"):  # once config.py grows the field
        monkeypatch.setattr(settings, "limit_ws_per_ip", 7)
    assert limits.limit("limit_ws_per_ip") == 7


class Conn:
    """Just enough of a Request/WebSocket for client_ip."""

    def __init__(self, peer: str | None, xff: str | None = None) -> None:
        self.client = type("C", (), {"host": peer})() if peer else None
        self.headers = {"x-forwarded-for": xff} if xff is not None else {}


def test_abu9_one_client_ip_rule_for_routes_and_sockets():
    from app import auth

    assert limits.client_ip is auth._client_ip
    assert limits.client_ip(Conn("127.0.0.1", "6.6.6.6, 198.51.100.2")) == "198.51.100.2"
    assert limits.client_ip(Conn("203.0.113.5", "198.51.100.1")) == "203.0.113.5"


def test_reset_all_clears_every_counter(clock):
    limits.TURNS_GLOBAL.hit("all")
    limits.CONNECTIONS.acquire("k", 2)
    limits.reset_all()
    assert limits.TURNS_GLOBAL.count("all") == 0 and limits.CONNECTIONS.total == 0


def test_auth_reset_rate_limit_also_resets_voice_limits(clock):
    from app import auth

    limits.TURNS_GLOBAL.hit("all")
    auth.reset_rate_limit()
    assert limits.TURNS_GLOBAL.count("all") == 0
