"""REL2: no test touches a real third-party service. Checks the conftest network guard."""

import socket

import httpx
import pytest
from app.config import get_settings

from tests.conftest import DUMMY_KEYS, NetworkBlocked


def test_rel2_connect_to_remote_ip_is_blocked(network_guard):
    s = socket.socket()
    try:
        with pytest.raises(NetworkBlocked):
            s.connect(("203.0.113.10", 443))
        assert s.connect_ex is not None
        with pytest.raises(NetworkBlocked):
            s.connect_ex(("198.51.100.1", 80))
    finally:
        s.close()
    assert len(network_guard) == 2
    network_guard.clear()  # expected violations; do not fail this test


def test_rel2_dns_lookup_of_remote_host_is_blocked(network_guard):
    with pytest.raises(NetworkBlocked):
        socket.getaddrinfo("api.anthropic.com", 443)
    assert network_guard == ["DNS lookup of api.anthropic.com"]
    network_guard.clear()


async def test_rel2_httpx_to_real_host_is_blocked_and_recorded(network_guard):
    async with httpx.AsyncClient() as c:
        with pytest.raises(httpx.HTTPError):
            await c.get("https://api.deepgram.com/v1/listen")
    assert network_guard and "api.deepgram.com" in network_guard[0]
    network_guard.clear()


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_rel2_local_lookups_allowed(network_guard, host):
    assert socket.getaddrinfo(host, 5432)
    assert network_guard == []


async def test_rel2_local_postgres_still_works():
    from app.db import pool

    assert await pool().fetchval("SELECT 1") == 1


def test_rel2_api_keys_are_dummies():
    s = get_settings()
    for env_name, value in DUMMY_KEYS.items():
        assert getattr(s, env_name.lower()) == value


def test_rel2_live_marker_registered(pytestconfig):
    markers = pytestconfig.getini("markers")
    assert any(m.startswith("live:") for m in markers)
    assert pytestconfig.getoption("--live") in (False, True)


@pytest.mark.live
async def test_live_sandbox_health_read_only():
    """Opt-in example (pytest --live): a read-only call to the real sandbox."""
    async with httpx.AsyncClient(base_url=get_settings().kit_base_url, timeout=10) as c:
        assert (await c.get("/v1/health")).status_code == 200
