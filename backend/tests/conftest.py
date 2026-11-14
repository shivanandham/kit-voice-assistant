"""Shared test foundation.

* Database: the test database comes from ``TEST_DATABASE_URL`` (default
  ``postgresql://localhost:5432/kit_assistant_test``) and is exported as ``DATABASE_URL`` before any
  app import. ``init_db()`` runs once; every table is truncated before each test.
* Network guard (autouse): any real outbound connection or DNS lookup for a non-local host fails the
  test, and the API keys in settings are dummy values (``DUMMY_KEYS``) so a leak is detectable.
  Tests marked ``@pytest.mark.live`` skip the guard, get the real keys, and are deselected unless
  pytest runs with ``--live``.
* ``fake_college``, ``app_client``, ``browser``: see ``tests/fakes/college.py``.
"""

import importlib
import ipaddress
import os
import socket
import sys
from urllib.parse import urlparse

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://localhost:5432/kit_assistant_test"
)
os.environ["DATABASE_URL"] = TEST_DATABASE_URL

DUMMY_KEYS = {
    "KIT_PARTICIPANT_KEY": "test-participant-key-not-real",
    "ANTHROPIC_API_KEY": "test-anthropic-key-not-real",
    "DEEPGRAM_API_KEY": "test-deepgram-key-not-real",
    "GEMINI_API_KEY": "test-gemini-key-not-real",
}
_REAL_KEY_ENV = {k: os.environ.get(k) for k in DUMMY_KEYS}
os.environ.update(DUMMY_KEYS)

import httpx
import pytest
import pytest_asyncio
import respx
from app import auth, db
from app.config import Settings, get_settings

from tests.fakes.college import FakeCollege

TABLES = "sessions, conversations, messages, tool_calls, pending_actions, documents, doc_chunks"

if "test" not in (urlparse(TEST_DATABASE_URL).path or ""):
    raise pytest.UsageError(
        "TEST_DATABASE_URL must name a database containing 'test'; tests truncate every table."
    )


# --------------------------------------------------------------------------- options and markers


def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", help="also run @pytest.mark.live tests")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live: talks to real third-party services; deselected unless --live"
    )


def pytest_collection_modifyitems(config, items):
    """Run every async test in the session loop; deselect live tests unless --live."""
    marker = pytest.mark.asyncio(loop_scope="session")
    keep, dropped = [], []
    for item in items:
        if item.get_closest_marker("live") and not config.getoption("--live"):
            dropped.append(item)
            continue
        if pytest_asyncio.is_async_test(item):
            item.add_marker(marker, append=False)
        keep.append(item)
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = keep


# --------------------------------------------------------------------------- database


@pytest_asyncio.fixture(autouse=True, loop_scope="session")
async def _clean_db():
    await db.init_db()
    await db.pool().execute(f"TRUNCATE {TABLES} RESTART IDENTITY CASCADE")
    auth.reset_rate_limit()
    yield


# --------------------------------------------------------------------------- network guard


class NetworkBlocked(OSError):
    """Raised for a blocked outbound connection (an OSError so client code fails as offline)."""


def _db_hosts() -> set[str]:
    host = urlparse(TEST_DATABASE_URL).hostname
    return {host} if host else set()


def _is_local(host) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode()
    host = str(host).split("%")[0]
    if host in {"localhost", ""} or host in _db_hosts():
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def network_guard(request, monkeypatch):
    """Fail the test on any real non-local connection; swap API keys for dummies."""
    settings = get_settings()
    if request.node.get_closest_marker("live"):
        for k, v in _REAL_KEY_ENV.items():
            if v is None:
                monkeypatch.delenv(k, raising=False)
            else:
                monkeypatch.setenv(k, v)
        real = Settings()
        for k in DUMMY_KEYS:
            monkeypatch.setattr(settings, k.lower(), getattr(real, k.lower()))
        yield []
        return

    for k, v in DUMMY_KEYS.items():
        monkeypatch.setattr(settings, k.lower(), v)
    violations: list[str] = []
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def check(address):
        if isinstance(address, tuple) and not _is_local(address[0]):
            violations.append(f"connect to {address[0]}:{address[1]}")
            raise NetworkBlocked(f"network access blocked in tests: {address!r}")

    def connect(self, address):
        check(address)
        return real_connect(self, address)

    def connect_ex(self, address):
        check(address)
        return real_connect_ex(self, address)

    def getaddrinfo(host, *args, **kwargs):
        if not _is_local(host):
            violations.append(f"DNS lookup of {host}")
            raise NetworkBlocked(f"network access blocked in tests: DNS {host!r}")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    yield violations
    if violations:
        pytest.fail("test tried to reach the network: " + "; ".join(sorted(set(violations))))


# --------------------------------------------------------------------------- fake college


@pytest.fixture
def fake_college():
    """The in-memory college API, mounted with respx (any other httpx request is an error)."""
    settings = get_settings()
    fake = FakeCollege(base_url=settings.kit_base_url, participant_key=settings.kit_participant_key)
    client_mod = sys.modules.get("app.kit.client")
    if client_mod is not None and hasattr(client_mod, "clear_today_cache"):
        client_mod.clear_today_cache()
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        fake.install(router)
        yield fake
    if client_mod is not None and hasattr(client_mod, "clear_today_cache"):
        client_mod.clear_today_cache()


# --------------------------------------------------------------------------- the real app

def load_main():
    """Import the real app (every module it needs exists; nothing is stubbed)."""
    return importlib.import_module("app.main")


async def _noop() -> None:
    return None


@pytest_asyncio.fixture(loop_scope="session")
async def asgi_app(fake_college, monkeypatch):
    """The real FastAPI app with its lifespan running and the college faked.

    The document sync loop is replaced by a no-op (knowledge tests drive it directly), the
    prompt-cache keep-warm task is off (llm_keep_warm=False) and close_db is a no-op so the shared
    test pool survives; everything else is real.
    """
    main = load_main()
    monkeypatch.setattr(main.store, "start_sync_loop", _noop)
    monkeypatch.setattr(main.store, "stop_sync_loop", _noop)
    monkeypatch.setattr(main, "close_db", _noop)
    monkeypatch.setattr(get_settings(), "llm_keep_warm", False)
    main._public_cache.clear()
    async with main.app.router.lifespan_context(main.app):
        yield main.app
    main._public_cache.clear()


@pytest_asyncio.fixture(loop_scope="session")
async def browser(asgi_app):
    """Factory: ``await browser()`` returns a new client (own cookie jar) over the app.

    Keyword arguments: ``base_url`` (default http://testserver) and ``client_ip``.
    """
    opened: list[httpx.AsyncClient] = []

    async def make(base_url: str = "http://testserver", client_ip: str = "127.0.0.1",
                   **kwargs) -> httpx.AsyncClient:
        transport = httpx.ASGITransport(app=asgi_app, raise_app_exceptions=False,
                                        client=(client_ip, 50000))
        c = httpx.AsyncClient(transport=transport, base_url=base_url, **kwargs)
        opened.append(c)
        return c

    yield make
    for c in opened:
        await c.aclose()


@pytest_asyncio.fixture(loop_scope="session")
async def app_client(browser):
    """An httpx.AsyncClient over the real app with the college faked (one browser)."""
    return await browser()
