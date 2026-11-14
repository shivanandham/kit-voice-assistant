"""Every abuse limit (ACCEPTANCE.md ABU1-ABU10) in one place, in memory, for one process.

Rule of thumb: IP-based limits apply only to visitors; a logged-in user is always limited by
login id, so people sharing a network never use up each other's allowance. ABU8's global daily
budget is the only thing logged-in users share.

Thresholds are settings (app.config, `.env`). Counters are bounded sliding windows on an
injectable clock (`_clock`); auth.py and main.py are thin callers.
"""

from __future__ import annotations

import json
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from fastapi import HTTPException

from app.config import get_settings
from app.db import pool

DAY = 86400.0
MAX_KEYS = 50_000
TRUSTED_PROXIES = {"127.0.0.1", "::1"}  # Caddy runs on the same host

# Test-time overrides of settings, by setting name (the acceptance tests use this name).
DEFAULTS: dict[str, float] = {}

DAILY_LIMIT = (
    "You've reached today's limit for questions. Please come back tomorrow, or contact the"
    " college office if it's urgent."
)
BUSY = (
    "I've reached today's limit for questions because it's very busy. Please try again later,"
    " or contact the college office if it's urgent."
)
NEW_CONVERSATION = "This conversation is getting very long. Please start a new conversation."
TOO_MANY_PENDING = (
    "There are already several pending actions waiting for a yes or no in this conversation. "
    "Ask the user to confirm or cancel those before preparing another."
)
TOO_MANY_ADMISSIONS = (
    "The daily limit for admission applications from here has been reached. "
    "Tell the user they can apply again tomorrow or contact the admissions office."
)
TOO_MANY_REQUESTS = "Too many requests. Please slow down and try again in a minute."

# The caller's IP for code that only sees a SessionCtx (tools); set by the voice socket.
client_ip_var: ContextVar[str | None] = ContextVar("client_ip", default=None)


def _default_clock() -> float:
    return time.monotonic()


_clock: Callable[[], float] = _default_clock


def now() -> float:
    """The limits clock (monotonic seconds); tests replace `_clock`."""
    return _clock()


def limit(name: str) -> float:
    """A threshold from settings (or a test override in DEFAULTS)."""
    if name in DEFAULTS:
        return DEFAULTS[name]
    return getattr(get_settings(), name)


def client_ip(conn: Any) -> str:
    """ABU9: the right-most X-Forwarded-For entry (the one Caddy appended), trusted only when
    the direct peer is the local proxy; otherwise the peer address. Same for HTTP and sockets."""
    peer = conn.client.host if conn.client else None
    forwarded = conn.headers.get("x-forwarded-for")
    if forwarded and peer in TRUSTED_PROXIES:
        last = forwarded.split(",")[-1].strip()
        if last:
            return last
    return peer or "unknown"


class SlidingWindow:
    """Per-key event timestamps over a sliding window, with a bound on the number of keys."""

    def __init__(self, window: float | str, max_keys: int = MAX_KEYS) -> None:
        """`window` is seconds, or the name of a limit setting holding seconds."""
        self._window = window
        self._max_keys = max_keys
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._hits)

    @property
    def window(self) -> float:
        """Window length in seconds."""
        return limit(self._window) if isinstance(self._window, str) else self._window

    def _prune(self, key: str) -> deque[float] | None:
        hits = self._hits.get(key)
        if hits is None:
            return None
        cutoff = now() - self.window
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if not hits:
            del self._hits[key]
            return None
        return hits

    def count(self, key: str) -> int:
        """Events for `key` inside the window."""
        hits = self._prune(key)
        return len(hits) if hits else 0

    def hit(self, key: str) -> None:
        """Record one event for `key`, evicting the least recently used keys if needed."""
        hits = self._prune(key)
        if hits is None:
            hits = self._hits[key] = deque()
        hits.append(now())
        self._hits.move_to_end(key)
        while len(self._hits) > self._max_keys:
            self._hits.popitem(last=False)

    def take(self, key: str, max_events: float) -> bool:
        """Record an event if fewer than `max_events` are in the window; False when full."""
        if self.count(key) >= max_events:
            return False
        self.hit(key)
        return True

    def reset(self) -> None:
        """Forget everything."""
        self._hits.clear()


class ConnectionGate:
    """Counts open voice sockets per key (visitor IP or login id) and overall (ABU1)."""

    def __init__(self) -> None:
        """Start with no open sockets."""
        self.open: dict[str, int] = {}
        self.total = 0

    def acquire(self, key: str, per_key: float) -> bool:
        """Reserve a socket slot; False when the key's cap or the overall cap is reached."""
        if self.total >= limit("limit_ws_total") or self.open.get(key, 0) >= per_key:
            return False
        self.open[key] = self.open.get(key, 0) + 1
        self.total += 1
        return True

    def release(self, key: str) -> None:
        """Free a slot taken by acquire()."""
        left = self.open.get(key, 0) - 1
        if left > 0:
            self.open[key] = left
        else:
            self.open.pop(key, None)
        self.total = max(0, self.total - 1)

    def reset(self) -> None:
        """Forget all sockets (tests)."""
        self.open.clear()
        self.total = 0


def socket_slot(ctx: Any, ip: str) -> tuple[str, float]:
    """ABU1 key and cap: visitors share their IP's slots; a logged-in user has their own."""
    if ctx.role == "visitor":
        return f"visitor:{ip}", limit("limit_ws_per_ip")
    return user_key(ctx), limit("limit_ws_per_user")


HTTP = SlidingWindow(60)
LOGIN_IP = SlidingWindow(60)
LOGIN_FAILURES = SlidingWindow("login_failure_window_seconds")
NEW_SESSIONS = SlidingWindow(3600)
TURNS_VISITOR_IP = SlidingWindow(DAY)
TURNS_USER = SlidingWindow(DAY)
TURNS_GLOBAL = SlidingWindow(DAY, max_keys=1)
ADMISSIONS_SESSION = SlidingWindow(DAY)
ADMISSIONS_IP = SlidingWindow(DAY)
CONNECTIONS = ConnectionGate()

_WINDOWS = (
    HTTP,
    LOGIN_IP,
    LOGIN_FAILURES,
    NEW_SESSIONS,
    TURNS_VISITOR_IP,
    TURNS_USER,
    TURNS_GLOBAL,
    ADMISSIONS_SESSION,
    ADMISSIONS_IP,
)


def reset_all() -> None:
    """Clear every counter (tests)."""
    for w in _WINDOWS:
        w.reset()
    _session_keys.clear()
    CONNECTIONS.reset()


# --------------------------------------------------------------------------- ABU2 / ABU7 / ABU8


async def conversation_turns(conversation_id: str | None) -> int:
    """User messages stored in a conversation (0 for an unknown or malformed id)."""
    try:
        cid = uuid.UUID(str(conversation_id))
    except ValueError:
        return 0
    return await pool().fetchval(
        "SELECT count(*) FROM messages WHERE conversation_id = $1 AND role = 'user'", cid
    )


def admit_turn(ctx: Any, ip: str, conversation_turns: int) -> str | None:
    """Check and record one turn; return a spoken-style refusal, or None when allowed.

    Synchronous on purpose: no await between the check and the record, and none before the
    caller hands over to the new turn. `conversation_turns` is the caller's running count.
    """
    if conversation_turns >= limit("limit_turns_per_conversation"):
        return NEW_CONVERSATION
    visitor = ctx.role == "visitor"
    budget = limit("limit_daily_turns" if visitor else "limit_daily_turns_users")
    if TURNS_GLOBAL.count("all") >= budget:
        return BUSY
    if visitor:  # per IP, so clearing cookies does not reset it
        window, key, cap = TURNS_VISITOR_IP, ip, limit("limit_visitor_turns_per_ip_per_day")
    else:  # per login id, across all of the user's sessions and devices
        window, key, cap = TURNS_USER, user_key(ctx), limit("limit_user_turns_per_day")
    if not window.take(key, cap):
        return DAILY_LIMIT
    TURNS_GLOBAL.hit("all")
    return None


def user_key(ctx: Any) -> str:
    """The per-person key for a logged-in session: the college login id."""
    login = (ctx.user or {}).get("login_id")
    return f"login:{login.strip().lower()}" if isinstance(login, str) and login else ctx.owner_key


# --------------------------------------------------------------------------- ABU6


async def pending_refusal(conversation_id: str, tool_name: str, args: dict) -> str | None:
    """Refuse a new pending action when the conversation already holds the maximum."""
    try:
        cid = uuid.UUID(conversation_id)
    except ValueError:
        return None
    others = await pool().fetchval(
        "SELECT count(*) FROM pending_actions WHERE conversation_id = $1 AND status = 'pending'"
        " AND NOT (tool_name = $2 AND args = $3::jsonb)",
        cid,
        tool_name,
        json.dumps(args, sort_keys=True),
    )
    return TOO_MANY_PENDING if others >= limit("limit_pending_per_conversation") else None


def admission_refusal(ctx: Any) -> str | None:
    """Refuse an admission application past the per-session or per-IP daily cap."""
    ip = client_ip_var.get()
    if ADMISSIONS_SESSION.count(ctx.session_id) >= limit("limit_admissions_per_session_per_day"):
        return TOO_MANY_ADMISSIONS
    if ip and ADMISSIONS_IP.count(ip) >= limit("limit_admissions_per_ip_per_day"):
        return TOO_MANY_ADMISSIONS
    return None


def record_admission(ctx: Any) -> None:
    """Count one submitted admission application."""
    ADMISSIONS_SESSION.hit(ctx.session_id)
    if ip := client_ip_var.get():
        ADMISSIONS_IP.hit(ip)


# --------------------------------------------------------------------------- ABU10


class BodySizeLimit:
    """ASGI middleware: refuse HTTP bodies over limit_http_body_bytes with 413.

    A declared Content-Length is checked before anything is read; a streamed body is counted
    as it arrives, and once it passes the limit the 413 is sent and the app sees a disconnect.
    """

    def __init__(self, app: Any) -> None:
        """Wrap the ASGI app."""
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        """Pass the request through, or answer 413 when the body is too large."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        cap = int(limit("limit_http_body_bytes"))
        headers = dict(scope.get("headers") or [])
        declared = headers.get(b"content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > cap):
            await _too_large(send)
            return
        seen = 0
        started = refused = False

        async def counted_receive() -> dict:
            nonlocal seen, refused
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > cap:  # answer now and tell the app the client went away
                    if not started and not refused:
                        refused = True
                        await _too_large(send)
                    return {"type": "http.disconnect"}
            return message

        async def tracked_send(message: dict) -> None:
            nonlocal started
            if refused:
                return
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        await self.app(scope, counted_receive, tracked_send)


async def _too_large(send: Callable) -> None:
    body = json.dumps({"detail": "That request is too large."}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


# --------------------------------------------------------------------------- ABU4 / ABU5

_session_keys: OrderedDict[str, tuple[str | None, float]] = OrderedDict()
SESSION_KEY_TTL_S = 30.0


async def _login_key_for(token: str | None) -> str | None:
    """The login key behind a session cookie (cached briefly), or None for visitors."""
    if not token:
        return None
    cached = _session_keys.get(token)
    if cached and cached[1] > now():
        return cached[0]
    from app.auth import _lookup  # late import: auth imports this module

    ctx = await _lookup(token)
    key = user_key(ctx) if ctx is not None and ctx.role != "visitor" else None
    _session_keys[token] = (key, now() + SESSION_KEY_TTL_S)
    _session_keys.move_to_end(token)
    while len(_session_keys) > MAX_KEYS:
        _session_keys.popitem(last=False)
    return key


async def http_allowed(request: Any) -> bool:
    """ABU5: count one HTTP request, per login id for logged-in sessions, else per visitor IP."""
    from app.auth import COOKIE_NAME

    key = await _login_key_for(request.cookies.get(COOKIE_NAME)) or f"ip:{client_ip(request)}"
    return HTTP.take(key, limit("http_limit_per_minute"))


def check_new_session(conn: Any) -> None:
    """ABU4: raise 429 when this visitor IP has created too many anonymous sessions this hour."""
    if not NEW_SESSIONS.take(client_ip(conn), limit("anon_sessions_per_hour")):
        raise HTTPException(429, "Too many new sessions from your network. Please try again later.")


def check_login_ip(conn: Any) -> None:
    """ABU5: raise 429 past login_limit_per_minute login attempts from one IP."""
    if not LOGIN_IP.take(client_ip(conn), limit("login_limit_per_minute")):
        raise HTTPException(429, "Too many login attempts. Please wait a minute.")


def _account(login_id: str) -> str:
    return login_id.strip().lower()[:200]


def check_login_id(login_id: str) -> None:
    """ABU5: raise 429 while this login id has too many recent failures."""
    if LOGIN_FAILURES.count(_account(login_id)) >= limit("login_failures_per_id"):
        minutes = max(1, round(limit("login_failure_window_seconds") / 60))
        raise HTTPException(
            429, f"Too many failed attempts for this account. Please wait {minutes} minutes."
        )


def record_login_failure(login_id: str) -> None:
    """ABU5: count a failed login (wrong credentials, not an outage) for this login id."""
    LOGIN_FAILURES.hit(_account(login_id))
