"""Cookie sessions stored hashed in Postgres, college login, and WebSocket origin check."""

import hashlib
import json
import secrets
import time  # noqa: F401 - tests freeze auth.time.monotonic; app.limits reads it
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Request, Response, WebSocket
from pydantic import BaseModel

from app import limits
from app.config import get_settings
from app.db import pool

COOKIE_NAME = "kit_session"
KIT_TOKEN_MARGIN_SECONDS = 60
LOGIN_WINDOW_SECONDS = limits.LOGIN_IP.window  # rate-limit rules live in app.limits
UNAVAILABLE = "The college system is unavailable. Try again shortly."
EXPIRED_PRUNE_BATCH = 100  # expired session rows deleted per new session
# The only profile fields kept from the college (CONTRACT.md SessionCtx.user). Anything else,
# e.g. free-text `notes` that may carry injected instructions, never reaches a client or a prompt.
PROFILE_FIELDS = ("id", "login_id", "full_name", "role", "department", "class_id", "class_code",
                  "programme", "year_of_study")

router = APIRouter()


@dataclass(frozen=True)
class SessionCtx:
    """Who is making a request. kit_token is the college JWT and never leaves the server."""

    session_id: str
    owner_key: str
    role: Literal["visitor", "student", "teacher"]
    user: dict | None
    kit_token: str | None


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def profile(user: dict | None) -> dict | None:
    """The allow-listed profile: PROFILE_FIELDS only, scalar values only (others become None)."""
    if not isinstance(user, dict):
        return None
    scalar = (str, int, float, bool)
    return {
        k: v if v is None or isinstance(v, scalar) else None
        for k, v in ((k, user.get(k)) for k in PROFILE_FIELDS)
    }


def _ctx_from_row(row) -> SessionCtx:
    user = row["user_json"]
    if isinstance(user, str):
        user = json.loads(user)
    return SessionCtx(
        session_id=row["id"],
        owner_key=row["owner_key"],
        role=row["role"],
        user=profile(user),
        kit_token=row["kit_token"],
    )


async def _lookup(token: str | None) -> SessionCtx | None:
    if not token:
        return None
    row = await pool().fetchrow(
        "SELECT * FROM sessions WHERE id = $1 AND expires_at > now()", _hash(token)
    )
    if row is None:
        return None
    expiry = row["kit_token_expires_at"]
    if expiry is not None and expiry - timedelta(seconds=KIT_TOKEN_MARGIN_SECONDS) <= datetime.now(
        UTC
    ):
        await invalidate_session(row["id"])
        return None
    return _ctx_from_row(row)


async def invalidate_session(session_id: str) -> None:
    """Delete a session (e.g. when the college API rejects its token)."""
    await pool().execute("DELETE FROM sessions WHERE id = $1", session_id)


def _set_cookie(response: Response, token: str) -> None:
    settings = get_settings()
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=settings.session_ttl_hours * 3600,
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
        path="/",
    )


async def _create_session(
    owner_key: str,
    role: str,
    user: dict | None = None,
    kit_token: str | None = None,
    kit_expires_in: int | None = None,
) -> tuple[str, SessionCtx]:
    """Insert a session under a fresh random token; returns (cookie token, ctx).

    Only the allow-listed profile is stored, and a bounded batch of expired rows is pruned.
    """
    user = profile(user)
    await pool().execute(
        "DELETE FROM sessions WHERE id IN"
        " (SELECT id FROM sessions WHERE expires_at < now() LIMIT $1)",
        EXPIRED_PRUNE_BATCH,
    )
    token = secrets.token_urlsafe(32)
    sid = _hash(token)
    now = datetime.now(UTC)
    kit_expiry = now + timedelta(seconds=kit_expires_in) if kit_expires_in else None
    expires = now + timedelta(hours=get_settings().session_ttl_hours)
    await pool().execute(
        "INSERT INTO sessions (id, owner_key, role, user_json, kit_token,"
        " kit_token_expires_at, expires_at) VALUES ($1,$2,$3,$4,$5,$6,$7)",
        sid,
        owner_key,
        role,
        json.dumps(user) if user else None,
        kit_token,
        kit_expiry,
        expires,
    )
    return token, SessionCtx(sid, owner_key, role, user, kit_token)  # type: ignore[arg-type]


async def session_from_request(request: Request, response: Response) -> SessionCtx:
    """Return the caller's session, creating a visitor session and cookie if there is none.

    New anonymous sessions are limited per IP (ABU4); over the limit this raises 429.
    """
    ctx = await _lookup(request.cookies.get(COOKIE_NAME))
    if ctx:
        return ctx
    limits.check_new_session(request)
    token, ctx = await _create_session(f"visitor:{secrets.token_hex(12)}", "visitor")
    _set_cookie(response, token)
    return ctx


async def session_from_websocket(ws: WebSocket) -> SessionCtx | None:
    """Return the session named by the cookie, or None if missing or expired."""
    return await _lookup(ws.cookies.get(COOKIE_NAME))


def origin_allowed(origin: str | None) -> bool:
    """Check a WebSocket Origin header against public_origin (localhost too when not secure)."""
    if not origin:
        return False
    settings = get_settings()
    if origin.rstrip("/").lower() == settings.public_origin.rstrip("/").lower():
        return True
    if not settings.cookie_secure:
        return _local_origin(origin)
    return False


def _local_origin(origin: str) -> bool:
    """True for a bare http(s) origin on a loopback host, e.g. http://localhost:8000."""
    try:
        parts = urlparse(origin)
        parts.port  # noqa: B018 - raises ValueError for a malformed port
    except ValueError:
        return False
    return (
        parts.scheme in ("http", "https")
        and parts.hostname in {"localhost", "127.0.0.1", "::1"}
        and parts.path in ("", "/")
        and not (parts.query or parts.fragment or parts.username or parts.password)
    )


# --- rate limits: thin callers into app.limits ---

_client_ip = limits.client_ip  # ABU9: the one client-IP rule


def reset_rate_limit() -> None:
    """Forget every recorded limit event (used by tests)."""
    limits.reset_all()


async def login_rate_limit(request: Request) -> None:
    """Dependency: ABU5 login attempts per IP, else 429."""
    limits.check_login_ip(request)


class LoginBody(BaseModel):
    """Login form payload."""

    login_id: str
    password: str


@router.post("/api/login", dependencies=[Depends(login_rate_limit)])
async def login(body: LoginBody, request: Request, response: Response) -> dict:
    """Log in against the college API and issue a brand-new session."""
    from app.kit.client import KitClient, KitError

    limits.check_login_id(body.login_id)
    try:
        data = await KitClient().login(body.login_id, body.password)
    except KitError as exc:
        if not exc.status or exc.status >= 500:  # status 0: unreachable or timed out
            raise HTTPException(503, UNAVAILABLE) from None
        limits.record_login_failure(body.login_id)
        raise HTTPException(401, exc.message or "Login failed") from None
    except Exception:
        raise HTTPException(503, UNAVAILABLE) from None

    # The role and identity come only from the college's answer; refuse anything unexpected.
    user = data.get("user") if isinstance(data, dict) else None
    kit_token = data.get("access_token") if isinstance(data, dict) else None
    if not (
        isinstance(user, dict)
        and user.get("role") in ("student", "teacher")
        and isinstance(user.get("login_id"), str)
        and user["login_id"]
        and isinstance(kit_token, str)
        and kit_token
    ):
        raise HTTPException(503, UNAVAILABLE)
    expires_in = data.get("expires_in")
    if not isinstance(expires_in, int) or isinstance(expires_in, bool):
        expires_in = None

    old = request.cookies.get(COOKIE_NAME)
    if old:
        await pool().execute("DELETE FROM sessions WHERE id = $1", _hash(old))
    token, ctx = await _create_session(
        f"user:{user['login_id']}",
        user["role"],
        user,
        kit_token,
        expires_in,
    )
    _set_cookie(response, token)
    return {"user": ctx.user}


@router.post("/api/logout")
async def logout(request: Request, response: Response) -> dict:
    """Delete the session and clear the cookie."""
    token = request.cookies.get(COOKIE_NAME)
    if token:
        await pool().execute("DELETE FROM sessions WHERE id = $1", _hash(token))
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"ok": True}


@router.get("/api/me")
async def me(request: Request, response: Response) -> dict:
    """Current role and user (null for visitors)."""
    ctx = await session_from_request(request, response)
    return {"role": ctx.role, "user": ctx.user}
