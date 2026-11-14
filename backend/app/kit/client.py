"""Async client for the KIT college API (https://ht.knowvally.com).

One shared httpx.AsyncClient is reused by every KitClient instance. Every call carries the
X-Participant-Key header; logged-in calls add the college JWT as a bearer token. Errors are raised
as KitError carrying the college's own code and message, so callers can show them verbatim.
"""

import asyncio
import re
import time
from typing import Any

import httpx

from app.config import get_settings

TIMEOUT = httpx.Timeout(10.0, connect=5.0)
GET_RETRIES = 2
RETRY_BACKOFF_SECONDS = 0.4
TODAY_TTL_SECONDS = 60.0

JWT_RE = re.compile(r"eyJ[\w-]+\.[\w-]+\.[\w-]*")
BEARER_RE = re.compile(r"(?i)bearer\s+\S+")
REDACTED = "[redacted]"

_http: httpx.AsyncClient | None = None
_today_cache: tuple[float, dict] | None = None


class KitError(Exception):
    """An error from the college API (or failure to reach it).

    status is the HTTP status (0 when the API could not be reached); code, message and field come
    from the college's error body {"error": {"code", "message", "field"}}.
    """

    def __init__(self, status: int, code: str, message: str, field: str | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.field = field

    def to_dict(self) -> dict:
        """Compact form for tool results."""
        out: dict[str, Any] = {"error": self.message, "code": self.code}
        if self.field:
            out["field"] = self.field
        return out


def _shared_http() -> httpx.AsyncClient:
    global _http
    if _http is None or _http.is_closed:
        _http = httpx.AsyncClient(base_url=get_settings().kit_base_url, timeout=TIMEOUT)
    return _http


async def close_http() -> None:
    """Close the shared HTTP client (app shutdown)."""
    global _http
    if _http is not None:
        await _http.aclose()
        _http = None


def redact(text: str, secrets: tuple[str | None, ...] = ()) -> str:
    """Remove the participant key, the given secrets and anything shaped like a JWT or bearer token."""
    for secret in (get_settings().kit_participant_key, *secrets):
        if secret:
            text = text.replace(secret, REDACTED)
    return JWT_RE.sub(REDACTED, BEARER_RE.sub(f"Bearer {REDACTED}", text))


def _error_from_response(resp: httpx.Response, secrets: tuple[str | None, ...] = ()) -> KitError:
    err = _parse_error(resp)
    err.message = redact(err.message, secrets)
    err.args = (err.message,)
    if err.field:
        err.field = redact(err.field, secrets)
    return err


def _parse_error(resp: httpx.Response) -> KitError:
    try:
        body = resp.json()
    except ValueError:
        body = None
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        err = body["error"]
        return KitError(
            resp.status_code,
            str(err.get("code") or f"http_{resp.status_code}"),
            str(err.get("message") or "The college system returned an error."),
            err.get("field"),
        )
    if isinstance(body, dict) and isinstance(body.get("detail"), list) and body["detail"]:
        first = body["detail"][0]
        loc = first.get("loc") or []
        return KitError(resp.status_code, "invalid_input", str(first.get("msg", "Invalid input.")),
                        str(loc[-1]) if loc else None)
    return KitError(resp.status_code, f"http_{resp.status_code}",
                    f"The college system returned an unexpected error (HTTP {resp.status_code}).")


class KitClient:
    """Thin async wrapper: one method per college endpoint, returning parsed JSON."""

    def __init__(self, token: str | None = None):
        self._token = token

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json: Any = None,
        idempotency_key: str | None = None,
        retry_post: bool = True,
        raw: bool = False,
    ) -> Any:
        headers = {"X-Participant-Key": get_settings().kit_participant_key}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        if params:
            params = {k: v for k, v in params.items() if v is not None}
        retryable = method == "GET" or (idempotency_key is not None and retry_post)
        attempts = 1 + (GET_RETRIES if retryable else 0)
        last: KitError | None = None
        for attempt in range(attempts):
            if attempt:
                await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)
            try:
                resp = await _shared_http().request(
                    method, path, params=params or None, json=json, headers=headers
                )
            except httpx.TimeoutException:
                last = KitError(0, "timeout", "The college system took too long to respond.")
                continue
            except httpx.TransportError:
                last = KitError(0, "unavailable", "The college system could not be reached.")
                continue
            if resp.status_code >= 500:
                last = _error_from_response(resp, (self._token,))
                continue
            if resp.status_code >= 400:
                raise _error_from_response(resp, (self._token,))
            if raw:
                return resp.content
            return resp.json() if resp.content else None
        assert last is not None
        raise last

    async def _get(self, path: str, **params: Any) -> Any:
        return await self._request("GET", path, params=params)

    # Public
    async def health(self) -> dict:
        """GET /v1/health."""
        return await self._get("/v1/health")

    async def today(self) -> dict:
        """GET /v1/today: the college's current date (not the real date)."""
        return await self._get("/v1/today")

    async def college(self) -> dict:
        """GET /v1/college."""
        return await self._get("/v1/college")

    async def courses(self) -> list[dict]:
        """GET /v1/courses."""
        return await self._get("/v1/courses")

    async def documents(self) -> list[dict]:
        """GET /v1/documents."""
        return await self._get("/v1/documents")

    async def document_file(self, id: int) -> bytes:
        """GET /v1/documents/{id}/file as raw PDF bytes."""
        return await self._request("GET", f"/v1/documents/{int(id)}/file", raw=True)

    async def submit_admission(self, payload: dict, idempotency_key: str) -> dict:
        """POST /v1/admissions/applications."""
        return await self._request("POST", "/v1/admissions/applications", json=payload,
                                   idempotency_key=idempotency_key)

    async def login(self, login_id: str, password: str) -> dict:
        """POST /v1/auth/login -> {access_token, token_type, expires_in, user}."""
        return await self._request("POST", "/v1/auth/login",
                                   json={"login_id": login_id, "password": password})

    async def placement_drives(self) -> list[dict]:
        """GET /v1/placements/drives."""
        return await self._get("/v1/placements/drives")

    # Logged-in user
    async def me(self) -> dict:
        """GET /v1/me."""
        return await self._get("/v1/me")

    async def academics(self) -> dict:
        """GET /v1/me/academics (students)."""
        return await self._get("/v1/me/academics")

    async def subjects(self) -> list[dict]:
        """GET /v1/me/subjects (students)."""
        return await self._get("/v1/me/subjects")

    async def timetable(self) -> list[dict]:
        """GET /v1/me/timetable: weekly slots, weekday 1=Mon..6=Sat (students)."""
        return await self._get("/v1/me/timetable")

    async def calendar(self, from_date: str | None = None, to_date: str | None = None) -> list[dict]:
        """GET /v1/calendar?from&to (students). Without dates the college returns upcoming events."""
        return await self._get("/v1/calendar", **{"from": from_date, "to": to_date})

    async def attendance(self) -> dict:
        """GET /v1/me/attendance (students)."""
        return await self._get("/v1/me/attendance")

    async def attendance_sessions(
        self, subject: str | None = None, from_date: str | None = None, to_date: str | None = None
    ) -> list[dict]:
        """GET /v1/me/attendance/sessions?subject&from&to (students)."""
        return await self._get("/v1/me/attendance/sessions",
                               **{"subject": subject, "from": from_date, "to": to_date})

    async def my_requests(self) -> list[dict]:
        """GET /v1/me/requests (students)."""
        return await self._get("/v1/me/requests")

    async def create_request(self, payload: dict, idempotency_key: str) -> dict:
        """POST /v1/me/requests: leave or on-duty request (students)."""
        return await self._request("POST", "/v1/me/requests", json=payload,
                                   idempotency_key=idempotency_key)

    async def get_request(self, id: int) -> dict:
        """GET /v1/me/requests/{id} (students; own requests only)."""
        return await self._get(f"/v1/me/requests/{int(id)}")

    async def cancel_request(self, id: int, idempotency_key: str | None = None) -> dict:
        """POST /v1/me/requests/{id}/cancel (students). The key is sent but never retried on."""
        return await self._request("POST", f"/v1/me/requests/{int(id)}/cancel",
                                   idempotency_key=idempotency_key, retry_post=False)

    async def register_drive(self, id: int, idempotency_key: str | None = None) -> dict:
        """POST /v1/placements/drives/{id}/register (students). The key is sent, never retried on."""
        return await self._request("POST", f"/v1/placements/drives/{int(id)}/register",
                                   idempotency_key=idempotency_key, retry_post=False)

    async def my_placements(self) -> list[dict]:
        """GET /v1/me/placements (students)."""
        return await self._get("/v1/me/placements")

    # Teachers
    async def teacher_classes(self) -> list[dict]:
        """GET /v1/teacher/classes: classes the teacher may act on today."""
        return await self._get("/v1/teacher/classes")

    async def teacher_requests(self, status: str | None = None) -> list[dict]:
        """GET /v1/teacher/requests?status."""
        return await self._get("/v1/teacher/requests", status=status)

    async def class_attendance(self, class_id: int) -> dict:
        """GET /v1/teacher/classes/{class_id}/attendance."""
        return await self._get(f"/v1/teacher/classes/{int(class_id)}/attendance")

    async def approve_request(
        self, id: int, comment: str | None = None, idempotency_key: str | None = None
    ) -> dict:
        """POST /v1/teacher/requests/{id}/approve. The key is sent but never retried on."""
        return await self._request("POST", f"/v1/teacher/requests/{int(id)}/approve",
                                   json={"comment": comment}, idempotency_key=idempotency_key,
                                   retry_post=False)

    async def reject_request(
        self, id: int, reason: str, idempotency_key: str | None = None
    ) -> dict:
        """POST /v1/teacher/requests/{id}/reject. The key is sent but never retried on."""
        return await self._request("POST", f"/v1/teacher/requests/{int(id)}/reject",
                                   json={"reason": reason}, idempotency_key=idempotency_key,
                                   retry_post=False)


async def college_today() -> dict:
    """The college date from /v1/today, cached for 60 s: {date, weekday, weekday_name, timezone}."""
    global _today_cache
    now = time.monotonic()
    if _today_cache and now - _today_cache[0] < TODAY_TTL_SECONDS:
        return _today_cache[1]
    data = await KitClient().today()
    _today_cache = (now, data)
    return data


def clear_today_cache() -> None:
    """Forget the cached college date (tests, or after a known date change)."""
    global _today_cache
    _today_cache = None
