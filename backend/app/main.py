"""FastAPI app: lifespan, routers, public college endpoints, pages, security headers."""

import asyncio
import contextlib
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app import auth, history, limits
from app.config import get_settings
from app.db import close_db, init_db, pool
from app.kit.client import KitClient, close_http
from app.knowledge import store
from app.voice import session as voice_session

log = logging.getLogger(__name__)
FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
PUBLIC_CACHE_SECONDS = 300
WARM_INTERVAL_SECONDS = 240  # the prompt cache lives five minutes
WARM_FIRST_SYNC_TIMEOUT = 120  # warm anyway if the first document sync takes longer
WARM_START_DELAY = 20  # used only when the store offers no first-sync signal

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "microphone=(self), camera=(), geolocation=()",
}
CSP = (
    "default-src 'self'; connect-src 'self' {ws}; img-src 'self' data:; "
    "media-src 'self' blob:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
)


def _csp() -> str:
    """The CSP, allowing WebSockets only to our own public origin (wss:// for https)."""
    origin = get_settings().public_origin.rstrip("/")
    scheme, sep, rest = origin.partition("://")
    ws = {"https": "wss", "http": "ws"}.get(scheme.lower()) if sep else None
    return CSP.format(ws=f"{ws}://{rest}" if ws else "").replace("'self' ;", "'self';")


async def _keep_warm() -> None:
    """Warm the model's prompt cache after the first document sync, then every few minutes."""
    from app.agent import loop as agent_loop  # imported late: built as a separate module

    wait_first_sync = getattr(store, "wait_first_sync", None)
    try:
        if wait_first_sync is not None:
            await asyncio.wait_for(wait_first_sync(), WARM_FIRST_SYNC_TIMEOUT)
        else:
            await asyncio.sleep(WARM_START_DELAY)
    except asyncio.CancelledError:
        raise
    except Exception:
        log.warning("waiting for the first document sync failed; warming anyway")
    while True:
        try:
            await agent_loop.warm_cache()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.warning("prompt cache warm-up failed", exc_info=True)
        await asyncio.sleep(WARM_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the database, the document sync loop and the cache keep-warm task; on shutdown stop
    them and close the college HTTP client and the database."""
    await init_db()
    await store.start_sync_loop()
    warm = asyncio.create_task(_keep_warm()) if get_settings().llm_keep_warm else None
    try:
        yield
    finally:
        if warm is not None:
            warm.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await warm
        try:
            await store.stop_sync_loop()
        finally:
            try:
                await close_http()
            finally:
                await close_db()


# No /docs, /redoc or /openapi.json: the API is private to our own frontend.
app = FastAPI(
    title="KIT Voice Assistant", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None
)
app.include_router(auth.router)
app.include_router(history.router)
app.include_router(voice_session.router)
app.add_middleware(limits.BodySizeLimit)  # ABU10; added first so security headers wrap it


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Per-IP request limit (ABU5, static files excluded) and security headers on every
    response, including unexpected 500s."""
    try:
        if not request.url.path.startswith("/static/") and not await limits.http_allowed(request):
            response = JSONResponse(
                {"detail": "Too many requests. Please slow down and try again in a minute."},
                status_code=429,
            )
        else:
            response = await call_next(request)
    except Exception:
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        response = PlainTextResponse("Internal Server Error", status_code=500)
    for key, value in {**SECURITY_HEADERS, "Content-Security-Policy": _csp()}.items():
        response.headers.setdefault(key, value)
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 without echoing the submitted values (they may contain a password)."""
    errors = [{"loc": e.get("loc"), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()]
    return JSONResponse({"detail": errors}, status_code=422)


@app.get("/healthz")
async def healthz() -> dict:
    """Liveness plus database check."""
    try:
        await pool().fetchval("SELECT 1")
        db_state = "ok"
    except Exception:
        db_state = "down"
    return {
        "status": "ok" if db_state == "ok" else "degraded",
        "db": db_state,
        "doc_mode": get_settings().doc_mode,
    }


_public_cache: dict[str, tuple[float, object]] = {}


async def _cached_public(key: str):
    cached = _public_cache.get(key)
    if cached and time.monotonic() - cached[0] < PUBLIC_CACHE_SECONDS:
        return cached[1]
    try:
        data = await getattr(KitClient(), key)()
    except Exception:
        raise HTTPException(503, "The college system is unavailable right now.") from None
    _public_cache[key] = (time.monotonic(), data)
    return data


@app.get("/api/public/college")
async def public_college():
    """College profile, cached for five minutes."""
    return await _cached_public("college")


@app.get("/api/public/courses")
async def public_courses():
    """Course catalogue, cached for five minutes."""
    return await _cached_public("courses")


async def _page(request: Request, name: str, *, login_required: bool = False) -> Response:
    # Every page goes through session_from_request so a cookie exists before any WebSocket opens.
    if get_settings().ui_variant == "app":
        name = "app/index.html"  # the built React UI (ui/), one index for every page
    page = FileResponse(FRONTEND / name)
    ctx = await auth.session_from_request(request, page)
    if login_required and ctx.role == "visitor":
        redirect = RedirectResponse("/login", status_code=303)
        for raw in page.headers.getlist("set-cookie"):
            redirect.headers.append("set-cookie", raw)
        return redirect
    return page


@app.get("/")
async def index(request: Request):
    """Home page."""
    return await _page(request, "index.html")


@app.get("/login")
async def login_page(request: Request):
    """Login page."""
    return await _page(request, "login.html")


@app.get("/assistant")
async def assistant_page(request: Request):
    """Assistant page; redirects to /login when not logged in."""
    return await _page(request, "assistant.html", login_required=True)


@app.get("/history")
async def history_page(request: Request):
    """History page."""
    return await _page(request, "history.html")


app.mount("/static", StaticFiles(directory=FRONTEND / "static", check_dir=False), name="static")
app.mount("/ui-assets", StaticFiles(directory=FRONTEND / "app", check_dir=False), name="ui-assets")
