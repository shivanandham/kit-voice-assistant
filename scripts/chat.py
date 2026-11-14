"""Text REPL for the assistant loop: `uv run python scripts/chat.py [--login arjun]`.

Prints streamed events with timings (time to first text, total) and per-call cache usage.
"""

import argparse
import asyncio
import json
import logging
import secrets
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app import db, history
from app.agent import loop
from app.auth import SessionCtx, _create_session
from app.kit.client import KitClient

PASSWORD = "Kit@2026"


class _UsageLine(logging.Handler):
    """Echo the loop's per-call usage log lines in a compact form."""

    def emit(self, record: logging.LogRecord) -> None:
        print(f"\n  [{record.getMessage()}]", flush=True)


async def _session(login_id: str | None) -> SessionCtx:
    """A visitor session, or a logged-in one via the college API."""
    if not login_id:
        _, ctx = await _create_session(f"visitor:{secrets.token_hex(12)}", "visitor")
        return ctx
    data = await KitClient().login(login_id, PASSWORD)
    user = data["user"]
    _, ctx = await _create_session(
        f"user:{user['login_id']}", user["role"], user, data["access_token"], data.get("expires_in")
    )
    return ctx


async def main() -> None:
    """Initialise the database, open a session and chat until EOF or 'quit'."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--login", help="college login id (password Kit@2026); visitor if omitted")
    parser.add_argument("--warm", action="store_true",
                        help="pre-warm the prompt cache first, as the app does at startup")
    args = parser.parse_args()

    log = logging.getLogger("app.agent.loop")
    log.setLevel(logging.INFO)
    log.addHandler(_UsageLine())
    log.propagate = False

    await db.init_db()
    try:
        if args.warm:
            started = time.monotonic()
            await loop.warm_cache()
            print(f"Cache warmed in {time.monotonic() - started:.2f}s")
        ctx = await _session(args.login)
        cid = await history.create_conversation(ctx)
        name = (ctx.user or {}).get("full_name", "visitor")
        print(f"Session: {ctx.role} ({name}), conversation {cid}")
        print("WARNING: saying yes to a pending action makes the assistant call "
              "confirm_pending_action, which performs a REAL write on the college sandbox.")
        print("Type a message, or 'quit'.\n")
        while True:
            try:
                line = await asyncio.to_thread(input, "you> ")
            except EOFError:
                break
            if line.strip().lower() in {"quit", "exit"}:
                break
            if not line.strip():
                continue
            t0 = time.monotonic()
            first = None
            print("kit> ", end="", flush=True)
            async for event in loop.run_turn(ctx, cid, line):
                if event["type"] == "text_delta":
                    first = first or time.monotonic() - t0
                    print(event["text"], end="", flush=True)
                elif event["type"] == "tool":
                    flag = " ERROR" if event.get("is_error") else ""
                    print(f"\n  <{event['name']} {event['state']}{flag}>", flush=True)
                elif event["type"] == "citations":
                    print(f"\n  sources: {json.dumps(event['sources'])}", flush=True)
                elif event["type"] == "error":
                    print(f"\n  !error: {event['message']}")
            ttft = f"{first:.2f}s" if first else "-"
            print(f"\n  (first text {ttft}, total {time.monotonic() - t0:.2f}s)\n")
    finally:
        await db.close_db()


if __name__ == "__main__":
    asyncio.run(main())
