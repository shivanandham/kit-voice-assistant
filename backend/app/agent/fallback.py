"""Fallback model for when the primary Claude call fails: one Gemini call, no tools."""

import asyncio
import logging

from google import genai
from google.genai import types

from app.config import get_settings

log = logging.getLogger(__name__)

FALLBACK_TIMEOUT = 12.0
FALLBACK_NOTE = (
    "\n\nThe main assistant service is temporarily unavailable, so you are answering without "
    "tools. Actions (submitting, cancelling, approving, registering) and live record lookups "
    "(attendance, timetable, calendar, requests, placements) are not available right now. Answer "
    "only from the documents and the conversation above, and say plainly when something needs "
    "the live college system and that the person can try again in a few minutes."
)
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """The shared google-genai client, created on first use."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=get_settings().gemini_api_key)
    return _client


def _plain(content) -> str:
    """Text of a message's content blocks (tool calls and thinking are left out)."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        kind = block.get("type") if isinstance(block, dict) else getattr(block, "type", "")
        if kind == "text":
            parts.append(block["text"] if isinstance(block, dict) else block.text)
    return " ".join(parts)


async def answer(system: list[dict], messages: list[dict]) -> str | None:
    """Answer the conversation's last user message with the fallback model; None on failure."""
    try:
        settings = get_settings()
        instructions = "\n\n".join(b["text"] for b in system) + FALLBACK_NOTE
        contents = []
        for m in messages:
            text = _plain(m["content"]).strip()
            if text and m["role"] in ("user", "assistant"):
                role = "user" if m["role"] == "user" else "model"
                contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
        if not contents:
            return None
        response = await asyncio.wait_for(
            _get_client().aio.models.generate_content(
                model=settings.llm_fallback_model,
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=instructions,
                                                   max_output_tokens=1024),
            ),
            FALLBACK_TIMEOUT,
        )
        return (response.text or "").strip() or None
    except Exception:
        log.exception("fallback model failed")
        return None
