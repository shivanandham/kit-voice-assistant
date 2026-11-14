"""Conversation history, always scoped by owner_key."""

import json
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response

from app.auth import SessionCtx, session_from_request
from app.db import pool

router = APIRouter()
TITLE_LIMIT = 60


async def create_conversation(ctx: SessionCtx) -> str:
    """Start a conversation owned by this session's owner; returns its uuid."""
    cid = uuid.uuid4()
    await pool().execute(
        "INSERT INTO conversations (id, owner_key, role) VALUES ($1, $2, $3)",
        cid,
        ctx.owner_key,
        ctx.role,
    )
    return str(cid)


def _uuid(value: object) -> uuid.UUID | None:
    if not isinstance(value, str):
        return None
    try:
        return uuid.UUID(value)
    except ValueError:
        return None


async def conversation_owned(ctx: SessionCtx, conversation_id: str) -> bool:
    """True if the conversation exists and belongs to this owner."""
    cid = _uuid(conversation_id)
    if cid is None:
        return False
    return bool(
        await pool().fetchval(
            "SELECT 1 FROM conversations WHERE id = $1 AND owner_key = $2", cid, ctx.owner_key
        )
    )


async def add_message(
    conversation_id: str, role: str, content: str, citations: list[dict] | None = None
) -> None:
    """Append a message (with optional citations), set the title from the first user message,
    bump updated_at."""
    cid = uuid.UUID(conversation_id)
    title = " ".join(content.split())[:TITLE_LIMIT] if role == "user" else None
    async with pool().acquire() as conn, conn.transaction():
        await conn.execute(
            "INSERT INTO messages (conversation_id, role, content, citations)"
            " VALUES ($1, $2, $3, $4::jsonb)",
            cid,
            role,
            content,
            json.dumps(citations) if citations is not None else None,
        )
        await conn.execute(
            "UPDATE conversations SET updated_at = now(),"
            " title = COALESCE(title, NULLIF($2, '')) WHERE id = $1",
            cid,
            title,
        )


async def recent_messages(conversation_id: str, limit: int = 40) -> list[dict]:
    """The last `limit` messages, oldest first, as [{role, content}]."""
    rows = await pool().fetch(
        "SELECT role, content FROM (SELECT id, role, content FROM messages"
        " WHERE conversation_id = $1 ORDER BY id DESC LIMIT $2) t ORDER BY id",
        uuid.UUID(conversation_id),
        limit,
    )
    return [{"role": r["role"], "content": r["content"]} for r in rows]


async def log_tool_call(
    conversation_id: str, name: str, input: dict, output: Any, is_error: bool
) -> None:
    """Record a tool call for auditing."""
    await pool().execute(
        "INSERT INTO tool_calls (conversation_id, name, input, output, is_error)"
        " VALUES ($1, $2, $3::jsonb, $4::jsonb, $5)",
        uuid.UUID(conversation_id),
        name,
        json.dumps(input, default=str),
        json.dumps(output, default=str),
        is_error,
    )


@router.get("/api/history")
async def list_history(request: Request, response: Response) -> list[dict]:
    """The caller's conversations, newest first (empty ones are skipped)."""
    ctx = await session_from_request(request, response)
    rows = await pool().fetch(
        "SELECT c.id, c.title, c.started_at, c.updated_at, count(m.id) AS message_count"
        " FROM conversations c JOIN messages m ON m.conversation_id = c.id"
        " WHERE c.owner_key = $1 GROUP BY c.id ORDER BY c.updated_at DESC",
        ctx.owner_key,
    )
    return [
        {
            "id": str(r["id"]),
            "title": r["title"],
            "started_at": r["started_at"].isoformat(),
            "updated_at": r["updated_at"].isoformat(),
            "message_count": r["message_count"],
        }
        for r in rows
    ]


@router.get("/api/history/{conversation_id}")
async def get_history(conversation_id: str, request: Request, response: Response) -> dict:
    """One conversation with its messages; 404 unless the caller owns it."""
    ctx = await session_from_request(request, response)
    cid = _uuid(conversation_id)
    row = None
    if cid is not None:
        row = await pool().fetchrow(
            "SELECT id, title FROM conversations WHERE id = $1 AND owner_key = $2",
            cid,
            ctx.owner_key,
        )
    if row is None:
        raise HTTPException(404, "Conversation not found")
    msgs = await pool().fetch(
        "SELECT role, content, created_at, citations FROM messages WHERE conversation_id = $1"
        " ORDER BY id",
        cid,
    )
    return {
        "id": str(row["id"]),
        "title": row["title"],
        "messages": [
            {
                "role": m["role"],
                "content": m["content"],
                "created_at": m["created_at"].isoformat(),
                "citations": json.loads(m["citations"]) if m["citations"] else None,
            }
            for m in msgs
        ],
    }
