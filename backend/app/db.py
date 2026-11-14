"""asyncpg pool, pgvector codec and idempotent schema setup."""

from pathlib import Path

import asyncpg
from pgvector.asyncpg import register_vector

from app.config import get_settings

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
_pool: asyncpg.Pool | None = None


async def _init_connection(conn: asyncpg.Connection) -> None:
    await register_vector(conn)


async def init_db() -> None:
    """Create the pool (once), make sure the vector extension exists, and apply schema.sql."""
    global _pool
    if _pool is not None:
        return
    dsn = get_settings().database_url
    # The extension must exist before the codec can be registered on pooled connections.
    boot = await asyncpg.connect(dsn)
    try:
        await boot.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await boot.execute(SCHEMA_PATH.read_text())
    finally:
        await boot.close()
    _pool = await asyncpg.create_pool(dsn, min_size=1, max_size=10, init=_init_connection)


def pool() -> asyncpg.Pool:
    """Return the open pool; raises if init_db() has not run."""
    if _pool is None:
        raise RuntimeError("database not initialised")
    return _pool


async def close_db() -> None:
    """Close the pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
