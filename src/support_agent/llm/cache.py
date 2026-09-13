"""sha256(key) -> response cache, sqlite-backed. Keeps eval/dev runs cheap and deterministic.

The caller decides what goes into the key. It must cover everything that changes the
meaning of a response - the model and the response schema, not just the prompt text -
or a replayed entry silently answers a question that was never asked.
"""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from pathlib import Path

# A notebook and a batch run touching the same file is normal here; without these
# the second writer gets "database is locked" instead of simply waiting its turn.
_BUSY_TIMEOUT_S = 30.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_cache (
    prompt_hash TEXT PRIMARY KEY,
    response TEXT NOT NULL
);
"""


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class LLMCache:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=_BUSY_TIMEOUT_S)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get(self, key: str) -> str | None:
        digest = hash_key(key)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT response FROM llm_cache WHERE prompt_hash = ?", (digest,)
            ).fetchone()
        return row[0] if row else None

    def set(self, key: str, response: str) -> None:
        digest = hash_key(key)
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache (prompt_hash, response) VALUES (?, ?)",
                (digest, response),
            )
