"""SQLite storage for decision records. One table, append-only."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from support_agent.storage.schemas import Decision

# A notebook and a batch run write here at the same time routinely; without these
# the second one gets "database is locked" instead of simply waiting its turn.
_BUSY_TIMEOUT_S = 30.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    decision_id TEXT PRIMARY KEY,
    tweet_id TEXT NOT NULL,
    message TEXT NOT NULL,
    intent TEXT NOT NULL,
    language TEXT NOT NULL,
    classify_confidence REAL NOT NULL,
    top1_sim REAL NOT NULL,
    is_repeat INTEGER NOT NULL,
    reply TEXT,
    grounded_pair_ids TEXT NOT NULL,
    action TEXT NOT NULL,
    reason TEXT NOT NULL,
    artifact_version TEXT NOT NULL,
    intents_version TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    created_at TEXT NOT NULL
);
"""


# CREATE TABLE IF NOT EXISTS silently does nothing to a table that already exists,
# so a database written before a column was added keeps the old shape and every
# insert fails. Columns added after the first release go here.
_ADDED_COLUMNS = {
    "language": "language TEXT NOT NULL DEFAULT 'unknown'",
    "top1_sim": "top1_sim REAL NOT NULL DEFAULT 0.0",
    "is_repeat": "is_repeat INTEGER NOT NULL DEFAULT 0",
}


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    present = {row[1] for row in conn.execute("PRAGMA table_info(decisions)")}
    for column, ddl in _ADDED_COLUMNS.items():
        if column not in present:
            conn.execute(f"ALTER TABLE decisions ADD COLUMN {ddl}")


@contextmanager
def connect(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=_BUSY_TIMEOUT_S)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        conn.execute(_SCHEMA)
        _add_missing_columns(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def insert_decision(db_path: Path, decision: Decision) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO decisions (
                decision_id, tweet_id, message, intent, language, classify_confidence,
                top1_sim, is_repeat, reply, grounded_pair_ids, action, reason,
                artifact_version, intents_version, latency_ms, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision.decision_id,
                decision.tweet_id,
                decision.message,
                decision.intent,
                decision.language,
                decision.classify_confidence,
                decision.top1_sim,
                int(decision.is_repeat),
                decision.reply,
                json.dumps(decision.grounded_pair_ids),
                decision.action,
                decision.reason,
                decision.artifact_version,
                decision.intents_version,
                decision.latency_ms,
                decision.created_at.isoformat(),
            ),
        )


def fetch_all(db_path: Path) -> list[sqlite3.Row]:
    with connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute("SELECT * FROM decisions ORDER BY created_at").fetchall()
