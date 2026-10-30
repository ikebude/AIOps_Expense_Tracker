"""SQLite persistence (stdlib only; parameterised queries only)."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from ml.config import get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS expenses (
  id TEXT PRIMARY KEY,
  date TEXT NOT NULL,
  merchant TEXT NOT NULL DEFAULT '',
  narration TEXT NOT NULL,
  channel TEXT NOT NULL DEFAULT 'POS',
  bank TEXT NOT NULL DEFAULT '',
  amount_kobo INTEGER NOT NULL CHECK (amount_kobo > 0),
  category TEXT NOT NULL,
  category_source TEXT NOT NULL,
  category_confidence REAL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_expenses_date ON expenses(date);
CREATE TABLE IF NOT EXISTS corrections (
  id INTEGER PRIMARY KEY AUTOINCREMENT, merchant TEXT, narration TEXT NOT NULL, channel TEXT, bank TEXT,
  category TEXT NOT NULL, created_at TEXT NOT NULL, applied INTEGER NOT NULL DEFAULT 0
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    path = get_settings().db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        with conn:
            yield conn
    finally:
        conn.close()


# ---- corrections -------------------------------------------------------------

def add_correction(merchant: str, narration: str, channel: str, bank: str, category: str) -> int:
    """Insert a correction and return the number of pending (unapplied) ones."""
    with connect() as c:
        c.execute(
            "INSERT INTO corrections (merchant, narration, channel, bank, category, created_at) VALUES (?,?,?,?,?,?)",
            (merchant, narration, channel, bank, category, now_iso()),
        )
        return c.execute("SELECT COUNT(*) FROM corrections WHERE applied = 0").fetchone()[0]


def pending_count() -> int:
    with connect() as c:
        return c.execute("SELECT COUNT(*) FROM corrections WHERE applied = 0").fetchone()[0]


def all_corrections() -> list[dict[str, Any]]:
    with connect() as c:
        rows = c.execute(
            "SELECT id, merchant, narration, channel, bank, category, applied FROM corrections ORDER BY id"
        ).fetchall()
    return [dict(r) for r in rows]


def mark_applied(max_id: int) -> None:
    with connect() as c:
        c.execute("UPDATE corrections SET applied = 1 WHERE id <= ?", (max_id,))
