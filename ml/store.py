"""SQLite persistence (stdlib only; parameterised queries only)."""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
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


# ---- expenses ----------------------------------------------------------------

def to_kobo(amount_ngn: float) -> int:
    return int((Decimal(str(amount_ngn)) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _row_to_expense(r: sqlite3.Row) -> dict[str, Any]:
    d = dict(r)
    d["amount_ngn"] = d.pop("amount_kobo") / 100
    d.pop("created_at", None)
    return d


def count_expenses() -> int:
    with connect() as c:
        return c.execute("SELECT COUNT(*) FROM expenses").fetchone()[0]


def has_seed_rows() -> bool:
    with connect() as c:
        return c.execute("SELECT 1 FROM expenses WHERE category_source = 'seed' LIMIT 1").fetchone() is not None


def insert_expense(
    *, date: str, merchant: str, narration: str, channel: str, bank: str, amount_ngn: float,
    category: str, category_source: str, category_confidence: float | None, exp_id: str | None = None,
) -> dict[str, Any]:
    exp_id = exp_id or f"EXP-{uuid.uuid4().hex[:12]}"
    created = now_iso()
    with connect() as c:
        c.execute(
            "INSERT INTO expenses (id, date, merchant, narration, channel, bank, amount_kobo, category,"
            " category_source, category_confidence, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (exp_id, date, merchant, narration, channel, bank, to_kobo(amount_ngn), category,
             category_source, category_confidence, created),
        )
        row = c.execute("SELECT * FROM expenses WHERE id = ?", (exp_id,)).fetchone()
    return _row_to_expense(row)


def seed_rows(rows: list[dict[str, Any]]) -> None:
    """Bulk insert dataset rows as category_source='seed' (only into an empty table)."""
    created = now_iso()
    with connect() as c:
        c.executemany(
            "INSERT OR IGNORE INTO expenses (id, date, merchant, narration, channel, bank, amount_kobo,"
            " category, category_source, category_confidence, created_at) VALUES (?,?,?,?,?,?,?,?,'seed',NULL,?)",
            [
                (r["tx_id"], r["date"], r["merchant"], r["narration"], r["channel"], r["bank"],
                 to_kobo(r["amount_ngn"]), r["category"], created)
                for r in rows
            ],
        )


def list_expenses(month: str | None, limit: int, offset: int) -> tuple[list[dict[str, Any]], int]:
    where, args = ("WHERE substr(date, 1, 7) = ?", [month]) if month else ("", [])
    with connect() as c:
        total = c.execute(f"SELECT COUNT(*) FROM expenses {where}", args).fetchone()[0]  # noqa: S608 (fixed fragment)
        rows = c.execute(
            f"SELECT * FROM expenses {where} ORDER BY date DESC, created_at DESC, id DESC LIMIT ? OFFSET ?",  # noqa: S608
            [*args, limit, offset],
        ).fetchall()
    return [_row_to_expense(r) for r in rows], total


def all_expenses() -> list[dict[str, Any]]:
    with connect() as c:
        rows = c.execute("SELECT * FROM expenses ORDER BY date, id").fetchall()
    return [_row_to_expense(r) for r in rows]


def available_months() -> list[str]:
    with connect() as c:
        rows = c.execute("SELECT DISTINCT substr(date, 1, 7) AS m FROM expenses ORDER BY m DESC").fetchall()
    return [r["m"] for r in rows]


def month_totals(month: str) -> tuple[int, int, list[dict[str, Any]]]:
    """Return (total_kobo, count, by_category) for a month, summed in integer kobo."""
    with connect() as c:
        rows = c.execute(
            "SELECT category, SUM(amount_kobo) AS kobo, COUNT(*) AS n FROM expenses"
            " WHERE substr(date, 1, 7) = ? GROUP BY category ORDER BY kobo DESC, category",
            (month,),
        ).fetchall()
    cats = [{"category": r["category"], "total_ngn": r["kobo"] / 100, "count": r["n"]} for r in rows]
    return sum(r["kobo"] for r in rows), sum(r["n"] for r in rows), cats
