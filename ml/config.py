"""Runtime settings read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _path(name: str, default: str) -> Path:
    p = Path(os.environ.get(name, "") or default)
    return p if p.is_absolute() else REPO_ROOT / p


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    api_key: str
    allowed_origins: list[str]
    db_path: Path
    model_path: Path
    dataset_path: Path
    demo_data: bool
    rate_limit_per_minute: int
    feedback_rate_limit_per_minute: int
    feedback_batch_size: int


def get_settings() -> Settings:
    """Read settings from os.environ on every call (no caching) so tests can monkeypatch."""
    origins = os.environ.get("ALLOWED_ORIGINS", "") or "http://localhost:5173"
    return Settings(
        api_key=os.environ.get("EXPENSE_API_KEY", ""),
        allowed_origins=[o.strip() for o in origins.split(",") if o.strip()],
        db_path=_path("EXPENSES_DB_PATH", "data/expenses.db"),
        model_path=_path("MODEL_PATH", "models/categoriser.joblib"),
        dataset_path=_path("DATASET_PATH", "data/ngn_expenses_labelled.csv"),
        # Owner decision: synthetic demo rows load only when DEMO_DATA=1 (off by default).
        demo_data=os.environ.get("DEMO_DATA", "").strip().lower() in ("1", "true", "yes"),
        rate_limit_per_minute=_int("RATE_LIMIT_PER_MINUTE", 60),
        feedback_rate_limit_per_minute=_int("FEEDBACK_RATE_LIMIT_PER_MINUTE", 10),
        feedback_batch_size=_int("FEEDBACK_BATCH_SIZE", 20),
    )
