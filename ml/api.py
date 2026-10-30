"""FastAPI Service for AIOps Expense Tracker ML & Analytics."""

from __future__ import annotations

import csv
import datetime as dt
import logging
import threading
from contextlib import asynccontextmanager
import time
from decimal import Decimal
from typing import Any, Literal

import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from ml.analytics import detect_spending_anomalies, forecast_category_spend
from ml.categoriser import ExpenseCategoriser
from ml.config import get_settings
from ml import store
from ml.data import CATEGORIES, generate_ngn_expenses_dataset
from ml.security import (
    limiter,
    rate_limit_default,
    rate_limit_feedback,
    require_api_key,
)
from ml.ingestion import extract_receipt_fields, parse_bank_csv

_STATE: dict[str, Any] = {}
_STATE_LOCK = threading.Lock()
_BASE_ROWS = 450
MAX_CSV_ROWS = 5000  # each row is classified synchronously (~5 ms); bounds request time
log = logging.getLogger("ml.api")


def reset_state() -> None:
    """Clear cached state (used by tests)."""
    with _STATE_LOCK:
        _STATE.clear()
    limiter.reset()


def _get_state() -> dict[str, Any]:
    with _STATE_LOCK:
        if "df" not in _STATE:
            settings = get_settings()
            if settings.dataset_path.exists():
                df = pd.read_csv(settings.dataset_path)
            else:
                df = generate_ngn_expenses_dataset(n_rows=600, seed=42)
            model = None
            # NOTE: joblib files are pickles; only load from a trusted local path.
            if settings.model_path.exists():
                try:
                    model = ExpenseCategoriser.load(settings.model_path)
                except Exception:  # noqa: BLE001
                    log.warning("Model file failed to load; retraining", exc_info=True)
            if model is None:
                model = ExpenseCategoriser(confidence_threshold=0.45, random_state=42)
                model.fit(df.iloc[:_BASE_ROWS])
                settings.model_path.parent.mkdir(parents=True, exist_ok=True)
                model.save(settings.model_path)
            _STATE["df"] = df
            _STATE["model"] = model
            if settings.demo_data and store.count_expenses() == 0:
                store.seed_rows(df.to_dict(orient="records"))
        return _STATE


@asynccontextmanager
async def lifespan(_: FastAPI):
    _get_state()
    yield


app = FastAPI(
    title="AIOps Expense Tracker API",
    description="NGN-first expense auto-categorisation, receipt-text parsing, bank CSV ingestion, MAD anomaly detection, and 3-month spend forecasting.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
    allow_credentials=False,
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """422 without echoing inputs: a JSON number like 1e400 parses to inf, which
    the default handler cannot serialise (it returned 500)."""
    errors = [{k: v for k, v in e.items() if k in ("type", "loc", "msg")} for e in exc.errors()]
    return JSONResponse({"detail": errors}, status_code=422)


Category = Literal[tuple(CATEGORIES)]  # type: ignore[valid-type]


class CategoriseItemRequest(BaseModel):
    merchant: str = Field(default="", max_length=120, description="Merchant or beneficiary name")
    narration: str = Field(..., min_length=1, max_length=500, description="Bank statement narration or expense description")
    channel: str = Field(default="POS", max_length=120, description="Payment channel: POS, NIP, USSD, WEB, BANK_FEE")
    bank: str = Field(default="GTBank", max_length=120, description="Bank or wallet name")


class FeedbackItemRequest(CategoriseItemRequest):
    true_category: Category = Field(..., description="User-corrected category label")


class ReceiptParseRequest(BaseModel):
    raw_text: str = Field(
        ..., min_length=1, max_length=10_000,
        description="Raw receipt text (e.g. pasted or from an external OCR tool)",
    )


class CsvImportRequest(BaseModel):
    csv_text: str = Field(..., max_length=1_000_000, description="Raw CSV content from GTBank, Access, OPay, or Kuda")
    bank_hint: str = Field(default="auto", max_length=60, description="Bank format hint or 'auto'")


class AnomalyTxItem(BaseModel):
    tx_id: str = Field(default="TX-1", max_length=60)
    date: str = Field(default="2026-02-15", max_length=20)
    merchant: str = Field(default="", max_length=120)
    narration: str = Field(default="", max_length=500)
    category: Category
    amount_ngn: float = Field(..., gt=0, le=1e10)


class AnomalyDetectRequest(BaseModel):
    transactions: list[AnomalyTxItem] = Field(..., max_length=5000)
    z_threshold: float = Field(default=2.5, ge=1.0, le=10.0)


@app.get("/health")
def health_check() -> dict[str, Any]:
    state = _get_state()
    return {
        "status": "ok",
        "service": "aiops-expense-tracker-ml",
        "dataset_rows": len(state["df"]),
        "model_fitted": bool(state["model"].is_fitted),
        "demo_data": store.has_seed_rows(),
        "expenses_count": store.count_expenses(),
    }


@app.post("/v1/categorise", dependencies=[Depends(rate_limit_default)])
def categorise_expense(req: CategoriseItemRequest) -> dict[str, Any]:
    state = _get_state()
    model: ExpenseCategoriser = state["model"]
    return model.predict_one(
        merchant=req.merchant,
        narration=req.narration,
        channel=req.channel,
        bank=req.bank,
    )


_RETRAIN_LOCK = threading.Lock()


def retrain_from_queue() -> dict[str, Any]:
    """Build a NEW model from base rows + every accepted correction, then swap it in.

    The live model is never mutated in place; on failure it is kept and the
    pending corrections stay queued.
    """
    if not _RETRAIN_LOCK.acquire(blocking=False):
        return {"status": "busy"}
    try:
        corrections = store.all_corrections()
        if not any(not c["applied"] for c in corrections):
            return {"status": "noop", "corrections_used": 0}
        started = time.monotonic()
        state = _get_state()
        corr_df = pd.DataFrame(corrections)[["merchant", "narration", "channel", "bank", "category"]]
        new_model = ExpenseCategoriser(confidence_threshold=0.45, random_state=42)
        try:
            new_model.fit_with_corrections(state["df"].iloc[:_BASE_ROWS], corr_df, correction_weight=5.0)
            new_model.save(get_settings().model_path)
        except Exception:  # noqa: BLE001
            log.exception("Retrain failed; previous model kept")
            return {"status": "failed"}
        with _STATE_LOCK:
            _STATE["model"] = new_model
        store.mark_applied(max(c["id"] for c in corrections))
        return {
            "status": "retrained",
            "corrections_used": len(corrections),
            "duration_ms": int((time.monotonic() - started) * 1000),
        }
    finally:
        _RETRAIN_LOCK.release()


@app.post(
    "/v1/categorise/feedback",
    status_code=202,
    dependencies=[Depends(require_api_key), Depends(rate_limit_feedback)],
)
def categorise_feedback(req: FeedbackItemRequest, background: BackgroundTasks) -> dict[str, Any]:
    _get_state()
    pending = store.add_correction(req.merchant, req.narration, req.channel, req.bank, req.true_category)
    batch = get_settings().feedback_batch_size
    status = "queued"
    if pending >= batch:
        background.add_task(retrain_from_queue)
        status = "queued_retrain_scheduled"
    return {"status": status, "pending_count": pending, "batch_size": batch}


@app.post(
    "/v1/categorise/retrain",
    dependencies=[Depends(require_api_key), Depends(rate_limit_feedback)],
)
def categorise_retrain() -> Any:
    result = retrain_from_queue()
    if result["status"] == "busy":
        return JSONResponse({"detail": "A retrain is already running"}, status_code=409)
    if result["status"] == "failed":
        return JSONResponse({"detail": "Retrain failed; previous model kept"}, status_code=500)
    return result


@app.post("/v1/receipt/parse", dependencies=[Depends(rate_limit_default)])
def parse_receipt_endpoint(req: ReceiptParseRequest) -> dict[str, Any]:
    parsed = extract_receipt_fields(req.raw_text)
    state = _get_state()
    model: ExpenseCategoriser = state["model"]
    cat_pred = model.predict_one(
        merchant=parsed["merchant"],
        narration=req.raw_text,
        channel=parsed["payment_channel"],
    )
    parsed["suggested_category"] = cat_pred["category"]
    parsed["category_confidence"] = cat_pred["confidence"]
    return parsed


@app.post(
    "/v1/csv/import",
    dependencies=[Depends(require_api_key), Depends(rate_limit_default)],
)
def import_bank_csv_endpoint(req: CsvImportRequest) -> dict[str, Any]:
    try:
        rows = parse_bank_csv(req.csv_text, bank_hint=req.bank_hint)
    except csv.Error as e:  # e.g. a field over the csv module's 131072-char limit
        raise HTTPException(422, f"Unreadable CSV: {e}") from None
    if len(rows) > MAX_CSV_ROWS:
        raise HTTPException(422, f"CSV has {len(rows)} rows; the limit is {MAX_CSV_ROWS} per request")
    state = _get_state()
    model: ExpenseCategoriser = state["model"]
    enriched: list[dict[str, Any]] = []
    for r in rows:
        pred = model.predict_one(
            merchant=r["merchant"],
            narration=r["narration"],
            channel=r["channel"],
            bank=r["bank"],
        )
        enriched.append({**r, **pred})
    return {
        "imported_count": len(enriched),
        "transactions": enriched,
    }


@app.post("/v1/anomalies", dependencies=[Depends(rate_limit_default)])
def detect_anomalies_endpoint(req: AnomalyDetectRequest) -> dict[str, Any]:
    state = _get_state()
    base_df: pd.DataFrame = state["df"]
    from ml.analytics import compute_category_mad_stats

    ref_stats = compute_category_mad_stats(base_df)
    in_df = pd.DataFrame([t.model_dump() for t in req.transactions])
    if in_df.empty:
        return {"flagged_count": 0, "results": []}
    scored_df = detect_spending_anomalies(in_df, z_threshold=req.z_threshold, reference_stats=ref_stats)
    results = scored_df.to_dict(orient="records")
    flagged = [r for r in results if r["anomaly_flag"] == 1]
    return {
        "evaluated_count": len(results),
        "flagged_count": len(flagged),
        "z_threshold": req.z_threshold,
        "results": results,
    }


def _store_df() -> pd.DataFrame:
    rows = store.all_expenses()
    if not rows:
        return pd.DataFrame(columns=["date", "month", "category", "amount_ngn"])
    df = pd.DataFrame(rows)
    df["month"] = df["date"].str.slice(0, 7)
    return df


@app.get("/v1/forecast")
def get_spend_forecast(horizon_months: int = Query(3, ge=1, le=12)) -> dict[str, Any]:
    _get_state()
    df = _store_df()
    if df["month"].nunique() < 3:
        raise HTTPException(422, "Need at least 3 months of history to forecast")
    return forecast_category_spend(df, horizon_months=horizon_months)


@app.get("/v1/categories")
def list_categories() -> dict[str, Any]:
    return {"categories": list(CATEGORIES)}


MONTH_RE = r"^\d{4}-(0[1-9]|1[0-2])$"


@app.get("/v1/expenses")
def list_expenses(
    month: str | None = Query(None, pattern=MONTH_RE),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    from ml.analytics import compute_category_mad_stats

    _get_state()
    items, total = store.list_expenses(month, limit, offset)
    if items:
        stats = compute_category_mad_stats(_store_df())
        scored = detect_spending_anomalies(pd.DataFrame(items), z_threshold=2.5, reference_stats=stats)
        for it, (_, sc) in zip(items, scored.iterrows()):
            it["anomaly"] = {
                "flag": bool(sc["anomaly_flag"]),
                "z_mad": float(sc["z_mad"]),
                "reason": str(sc["anomaly_reason"]),
            }
    return {"items": items, "total": total}


class ExpenseCreate(BaseModel):
    date: dt.date
    merchant: str = Field(default="", max_length=120)
    narration: str = Field(..., min_length=1, max_length=500)
    amount_ngn: float = Field(..., gt=0, le=1e10)
    channel: str = Field(default="POS", max_length=20)
    bank: str = Field(default="", max_length=60)
    category: Category | None = None

    @field_validator("date")
    @classmethod
    def _not_future(cls, v: dt.date) -> dt.date:
        if v > dt.date.today() + dt.timedelta(days=1):
            raise ValueError("date cannot be in the future")
        return v

    @field_validator("amount_ngn")
    @classmethod
    def _two_dp(cls, v: float) -> float:
        if Decimal(str(v)).as_tuple().exponent < -2:  # type: ignore[operator]
            raise ValueError("amount_ngn allows at most 2 decimal places")
        return v


@app.post(
    "/v1/expenses",
    status_code=201,
    dependencies=[Depends(require_api_key), Depends(rate_limit_default)],
)
def create_expense(req: ExpenseCreate) -> dict[str, Any]:
    from ml.analytics import compute_category_mad_stats

    state = _get_state()
    if req.category:
        category, source, conf = req.category, "user", None
    else:
        pred = state["model"].predict_one(
            merchant=req.merchant, narration=req.narration, channel=req.channel, bank=req.bank or "GTBank"
        )
        category, source, conf = pred["category"], pred["source"], pred["confidence"]
    row = store.insert_expense(
        date=req.date.isoformat(), merchant=req.merchant, narration=req.narration, channel=req.channel,
        bank=req.bank, amount_ngn=req.amount_ngn, category=category, category_source=source,
        category_confidence=conf,
    )
    scored = detect_spending_anomalies(
        pd.DataFrame([row]), z_threshold=2.5, reference_stats=compute_category_mad_stats(_store_df())
    ).iloc[0]
    row["anomaly"] = {
        "flag": bool(scored["anomaly_flag"]),
        "z_mad": float(scored["z_mad"]),
        "reason": str(scored["anomaly_reason"]),
    }
    return row


@app.get("/v1/summary")
def get_summary(month: str | None = Query(None, pattern=MONTH_RE)) -> dict[str, Any]:
    _get_state()
    months = store.available_months()
    month = month or (months[0] if months else None)
    if month is None:
        return {"month": None, "available_months": [], "total_ngn": 0, "count": 0, "by_category": [],
                "previous_month": None, "previous_total_ngn": None, "change_pct": None}
    total_k, count, cats = store.month_totals(month)
    prev_dt = (dt.date.fromisoformat(month + "-01") - dt.timedelta(days=1))
    prev = prev_dt.strftime("%Y-%m")
    prev_k, prev_n, _ = store.month_totals(prev)
    has_prev = prev_n > 0
    return {
        "month": month,
        "available_months": months,
        "total_ngn": total_k / 100,
        "count": count,
        "by_category": cats,
        "previous_month": prev if has_prev else None,
        "previous_total_ngn": prev_k / 100 if has_prev else None,
        "change_pct": round((total_k - prev_k) / prev_k * 100, 2) if prev_k else None,
    }
