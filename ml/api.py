"""FastAPI Service for AIOps Expense Tracker ML & Analytics."""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
import time
from typing import Any, Literal

import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

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
    rows = parse_bank_csv(req.csv_text, bank_hint=req.bank_hint)
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


@app.get("/v1/forecast")
def get_spend_forecast(horizon_months: int = Query(3, ge=1, le=12)) -> dict[str, Any]:
    state = _get_state()
    return forecast_category_spend(state["df"], horizon_months=horizon_months)
