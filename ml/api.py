"""FastAPI Service for AIOps Expense Tracker ML & Analytics."""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.analytics import detect_spending_anomalies, forecast_category_spend
from ml.categoriser import ExpenseCategoriser
from ml.config import get_settings
from ml.data import generate_ngn_expenses_dataset
from ml.ingestion import extract_receipt_fields, parse_bank_csv

_STATE: dict[str, Any] = {}
_STATE_LOCK = threading.Lock()
_BASE_ROWS = 450
log = logging.getLogger("ml.api")


def reset_state() -> None:
    """Clear cached state (used by tests)."""
    with _STATE_LOCK:
        _STATE.clear()


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

class CategoriseItemRequest(BaseModel):
    merchant: str = Field(default="", description="Merchant or beneficiary name")
    narration: str = Field(..., description="Bank statement narration or expense description")
    channel: str = Field(default="POS", description="Payment channel: POS, NIP, USSD, WEB, BANK_FEE")
    bank: str = Field(default="GTBank", description="Bank or wallet name")


class FeedbackItemRequest(BaseModel):
    merchant: str = Field(default="", description="Merchant name")
    narration: str = Field(..., description="Bank narration string")
    channel: str = Field(default="POS", description="Channel")
    bank: str = Field(default="GTBank", description="Bank")
    true_category: str = Field(..., description="User-corrected category label")


class ReceiptParseRequest(BaseModel):
    raw_text: str = Field(..., description="Raw OCR text extracted from receipt or POS slip")


class CsvImportRequest(BaseModel):
    csv_text: str = Field(..., description="Raw CSV content from GTBank, Access, OPay, or Kuda")
    bank_hint: str = Field(default="auto", description="Bank format hint or 'auto'")


class AnomalyTxItem(BaseModel):
    tx_id: str = "TX-1"
    date: str = "2026-02-15"
    merchant: str = ""
    narration: str = ""
    category: str
    amount_ngn: float


class AnomalyDetectRequest(BaseModel):
    transactions: list[AnomalyTxItem]
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


@app.post("/v1/categorise")
def categorise_expense(req: CategoriseItemRequest) -> dict[str, Any]:
    state = _get_state()
    model: ExpenseCategoriser = state["model"]
    return model.predict_one(
        merchant=req.merchant,
        narration=req.narration,
        channel=req.channel,
        bank=req.bank,
    )


@app.post("/v1/categorise/feedback")
def categorise_feedback(req: FeedbackItemRequest) -> dict[str, Any]:
    state = _get_state()
    model: ExpenseCategoriser = state["model"]
    base_df: pd.DataFrame = state["df"].iloc[:450]
    corr_df = pd.DataFrame(
        [
            {
                "merchant": req.merchant,
                "narration": req.narration,
                "channel": req.channel,
                "bank": req.bank,
                "category": req.true_category,
            }
        ]
    )
    model.fit_with_corrections(base_df, corr_df, correction_weight=5.0)
    updated = model.predict_one(
        merchant=req.merchant,
        narration=req.narration,
        channel=req.channel,
        bank=req.bank,
    )
    return {
        "status": "retrained",
        "corrections_count": len(model.correction_history),
        "updated_prediction": updated,
    }


@app.post("/v1/receipt/parse")
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


@app.post("/v1/csv/import")
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


@app.post("/v1/anomalies")
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
def get_spend_forecast(horizon_months: int = 3) -> dict[str, Any]:
    state = _get_state()
    return forecast_category_spend(state["df"], horizon_months=horizon_months)
