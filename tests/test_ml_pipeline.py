"""Unit and integration tests for AIOps Expense Tracker ML & Analytics Pipeline."""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ml.analytics import (
    compute_category_mad_stats,
    detect_spending_anomalies,
    forecast_category_spend,
)
from ml.api import app, reset_state
from ml.categoriser import COLD_START_KEYWORD_RULES, ExpenseCategoriser
from ml.data import generate_ngn_expenses_dataset
from ml.ingestion import extract_receipt_fields, parse_bank_csv


def test_dataset_generation():
    df = generate_ngn_expenses_dataset(n_rows=600, seed=42)
    assert len(df) == 600
    expected_cols = [
        "tx_id",
        "date",
        "merchant",
        "narration",
        "channel",
        "bank",
        "amount_ngn",
        "category",
        "is_anomaly",
    ]
    for col in expected_cols:
        assert col in df.columns
    assert df["category"].nunique() == 10
    assert (df["amount_ngn"] > 0).all()


def test_categoriser_training_and_inference():
    df = generate_ngn_expenses_dataset(n_rows=600, seed=42)
    train_df = df.iloc[:450]
    test_df = df.iloc[450:]

    clf = ExpenseCategoriser(confidence_threshold=0.45, random_state=42)
    clf.fit(train_df)
    assert clf.is_fitted

    pred = clf.predict_one(
        merchant="SHOPRITE IKEJA",
        narration="POS PURCHASE SHOPRITE IKEJA CITY MALL",
        channel="POS",
        bank="GTBank",
    )
    assert pred["category"] == "Feeding & Groceries"
    assert 0.0 <= pred["confidence"] <= 1.0
    assert pred["source"] in ("ml_tfidf_logreg", "rule_fallback")

    # Batch evaluation
    preds = clf.predict_df(test_df)
    assert len(preds) == len(test_df)
    assert all("category" in p and 0.0 <= p["confidence"] <= 1.0 for p in preds)


def test_categoriser_cold_start_rules():
    clf = ExpenseCategoriser(confidence_threshold=0.99, random_state=42)
    # Even without fit, rules should apply
    pred = clf.predict_one(
        merchant="UBER TRIP",
        narration="UBER TRIP LAGOS ISLAND",
        channel="WEB",
        bank="Access Bank",
    )
    assert pred["category"] == "Transport & Fuel"
    assert pred["source"] == "rule_fallback"


def test_online_feedback_retraining():
    df = generate_ngn_expenses_dataset(n_rows=600, seed=42)
    train_df = df.iloc[:450]

    clf = ExpenseCategoriser(confidence_threshold=0.45, random_state=42)
    clf.fit(train_df)

    correction = pd.DataFrame(
        [
            {
                "merchant": "UNKNOWN VENDOR XYZ",
                "narration": "PAYMENT TO UNKNOWN VENDOR XYZ",
                "channel": "NIP",
                "bank": "Kuda",
                "category": "Equipment & Repairs",
            }
        ]
    )
    clf.fit_with_corrections(train_df, correction, correction_weight=10.0)
    new_pred = clf.predict_one(
        merchant="UNKNOWN VENDOR XYZ",
        narration="PAYMENT TO UNKNOWN VENDOR XYZ",
        channel="NIP",
        bank="Kuda",
    )
    assert new_pred["category"] == "Equipment & Repairs"


@pytest.mark.parametrize(
    "hostile",
    [
        "VAT" + " " * 9990 + "x",
        "VAT:" + " " * 9990 + "x",
        "TOTAL" + " " * 9990 + "x",
        "DATE" + " " * 9990 + "x",
        "vat " * 2500,
    ],
)
def test_receipt_parser_is_linear_on_hostile_text(hostile):
    # Audit 2026-11-01: "DATE" + ~2,500 spaces + "x" took longer than 15 s because the
    # optional prefix "\s*:?\s*" backtracked cubically; /v1/receipt/parse is unauthenticated.
    import time

    t0 = time.monotonic()
    extract_receipt_fields(hostile[:10_000])
    assert time.monotonic() - t0 < 1.0


def test_receipt_fields_with_spacing_variants():
    text = "Shop\nDate :  15/01/2026\nTotal Paid :  NGN  2,000.50\nVAT  (7.5%) :  N 140.00"
    f = extract_receipt_fields(text)
    assert f["date"] == "2026-01-15"
    assert f["total_ngn"] == 2000.50
    assert f["vat_ngn"] == 140.0


def test_receipt_ocr_extraction():
    sample_receipt = """
    *** SHOPRITE IKEJA CITY MALL ***
    DATE: 2026-01-15 14:32:00
    POS TERMINAL: 20394812
    TOTAL: NGN 24,500.00
    VAT (7.5%): NGN 1,709.30
    PAYMENT: POS DEBIT CARD
    THANK YOU FOR SHOPPING WITH US
    """
    fields = extract_receipt_fields(sample_receipt)
    assert "SHOPRITE" in fields["merchant"].upper()
    assert fields["date"] == "2026-01-15"
    assert fields["total_ngn"] == 24500.0
    assert fields["vat_ngn"] == pytest.approx(1709.3, rel=1e-2)
    assert fields["payment_channel"] == "POS"


def test_bank_csv_parsing():
    gtbank_csv = """Trans Date,Value Date,Description,Debit,Credit,Balance
01-Feb-2026,01-Feb-2026,POS PUR SHOPRITE IKEJA,15000.00,,450000.00
02-Feb-2026,02-Feb-2026,USSD MTN AIRTIME,2000.00,,448000.00
"""
    rows = parse_bank_csv(gtbank_csv, bank_hint="GTBank")
    assert len(rows) == 2
    assert rows[0]["merchant"] != ""
    assert rows[0]["amount_ngn"] == 15000.0
    assert rows[1]["amount_ngn"] == 2000.0


def test_mad_anomaly_detection():
    df = generate_ngn_expenses_dataset(n_rows=600, seed=42)
    stats = compute_category_mad_stats(df)
    assert len(stats) == 10

    test_txs = pd.DataFrame(
        [
            {
                "tx_id": "TEST-1",
                "date": "2026-02-15",
                "merchant": "STAMP DUTY",
                "narration": "NIP STAMP DUTY CHARGE",
                "category": "POS & Bank Charges",
                "amount_ngn": 50000.0,  # massive anomaly for bank charges (median is ~262)
            },
            {
                "tx_id": "TEST-2",
                "date": "2026-02-15",
                "merchant": "INTERSWITCH SURCHARGE",
                "narration": "ATM WITHDRAWAL SURCHARGE",
                "category": "POS & Bank Charges",
                "amount_ngn": 100.0,  # normal
            },
        ]
    )
    scored = detect_spending_anomalies(test_txs, z_threshold=2.5, reference_stats=stats)
    assert scored.loc[scored["tx_id"] == "TEST-1", "anomaly_flag"].iloc[0] == 1
    assert scored.loc[scored["tx_id"] == "TEST-2", "anomaly_flag"].iloc[0] == 0


def test_forecast_category_spend():
    df = generate_ngn_expenses_dataset(n_rows=600, seed=42)
    res = forecast_category_spend(df, horizon_months=3)
    assert res["horizon_months"] == 3
    assert len(res["future_months"]) == 3
    assert "category_forecasts" in res
    assert len(res["category_forecasts"]) == 10
    for cat, forecasts in res["category_forecasts"].items():
        assert len(forecasts) == 3
        for f in forecasts:
            assert f["forecast_ngn"] > 0
            assert f["ci80_low_ngn"] <= f["forecast_ngn"] <= f["ci80_high_ngn"]


def test_fastapi_endpoints(monkeypatch, tmp_path):
    # Demo data is opt-in (DEMO_DATA=1); use an isolated DB so the forecast has history.
    monkeypatch.setenv("DEMO_DATA", "1")
    monkeypatch.setenv("EXPENSES_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "m.joblib"))
    reset_state()
    client = TestClient(app)

    # Health
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    # Categorise
    cat_resp = client.post(
        "/v1/categorise",
        json={
            "merchant": "SHOPRITE",
            "narration": "POS PURCHASE SHOPRITE IKEJA",
            "channel": "POS",
            "bank": "GTBank",
        },
    )
    assert cat_resp.status_code == 200
    assert "category" in cat_resp.json()

    # Receipt parse
    rcpt_resp = client.post(
        "/v1/receipt/parse",
        json={
            "raw_text": "TOTAL: NGN 12,000.00\nDATE: 2026-02-10\nMERCHANT: NNPC FILLING STATION\nPAYMENT: POS"
        },
    )
    assert rcpt_resp.status_code == 200
    data = rcpt_resp.json()
    assert data["total_ngn"] == 12000.0
    assert "suggested_category" in data

    # Forecast
    fc_resp = client.get("/v1/forecast?horizon_months=3")
    assert fc_resp.status_code == 200
    assert "forecast_monthly_total_ngn" in fc_resp.json()
