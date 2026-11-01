"""Auth, validation, retrain safety, rate limit and CORS tests for the API."""

from __future__ import annotations

from ml.categoriser import ExpenseCategoriser

KEY = {"X-API-Key": "s3cret"}
FB = {"narration": "POS PURCHASE ACME", "true_category": "Miscellaneous"}
SHOP = {"narration": "POS PURCHASE SHOPRITE IKEJA"}


def _still_sane(c):
    r = c.post("/v1/categorise", json=SHOP).json()
    assert isinstance(r["confidence"], float) and 0 <= r["confidence"] <= 1
    assert r["category"] == "Feeding & Groceries"


def test_feedback_disabled_without_key(make_client):
    c = make_client()
    assert c.post("/v1/categorise/feedback", json=FB, headers=KEY).status_code == 503


def test_feedback_requires_correct_key(make_client):
    c = make_client(EXPENSE_API_KEY="s3cret")
    assert c.post("/v1/categorise/feedback", json=FB).status_code == 401
    assert c.post("/v1/categorise/feedback", json=FB, headers={"X-API-Key": "nope"}).status_code == 401
    assert c.post("/v1/categorise/feedback?api_key=s3cret", json=FB).status_code == 401


def test_invalid_category_rejected_and_model_intact(make_client):
    c = make_client(EXPENSE_API_KEY="s3cret")
    bad = {**FB, "true_category": "NOT_A_CATEGORY"}
    assert c.post("/v1/categorise/feedback", json=bad, headers=KEY).status_code == 422
    _still_sane(c)


def test_queue_then_retrain_then_noop(make_client):
    c = make_client(EXPENSE_API_KEY="s3cret")
    r = c.post("/v1/categorise/feedback", json=FB, headers=KEY)
    assert r.status_code == 202
    assert r.json()["status"] == "queued" and r.json()["pending_count"] == 1
    r = c.post("/v1/categorise/retrain", headers=KEY)
    assert r.status_code == 200 and r.json()["status"] == "retrained"
    assert r.json()["corrections_used"] == 1
    assert c.post("/v1/categorise/retrain", headers=KEY).json()["status"] == "noop"


def test_retrain_failure_keeps_old_model(make_client, monkeypatch):
    c = make_client(EXPENSE_API_KEY="s3cret")
    c.post("/v1/categorise/feedback", json=FB, headers=KEY)

    def boom(*a, **k):
        raise ValueError("boom")

    monkeypatch.setattr(ExpenseCategoriser, "fit_with_corrections", boom)
    assert c.post("/v1/categorise/retrain", headers=KEY).status_code == 500
    _still_sane(c)


def test_feedback_rate_limited(make_client):
    c = make_client(EXPENSE_API_KEY="s3cret", FEEDBACK_RATE_LIMIT_PER_MINUTE=2)
    codes = [c.post("/v1/categorise/feedback", json=FB, headers=KEY).status_code for _ in range(3)]
    assert codes == [202, 202, 429]
    r = c.post("/v1/categorise/feedback", json=FB, headers=KEY)
    assert "retry-after" in r.headers


def test_input_bounds(make_client):
    c = make_client()
    assert c.post("/v1/categorise", json={"narration": "x" * 501}).status_code == 422
    assert c.post("/v1/receipt/parse", json={"raw_text": "x" * 10_001}).status_code == 422
    assert c.get("/v1/forecast?horizon_months=0").status_code == 422
    assert c.get("/v1/forecast?horizon_months=13").status_code == 422
    tx = {"category": "Miscellaneous", "amount_ngn": -5}
    assert c.post("/v1/anomalies", json={"transactions": [tx]}).status_code == 422


def test_non_finite_number_is_422_not_500(make_client):
    # A raw JSON 1e400 parses to inf; the default 422 handler could not serialise it (500).
    c = make_client()
    body = '{"transactions": [{"category": "Miscellaneous", "amount_ngn": 1e400}]}'
    r = c.post("/v1/anomalies", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert "input" not in r.json()["detail"][0]


def test_csv_import_requires_key(make_client):
    c = make_client(EXPENSE_API_KEY="s3cret")
    body = {"csv_text": "a,b\n1,2"}
    assert c.post("/v1/csv/import", json=body).status_code == 401


def test_csv_import_hostile_files(make_client):
    c = make_client(EXPENSE_API_KEY="s3cret")
    head = "Trans. Date,Value Date,Remarks,Debit,Credit,Balance\n"
    row = "02-Jan-2026,02-Jan-2026,POS SHOPRITE,5000,,1\n"
    imp = lambda text: c.post("/v1/csv/import", json={"csv_text": text}, headers=KEY)  # noqa: E731
    # Field over the csv module limit: was an unhandled csv.Error (500).
    assert imp(head + "02-Jan-2026,02-Jan-2026," + "A" * 200_000 + ",5000,,1\n").status_code == 422
    # Row cap bounds the synchronous classification time.
    assert imp(head + row * 5001).status_code == 422
    # A 400-digit amount overflowed to inf and came back as amount_ngn: null; now the row is skipped.
    assert imp(head + "02-Jan-2026,02-Jan-2026,POS SHOPRITE," + "9" * 400 + ",,1\n").json()["imported_count"] == 0
    # UTF-8 BOM no longer hides the date header (the date silently became today).
    r = imp("﻿" + head + row).json()
    assert r["transactions"][0]["date"] == "2026-01-02"


def test_cors_allowlist(make_client):
    c = make_client()
    evil = c.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in evil.headers
    ok = c.get("/health", headers={"Origin": "http://localhost:5173"})
    assert ok.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_csv_import_of_5000_rows_is_batched(make_client):
    # Audit 2026-11-01: rows were classified one predict_proba call at a time (~28 s for 5,000).
    import time

    from ml.categoriser import ExpenseCategoriser
    from ml.data import generate_ngn_expenses_dataset

    df = generate_ngn_expenses_dataset(n_rows=600, seed=42)
    model = ExpenseCategoriser(confidence_threshold=0.45, random_state=42).fit(df.iloc[:450])
    rows = df.iloc[450:].to_dict(orient="records")
    assert model.predict_many(rows) == [
        model.predict_one(str(r["merchant"]), str(r["narration"]), str(r["channel"]), str(r["bank"])) for r in rows
    ]

    c = make_client(EXPENSE_API_KEY="k" * 32, RATE_LIMIT_PER_MINUTE=1000)
    csv_text = "Remarks,Debit\n" + "POS PUR SHOPRITE LEKKI,1500\n" * 5000
    t0 = time.monotonic()
    r = c.post("/v1/csv/import", json={"csv_text": csv_text}, headers={"X-API-Key": "k" * 32})
    assert r.status_code == 200 and r.json()["imported_count"] == 5000
    assert time.monotonic() - t0 < 10
