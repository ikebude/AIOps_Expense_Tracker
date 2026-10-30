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


def test_csv_import_requires_key(make_client):
    c = make_client(EXPENSE_API_KEY="s3cret")
    body = {"csv_text": "a,b\n1,2"}
    assert c.post("/v1/csv/import", json=body).status_code == 401


def test_cors_allowlist(make_client):
    c = make_client()
    evil = c.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in evil.headers
    ok = c.get("/health", headers={"Origin": "http://localhost:5173"})
    assert ok.headers["access-control-allow-origin"] == "http://localhost:5173"
