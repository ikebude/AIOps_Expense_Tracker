"""Expense store and dashboard endpoint tests."""

from __future__ import annotations

KEY = {"X-API-Key": "s3cret"}


def _post(c, **kw):
    body = {"date": "2026-03-05", "narration": "POS PURCHASE SHOPRITE IKEJA", "amount_ngn": 100}
    body.update(kw)
    return c.post("/v1/expenses", json=body, headers=KEY)


def test_seeded_list_and_month_filter(make_client):
    c = make_client(DEMO_DATA="1")
    assert c.get("/v1/expenses?limit=5").json()["total"] == 600
    assert c.get("/v1/expenses?month=2026-02").json()["total"] == 100
    assert c.get("/health").json()["demo_data"] is True


def test_demo_data_off_by_default(make_client):
    c = make_client()
    assert c.get("/v1/expenses").json() == {"items": [], "total": 0}
    h = c.get("/health").json()
    assert h["demo_data"] is False and h["expenses_count"] == 0
    s = c.get("/v1/summary").json()
    assert s["month"] is None and s["total_ngn"] == 0 and s["by_category"] == []


def test_post_requires_key(make_client):
    c = make_client(EXPENSE_API_KEY="s3cret")
    body = {"date": "2026-03-05", "narration": "x", "amount_ngn": 1}
    assert c.post("/v1/expenses", json=body).status_code == 401
    c2 = make_client()
    assert c2.post("/v1/expenses", json=body).status_code == 503


def test_post_auto_categorises(make_client):
    c = make_client(EXPENSE_API_KEY="s3cret")
    r = _post(c, amount_ngn=1234.5)
    assert r.status_code == 201
    j = r.json()
    assert j["category_source"] in ("ml_tfidf_logreg", "rule_fallback")
    assert j["amount_ngn"] == 1234.5 and "anomaly" in j


def test_post_user_category(make_client):
    c = make_client(EXPENSE_API_KEY="s3cret")
    j = _post(c, category="Miscellaneous").json()
    assert j["category"] == "Miscellaneous" and j["category_source"] == "user"


def test_post_validation(make_client):
    c = make_client(EXPENSE_API_KEY="s3cret")
    for bad in (0, -1, 1.234, "abc"):
        assert _post(c, amount_ngn=bad).status_code == 422
    assert _post(c, date="2026-13-01").status_code == 422
    assert _post(c, date="2999-01-01").status_code == 422


def test_summary_exact_kobo_sum(make_client):
    c = make_client(EXPENSE_API_KEY="s3cret")
    _post(c, amount_ngn=0.10)
    _post(c, amount_ngn=0.20)
    s = c.get("/v1/summary?month=2026-03").json()
    assert s["total_ngn"] == 0.3 and s["count"] == 2


def test_summary_change_vs_previous(make_client):
    c = make_client(EXPENSE_API_KEY="s3cret")
    _post(c, date="2026-02-10", amount_ngn=100)
    _post(c, date="2026-03-10", amount_ngn=150)
    s = c.get("/v1/summary").json()
    assert s["month"] == "2026-03" and s["previous_month"] == "2026-02" and s["change_pct"] == 50.0


def test_hostile_strings_stored_literally(make_client):
    c = make_client(EXPENSE_API_KEY="s3cret")
    for text in ("'); DROP TABLE expenses;--", "<img src=x onerror=alert(1)>"):
        assert _post(c, narration=text).json()["narration"] == text
    items = c.get("/v1/expenses").json()
    assert items["total"] == 2
    assert {i["narration"] for i in items["items"]} == {"'); DROP TABLE expenses;--", "<img src=x onerror=alert(1)>"}


def test_forecast_needs_three_months(make_client):
    c = make_client()
    assert c.get("/v1/forecast").status_code == 422


def test_seeded_forecast_runs(make_client):
    # The store buckets by calendar date; the benchmark buckets by the dataset's own
    # `month` column (3 boundary rows differ), so WAPE here (0.0778) is not the 0.081 benchmark.
    c = make_client(DEMO_DATA="1")
    j = c.get("/v1/forecast").json()
    assert 0 < j["holdout_wape"] < 0.25


def test_categories(make_client):
    c = make_client()
    assert len(c.get("/v1/categories").json()["categories"]) == 10
