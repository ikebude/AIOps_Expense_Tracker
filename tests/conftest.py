"""Shared fixtures: isolated DB/model paths per test, one trained model reused."""

from __future__ import annotations

import shutil

import pytest
from fastapi.testclient import TestClient

from ml import api
from ml.categoriser import ExpenseCategoriser
from ml.data import generate_ngn_expenses_dataset


@pytest.fixture(scope="session")
def trained_model_file(tmp_path_factory):
    df = generate_ngn_expenses_dataset(n_rows=600, seed=42)
    model = ExpenseCategoriser(confidence_threshold=0.45, random_state=42)
    model.fit(df.iloc[:450])
    path = tmp_path_factory.mktemp("model") / "base.joblib"
    model.save(path)
    return path


@pytest.fixture
def make_client(monkeypatch, tmp_path, trained_model_file):
    """Factory: make_client(**env) -> context-managed TestClient with isolated state."""
    stack = []

    def _make(**env):
        model_path = tmp_path / "m.joblib"
        shutil.copy(trained_model_file, model_path)
        monkeypatch.setenv("EXPENSES_DB_PATH", str(tmp_path / "t.db"))
        monkeypatch.setenv("MODEL_PATH", str(model_path))
        monkeypatch.setenv("EXPENSE_API_KEY", "")
        monkeypatch.delenv("DEMO_DATA", raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, str(v))
        api.reset_state()
        c = TestClient(api.app)
        c.__enter__()
        stack.append(c)
        return c

    yield _make
    for c in stack:
        c.__exit__(None, None, None)
    api.reset_state()
