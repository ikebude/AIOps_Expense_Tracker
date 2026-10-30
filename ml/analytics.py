"""Per-Category Robust Z-Score (MAD) Anomaly Detector & 3-Month Spend Forecaster."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from ml.data import CATEGORIES


def compute_category_mad_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute per-category Median and Median Absolute Deviation (MAD) in NGN."""
    stats: dict[str, dict[str, float]] = {}
    for cat, grp in df.groupby("category"):
        vals = grp["amount_ngn"].astype(float).to_numpy()
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        if mad < 1e-6:
            q75, q25 = np.percentile(vals, [75, 25])
            mad = max(float((q75 - q25) / 1.349), 1.0)
        stats[str(cat)] = {
            "median_ngn": round(med, 2),
            "mad_ngn": round(mad, 2),
            "count": int(len(vals)),
        }
    return stats


def detect_spending_anomalies(
    df: pd.DataFrame,
    z_threshold: float = 2.5,
    reference_stats: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Flag unusual spending spikes per category using robust MAD z-score:

    `z_mad = 0.6745 * (x - median_c) / MAD_c`
    """
    out = df.copy()
    stats = reference_stats if reference_stats is not None else compute_category_mad_stats(out)

    z_scores: list[float] = []
    medians: list[float] = []
    mads: list[float] = []
    flags: list[int] = []
    reasons: list[str] = []

    for row in out.to_dict(orient="records"):
        cat = str(row.get("category", "Miscellaneous"))
        amt = float(row.get("amount_ngn", 0.0))
        cat_stat = stats.get(cat, {"median_ngn": amt, "mad_ngn": max(amt * 0.2, 1.0)})
        med = cat_stat["median_ngn"]
        mad = max(cat_stat["mad_ngn"], 1.0)
        z_mad = 0.6745 * (amt - med) / mad
        is_flagged = int(z_mad >= z_threshold)

        z_scores.append(round(float(z_mad), 3))
        medians.append(med)
        mads.append(mad)
        flags.append(is_flagged)
        if is_flagged:
            ratio = amt / max(med, 1.0)
            reasons.append(
                f"₦{amt:,.0f} is {ratio:.1f}x category median (₦{med:,.0f}) in '{cat}' (z_mad={z_mad:.2f})"
            )
        else:
            reasons.append("Within normal category band")

    out["category_median_ngn"] = medians
    out["category_mad_ngn"] = mads
    out["z_mad"] = z_scores
    out["anomaly_flag"] = flags
    out["anomaly_reason"] = reasons
    return out


def evaluate_anomaly_detector(df: pd.DataFrame, z_threshold: float = 2.5) -> dict[str, Any]:
    """Compare per-category MAD z-score against naive global mean/std z-score."""
    scored = detect_spending_anomalies(df, z_threshold=z_threshold)
    y_true = scored["is_anomaly"].astype(int).to_numpy()
    y_pred_mad = scored["anomaly_flag"].astype(int).to_numpy()

    # Naive global z-score baseline (gets skewed by Rent & Housing)
    all_amts = scored["amount_ngn"].astype(float).to_numpy()
    global_mean = float(np.mean(all_amts))
    global_std = max(float(np.std(all_amts)), 1.0)
    z_global = (all_amts - global_mean) / global_std
    y_pred_global = (z_global >= z_threshold).astype(int)

    return {
        "z_threshold": z_threshold,
        "total_transactions": int(len(scored)),
        "true_anomalies": int(np.sum(y_true)),
        "flagged_anomalies_mad": int(np.sum(y_pred_mad)),
        "mad_precision": round(float(precision_score(y_true, y_pred_mad, zero_division=0)), 4),
        "mad_recall": round(float(recall_score(y_true, y_pred_mad, zero_division=0)), 4),
        "mad_f1": round(float(f1_score(y_true, y_pred_mad, zero_division=0)), 4),
        "baseline_global_z_precision": round(float(precision_score(y_true, y_pred_global, zero_division=0)), 4),
        "baseline_global_z_recall": round(float(recall_score(y_true, y_pred_global, zero_division=0)), 4),
        "baseline_global_z_f1": round(float(f1_score(y_true, y_pred_global, zero_division=0)), 4),
    }


def _ets_forecast_series(
    values: np.ndarray,
    horizon: int = 3,
    alpha: float = 0.55,
    beta: float = 0.15,
    phi: float = 0.85,
) -> tuple[np.ndarray, float]:
    """Level + damped-trend exponential smoothing forecaster for monthly spend series.

    Returns `(forecast_array, residual_std)`.
    """
    n = len(values)
    if n == 0:
        return np.zeros(horizon), 0.0
    if n == 1:
        return np.repeat(values[0], horizon), float(values[0] * 0.1)

    level = float(values[0])
    trend = float(values[1] - values[0]) * 0.5
    fitted: list[float] = [level]

    for t in range(1, n):
        y_t = float(values[t])
        pred_t = level + phi * trend
        fitted.append(pred_t)
        prev_level = level
        level = alpha * y_t + (1.0 - alpha) * (prev_level + phi * trend)
        trend = beta * (level - prev_level) + (1.0 - beta) * phi * trend

    residuals = values - np.array(fitted)
    res_std = float(np.std(residuals)) if n > 2 else float(np.mean(values) * 0.1)

    forecasts: list[float] = []
    phi_sum = 0.0
    for h in range(1, horizon + 1):
        phi_sum += phi**h
        f_h = max(0.0, level + phi_sum * trend)
        forecasts.append(round(f_h, 2))

    return np.array(forecasts), round(res_std, 2)


def forecast_category_spend(df: pd.DataFrame, horizon_months: int = 3) -> dict[str, Any]:
    """Compute 3-month category & total NGN spend forecast plus holdout backtest MAE & WAPE.

    Model: level + damped-trend exponential smoothing with a fixed December x1.14 index;
    holdout actuals exclude MAD-flagged spikes. Baseline: naive last value.
"""
    work = df.copy()
    if "month" not in work.columns:
        work["month"] = pd.to_datetime(work["date"]).dt.strftime("%Y-%m")

    # Exclude anomalous spending spikes (flagged by per-category MAD z-score >= 2.5) before fitting baseline ETS
    scored = detect_spending_anomalies(work, z_threshold=2.5)
    clean_work = scored[scored["anomaly_flag"] == 0].copy()

    monthly_pivot = (
        clean_work.pivot_table(
            index="month",
            columns="category",
            values="amount_ngn",
            aggfunc="sum",
            fill_value=0.0,
        )
        .sort_index()
    )
    for cat in CATEGORIES:
        if cat not in monthly_pivot.columns:
            monthly_pivot[cat] = 0.0
    monthly_pivot = monthly_pivot[CATEGORIES]

    months = list(monthly_pivot.index)
    # Nigerian festive seasonal index (December has ~1.14x organic spend bump)
    seasonal_idx = {m: (1.14 if str(m).endswith("-12") else 1.00) for m in months}

    last_dt = pd.to_datetime(f"{months[-1]}-01")
    future_months = [
        (last_dt + pd.DateOffset(months=i)).strftime("%Y-%m")
        for i in range(1, horizon_months + 1)
    ]

    # Temporal holdout backtest: train on first (M - 2) months, evaluate on last 2 months across all 10 categories
    n_months = len(months)
    holdout_k = min(2, max(1, n_months - 3))
    train_pivot = monthly_pivot.iloc[:-holdout_k]
    test_pivot = monthly_pivot.iloc[-holdout_k:]
    train_months = list(train_pivot.index)
    test_months = list(test_pivot.index)

    actuals_all: list[float] = []
    preds_all: list[float] = []
    naive_preds_all: list[float] = []

    for cat in CATEGORIES:
        raw_tr = train_pivot[cat].to_numpy(dtype=float)
        deseason_tr = np.array([v / seasonal_idx[m] for v, m in zip(raw_tr, train_months)])
        te_vals = test_pivot[cat].to_numpy(dtype=float)
        fc_deseason, _ = _ets_forecast_series(deseason_tr, horizon=holdout_k)
        fc_vals = np.array([v * (1.14 if str(m).endswith("-12") else 1.0) for v, m in zip(fc_deseason, test_months)])
        naive_fc = np.repeat(raw_tr[-1], holdout_k)
        actuals_all.extend(te_vals.tolist())
        preds_all.extend(fc_vals.tolist())
        naive_preds_all.extend(naive_fc.tolist())

    act_arr = np.array(actuals_all)
    pred_arr = np.array(preds_all)
    naive_arr = np.array(naive_preds_all)

    mae_ngn = round(float(np.mean(np.abs(act_arr - pred_arr))), 2)
    wape = round(float(np.sum(np.abs(act_arr - pred_arr)) / max(np.sum(act_arr), 1.0)), 4)

    naive_mae_ngn = round(float(np.mean(np.abs(act_arr - naive_arr))), 2)
    naive_wape = round(float(np.sum(np.abs(act_arr - naive_arr)) / max(np.sum(act_arr), 1.0)), 4)

    # Fit on full 6 months to produce next 3 months forecast
    category_forecasts: dict[str, list[dict[str, Any]]] = {}
    monthly_totals_forecast = {m: 0.0 for m in future_months}

    for cat in CATEGORIES:
        raw_series = monthly_pivot[cat].to_numpy(dtype=float)
        deseason_series = np.array([v / seasonal_idx[m] for v, m in zip(raw_series, months)])
        fc_vals, res_std = _ets_forecast_series(deseason_series, horizon=horizon_months)
        cat_list: list[dict[str, Any]] = []
        for m_str, val in zip(future_months, fc_vals):
            s_mult = 1.14 if m_str.endswith("-12") else 1.00
            adj_val = val * s_mult
            low_80 = round(max(0.0, float(adj_val - 1.28 * res_std)), 2)
            high_80 = round(float(adj_val + 1.28 * res_std), 2)
            cat_list.append(
                {
                    "month": m_str,
                    "forecast_ngn": round(float(adj_val), 2),
                    "ci80_low_ngn": low_80,
                    "ci80_high_ngn": high_80,
                }
            )
            monthly_totals_forecast[m_str] = round(monthly_totals_forecast[m_str] + float(adj_val), 2)
        category_forecasts[cat] = cat_list

    historical_totals = {
        str(m): round(float(monthly_pivot.loc[m].sum()), 2) for m in months
    }

    return {
        "model": "damped_trend_ets_fixed_dec_index",
        "horizon_months": horizon_months,
        "future_months": future_months,
        "holdout_months_evaluated": holdout_k,
        "holdout_mae_ngn": mae_ngn,
        "holdout_wape": wape,
        "baseline_naive_last_value_mae_ngn": naive_mae_ngn,
        "baseline_naive_last_value_wape": naive_wape,
        "historical_monthly_total_ngn": historical_totals,
        "forecast_monthly_total_ngn": monthly_totals_forecast,
        "category_forecasts": category_forecasts,
    }
