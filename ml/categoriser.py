"""ML Auto-Categoriser (Word + Char-WB TF-IDF + Calibrated Logistic Regression) with Cold-Start Rules and Online Retraining."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from ml.data import CATEGORIES

# Keyword rules used both as cold-start fallback (when ML confidence < 0.45) and baseline comparator
COLD_START_KEYWORD_RULES: dict[str, tuple[str, ...]] = {
    "Feeding & Groceries": (
        "shoprite",
        "spar",
        "bokku",
        "chicken republic",
        "mama put",
        "buka",
        "grocery",
        "supermarket",
        "bread",
        "amala",
        "sweet sensation",
        "canteen",
    ),
    "Transport & Fuel": (
        "nnpc",
        "totalenergies",
        "mobil",
        "pms",
        "fuel",
        "petrol",
        "diesel",
        "bolt",
        "uber",
        "lcc",
        "toll",
        "cowry",
        "brt",
        "air peace",
        "ibom air",
        "gigm",
        "flight",
    ),
    "Data & Utilities": (
        "mtn",
        "airtel",
        "ekedc",
        "ikedc",
        "electricity",
        "meter",
        "token",
        "buypower",
        "spectranet",
        "fibre",
        "dstv",
        "gotv",
        "multichoice",
        "lawma",
        "vtu",
        "data bundle",
    ),
    "Rent & Housing": (
        "rent",
        "landlord",
        "estate",
        "service charge",
        "tenancy",
        "caution deposit",
        "coworking",
        "workstation",
        "lease",
        "apartment",
    ),
    "Family & Dependents": (
        "mum",
        "upkeep",
        "parents",
        "school fees",
        "tuition",
        "corona school",
        "sibling",
        "remittance",
        "dependent",
        "allowance",
        "baby care",
    ),
    "Business SaaS & Cloud": (
        "aws",
        "amazon web services",
        "digitalocean",
        "github",
        "openai",
        "anthropic",
        "claude",
        "google workspace",
        "cloudflare",
        "paystack",
        "saas",
        "cloud hosting",
    ),
    "POS & Bank Charges": (
        "stamp duty",
        "emtl",
        "sms alert",
        "card maintenance",
        "transfer fee",
        "withdrawal charge",
        "pos agent",
        "account maintenance",
        "commission",
    ),
    "Healthcare": (
        "medplus",
        "healthplus",
        "pharmacy",
        "reddington",
        "hospital",
        "luth",
        "synlab",
        "clinic",
        "axa mansard",
        "hmo",
        "dental",
        "medical",
        "prescription",
    ),
    "Equipment & Repairs": (
        "computer village",
        "ssd",
        "laptop",
        "slot systems",
        "screen replacement",
        "mikano",
        "generator",
        "inverter",
        "battery",
        "thermocool",
        "compressor",
        "repair",
    ),
    "Miscellaneous": (
        "terra kulture",
        "filmhouse",
        "cinema",
        "landmark beach",
        "jumia",
        "barbershop",
        "grooming",
        "charity",
        "tithe",
    ),
}


def normalize_text(text: str) -> str:
    """Normalize Nigerian bank narration strings while preserving merchant and channel tokens."""
    if not text:
        return ""
    cleaned = text.lower()
    # Strip long numeric reference numbers (6+ digits) and REF codes
    cleaned = re.sub(r"\bref\s*\d+\b", " ", cleaned)
    cleaned = re.sub(r"\b\d{6,}\b", " ", cleaned)
    cleaned = re.sub(r"[/|_\-,:;]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def compose_feature_text(
    merchant: str = "",
    narration: str = "",
    channel: str = "",
    bank: str = "",
) -> str:
    """Combine merchant, narration, channel, and bank into a normalized feature string."""
    raw = f"{merchant} {narration} {channel} {bank}"
    return normalize_text(raw)


def rule_based_predict_one(text: str) -> tuple[str, float, str]:
    """Predict expense category using deterministic keyword rules.

    Returns (category, pseudo_confidence, matched_keyword).
    """
    norm = normalize_text(text)
    best_cat = "Miscellaneous"
    best_hits = 0
    matched_kw = "default_fallback"

    for cat, keywords in COLD_START_KEYWORD_RULES.items():
        hits = 0
        first_kw = ""
        for kw in keywords:
            if kw in norm:
                hits += 1
                if not first_kw:
                    first_kw = kw
        if hits > best_hits:
            best_hits = hits
            best_cat = cat
            matched_kw = first_kw

    if best_hits == 0:
        return "Miscellaneous", 0.35, "no_keyword_match"
    conf = min(0.85, 0.55 + 0.12 * (best_hits - 1))
    return best_cat, round(conf, 4), matched_kw


class ExpenseCategoriser:
    """Hybrid TF-IDF (word + char_wb) + Calibrated Logistic Regression with Cold-Start Rules."""

    def __init__(self, confidence_threshold: float = 0.45, random_state: int = 42) -> None:
        self.confidence_threshold = confidence_threshold
        self.random_state = random_state
        self.feature_union = FeatureUnion(
            transformer_list=[
                (
                    "word_tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=1,
                        max_features=6000,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char_tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=2,
                        max_features=8000,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
        self.base_lr = LogisticRegression(
            C=2.5,
            class_weight="balanced",
            max_iter=1000,
            random_state=self.random_state,
        )
        self.classifier = CalibratedClassifierCV(
            estimator=self.base_lr,
            method="sigmoid",
            cv=3,
        )
        self.pipeline = Pipeline(
            [
                ("features", self.feature_union),
                ("clf", self.classifier),
            ]
        )
        self.is_fitted = False
        self.correction_history: list[dict[str, Any]] = []

    @staticmethod
    def _df_to_texts(df: pd.DataFrame) -> list[str]:
        merchants = df["merchant"].fillna("").astype(str) if "merchant" in df else pd.Series([""] * len(df))
        narrations = df["narration"].fillna("").astype(str) if "narration" in df else pd.Series([""] * len(df))
        channels = df["channel"].fillna("").astype(str) if "channel" in df else pd.Series([""] * len(df))
        banks = df["bank"].fillna("").astype(str) if "bank" in df else pd.Series([""] * len(df))
        return [
            compose_feature_text(m, n, c, b)
            for m, n, c, b in zip(merchants, narrations, channels, banks)
        ]

    def fit(self, df: pd.DataFrame, sample_weight: np.ndarray | None = None) -> "ExpenseCategoriser":
        """Fit the TF-IDF + Calibrated LogisticRegression pipeline."""
        texts = self._df_to_texts(df)
        y = df["category"].astype(str).to_numpy()
        fit_params: dict[str, Any] = {}
        if sample_weight is not None:
            fit_params["clf__sample_weight"] = sample_weight
        self.pipeline.fit(texts, y, **fit_params)
        self.is_fitted = True
        return self

    def fit_with_corrections(
        self,
        base_df: pd.DataFrame,
        corrections_df: pd.DataFrame,
        correction_weight: float = 4.0,
    ) -> "ExpenseCategoriser":
        """Retrain model incorporating user corrections with higher sample weight."""
        combined = pd.concat([base_df, corrections_df], ignore_index=True)
        weights = np.ones(len(combined), dtype=float)
        weights[len(base_df) :] = correction_weight
        self.fit(combined, sample_weight=weights)
        # Record history only after a successful fit so a failed fit leaves no trace.
        for row in corrections_df.to_dict(orient="records"):
            self.correction_history.append(row)
        return self

    def predict_one(
        self,
        merchant: str = "",
        narration: str = "",
        channel: str = "POS",
        bank: str = "GTBank",
    ) -> dict[str, Any]:
        """Predict category for a single transaction with cold-start rule fallback."""
        text = compose_feature_text(merchant, narration, channel, bank)
        if not self.is_fitted:
            cat, conf, kw = rule_based_predict_one(text)
            return {
                "category": cat,
                "confidence": conf,
                "source": "rule_fallback",
                "reason": f"Cold-start keyword match ({kw})",
            }

        probs = self.pipeline.predict_proba([text])[0]
        classes = list(self.pipeline.classes_)
        best_idx = int(np.argmax(probs))
        best_prob = float(probs[best_idx])
        best_cat = str(classes[best_idx])

        if best_prob < self.confidence_threshold:
            rule_cat, rule_conf, kw = rule_based_predict_one(text)
            if kw != "no_keyword_match":
                return {
                    "category": rule_cat,
                    "confidence": round(max(best_prob, rule_conf), 4),
                    "source": "rule_fallback",
                    "reason": f"Low ML confidence ({best_prob:.2f} < {self.confidence_threshold}); matched keyword '{kw}'",
                }

        return {
            "category": best_cat,
            "confidence": round(best_prob, 4),
            "source": "ml_tfidf_logreg",
            "reason": f"Calibrated TF-IDF + LogReg probability {best_prob:.2f}",
        }

    def predict_df(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Predict categories and metadata for a DataFrame of transactions."""
        results: list[dict[str, Any]] = []
        for row in df.to_dict(orient="records"):
            res = self.predict_one(
                merchant=str(row.get("merchant", "")),
                narration=str(row.get("narration", "")),
                channel=str(row.get("channel", "")),
                bank=str(row.get("bank", "")),
            )
            results.append(res)
        return results

    def save(self, path: str | Path = "models/categoriser.joblib") -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out)
        return out

    @classmethod
    def load(cls, path: str | Path = "models/categoriser.joblib") -> "ExpenseCategoriser":
        return joblib.load(Path(path))


def compute_ece(y_true: np.ndarray, probs: np.ndarray, classes: list[str], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) for multiclass probabilities."""
    confidences = np.max(probs, axis=1)
    pred_indices = np.argmax(probs, axis=1)
    preds = np.array([classes[i] for i in pred_indices])
    accuracies = (preds == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = float(np.mean(in_bin))
        if prop_in_bin > 0:
            avg_conf = float(np.mean(confidences[in_bin]))
            avg_acc = float(np.mean(accuracies[in_bin]))
            ece += abs(avg_acc - avg_conf) * prop_in_bin
    return round(float(ece), 4)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Any,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute 95% bootstrap confidence interval for a classification metric."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores.append(float(metric_fn(y_true[idx], y_pred[idx])))
    low, high = np.percentile(scores, [2.5, 97.5])
    return round(float(low), 4), round(float(high), 4)


def evaluate_categoriser(df: pd.DataFrame, seed: int = 42) -> tuple[ExpenseCategoriser, dict[str, Any]]:
    """Train and evaluate the categoriser on a strict chronological 75/25 split and 5-fold CV."""
    # Ensure chronological sort
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.75)  # 450 train / 150 test
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()

    model = ExpenseCategoriser(confidence_threshold=0.45, random_state=seed)
    model.fit(train_df)

    # Evaluate main hybrid model on chronological test split
    test_preds_meta = model.predict_df(test_df)
    y_true = test_df["category"].astype(str).to_numpy()
    y_pred = np.array([r["category"] for r in test_preds_meta])
    sources = [r["source"] for r in test_preds_meta]

    macro_f1 = round(float(f1_score(y_true, y_pred, average="macro")), 4)
    weighted_f1 = round(float(f1_score(y_true, y_pred, average="weighted")), 4)
    accuracy = round(float(accuracy_score(y_true, y_pred)), 4)

    f1_ci_low, f1_ci_high = bootstrap_ci(
        y_true,
        y_pred,
        lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
        n_boot=1000,
        seed=seed,
    )
    acc_ci_low, acc_ci_high = bootstrap_ci(
        y_true,
        y_pred,
        lambda yt, yp: accuracy_score(yt, yp),
        n_boot=1000,
        seed=seed,
    )

    # Calibration check on test set
    test_texts = ExpenseCategoriser._df_to_texts(test_df)
    test_probs = model.pipeline.predict_proba(test_texts)
    ece = compute_ece(y_true, test_probs, list(model.pipeline.classes_))

    # Evaluate rule-only baseline on the same chronological test split
    rule_preds = np.array([rule_based_predict_one(t)[0] for t in test_texts])
    rule_macro_f1 = round(float(f1_score(y_true, rule_preds, average="macro", zero_division=0)), 4)
    rule_accuracy = round(float(accuracy_score(y_true, rule_preds)), 4)

    # 5-fold Stratified Cross-Validation across full dataset
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_f1s: list[float] = []
    cv_accs: list[float] = []
    for tr_idx, va_idx in skf.split(df_sorted, df_sorted["category"]):
        cv_model = ExpenseCategoriser(confidence_threshold=0.45, random_state=seed)
        cv_model.fit(df_sorted.iloc[tr_idx])
        va_preds = [r["category"] for r in cv_model.predict_df(df_sorted.iloc[va_idx])]
        va_true = df_sorted.iloc[va_idx]["category"].astype(str).to_numpy()
        cv_f1s.append(float(f1_score(va_true, va_preds, average="macro")))
        cv_accs.append(float(accuracy_score(va_true, va_preds)))

    # Per-category F1 breakdown
    report_dict = classification_report(y_true, y_pred, labels=CATEGORIES, output_dict=True, zero_division=0)
    per_category_f1 = {
        cat: {
            "f1": round(float(report_dict[cat]["f1-score"]), 4),
            "precision": round(float(report_dict[cat]["precision"]), 4),
            "recall": round(float(report_dict[cat]["recall"]), 4),
            "support": int(report_dict[cat]["support"]),
        }
        for cat in CATEGORIES
    }

    # Test online learning / user corrections adaptation on a novel merchant narration
    novel_Corrections = pd.DataFrame(
        [
            {
                "merchant": "ALARA CONCEPT STORE VI",
                "narration": "POS PUR ALARA CONCEPT STORE VICTORIA ISLAND",
                "channel": "POS",
                "bank": "GTBank",
                "category": "Feeding & Groceries",
            },
            {
                "merchant": "ALARA CAFE VI",
                "narration": "NIP/GTB/ALARA NOK RESTAURANT VI DINNER",
                "channel": "NIP",
                "bank": "GTBank",
                "category": "Feeding & Groceries",
            },
        ]
    )
    retrained_model = ExpenseCategoriser(confidence_threshold=0.45, random_state=seed)
    retrained_model.fit_with_corrections(train_df, novel_Corrections, correction_weight=5.0)
    adapted_pred = retrained_model.predict_one(
        merchant="ALARA CONCEPT STORE VI",
        narration="POS PUR ALARA CONCEPT STORE VI LUNCH",
        channel="POS",
        bank="GTBank",
    )

    metrics = {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "split_type": "chronological_75_25",
        "macro_f1": macro_f1,
        "macro_f1_ci_95": [f1_ci_low, f1_ci_high],
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
        "accuracy_ci_95": [acc_ci_low, acc_ci_high],
        "ece": ece,
        "cv_5fold_macro_f1_mean": round(float(np.mean(cv_f1s)), 4),
        "cv_5fold_macro_f1_std": round(float(np.std(cv_f1s)), 4),
        "cv_5fold_accuracy_mean": round(float(np.mean(cv_accs)), 4),
        "baseline_rules_macro_f1": rule_macro_f1,
        "baseline_rules_accuracy": rule_accuracy,
        "f1_lift_over_rules": round(macro_f1 - rule_macro_f1, 4),
        "rule_fallback_rate_test": round(float(sources.count("rule_fallback") / len(sources)), 4),
        "online_retrain_adaptation_Category": adapted_pred["category"],
        "online_retrain_adaptation_confidence": adapted_pred["confidence"],
        "per_category": per_category_f1,
    }
    return model, metrics
