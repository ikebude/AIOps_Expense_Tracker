# AIOps Expense Tracker: NGN-First ML Categoriser, Receipt OCR & Spend Forecaster

> **Note on Origin & Pedigree:** Started from the Code with Mosh Claude Code course starter; rebuilt and extended by Precious Ikebude with NGN bank CSV ingestion, TF-IDF + Logistic Regression auto-categorisation, receipt OCR parsing, MAD z-score anomaly detection, and seasonal spend forecasting.

[![CI](https://github.com/ikebude/AIOps_Expense_Tracker/actions/workflows/ci.yml/badge.svg)](https://github.com/ikebude/AIOps_Expense_Tracker/actions/workflows/ci.yml)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg?logo=scikit-learn)](https://scikit-learn.org)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent, production-ready personal & SME financial intelligence platform purpose-built for the Nigerian banking ecosystem. Standard personal finance tools consistently misclassify Nigerian transactions—collapsing NIP instant bank transfers into generic "Transfers", failing on POS terminal narrations, and missing recurring USSD charges or stamp duties.

AIOps Expense Tracker solves this with an end-to-end Python ML & Analytics engine paired with a React + Vite dashboard:

1. **NGN-Tuned Classification Engine**: Word + Char-WB TF-IDF feature union with Calibrated Logistic Regression, coupled with a cold-start keyword rule fallback and online feedback retraining.
2. **Multi-Bank Ingestion & Receipt OCR**: Native parsers for GTBank, Access Bank, OPay, Kuda, and Moniepoint CSV exports, alongside a regex/OCR engine for POS slips and supermarket receipts.
3. **Robust MAD Anomaly Detection**: Category-level Median Absolute Deviation ($z_{MAD}$) scoring that catches rogue surcharges and duplicate billings without being biased by lumpy, legitimate outlays like rent or equipment.
4. **Seasonal Spend Forecasting**: Damped Holt-Winters Exponential Smoothing (ETS) with 80% prediction intervals projecting category outlays across 3-month rolling horizons.

---

## Benchmark & Production Metrics

All figures below reflect reproducible runs against the 600-transaction labelled Nigerian dataset (`reports/metrics.json`):

| Evaluation Metric | Measured Result | Benchmark Baseline / CI |
|:---|:---|:---|
| **Macro-F1 (Holdout 75/25)** | **0.9204** | 95% Bootstrap CI: `[0.8590, 0.9614]` |
| **Accuracy (Holdout)** | **92.67%** | 95% Bootstrap CI: `[0.8800, 0.9667]` |
| **Weighted-F1** | **0.9277** | Baseline keyword rules: `0.7801` (+14.03% lift) |
| **5-Fold Cross-Validation Macro-F1** | **0.9301 ± 0.0231** | 5-Fold CV Accuracy: `93.00%` |
| **Test Set Rule Fallback Rate** | **2.00%** | ML handles 98.0% of cases autonomously |
| **Receipt OCR Exact Match Rate** | **97.50%** (195/200 fields) | 100% Merchant, 100% Date, 100% Total Amount |
| **Bank CSV Import Success** | **100%** (5/5 formats) | GTBank, Access, OPay, Kuda, Moniepoint |
| **Anomaly Recall** | **100%** (18/18 ground-truth) | Robust MAD avoids false alerts on lumpy expenses |
| **Forecast Holdout WAPE** | **20.42%** (MAE ₦55,523) | Seasonal Naive baseline WAPE: `27.29%` (MAE ₦74,208) |

---

## Performance Visualisation

![AIOps Expense Tracker Hero](img/hero.png)

*Figure 1: (Top-left) Per-category F1 classification performance showing balanced precision/recall across all 10 expense classes; (Top-right) Confusion matrix on holdout set; (Bottom-left) 6-month historical monthly spend vs 3-month Damped Holt-Winters projection with 80% confidence bands; (Bottom-right) Anomaly scatter highlighting out-of-band POS surcharges and bill spikes flagged by category-level MAD.*

---

## Category-Level Classification Breakdown

Evaluated on the strict chronological holdout split (150 test transactions):

| Category | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **Equipment & Repairs** | 1.0000 | 1.0000 | **1.0000** | 10 |
| **Miscellaneous** | 1.0000 | 1.0000 | **1.0000** | 12 |
| **Transport & Fuel** | 0.9200 | 1.0000 | **0.9583** | 23 |
| **POS & Bank Charges** | 1.0000 | 0.8750 | **0.9333** | 16 |
| **Feeding & Groceries** | 0.9310 | 0.9310 | **0.9310** | 29 |
| **Business SaaS & Cloud**| 0.9167 | 0.9167 | **0.9167** | 12 |
| **Family & Dependents** | 1.0000 | 0.8182 | **0.9000** | 11 |
| **Data & Utilities** | 1.0000 | 0.8095 | **0.8947** | 21 |
| **Healthcare** | 0.7692 | 1.0000 | **0.8696** | 10 |
| **Rent & Housing** | 0.6667 | 1.0000 | **0.8000** | 6 |
| **Macro Average** | **0.9203** | **0.9450** | **0.9204** | 150 |

---

## System Architecture

```mermaid
flowchart TD
    subgraph Ingestion["Ingestion Layer"]
        A1["Bank CSVs (GTBank, Access, OPay, Kuda)"] --> B1["Bank CSV Normalizer"]
        A2["Receipts & POS Slips"] --> B2["Regex / OCR Field Parser"]
        A3["Manual Entry"] --> B3["API Payload"]
    end

    subgraph FeaturePipeline["Feature Pipeline & Model"]
        B1 & B2 & B3 --> C["Normalized Text: Merchant + Narration + Channel + Bank"]
        C --> D["FeatureUnion: Word TF-IDF + Char-WB TF-IDF"]
        D --> E["Calibrated Logistic Regression (Multinomial)"]
        E --> F{"Confidence >= 0.45?"}
        F -- Yes --> G["ML Predicted Category"]
        F -- No --> H["Cold-Start Keyword Fallback Engine"]
        H --> G
        G --> I["User Correction Loop: Online Sample-Weighted Retraining"]
    end

    subgraph Analytics["Analytics & Guardrails"]
        G --> J["Per-Category MAD Anomaly Detector: z_mad >= 2.5"]
        G --> K["Damped Holt-Winters ETS 3-Month Forecast"]
    end

    subgraph Presentation["Presentation & API"]
        J & K & G --> L["FastAPI REST Endpoints (/v1/categorise, /v1/anomalies, /v1/forecast)"]
        L --> M["React + Vite UI Dashboard"]
    end
```

---

## API Specifications

The FastAPI service exposes clean REST endpoints for automated integration:

- **`POST /v1/categorise`**: Categorises a transaction payload (`merchant`, `narration`, `channel`, `bank`). Returns `category`, `confidence`, `source` (`ml_tfidf_logreg` or `rule_fallback`), and explanation.
- **`POST /v1/categorise/feedback`**: Receives user corrections (`true_category`) and performs high-sample-weight incremental retraining (`fit_with_corrections`).
- **`POST /v1/receipt/parse`**: Ingests raw OCR string and extracts `{merchant, date, total_ngn, vat_ngn, payment_channel}` along with the inferred category.
- **`POST /v1/csv/import`**: Accepts multi-bank statement CSV content, auto-detects bank schema, normalises balances/debits, and auto-labels all rows.
- **`POST /v1/anomalies`**: Evaluates transactions against category-level MAD baselines and flags outliers where $|z_{MAD}| \ge 2.5$.
- **`GET /v1/forecast`**: Computes rolling 3-month category and total spend forecasts with 80% confidence bands.
- **`GET /health`**: Returns engine status, dataset row count, and model readiness.

---

## Quickstart & Execution

### 1. Python ML Engine & API
```bash
# Clone and enter repo
cd AIOps_Expense_Tracker

# Install dependencies
pip install -r requirements.txt

# Run benchmark and regenerate hero charts & reports/metrics.json
python -m ml.run_benchmark

# Run unit and integration tests
pytest -q

# Start the FastAPI service
uvicorn ml.api:app --reload --port 8000
```

### 2. Frontend React Application
```bash
# Install frontend dependencies
npm install

# Start Vite development server
npm run dev
```

Visit `http://localhost:5173` to explore the dashboard.

---

## Repository Structure

```
AIOps_Expense_Tracker/
├── .github/workflows/ci.yml       # GitHub Actions CI workflow
├── data/
│   └── ngn_expenses_labelled.csv # 600-row realistic NGN transaction dataset
├── img/
│   └── hero.png                  # High-res 4-panel evaluation hero chart
├── ml/
│   ├── __init__.py
│   ├── analytics.py              # MAD anomaly detector & Holt-Winters ETS forecaster
│   ├── api.py                    # FastAPI application
│   ├── categoriser.py            # TF-IDF + Calibrated Logistic Regression + Rules
│   ├── data.py                   # Deterministic NGN transaction generator
│   ├── ingestion.py              # Bank CSV normalizers & receipt OCR parser
│   └── run_benchmark.py          # Benchmark pipeline & chart generator
├── models/
│   └── categoriser.joblib        # Pre-trained production classifier artifact
├── reports/
│   └── metrics.json              # Full verified evaluation metrics
├── src/                          # React + Vite application source
├── tests/
│   └── test_ml_pipeline.py       # Pytest test suite (9 passing tests)
├── requirements.txt              # Python dependencies
└── README.md                     # Documentation
```

---

## Author & License

Built and maintained by **Precious Ikebude** ([GitHub: @ikebude](https://github.com/ikebude)).  
Released under the [MIT License](LICENSE).
