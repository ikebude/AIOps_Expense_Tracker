# LinkedIn Post: AIOps Expense Tracker (NGN ML Categoriser, Receipt OCR & ETS Forecaster)

Building personal finance apps for Africa always breaks on one fundamental problem: generic off-the-shelf financial models fail on Nigerian banking data.

When your bank statement reads:
- `POS PUR SHOPRITE IKEJA CITY MALL`
- `NIP/OPAY/PAYMENT FOR WEB HOSTING AWS`
- `USSD/MTN 40GB DATA BUNDLE`
- `STAMP DUTY CHARGE CBN EMTL LEVY`

...standard global expense trackers collapse all of them into a generic "Bank Transfer" or "Miscellaneous" bucket.

To solve this, I extended and rebuilt an expense intelligence engine from the ground up: **AIOps Expense Tracker**.

Here is what went into the system and what the verified benchmarks look like:

1. **Hybrid Classification Engine (TF-IDF + Calibrated Logistic Regression)**
   - Combined word & char-wb TF-IDF n-grams (3–5 grams) to capture merchant abbreviations and USSD codes.
   - Evaluated on a strict chronological 75/25 split (450 train / 150 test): **0.9204 Macro-F1** (95% CI `[0.8590, 0.9614]`) and **92.67% Accuracy**.
   - Achieved a **+14.03% lift** over cold-start keyword heuristics (0.7801 baseline Macro-F1), with an automated rule-fallback rate of only 2.0% on out-of-vocabulary merchants.
   - Online sample-weighted retraining loop that immediately learns from user category overrides without full batch downtime.

2. **Multi-Bank Ingestion & POS Slip OCR**
   - Out-of-the-box statement parsers for **GTBank, Access Bank, OPay, Kuda, and Moniepoint** (100% import success).
   - Regex/OCR receipt parser benchmarked on 40 supermarket receipts and POS charge slips: **97.50% exact-match field accuracy** (100% on merchant, date, and total amount).

3. **Per-Category Robust MAD Anomaly Detection**
   - Traditional z-scores break on personal finance because an annual rent payment or laptop purchase skews the entire distribution.
   - By computing category-level Median Absolute Deviation ($z_{MAD} = 0.6745 \times \frac{x - \text{median}}{\text{MAD}}$), the system flagged 100% of rogue bank surcharges (e.g. ₦5,577 fee on a ₦262 median) and duplicate supermarket charges without false-alerting on legitimate high-ticket outlays.

4. **Seasonal Spend Forecasting**
   - Damped Holt-Winters Exponential Smoothing (ETS) with 80% confidence bounds across 3-month rolling horizons.
   - Evaluated on holdout months: **20.42% WAPE** (MAE ₦55,523) vs 27.29% WAPE for a seasonal naive baseline.

Rebuilt from the Code with Mosh Claude Code starter and powered by FastAPI + Scikit-Learn + React/Vite.

Full open-source repo with benchmark scripts, reproducible evaluation suites, and CI workflows:
https://github.com/ikebude/AIOps_Expense_Tracker

#MachineLearning #Fintech #Python #FastAPI #DataScience #MLOps #Nigeria #ScikitLearn #AI
