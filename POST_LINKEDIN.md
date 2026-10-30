# LinkedIn Post: AIOps Expense Tracker (NGN ML Categoriser, Receipt-Text Parser & Spend Forecaster)

Personal finance tools often stumble on Nigerian bank data. When a statement reads:
- `POS PUR SHOPRITE IKEJA CITY MALL`
- `NIP/OPAY/PAYMENT FOR WEB HOSTING AWS`
- `USSD/MTN 40GB DATA BUNDLE`
- `STAMP DUTY CHARGE CBN EMTL LEVY`

...generic trackers tend to drop them into "Transfers" or "Miscellaneous".

I rebuilt an expense tracker around that problem: **AIOps Expense Tracker**. Important caveat first: every number below is measured on a synthetic, deterministically generated dataset (seed 42). It shows the pipeline works, not real-world accuracy.

1. **Classifier (TF-IDF + calibrated logistic regression)**
   - Chronological 75/25 split (450 train / 150 test): **0.9204 macro-F1** (95% CI 0.8590 to 0.9614), 92.67% accuracy.
   - Keyword-rules baseline macro-F1 is 0.7801, so +0.1403 absolute macro-F1. Calibration is imperfect (ECE 0.1486).
   - User corrections are queued and a new model is trained and swapped in only if training succeeds.

2. **Ingestion**
   - CSV parsers for GTBank, Access Bank, OPay and Kuda (5/5 smoke-test rows, one tiny sample per format).
   - A regex parser for pasted receipt text: 97.5% field exact match (195/200 fields) on 40 synthetic receipts. It parses text, not images; there is no OCR.

3. **Per-category MAD anomaly detection**
   - Median absolute deviation per category avoids rent-style outliers skewing the baseline.
   - In-sample on synthetic data with 18 injected spikes: precision 0.947, recall 1.0, F1 0.973, against F1 0.688 for a global z-score.

4. **Spend forecast**
   - Level + damped-trend exponential smoothing with a fixed December index.
   - 2-month holdout: WAPE 8.1% vs 11.2% for a naive last-value baseline.

There is also a mobile-first NGN dashboard (expenses only), API-key-protected write endpoints, rate limits, and CI that fails if the committed results are stale.

Started from the Code with Mosh Claude Code course starter. Repo with benchmark scripts and tests:
https://github.com/ikebude/AIOps_Expense_Tracker

#MachineLearning #Fintech #Python #FastAPI #DataScience #Nigeria #ScikitLearn
