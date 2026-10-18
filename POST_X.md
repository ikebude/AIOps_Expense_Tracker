# X (Twitter) Thread: AIOps Expense Tracker

1/5
Most personal finance apps break on Nigerian bank data.
"NIP/OPAY/SHOPRITE", "USSD/MTN 40GB DATA", "STAMP DUTY CBN LEVY" get lumped into "Transfers".

I built an NGN-first ML classification & spend forecasting engine: AIOps Expense Tracker 🇳🇬📊

Here are the numbers 👇

2/5
ML Engine:
• Word + Char-WB TF-IDF (3-5 grams) + Calibrated Logistic Regression
• Evaluated on chronological 75/25 holdout: 0.9204 Macro-F1 (95% CI 0.859–0.961)
• 92.67% accuracy, +14% lift over heuristic rules
• Online sample-weighted retraining on user feedback

3/5
Ingestion & OCR:
• Native statement parsers for GTBank, Access Bank, OPay, Kuda & Moniepoint
• Receipt OCR benchmark: 97.5% field exact-match rate across 40 receipts (100% on merchant, date & total amount)

4/5
Analytics & Forecasting:
• Category-level Median Absolute Deviation (MAD) anomaly detector catches rogue POS fees without rent-skew false alerts
• Damped Holt-Winters ETS 3-month forecast: 20.4% holdout WAPE vs 27.3% seasonal naive baseline

5/5
Rebuilt from the @moshhamedani Claude Code starter with Python, FastAPI & React.

100% reproducible with automated pytest suite and CI.
Code + Benchmark: https://github.com/ikebude/AIOps_Expense_Tracker
