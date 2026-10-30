# X (Twitter) Thread: AIOps Expense Tracker

1/5
Most personal finance apps stumble on Nigerian bank data.
"NIP/OPAY/SHOPRITE", "USSD/MTN 40GB DATA", "STAMP DUTY CBN LEVY" get lumped into "Transfers".

I built an NGN-first expense tracker with ML categorisation and a spend forecast: AIOps Expense Tracker.

All numbers below are on a synthetic dataset (seed 42): they show the pipeline works, not real-world accuracy.

2/5
Classifier:
• Word + Char-WB TF-IDF + calibrated logistic regression
• Chronological 75/25 holdout: 0.9204 macro-F1 (95% CI 0.859 to 0.961), 92.67% accuracy
• Rules baseline macro-F1 0.7801 (+0.1403 absolute)

3/5
Ingestion:
• CSV parsers for GTBank, Access Bank, OPay & Kuda (5/5 smoke rows)
• Receipt-text parser (text in, fields out, no OCR): 97.5% field exact match on 40 synthetic receipts

4/5
Analytics:
• Per-category MAD anomaly detector: F1 0.973 vs 0.688 for a global z-score (in-sample, 18 injected spikes)
• Damped-trend ETS forecast: 8.1% holdout WAPE vs 11.2% naive last-value

5/5
Rebuilt from the @moshhamedani Claude Code starter with Python, FastAPI & React.

Tests and CI included; CI fails if committed results are stale.
Code + benchmark: https://github.com/ikebude/AIOps_Expense_Tracker
