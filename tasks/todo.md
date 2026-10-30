# Build plan — expense workstream (tasks/BUILD_SPEC.md)

- [x] 1 Pin Python deps, CI on 3.13
- [x] 2 Untrack model, repo-relative config, startup build
- [x] 3 Harden feedback/retrain, bounds, rate limit, CORS
- [x] 4 SQLite expense store + dashboard endpoints
- [x] 5 Honest benchmark labels, regenerate artefacts
- [x] 6 Frontend tooling (audit fix, vitest)
- [x] 7 Real NGN dashboard
- [x] 8 CI frontend + freshness check
- [x] 9 Dockerfiles, compose, .env.example
- [x] 10 LICENSE
- [x] 11 README and posts
- [x] 12 Review

## Review
Final verification (2026-10-30):
- Pins verified: fastapi 0.162.0, uvicorn 0.54.0, pydantic 2.14.2, pandas 3.1.0, numpy 2.5.4, scikit-learn 1.9.1, joblib 1.6.0, matplotlib 3.11.2, pytest 9.2.0, httpx 0.28.1.
- `python -m ml.run_benchmark` run twice: byte-identical data/ngn_expenses_labelled.csv (51d644cf...) and reports/metrics.json (c9a47b45...); `git diff --exit-code` on both -> ARTEFACTS_FRESH. Numbers equal the spec's "regenerated truth" column.
- pytest: 30 passed, 0 failed, no InconsistentVersionWarning (baseline was 9 passed).
- npm: `npm ci`, lint clean, vitest 29 passed (3 files), build OK, `npm audit` found 0 vulnerabilities.
- `docker compose config` -> COMPOSE_OK. Images NOT built (no Docker daemon in this environment).
- No .joblib/.db/.env tracked; no dangerouslySetInnerHTML or f-string SQL.
- Not verified here: a live browser run of the dashboard (component tests with mocked fetch only), CI on GitHub.
- Deviations: see tasks/BUILD_SPEC.md "Builder notes" (DEMO_DATA opt-in, wider API-key auth, calendar-date vs `month` bucketing, Sonnet trailer, items 6-7 and 8-10 committed together, item 4 pushed once with 2 red tests then fixed).
