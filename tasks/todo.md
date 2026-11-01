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
- Deviations: see tasks/BUILD_SPEC.md "Builder notes" (DEMO_DATA opt-in, wider API-key auth, calendar-date vs `month` bucketing, different attribution trailer, items 6-7 and 8-10 committed together, item 4 pushed once with 2 red tests then fixed).

## Tester review (2026-10-30)
Verdict PASS-WITH-NOTES. pytest 32 passed, vitest 29 passed, lint clean, build OK, npm audit 0, compose config OK, artefacts fresh (sha256 identical twice), stale check proven to fail on a tampered number. Full stack driven in Playwright at 320/390/1280 light+dark, axe 0 violations. Fixes: 78cb723, 45e8d48, 8309e31, 8539231, 1c7115d. Open findings in tasks/TEST_PLAN.md "Tester findings".

## Audit plan (2026-11-01, branch audit-2026-11-01)
Owner check-in not possible during the audit run; the plan follows the org audit brief and this repo's CLAUDE.md. Findings go to tasks/AUDIT_2026-11-01.md.
- [x] Baseline: pytest, vitest, lint, build, npm audit, pip-audit, benchmark freshness
- [x] Security probe: API key, rate limit, CORS, CSV/receipt parsing limits, feedback/retrain abuse
- [x] UX: dashboard at 390 and 1280, light and dark (preinstalled Chromium), look at screenshots
- [x] Docs: README (2-minute read, course-starter origin stated, synthetic labels), POST_X/POST_LINKEDIN accuracy
- [x] Fix confirmed issues with tests; re-run everything; write audit report and lessons

### Audit review (2026-11-01)
pytest 32 -> 39 passed; vitest 29 passed; lint clean; build OK; npm audit: 0 prod, 6 moderate dev-only (no fix); pip-audit clean; benchmark artefacts byte-identical after all changes.
Fixed: receipt-parser ReDoS (unauthenticated, >15 s per request), per-row CSV classification (~28 s for 5,000 rows), forecast buried below the list, README test counts and receipt caveat, dashboard screenshot.
Open and owner decisions: see tasks/AUDIT_2026-11-01.md.
