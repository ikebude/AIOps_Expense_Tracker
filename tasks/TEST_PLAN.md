# TEST_PLAN: AIOps Expense Tracker (workstream: expense)

The tester runs this plan independently and does **not** trust the builder's report. Run everything from `/home/user/AIOps_Expense_Tracker` on branch `ccr-7d21c86c-pnbh8t`, starting from a fresh `git pull`.
Python: `PY=/tmp/claude-0/-home-user/e6ff3af4-b759-5a42-ae4a-d7f12c188d23/scratchpad/venv/bin/python`. The system python has no pytest. Alternatively create a fresh `.venv` from `requirements-dev.txt`. That is preferred, because it proves the pins resolve.
Chrome for screenshots: `/root/.cache/puppeteer/chrome/linux-156.0.8078.47/chrome-linux64/chrome` (headless). The Playwright CLI 1.63 is reachable via `npx`, but its browsers are not cached.
Docker: the CLI and Compose are present, but there is **no daemon**. Image builds cannot be verified here.

Planner baseline for comparison (HEAD `af5edf2`): pytest 9 passed / 8 warnings, eslint clean, vite build OK, npm audit 16 vulns (12 high).

## A. Gate commands (expected results)

| # | Command | Expected |
|---|---|---|
| A1 | `git log --oneline af5edf2..HEAD` | One commit per spec item (about 11–12); authored by `ikebude <precious.ikebude@outlook.com>`; each ends with the two trailer lines; no model name elsewhere (`git log -p af5edf2..HEAD \| grep -n "<model names>" \| grep -v "Co-Authored-By\|Claude-Session"` → only legitimate hits, such as the keyword-rule list in `ml/categoriser.py`, which was already there) |
| A2 | `cat requirements.txt requirements-dev.txt` | Exact `==` pins: fastapi 0.162.0, uvicorn 0.54.0, pydantic 2.14.2, pandas 3.1.0, numpy 2.5.4, scikit-learn 1.9.1, joblib 1.6.0; dev: matplotlib 3.11.2, pytest 9.2.0, httpx 0.28.1 |
| A3 | `$PY -m pytest -q -W error::sklearn.exceptions.InconsistentVersionWarning` | 0 failed; count = 9 original + new (spec asks for 8 security + 7 expense cases, so ≥24 total); no InconsistentVersionWarning |
| A4 | `$PY -m ml.run_benchmark; git diff --exit-code -- reports/metrics.json data/ngn_expenses_labelled.csv` | Exit 0 (committed artefacts are fresh). Run twice; `sha256sum` identical |
| A5 | `npm ci && npm run lint && npm test && npm run build` | All green; vitest ≥15 tests, 0 failed |
| A6 | `npm audit` | `found 0 vulnerabilities` |
| A7 | `grep -E '"(vitest\|jsdom)"' package.json` | vitest 5.0.5 (not ≤4.1.12), jsdom 29.x (not 30.x) |
| A8 | `docker compose config >/dev/null && echo OK` | OK |
| A9 | `git ls-files \| grep -E '\.joblib$\|\.db$\|(^\|/)\.env$'` | No output (`.env.example` is allowed) |
| A10 | `test -f LICENSE && head -3 LICENSE` | MIT, "Copyright (c) 2026 Precious Ikebude" |
| A11 | `.env.example` vs code | Every `os.environ`/`getenv` key in `ml/` and every `import.meta.env.VITE_*` in `src/` appears in `.env.example` (`grep -rhoE "environ(\.get)?\(?\[?\"[A-Z_]+\|getenv\(\"[A-Z_]+\|VITE_[A-Z_]+" ml src \| sort -u`) |
| A12 | CI YAML | `.github/workflows/ci.yml` has a python job (3.13, requirements-dev, benchmark + `git diff --exit-code` + pytest) and a frontend job (node 22, npm ci, lint, test, build, audit). It parses as YAML |

## B. Metrics must equal the planner's regenerated values (`reports/metrics.json`)

dataset.total_spend_ngn **18776865.18**; categoriser macro_f1 **0.9204**, CI **[0.859, 0.9614]**, accuracy **0.9267**, weighted_f1 **0.9277**, ece **0.1486**, cv mean/std **0.9301/0.0231**, rules macro_f1/accuracy **0.7801/0.7533**, fallback **0.02**; receipt (key `ingestion.receipt_text_parsing`) field **0.975**, full **0.925**, vat **0.925**, channel **0.95**; csv **5/5**, `formats_tested` **4**; anomaly MAD P/R/F1 **0.9474/1.0/0.973**, flagged **19**, true **18**; global-z **0.7857/0.6111/0.6875**; forecast `model` **damped_trend_ets_fixed_dec_index**, holdout MAE **21761.02**, WAPE **0.081**, naive last-value MAE **30146.36**, WAPE **0.1122**.
Any deviation means either the algorithm was changed (forbidden by the spec) or the pins differ. That is a FAIL until explained.
Also: `grep -n "seasonal_naive\|receipt_ocr\|Holt" ml/*.py reports/metrics.json` → no hits.

## C. Adversarial API tests (run against `uvicorn ml.api:app --port 8765` with a temp DB)

Start: `EXPENSES_DB_PATH=$(mktemp -d)/t.db MODEL_PATH=$(mktemp -d)/m.joblib EXPENSE_API_KEY=s3cret-test FEEDBACK_RATE_LIMIT_PER_MINUTE=5 $PY -m uvicorn ml.api:app --port 8765` (run in the background), then use `curl -s -o /dev/stderr -w '%{http_code}\n'`.

| # | Attack / check | Expected |
|---|---|---|
| C1 | Restart **without** `EXPENSE_API_KEY`, then POST `/v1/categorise/feedback` with any header | **503** (fail closed) |
| C2 | Key set: no header / `X-API-Key: wrong` / `X-API-Key: ` (empty) / `?api_key=s3cret-test` / `Authorization: Bearer s3cret-test` | all **401** |
| C3 | Key prefix or longer key (`s3cret`, `s3cret-test-x`) | 401; code uses `hmac.compare_digest` (grep) |
| C4 | **Model-corruption regression** (the planner reproduced this on baseline: HTTP 500, then every categorise returned `confidence: null`, "probability nan"): correct key, `true_category: "NOT_A_CATEGORY"` | **422**; then `POST /v1/categorise {"narration":"POS PURCHASE SHOPRITE IKEJA"}` → category "Feeding & Groceries", numeric confidence in [0, 1] |
| C5 | Valid feedback | **202** `queued`, and the request returns in <200 ms (no synchronous retrain); `/v1/categorise/retrain` → 200 `retrained`; again → `noop` |
| C6 | Feedback flood: 6 valid calls inside 60 s with limit 5 | 6th → **429** with a `Retry-After` header |
| C7 | Retrain path never mutates the live model: read `retrain_from_queue` and confirm a **new** `ExpenseCategoriser` is fitted and then swapped under a lock; `fit_with_corrections` appends history only after a successful fit | Code review passes; pytest case 5 exists |
| C8 | Oversized inputs: narration 501 chars; `raw_text` 10,001; `csv_text` 1,000,001; 5,001 anomaly transactions | all **422**, with no 500 in server logs |
| C9 | `GET /v1/forecast?horizon_months=0`, `13`, `-1`, `abc` | **422** |
| C10 | Empty store (`SEED_DEMO_DATA=false`): `/v1/forecast` | **422** "Need at least 3 months of history" (not 500) |
| C11 | `POST /v1/expenses` amount `0`, `-1`, `1.234`, `"1e400"`, `"abc"`, `null`; date `2026-13-01`, `2026-02-30`, far future | all **422** |
| C12 | SQL injection: narration `'); DROP TABLE expenses;--`, merchant `" OR 1=1 --`, month query `2026-01' OR '1'='1` | stored and returned literally; month → 422 (regex); `GET /v1/expenses` still works; `grep -rnE "execute\(f\"\|execute\(.*%\|\.format\(" ml/store.py` → none |
| C13 | Money exactness: empty store, add ₦0.10 and ₦0.20 in one month | `/v1/summary` `total_ngn == 0.3` exactly; DB column is `amount_kobo INTEGER` |
| C14 | CORS: `curl -H 'Origin: http://evil.example' -I .../v1/expenses` | no `access-control-allow-origin`; with `Origin: http://localhost:5173` it is echoed; never `*` with credentials |
| C15 | `/health` from another CWD (`cd /tmp && $PY -m uvicorn --app-dir <repo> ml.api:app`) | 200, `model_fitted: true`, `demo_data: true` |
| C16 | Secrets hygiene | Key never logged (`grep -rn "api_key" ml \| grep -i "log\|print"` → none); `src/` contains no key/`X-API-Key` |

## D. Frontend checks

| # | Check | Expected |
|---|---|---|
| D1 | `grep -rn "dangerouslySetInnerHTML\|innerHTML" src` | none |
| D2 | `grep -rn '\$' src/*.jsx src/components` for currency usage | No `$` currency; formatting goes through `formatNGN` (`Intl.NumberFormat('en-NG',{currency:'NGN'})`) |
| D3 | XSS: create an expense with narration `<img src=x onerror=alert(1)>` via the API, then load the UI | Rendered as literal text; no dialog; no console errors |
| D4 | Concatenation regression | A unit test asserts `sumAmounts([{amount_ngn:27},{amount_ngn:31}]) === 58`; the request body in the add-expense test has `typeof amount_ngn === 'number'` |
| D5 | States | Stop the API → error state with Retry; restart → Retry recovers. Empty store → empty state. Slow network (DevTools throttle or delayed mock) → skeleton with `aria-busy` |
| D6 | Features end-to-end against the live API (`npm run dev`, `VITE_API_BASE=http://localhost:8765`) | list loads; add expense with "Auto-categorise" → row shows category + "auto" tag; receipt text parse prefills the form; month summary + change vs previous; at least one "Unusual" badge with reason text visible in seeded data (e.g. NGN-0431 INTERSWITCH ATM SURCHARGE ₦5,576.52); forecast chart renders with caption WAPE 8.1% labelled synthetic; "Demo data (synthetic)" pill visible |
| D7 | a11y | Every input has a `<label>`; status messages are in an `aria-live` region; the chart has `role="img"` + `<title>/<desc>` + a hidden data table; the anomaly state is not conveyed by colour alone; visible focus ring; keyboard-only add-expense flow works (Tab/Enter). If available, run `npx --yes @axe-core/cli@4 http://localhost:5173 --chromedriver-path …` or inject axe-core into headless Chrome; 0 serious/critical |
| D8 | Screenshots | Headless Chrome at **390×844** and **1280×800**, both **light** and **dark** (`--force-dark-mode` or `prefers-color-scheme` emulation, plus the in-app toggle). Save to the scratchpad (not the repo, unless README references them). Review: no horizontal scroll at 390 (also check 320), no clipped text, contrast is fine in both themes, the table→cards switch works, the chart is legible |
| D9 | `index.html` | title "AIOps Expense Tracker", meta description, `lang` set |

Screenshot command template:
`CH=/root/.cache/puppeteer/chrome/linux-156.0.8078.47/chrome-linux64/chrome; $CH --headless=new --no-sandbox --hide-scrollbars --window-size=390,844 --virtual-time-budget=8000 --screenshot=$SCRATCH/m-light.png http://localhost:5173/` (add `--force-dark-mode --enable-features=WebContentsForceDark:inversion_method/cielab_based` only as a fallback; prefer the toggle, or a Playwright/puppeteer script with `emulateMediaFeatures([{name:'prefers-color-scheme',value:'dark'}])` for real `prefers-color-scheme` emulation).

## E. Docs / links

| # | Check | Expected |
|---|---|---|
| E1 | Broken links: extract every `](...)` and `href`/`src` from `README.md`, `POST_*.md`, `index.html` | Relative paths exist in the repo (LICENSE, img/hero.png, .env.example …); external URLs return 2xx/3xx (`curl -sIL -o /dev/null -w '%{http_code}'`; the CI badge URL may 404 until a workflow runs on main, so note it rather than fail it) |
| E2 | Overclaim grep: `grep -niE "production-ready\|\bocr\b\|holt\|seasonal naive\|100% recall\|moniepoint\|\+14" README.md POST_*.md` | No hits, except a sentence saying OCR is out of scope / text-only |
| E3 | "synthetic" disclosure | README metrics section and both posts say the data is synthetic |
| E4 | Hero image | `img/hero.png` has 2 panels; no "80% Forecast Interval" band; title says "Damped-Trend ETS" and "Receipt-Text Parsing"; footer says synthetic |
| E5 | Untouched | `git diff af5edf2..HEAD --stat -- CLAUDE.md` is empty; the course-origin note and the `@moshhamedani` mention are still present (owner decision); "Precious Ikebude" spelled exactly |
| E6 | Test counts in README | Match the actual pytest/vitest totals from A3/A5 |

## F. README claim → evidence table (tester fills in the "Verified" column)

| README / post claim | Evidence source | Expected value | Verified |
|---|---|---|---|
| Macro-F1 holdout + 95% CI | metrics.json `categoriser.macro_f1`, `macro_f1_ci_95` | 0.9204 [0.859, 0.9614] | |
| Accuracy + CI | `categoriser.accuracy`, `accuracy_ci_95` | 0.9267 [0.88, 0.9667] | |
| Weighted-F1 (stated on its own) | `categoriser.weighted_f1` | 0.9277 | |
| Rules baseline macro-F1, lift (absolute) | `baseline_rules_macro_f1`, `f1_lift_over_rules` | 0.7801, +0.1403 | |
| ECE | `categoriser.ece` | 0.1486 | |
| 5-fold CV macro-F1 | `cv_5fold_macro_f1_mean/std` | 0.9301 ± 0.0231 | |
| Rule-fallback rate | `rule_fallback_rate_test` | 2.0% | |
| Receipt-text field / full match | `ingestion.receipt_text_parsing.*` | 97.5% (195/200) / 92.5% (37/40) | |
| Bank CSV | `ingestion.bank_csv` | 5/5 rows, 4 formats | |
| Anomaly MAD P/R/F1 vs global-z | `anomaly_detection.*` | 0.9474/1.0/0.973 vs 0.7857/0.6111/0.6875 | |
| Forecast WAPE/MAE vs naive | `forecasting.holdout_*`, `baseline_naive_last_value_*` | 8.1% / ₦21,761.02 vs 11.22% / ₦30,146.36 | |
| Dataset size / period / spend | `dataset.*` | 600 rows, 2025-09-01→2026-02-27, ₦18,776,865.18 | |
| Per-category table (if kept) | `categoriser.per_category` | each row matches | |
| Test counts | A3 / A5 output | actual numbers | |
| Any other number | must map to a file or be removed | — | |

## G. Verdict rules

- **FAIL** if any of these hold: A3–A6 not green; B mismatch; C1–C4 or C12 fail; any unverifiable number in README/posts; secrets/binaries committed; CLAUDE.md edited; tests skipped or deleted.
- **PASS-WITH-NOTES** for minor UI/a11y issues, Docker not buildable (expected: no daemon), or a CI badge that cannot be checked.
- Fix small defects directly (`fix: …` commits, same author and trailers). Log larger ones under "Tester findings" below with file:line and severity, then push.

## Tester findings

Tester run 2026-10-30 on top of `15de125`. Verdict: **PASS-WITH-NOTES**.

### Results
- A1 authored by ikebude, 2 trailers each. Note: builder commits carry a different `Co-Authored-By` model than RULES.md; history not rewritten (RULES 1). Model names removed from tasks/*.md.
- A2 pins resolve in a fresh venv (Python 3.13.17). A3 pytest **32 passed** (30 + 2 tester regressions), no InconsistentVersionWarning. A4 benchmark twice, sha256 identical, `git diff --exit-code` 0.
- A5 lint clean, vitest **29 passed**, build OK. A6 `npm audit` 0. A7 vitest 5.0.5 / jsdom 29.1.1. A9–A12 OK.
- **A8 failed before fix**: `docker compose config` errored without `.env` (fixed, see below).
- B: every metrics.json value equals the planner's table; no `seasonal_naive|receipt_ocr|Holt`.
- Stale-artefact check: committed `macro_f1` tampered 0.9204→0.9504 in a scratch clone; regenerate + `git diff --exit-code` → **exit 1**. Works. README numbers themselves are not machine-checked (verified by hand, table F all match).
- C1 503 ×4; C2/C3 all 401 (none, wrong, empty, query, Bearer, prefix, longer, case, unicode); C4 `NOT_A_CATEGORY`/""/lowercase → 422, categorise still "Feeding & Groceries" conf 0.815; C5 202 in 24 ms; C6 6th → 429 `Retry-After: 60`; C8 all 422 (50 MB body 422 in 0.2 s, 100k-deep JSON 400); C9/C10/C11 422; C12 SQLi on every field stored literally, no f-string SQL; C13 covered by test (0.1+0.2 == 0.3, INTEGER kobo); C14 evil origin/preflight gets no ACAO, allowed origin echoed, `null` origin refused, no credentials; C15 OK from /tmp; C16 OK.
- D (real stack: uvicorn DEMO_DATA=1 + built frontend + Playwright, 390/1280 × light/dark + 320): loading skeleton (aria-busy), error + Retry recovers, empty state, populated, receipt parse prefills (date, merchant, ₦18,750, 72% suggestion), add expense with key → row with "auto" tag, without key → "Invalid or missing API key". No horizontal scroll at 320/390/1280. XSS narration renders literally, 0 dialogs, 0 `<img>`. axe-core: **0 violations** in all four combos.

### Fixed by the tester
1. **High** `docker-compose.yml`: `.env` copied from `.env.example` overrode the image's `/data` paths with repo-relative ones → non-root API writes to root-owned `/app` (crash) and data lives outside the volume; `docker compose config` failed with no `.env`. Now `.env` optional and paths pinned to `/data` (78cb723).
2. **Medium** `ml/api.py`: raw JSON `1e400` → 500 (default 422 handler cannot serialise `inf`). Custom handler returns type/loc/msg, no input echo (45e8d48, test added).
3. **Medium** `/v1/csv/import`: field >131072 chars → unhandled `csv.Error` 500 → now 422; 15k rows took 85 s in one request → 5,000-row cap; 400-digit amount → `amount_ngn: null` → row skipped; UTF-8 BOM hid the date header so rows silently got today's date → BOM stripped (45e8d48, test added).
4. **Low/UI** forecast axis text ~5 px at 390 px and no y scale → ₦ y bounds, two-line month labels; amounts not tabular; table dates wrapped; dark "Unusual" badge 2.8:1 and light demo pill 3.8:1 contrast → 6.8:1 / 5.0:1 (8309e31).
5. **Low** hero.png legend covered the Rent & Housing "0.80" label (8539231). README test counts 30→32; `/v1/anomalies` baseline wording corrected (1c7115d).

### Open findings (not fixed)
| Sev | Where | Finding |
|---|---|---|
| Medium | `ml/security.py:51` + uvicorn defaults | uvicorn trusts `X-Forwarded-For` from 127.0.0.1, so any client reaching the API via localhost (or a same-host proxy that forwards client XFF) bypasses the per-IP limiter by spoofing XFF (reproduced: 62×429, then spoofed XFF → 200). Run with `--forwarded-allow-ips` set to the real proxy, or `--no-proxy-headers` when exposed directly. |
| Low | `ml/api.py` `list_expenses`, `get_summary`, `get_forecast` | GET endpoints are not rate-limited, and `/v1/expenses` recomputes MAD stats over the whole store on every call (cost grows with the store). |
| Low | `ml/security.py:34` | Limiter keeps one deque per distinct client key forever (unbounded memory with many IPs). |
| Low | `ml/ingestion.py` `normalize_date_str` | Unparseable CSV dates pass through as-is (`99-Foo-2026`) and missing dates become today; duplicate headers silently use one column. Should flag rows instead. |
| Low | `src/components/ExpenseForm.jsx:5` | Default date uses UTC (`toISOString`): in Lagos 00:00–01:00 it pre-fills yesterday. |
| Low | `src/App.jsx` `onCreated` | New row is prepended regardless of date order or the selected month filter. |
| Low | `deploy/nginx.conf`, Dockerfiles | No security headers (CSP, X-Content-Type-Options), no HEALTHCHECK; base images tag-pinned, not digest-pinned. Images not built (no daemon). |
| Info | `/v1/csv/import` | Formula-injection payloads (`=HYPERLINK`, `+cmd`, `@SUM`) are returned verbatim in JSON. Safe today (no CSV/XLSX export, React escapes); escape leading `= + - @` if an export is ever added. NUL and U+202E bidi characters are stored as given. |
| Info | E1 | External links (shields.io, docs sites, CI badge) unreachable through this sandbox's proxy; repo URL returns 200. CI not run on GitHub. |

