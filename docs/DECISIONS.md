# Decisions log

Choices made where SPEC.md was ambiguous, or where the implementation deliberately
departs from it. Each entry: decision, why, and how to change it.

### D-01 No synthetic data anywhere (owner instruction, 2026-11-11)
SPEC §2.3 asked for a synthetic M5-schema sample for CI. The owner replaced this with
"don't use any synthetic data". Consequences:
- There is no `make-sample` command and no `--sample` flag. The pipeline only ever runs on
  the real M5 files.
- Unit tests still need inputs. They use a few hand-written rows and small generated arrays
  inside the test files (`tests/conftest.py`) to check the maths, for example that the PVM
  effects sum exactly to the variance. These are test fixtures, not a dataset, and no result
  in the report comes from them.
- `python run.py check` = lint + unit tests + the full pipeline on a **real-data subset**
  (stores CA_1, TX_1 and WI_1, 3 origins; see `check:` in `config.yaml`) with the quality
  gates from SPEC §14.2. Without M5 files, `check` runs lint and tests and skips the gate.
- CI runs lint and unit tests on every push. The real-data gate runs in CI only if the
  repository has `KAGGLE_USERNAME` / `KAGGLE_KEY` secrets (the data cannot be committed).

### D-02 Full weeks only
Walmart weeks run Saturday to Friday (`wm_yr_wk`). Weeks with fewer than 7 days of sales
are dropped from every weekly table. In M5 that is the last week (2016-05-21/22, 2 days).
`docs/DATA_FACTS.md` (written by `run.py build`) records how many weeks were dropped.

### D-03 Fiscal calendar
FY ends 31 January, so `fiscal_year` = the calendar year it ends in (FY2016 = Feb 2015 to
Jan 2016). Each Walmart week is assigned whole to a fiscal year and month by its Tuesday
(the middle day, `fiscal.week_anchor_day`), so a week goes where most of its days are.
The default is calendar months (`fiscal.month_mode: calendar`), and a 4-4-5 pattern is
available (`445`; week 53 joins month 12). Calendar months were chosen because they
match how monthly actuals are usually reported, and M5 has no real 4-4-5 data. The logic
lives in Python (`fpa/fiscal.py`, unit-tested) and is loaded into DuckDB as
`stg_fiscal_weeks`.

### D-04 Revenue and "not yet listed"
`revenue = units x sell_price` of that store, item and week. On days without a price,
the item is "not listed": `is_listed = false` and revenue is 0. Check c04 fails the build if
any unit is sold without a price.

### D-05 AutoETS with a 52-week season
ETS cannot model a season longer than 24 periods. statsforecast, like R's forecast package,
silently drops the seasonality in that case. So `auto_ets` is AutoETS (`ZZN`) on the
STL-deseasonalised series, through statsforecast's `MSTL(season_length=52,
trend_forecaster=AutoETS)`. The seasonal component comes back on top of the forecast,
and the intervals come from the ETS part.

### D-06 ARIMA with Fourier terms by default
Seasonal ARIMA with m = 52 on about 5 years of data is slow and often unstable, as SPEC §6.2
anticipated. The default is `AutoARIMA(season_length=1)` with 6 yearly Fourier pairs as
regressors, which are known in advance. `models.arima.mode: seasonal` switches to
m = 52.

### D-07 LightGBM: direct strategy, global over all hierarchy nodes, without mlforecast
- **Direct, horizon-as-feature.** There is one model with `h` as a feature. Each training
  row is (node, origin o', h): its features come from weeks before o', and its target is
  week o' + h - 1. There is no recursion, so no error feedback, and leakage is easy to
  prove: `tests/test_features.py` puts a marker spike right after the origin and checks
  that the features don't change.
- **Global across all 133 nodes**, not only the 70 bottom series, with the level and
  store/state/category/department codes as categoricals. That way LightGBM has base
  forecasts at every level, which reconciliation needs. The target is scale-free (log1p,
  or divided by the trailing 13-week mean; both are compared on validation), so mixing
  levels is fine.
- **Implemented with `lightgbm` directly**, not `mlforecast`. This keeps the feature/target
  alignment explicit and testable, and drops one dependency.
- **Tuning** is a small grid in `config.yaml`, scored on validation origins only. optuna
  is not used.
- Same-week-last-year values relative to the target (`y[t-52]`, `y[t-104]`) are used
  alongside the origin lags in SPEC §6.3. They're always in the past for h <= 52.

### D-08 Prediction intervals
- Statistical models: native statsforecast intervals.
- Baselines, LightGBM, the ensemble and every reconciled forecast use **split-conformal
  intervals**. The score is |y - yhat| / (trailing 13-week mean of the node at the origin).
  The (1 - alpha) quantile, with the finite-sample correction ceil((n+1)(1-alpha))/n, is
  computed per (model, hierarchy level, horizon bucket 1-4 / 5-8 / 9-13) from the
  **validation** origins. Coverage is reported on test and hold-out origins only.
- For reconciled forecasts the method is to reconcile the point forecast, then calibrate
  the interval on the reconciled validation errors. This keeps intervals honest at every
  level without assuming normality.
- Budget forecasts run up to 53 weeks ahead. Beyond week 13, the last bucket's quantile
  is widened by sqrt(h / 11). This is an approximation and is documented in the report.

### D-09 Reconciliation in numpy instead of hierarchicalforecast
BottomUp, TopDown and MinTrace (ols, shrink) are about 100 lines of linear algebra
(`fpa/reconcile.py`). Writing them directly makes the maths visible, removes a dependency
whose API changes between versions, and the coherence of every method is unit-tested.
- TopDown uses forecast proportions along the store path (total, state, store, store x
  department), which is the strict tree inside the grouped hierarchy. Forecast proportions
  aren't defined for grouped structures in general.
- MinT shrink uses W = the Schafer-Strimmer shrinkage of the base model's in-sample
  one-step residual covariance. If fewer than 20 complete residual weeks exist, it falls
  back to structural scaling W = diag(S 1).

### D-10 Model selection
The selection metric is mean WAPE across the 8 hierarchy levels on validation origins.
- The ensemble is the simple average of the top 3 single models, excluding naive.
- The champion base is the better of the best single model and the ensemble.
- The reconciliation method is chosen the same way. The champion is "base + method", for
  example `lightgbm+mint_shrink`.

Test and hold-out origins are never used for any choice.

### D-11 Budget for the PVM bridge
M5 has no budget, so the budget is the frozen forecast (SPEC §8.1). For the PVM bridge
vs budget, the planned item units and prices are built in two steps:
- Each store x department budget is split across items by their revenue mix over the 52
  weeks before the budget.
- That revenue is converted to units at trailing prices: the unit value over the last 13
  weeks with sales, else over 52 weeks.

### D-12 Data for the public app
`python run.py export` writes aggregated outputs to `artifacts/demo/` (Parquet). These are
store x department weekly revenue, forecasts, metrics, variance, PVM by store x department,
and capital tables. There is no item-level data. The folder is **git-ignored** until the
owner has checked the Kaggle competition rules (SPEC §11.3 asks for her decision before
any derived real data is committed). Once that's decided, remove `artifacts/demo/*` from
`.gitignore` and commit the folder for Streamlit Community Cloud.

### D-13 `check` outputs are kept separate
The quick real-data run writes to `data/processed/check/`, `reports/check/` and
`exports/check/`, so it never overwrites the full run.

### D-14 Dependencies vs SPEC §4
- Removed: `mlforecast`, `hierarchicalforecast`, `utilsforecast` (D-07, D-09).
- Added: `pyarrow` (Parquet cache, which SPEC §13 asks for), `scipy` (Student-t p-value for
  Diebold-Mariano, already a statsforecast dependency) and `kaggle` (the CLI that
  `run.py download` calls).
- Data library: pandas everywhere. polars is not used.
- Python 3.11 or 3.12. The pinned numpy 1.26 has no 3.13 wheels.

### D-15 Price index
Laspeyres per store x department x week:
100 x sum_i p(i,t) q(i,0) / sum_i p(i,0) q(i,0), over items listed in week t that sold in
the base period.
- Base = the first fiscal year of full weeks (FY2012 in M5).
- p(i,0) = base-year unit value (revenue / units), q(i,0) = base-year units.
- The numerator and denominator are stored, so any aggregate node's index is exactly
  sum(num) / sum(den).

### D-16 Working capital
- sigma_LT for each store x department is the standard deviation of the forecast error
  summed over the lead time. It is estimated from the backtest origins (validation +
  test).
- The service-level check uses the hold-out origin.
- Everything is in revenue terms and converted to cost with the assumed COGS ratios. See
  the README limitations.

### D-17 Scaled metrics
- MASE is scaled by the in-sample mean absolute seasonal difference (lag 52), as SPEC
  §7.1 says.
- RMSSE follows the M5 convention: in-sample mean squared one-step difference.
- The revenue-weighted RMSSE (WRMSSE-style) weights each series by its revenue over the 52
  weeks before the origin. Unlike M5, the target is revenue, not units.

### D-18 Diebold-Mariano
Champion vs seasonal naive. The loss is absolute error summed over the bottom-level
series per (origin, horizon), on test and hold-out origins. The test uses the
Harvey-Leybourne-Newbold correction and a long-run variance with lag h - 1 = 12.
Overlapping origins make the observations dependent; the lag window only partly absorbs
this, and the report says so.

### D-19 Forward calendar
For the rolling forecast beyond the data, weeks are projected:
- Dates and the fiscal calendar are exact.
- Events and SNAP days are copied from 52 weeks earlier. This is an approximation for
  moving holidays such as Easter.
