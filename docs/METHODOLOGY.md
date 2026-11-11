# Methodology

Written for a finance reader first, with enough detail for a data scientist to check
the work. Design choices are logged in [DECISIONS.md](DECISIONS.md).

## 1. What is forecast, and why weekly revenue at store x department

FP&A plans and reports in money, so the target is **revenue in USD**, not units. M5 has
daily unit sales per item and store, and weekly shelf prices. For every item, store and
day, revenue = units sold x that week's price. This is summed to Walmart weeks (Saturday to
Friday) and to **store x department**: 10 stores x 7 departments = 70 bottom series.

That is the level where a retail finance team plans: a store manager owns a store, and a
category manager owns a department. It's also aggregated enough to be stable. A single
item sells a few units a day, mostly zeros, so it's very noisy. The bottom series roll up
into an 8-level hierarchy:

| Level | Nodes | Level | Nodes |
|---|---|---|---|
| L1 company | 1 | L5 department | 7 |
| L2 state | 3 | L6 state x category | 9 |
| L3 store | 10 | L7 store x category | 30 |
| L4 category | 3 | L8 store x department | 70 |

All 133 nodes are forecast 1 to 13 weeks ahead (one fiscal quarter). Monthly and
quarterly figures are sums of weeks under the fiscal calendar: the fiscal year ends
31 January, and each week belongs to the month that contains most of its days.

## 2. Backtesting: testing the forecast the way it will be used

A forecast is only useful if it is accurate on data it has not seen. Time series can't be
split at random: a random split lets the model "see" next month while predicting last
month, which flatters accuracy. Instead the project replays history:

1. Pick a **forecast origin**, for example the start of week 230.
2. Train on everything before it, and forecast the next 13 weeks.
3. Compare with what actually happened.
4. Move the origin forward 4 weeks and repeat (8 origins, expanding window).

The origins have three roles:

| Role | Used for |
|---|---|
| Validation (first 4 origins) | Choosing models, tuning LightGBM, calibrating intervals |
| Test (last 4 origins) | Reporting accuracy |
| Hold-out (last 13 weeks of data) | Final, untouched check |

Nothing about the test or hold-out origins influences any choice. Features at an
origin only use data up to that origin. The exceptions are things known in advance:
calendar, holidays, SNAP days, and optionally planned prices (`use_future_prices`, off by
default).

## 3. Models

| Model | In two sentences |
|---|---|
| Naive | Next week = last week. The simplest possible benchmark. |
| Seasonal naive | Next week = the same week last year. The primary benchmark: anything worth using must beat it. |
| Moving average (4, 13) | The average of the last 4 or 13 weeks, held flat. Smooths noise but ignores seasonality. |
| Run-rate | The last 13-week average x a seasonality index (each week's value relative to the run-rate in up to 3 prior years). This is the spreadsheet method many FP&A teams use, so it represents "today's process". |
| AutoETS | Exponential smoothing that picks its own trend and error form, applied after removing the yearly pattern with STL. It weights recent weeks more. |
| AutoARIMA | Autoregressive model with automatic order selection, plus yearly sine/cosine terms for seasonality. A classic statistical benchmark. |
| AutoTheta | Theta method (trend line + smoothed level) on a seasonally adjusted series. Strong in forecasting competitions. |
| LightGBM | One gradient-boosted tree model learnt across all 133 series. It uses recent revenue, same week last year, rolling statistics, calendar and holiday distances, event and SNAP counts, the price index and store/department identity, with a separate prediction for each horizon. |
| Ensemble | Simple average of the 3 best models on validation. Averaging often cancels individual models' errors. |

## 4. Prediction intervals and coverage

A forecast of USD 10M is more useful as "USD 10M, 80% likely between 9.4M and 10.6M".
The **80% interval** should contain the actual about 80% of the time. That share is the
**coverage**, and the report measures it on test and hold-out weeks. If coverage is 60%,
the intervals are too narrow and the forecast is overconfident. If it's 98%, they're
needlessly wide.

- Statistical models produce intervals from their own error model.
- Every other model, and the reconciled forecasts, use **conformal intervals**. The project
  looks at how big the errors were on the validation origins, relative to each series'
  size, for each hierarchy level and horizon range. It then sets the interval half-width
  so that 80% (or 95%) of those past errors would have fitted inside. This makes no
  normality assumption.

## 5. Reconciliation: making the numbers add up

Forecasting each node separately gives numbers that don't add up: the 10 store forecasts
won't sum to the company forecast. Finance can't publish that. Reconciliation adjusts
every node so that each parent equals the sum of its children.

| Method | How it works |
|---|---|
| Bottom-up | Forecast the 70 bottom series and add them up. Coherent, but ignores the cleaner signal at the top. |
| Top-down | Forecast the company total and split it along the store path in proportion to the lower-level forecasts. |
| MinT (ols / shrink) | Combine all 133 forecasts optimally. Each level gets a weight that reflects how reliable it has been (from the model's past errors), and the result is the closest coherent set of numbers. Usually the most accurate. |

The report shows accuracy at every level before and after reconciliation. A unit test and
the `check` gate both verify that every parent equals the sum of its children.

## 6. Accuracy measures

| Measure | Definition | How to read it |
|---|---|---|
| WAPE | sum of absolute errors / sum of actuals | "% error". The primary metric. |
| Bias | sum(forecast - actual) / sum(actual) | Positive = systematically over-forecasting. |
| MAE, RMSE | Mean absolute / root mean squared error, in USD | Size of errors in money. |
| MASE | MAE / in-sample MAE of seasonal naive | Below 1 = better than "same week last year". |
| Weighted RMSSE | RMSE scaled by each series' week-to-week volatility, averaged with revenue weights | Like the M5 competition metric, but on revenue rather than units. |
| Coverage, width, pinball, Winkler | Interval quality | See section 4. |
| FVA | WAPE(benchmark) - WAPE(model) | How many points of error each step removes. |

A Diebold-Mariano test checks whether the champion's advantage over seasonal naive is
statistically significant or could be noise.

## 7. Price-volume-mix bridge

Why did revenue differ from last year or from budget? For each store x department, with
items i, units q, unit prices p, base period 0 (prior year or budget) and actual period 1:

| Effect | Formula | What it means |
|---|---|---|
| Volume | (Q1 - Q0) x P0_bar | More or fewer units in total, valued at base prices and base mix |
| Mix | Q1 x sum_i (m1_i - m0_i) x p0_i | Selling relatively more expensive or cheaper items |
| Price | sum_i q1_i x (p1_i - p0_i) | Price changes on the units actually sold |

Here Q = sum of q, m_i = q_i / Q is the item's unit share, and P0_bar = R0 / Q0 is the
average base price.

The three effects add up **exactly** to actual revenue - base revenue, which is unit-tested.
Mix is measured within each department. Store and company figures are sums of the
department bridges.

For the budget comparison, planned item units and prices are derived from the budget
revenue, the trailing item mix and trailing prices (DECISIONS D-11).

## 8. Working capital: what better forecasts are worth

Retailers hold **safety stock** to cover forecast error during the replenishment lead
time. The bigger the error, the more stock is needed:

safety stock (at cost) = z(service level) x sigma_LT x COGS ratio

- z is 1.28, 1.64 or 2.05 for 90%, 95% or 98% service.
- sigma_LT is the standard deviation of the forecast error summed over the lead time
  (2 weeks), per store x department, measured on the backtest.
- COGS ratios are assumptions (FOODS 75%, HOBBIES 60%, HOUSEHOLD 65%).

The safety stock under the champion is compared with seasonal naive and run-rate. The
difference is **working capital released**, and 20% a year of it is the **carrying cost
saved**. A hold-out check measures how often actual demand stayed within forecast + safety
stock.

**Limits.**
- This is a stylised simulation: M5 has no costs, inventory or lead times, and it works on
  revenue at store x department rather than units per item.
- Real safety stock is set per item, so absolute USD figures are indicative.
- The relative comparison between methods is the meaningful part: lower forecast error
  means proportionally less buffer stock.

## 9. Price elasticity and scenarios

Per category, log(units) is regressed on log(price index), with controls:
- store x department fixed effects and trends;
- yearly seasonality (Fourier terms);
- event counts, SNAP days and store closures.

The coefficient is the elasticity, used to show revenue under +/-5% price changes:
revenue x (1 + dp)^(1 + elasticity). This is observational data: prices move with
promotions and seasons, not at random, so the estimate is an association, not a causal
effect. Event uplift is the average revenue of a holiday week relative to the
surrounding non-event weeks.
