-- 02_dim_date.sql
-- Purpose: daily date dimension with the fiscal calendar, events, SNAP and closure
-- flags, plus a weekly dimension (one row per full Walmart week) used by the models.
-- Needs `stg_fiscal_weeks` (built in Python from fpa.fiscal so the logic is unit-tested).
-- Template variables: {closure_expr}, {last_sales_day}

CREATE OR REPLACE TABLE dim_date AS
SELECT
    c.date,
    c.d,
    c.d_num,
    c.wm_yr_wk,
    f.week_start,
    f.week_end,
    c.year,
    c.month,
    f.fiscal_year,
    f.fiscal_quarter,
    f.fiscal_month,
    f.fiscal_week,
    f.fiscal_period,
    c.weekday,
    (c.event_name_1 IS NOT NULL OR c.event_name_2 IS NOT NULL) AS is_event,
    c.event_name_1,
    c.event_type_1,
    c.event_name_2,
    c.event_type_2,
    c.snap_ca,
    c.snap_tx,
    c.snap_wi,
    ({closure_expr})                          AS is_closure,
    f.is_full_week
FROM stg_calendar AS c
INNER JOIN stg_fiscal_weeks AS f
    ON f.wm_yr_wk = c.wm_yr_wk
WHERE c.d_num <= {last_sales_day};

-- Weekly dimension: full weeks only (partial first/last weeks are dropped, D-02).
CREATE OR REPLACE TABLE dim_week AS
WITH day_events AS (
    SELECT wm_yr_wk, event_name_1 AS event_name, event_type_1 AS event_type
    FROM dim_date WHERE event_name_1 IS NOT NULL
    UNION ALL
    SELECT wm_yr_wk, event_name_2, event_type_2
    FROM dim_date WHERE event_name_2 IS NOT NULL
),
week_events AS (
    SELECT
        wm_yr_wk,
        COUNT(*) FILTER (WHERE event_type = 'Sporting')  AS n_sporting,
        COUNT(*) FILTER (WHERE event_type = 'Cultural')  AS n_cultural,
        COUNT(*) FILTER (WHERE event_type = 'National')  AS n_national,
        COUNT(*) FILTER (WHERE event_type = 'Religious') AS n_religious,
        string_agg(DISTINCT event_name, '|') AS event_names
    FROM day_events
    GROUP BY wm_yr_wk
),
week_days AS (
    SELECT
        wm_yr_wk,
        SUM(snap_ca)                       AS snap_ca_days,
        SUM(snap_tx)                       AS snap_tx_days,
        SUM(snap_wi)                       AS snap_wi_days,
        MAX(CAST(is_closure AS INTEGER))   AS has_closure
    FROM dim_date
    GROUP BY wm_yr_wk
)
SELECT
    ROW_NUMBER() OVER (ORDER BY f.wm_yr_wk) - 1   AS t,
    f.wm_yr_wk,
    f.week_start,
    f.week_end,
    f.fiscal_year,
    f.fiscal_quarter,
    f.fiscal_month,
    f.fiscal_week,
    f.fiscal_period,
    COALESCE(e.n_sporting, 0)   AS n_sporting,
    COALESCE(e.n_cultural, 0)   AS n_cultural,
    COALESCE(e.n_national, 0)   AS n_national,
    COALESCE(e.n_religious, 0)  AS n_religious,
    COALESCE(e.event_names, '') AS event_names,
    d.snap_ca_days,
    d.snap_tx_days,
    d.snap_wi_days,
    d.has_closure
FROM stg_fiscal_weeks AS f
INNER JOIN week_days AS d ON d.wm_yr_wk = f.wm_yr_wk
LEFT JOIN week_events AS e ON e.wm_yr_wk = f.wm_yr_wk
WHERE f.is_full_week
ORDER BY f.wm_yr_wk;
