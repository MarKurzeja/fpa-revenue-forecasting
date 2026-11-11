-- 06_price_index.sql
-- Purpose: Laspeyres price index per store x department x week, and the core
-- modelling table fact_revenue_weekly (store x department x week).
--   index_t = 100 * SUM_i p(i,t) q(i,0) / SUM_i p(i,0) q(i,0)
-- over items listed in week t that sold in the base period. Base = first fiscal year
-- of full weeks; p_0 = base-year unit value (revenue / units), q_0 = base-year units.
-- Numerator and denominator are kept so any aggregate node gets an exact index.

CREATE OR REPLACE TABLE price_base AS
WITH base_weeks AS (
    SELECT wm_yr_wk
    FROM stg_fiscal_weeks
    WHERE is_full_week
      AND fiscal_year = (SELECT MIN(fiscal_year) FROM stg_fiscal_weeks WHERE is_full_week)
)
SELECT
    w.store_id,
    w.item_id,
    SUM(w.units)                                   AS q0,
    SUM(w.revenue) / NULLIF(SUM(w.units), 0)       AS p0
FROM fact_sales_weekly AS w
INNER JOIN base_weeks AS b
    ON b.wm_yr_wk = w.wm_yr_wk
GROUP BY w.store_id, w.item_id
HAVING SUM(w.units) > 0;

CREATE OR REPLACE TABLE price_index_weekly AS
SELECT
    w.wm_yr_wk,
    w.store_id,
    i.dept_id,
    SUM(w.avg_price * b.q0)                                  AS laspeyres_num,
    SUM(b.p0 * b.q0)                                         AS laspeyres_den,
    100.0 * SUM(w.avg_price * b.q0) / SUM(b.p0 * b.q0)       AS price_index
FROM fact_sales_weekly AS w
INNER JOIN price_base AS b
    ON b.store_id = w.store_id AND b.item_id = w.item_id
INNER JOIN dim_item AS i
    ON i.item_id = w.item_id
WHERE w.avg_price IS NOT NULL
GROUP BY w.wm_yr_wk, w.store_id, i.dept_id;

CREATE OR REPLACE TABLE fact_revenue_weekly AS
WITH agg AS (
    SELECT
        w.wm_yr_wk,
        w.store_id,
        i.dept_id,
        SUM(w.units)    AS units,
        SUM(w.revenue)  AS revenue
    FROM fact_sales_weekly AS w
    INNER JOIN dim_item AS i
        ON i.item_id = w.item_id
    GROUP BY w.wm_yr_wk, w.store_id, i.dept_id
)
SELECT
    a.wm_yr_wk,
    a.store_id,
    a.dept_id,
    a.units,
    a.revenue,
    pi.price_index,
    pi.laspeyres_num,
    pi.laspeyres_den
FROM agg AS a
LEFT JOIN price_index_weekly AS pi
    ON  pi.wm_yr_wk = a.wm_yr_wk
    AND pi.store_id = a.store_id
    AND pi.dept_id  = a.dept_id
ORDER BY a.store_id, a.dept_id, a.wm_yr_wk;
