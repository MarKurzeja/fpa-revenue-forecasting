-- 05_fact_weekly.sql
-- Purpose: weekly item x store fact on full Walmart weeks (Saturday..Friday).

CREATE OR REPLACE TABLE fact_sales_weekly AS
SELECT
    dd.wm_yr_wk,
    f.store_id,
    f.item_id,
    SUM(f.units)                                    AS units,
    AVG(f.sell_price)                               AS avg_price,   -- listed price (NULL if not listed)
    SUM(f.revenue)                                  AS revenue,
    SUM(CASE WHEN f.is_listed THEN 1 ELSE 0 END)    AS n_listed_days
FROM fact_sales_daily AS f
INNER JOIN dim_date AS dd
    ON dd.date = f.date
WHERE dd.is_full_week
GROUP BY dd.wm_yr_wk, f.store_id, f.item_id;
