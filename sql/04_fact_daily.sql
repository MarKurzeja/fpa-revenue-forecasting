-- 04_fact_daily.sql
-- Purpose: daily item x store fact with revenue = units x weekly sell price.
-- Days before an item's first price are "not listed" (is_listed = false, revenue 0),
-- not zero demand (SPEC §2.4).

CREATE OR REPLACE TABLE fact_sales_daily AS
SELECT
    dd.date,
    s.store_id,
    s.item_id,
    s.units,
    p.sell_price,
    CAST(s.units AS DOUBLE) * COALESCE(p.sell_price, 0.0) AS revenue,
    (p.sell_price IS NOT NULL)                           AS is_listed
FROM stg_sales_long AS s
INNER JOIN dim_date AS dd
    ON dd.d = s.d
LEFT JOIN stg_prices AS p
    ON  p.store_id = s.store_id
    AND p.item_id  = s.item_id
    AND p.wm_yr_wk = dd.wm_yr_wk;
