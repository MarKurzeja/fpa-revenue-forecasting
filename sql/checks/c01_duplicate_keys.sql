-- c01: no duplicate keys in fact tables. Returns one row per violating key.
SELECT 'fact_sales_daily' AS tbl, CAST(date AS VARCHAR) || '|' || store_id || '|' || item_id AS key, COUNT(*) AS n
FROM fact_sales_daily GROUP BY date, store_id, item_id HAVING COUNT(*) > 1
UNION ALL
SELECT 'fact_sales_weekly', CAST(wm_yr_wk AS VARCHAR) || '|' || store_id || '|' || item_id, COUNT(*)
FROM fact_sales_weekly GROUP BY wm_yr_wk, store_id, item_id HAVING COUNT(*) > 1
UNION ALL
SELECT 'fact_revenue_weekly', CAST(wm_yr_wk AS VARCHAR) || '|' || store_id || '|' || dept_id, COUNT(*)
FROM fact_revenue_weekly GROUP BY wm_yr_wk, store_id, dept_id HAVING COUNT(*) > 1
LIMIT 20;
