-- c03: sum of daily revenue = sum of weekly revenue for every store x full week,
-- and item-level weekly = store x department weekly.
WITH daily AS (
    SELECT dd.wm_yr_wk, f.store_id, SUM(f.revenue) AS rev
    FROM fact_sales_daily AS f
    INNER JOIN dim_date AS dd ON dd.date = f.date
    WHERE dd.is_full_week
    GROUP BY dd.wm_yr_wk, f.store_id
),
weekly AS (
    SELECT wm_yr_wk, store_id, SUM(revenue) AS rev
    FROM fact_sales_weekly GROUP BY wm_yr_wk, store_id
),
dept AS (
    SELECT wm_yr_wk, store_id, SUM(revenue) AS rev
    FROM fact_revenue_weekly GROUP BY wm_yr_wk, store_id
)
SELECT d.wm_yr_wk, d.store_id, d.rev AS daily_rev, w.rev AS weekly_rev, x.rev AS dept_rev
FROM daily AS d
FULL OUTER JOIN weekly AS w ON w.wm_yr_wk = d.wm_yr_wk AND w.store_id = d.store_id
FULL OUTER JOIN dept   AS x ON x.wm_yr_wk = d.wm_yr_wk AND x.store_id = d.store_id
WHERE d.rev IS NULL OR w.rev IS NULL OR x.rev IS NULL
   OR ABS(d.rev - w.rev) > 1e-6 * GREATEST(1.0, ABS(d.rev))
   OR ABS(d.rev - x.rev) > 1e-6 * GREATEST(1.0, ABS(d.rev))
LIMIT 20;
