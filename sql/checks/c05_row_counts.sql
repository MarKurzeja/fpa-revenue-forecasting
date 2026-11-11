-- c05: row counts match expectations within a tolerance.
-- Template variables: {expected_series}, {tolerance}
WITH counts AS (
    SELECT
        (SELECT COUNT(*) FROM stg_sales_wide)                                  AS n_series,
        (SELECT COUNT(*) FROM dim_date)                                        AS n_days,
        (SELECT COUNT(*) FROM dim_week)                                        AS n_weeks,
        (SELECT COUNT(*) FROM fact_sales_daily)                                AS n_daily,
        (SELECT COUNT(*) FROM fact_sales_weekly)                               AS n_weekly,
        (SELECT COUNT(*) FROM fact_revenue_weekly)                             AS n_rev,
        (SELECT COUNT(*) FROM (SELECT DISTINCT store_id, dept_id FROM fact_revenue_weekly)) AS n_bottom
)
SELECT 'n_series' AS what, n_series AS actual, {expected_series} AS expected FROM counts
WHERE ABS(n_series - {expected_series}) > {tolerance} * {expected_series}
UNION ALL
SELECT 'fact_sales_daily', n_daily, n_series * n_days FROM counts WHERE n_daily <> n_series * n_days
UNION ALL
SELECT 'fact_sales_weekly', n_weekly, n_series * n_weeks FROM counts WHERE n_weekly <> n_series * n_weeks
UNION ALL
SELECT 'fact_revenue_weekly', n_rev, n_bottom * n_weeks FROM counts WHERE n_rev <> n_bottom * n_weeks;
