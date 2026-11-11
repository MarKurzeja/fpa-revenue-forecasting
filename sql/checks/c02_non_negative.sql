-- c02: revenue >= 0 and units >= 0 in every fact table.
SELECT 'fact_sales_daily' AS tbl, COUNT(*) AS n FROM fact_sales_daily WHERE units < 0 OR revenue < 0 HAVING COUNT(*) > 0
UNION ALL
SELECT 'fact_sales_weekly', COUNT(*) FROM fact_sales_weekly WHERE units < 0 OR revenue < 0 HAVING COUNT(*) > 0
UNION ALL
SELECT 'fact_revenue_weekly', COUNT(*) FROM fact_revenue_weekly WHERE units < 0 OR revenue < 0 HAVING COUNT(*) > 0;
