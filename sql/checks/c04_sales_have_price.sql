-- c04: every sale with units > 0 has a price.
SELECT CAST(date AS VARCHAR) AS date, store_id, item_id, units
FROM fact_sales_daily
WHERE units > 0 AND sell_price IS NULL
LIMIT 20;
