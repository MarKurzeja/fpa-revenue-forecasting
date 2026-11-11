-- 03_dims.sql
-- Purpose: store, item and department dimensions.

CREATE OR REPLACE TABLE dim_store AS
SELECT DISTINCT store_id, state_id
FROM stg_sales_wide
ORDER BY store_id;

CREATE OR REPLACE TABLE dim_item AS
SELECT DISTINCT item_id, dept_id, cat_id
FROM stg_sales_wide
ORDER BY item_id;

CREATE OR REPLACE TABLE dim_dept AS
SELECT DISTINCT dept_id, cat_id
FROM stg_sales_wide
ORDER BY dept_id;
