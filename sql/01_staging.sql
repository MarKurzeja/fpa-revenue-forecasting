-- 01_staging.sql
-- Purpose: load the three raw M5 files into typed staging tables and reshape the
-- wide daily sales matrix (d_1 .. d_1941) into a long (item, store, day) table.
-- Template variables: {raw_dir}, {store_filter}

CREATE OR REPLACE TABLE stg_calendar AS
WITH calendar AS (
    SELECT
        *,
        CAST(ROW_NUMBER() OVER (ORDER BY CAST(date AS DATE)) AS INTEGER) AS d_num
    FROM read_csv('{raw_dir}/calendar.csv', header = true, auto_detect = true,
                  types = {{'event_name_1': 'VARCHAR', 'event_type_1': 'VARCHAR',
                            'event_name_2': 'VARCHAR', 'event_type_2': 'VARCHAR'}})
)
SELECT
    CAST(date AS DATE)                      AS date,
    'd_' || CAST(d_num AS VARCHAR)           AS d,
    d_num,
    CAST(wm_yr_wk AS INTEGER)               AS wm_yr_wk,
    weekday,
    CAST(wday AS INTEGER)                   AS wday,
    CAST(month AS INTEGER)                  AS month,
    CAST(year AS INTEGER)                   AS year,
    NULLIF(CAST(event_name_1 AS VARCHAR), '') AS event_name_1,
    NULLIF(CAST(event_type_1 AS VARCHAR), '') AS event_type_1,
    NULLIF(CAST(event_name_2 AS VARCHAR), '') AS event_name_2,
    NULLIF(CAST(event_type_2 AS VARCHAR), '') AS event_type_2,
    CAST(snap_CA AS INTEGER)                AS snap_ca,
    CAST(snap_TX AS INTEGER)                AS snap_tx,
    CAST(snap_WI AS INTEGER)                AS snap_wi
FROM calendar;

CREATE OR REPLACE TABLE stg_sales_wide AS
SELECT *
FROM read_csv('{raw_dir}/sales_train_evaluation.csv', header = true, auto_detect = true)
WHERE {store_filter};

CREATE OR REPLACE TABLE stg_prices AS
SELECT
    p.store_id,
    p.item_id,
    CAST(p.wm_yr_wk AS INTEGER)   AS wm_yr_wk,
    CAST(p.sell_price AS DOUBLE)  AS sell_price
FROM read_csv('{raw_dir}/sell_prices.csv', header = true, auto_detect = true) AS p
WHERE p.store_id IN (SELECT DISTINCT store_id FROM stg_sales_wide);

-- Wide -> long. UNPIVOT keeps every day, including zero-sales days.
CREATE OR REPLACE TABLE stg_sales_long AS
SELECT
    item_id,
    dept_id,
    cat_id,
    store_id,
    state_id,
    d,
    CAST(units AS INTEGER) AS units
FROM (
    UNPIVOT stg_sales_wide
    ON COLUMNS('^d_[0-9]+$')
    INTO NAME d VALUE units
) AS long;
