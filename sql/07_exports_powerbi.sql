-- 07_exports_powerbi.sql
-- Purpose: star-schema views for Power BI. Forecast, metric and PVM facts are
-- written from Python (fpa.export) because they come from the modelling layer.

CREATE OR REPLACE VIEW v_pbi_dim_date AS
SELECT wm_yr_wk, week_start, week_end, fiscal_year, fiscal_quarter, fiscal_month,
       fiscal_week, fiscal_period, n_sporting + n_cultural + n_national + n_religious AS n_events,
       event_names, has_closure
FROM dim_week;

CREATE OR REPLACE VIEW v_pbi_dim_store AS
SELECT store_id, state_id FROM dim_store;

CREATE OR REPLACE VIEW v_pbi_dim_dept AS
SELECT dept_id, cat_id FROM dim_dept;

CREATE OR REPLACE VIEW v_pbi_fact_actuals_weekly AS
SELECT wm_yr_wk, store_id, dept_id, units, revenue, price_index
FROM fact_revenue_weekly;
