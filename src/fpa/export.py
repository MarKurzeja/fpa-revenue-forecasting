"""Power BI CSV exports (SPEC §12) and aggregated artefacts for the app (SPEC §11.3)."""

from __future__ import annotations

import shutil
from pathlib import Path

from fpa import io
from fpa.config import Config
from fpa.evaluate import horizon_buckets, summarize
from fpa.features import extend_weeks


def _con(cfg: Config):
    from fpa.data.build import connect

    return connect(cfg.paths.duckdb_file, read_only=True)


def export_powerbi(cfg: Config) -> Path:
    out = cfg.paths.exports_dir
    out.mkdir(parents=True, exist_ok=True)
    con = _con(cfg)
    try:
        dim_week = con.execute("SELECT * FROM dim_week ORDER BY t").fetchdf()
        con.execute(f"COPY (SELECT * FROM v_pbi_dim_store) TO '{out / 'dim_store.csv'}' (HEADER)")
        con.execute(f"COPY (SELECT * FROM v_pbi_dim_dept) TO '{out / 'dim_dept.csv'}' (HEADER)")
        con.execute(
            f"COPY (SELECT * FROM v_pbi_fact_actuals_weekly) TO "
            f"'{out / 'fact_actuals_weekly.csv'}' (HEADER)"
        )
    finally:
        con.close()
    dd = extend_weeks(dim_week, cfg.horizon, cfg.fiscal)
    dd["is_actual"] = ~dd["is_projected"]
    dd[
        [
            "wm_yr_wk",
            "t",
            "week_start",
            "week_end",
            "fiscal_year",
            "fiscal_quarter",
            "fiscal_month",
            "fiscal_week",
            "fiscal_period",
            "event_names",
            "is_actual",
        ]
    ].to_csv(out / "dim_date.csv", index=False)

    a = cfg.paths.artifacts_dir
    plan = a / "planning"
    if io.table_exists(plan / "forecast_versions"):
        io.load_table(plan / "forecast_versions").to_csv(out / "dim_version.csv", index=False)
        vals = io.load_table(plan / "forecast_values")
        bottom = vals[vals["level"] == 8]
        bottom[
            [
                "version_id",
                "wm_yr_wk",
                "store_id",
                "dept_id",
                "revenue_point",
                "lo80",
                "hi80",
                "lo95",
                "hi95",
            ]
        ].to_csv(out / "fact_forecast_weekly.csv", index=False)
        vals[
            [
                "version_id",
                "wm_yr_wk",
                "node_id",
                "level",
                "revenue_point",
                "lo80",
                "hi80",
                "lo95",
                "hi95",
            ]
        ].to_csv(out / "fact_forecast_nodes.csv", index=False)
        io.load_table(plan / "pvm").to_csv(out / "fact_pvm.csv", index=False)
        io.load_table(plan / "variance_monthly").to_csv(
            out / "fact_variance_monthly.csv", index=False
        )
        io.load_table(plan / "commentary").to_csv(out / "fact_commentary.csv", index=False)
    if io.table_exists(a / "backtest_records"):
        rec = io.load_table(a / "backtest_records").dropna(subset=["y_true"])
        rec = horizon_buckets(rec, cfg.horizon_buckets)
        met = summarize(rec, ["model", "family", "role", "level", "horizon_bucket"], scaled=False)
        met.to_csv(out / "fact_metrics.csv", index=False)
    return out


def export_demo(cfg: Config) -> Path:
    """Small aggregated tables for the Streamlit app (no item-level data)."""
    from fpa.panel import load_panel

    out = cfg.paths.demo_dir
    out.mkdir(parents=True, exist_ok=True)
    panel = load_panel(cfg.paths.duckdb_file)
    io.save_table(panel.nodes, out / "nodes")
    io.save_table(panel.weeks, out / "weeks")
    io.save_table(panel.long("y").rename(columns={"y": "revenue"}), out / "actuals")
    a = cfg.paths.artifacts_dir
    if io.table_exists(a / "backtest_records"):
        rec = io.load_table(a / "backtest_records")
        keep = [
            "model",
            "family",
            "origin",
            "role",
            "node_id",
            "level",
            "h",
            "t",
            "wm_yr_wk",
            "y_true",
            "y_pred",
            "lo80",
            "hi80",
            "lo95",
            "hi95",
        ]
        io.save_table(rec[keep], out / "backtest")
        io.save_table(
            summarize(
                horizon_buckets(rec.dropna(subset=["y_true"]), cfg.horizon_buckets),
                ["model", "family", "role", "level", "horizon_bucket"],
                scaled=False,
            ),
            out / "metrics",
        )
    for src in ("selection.json", "coherence.json", "diebold_mariano.json"):
        if (a / src).exists():
            shutil.copy(a / src, out / src)
    for sub in ("planning", "capital"):
        d = a / sub
        if d.exists():
            for f in d.iterdir():
                shutil.copy(f, out / f"{sub}_{f.name}")
    return out
