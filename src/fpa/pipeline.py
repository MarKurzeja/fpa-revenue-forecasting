"""Pipeline steps behind the CLI (SPEC §13). Each step caches its outputs."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd

from fpa import io
from fpa import working_capital as wc
from fpa.config import Config
from fpa.evaluate import level_wape, summarize
from fpa.hierarchy import is_coherent
from fpa.metrics import diebold_mariano

log = logging.getLogger(__name__)


@contextmanager
def timed(step: str, timings: dict[str, float]):
    t0 = time.perf_counter()
    print(f"==> {step}")
    yield
    timings[step] = time.perf_counter() - t0
    print(f"<== {step} done in {timings[step]:.1f}s")


def _panel(cfg: Config):
    from fpa.panel import load_panel

    return load_panel(cfg.paths.duckdb_file)


def step_build(cfg: Config, force: bool = False) -> None:
    from fpa.data.build import build, connect, write_data_facts
    from fpa.data.download import has_raw_data

    if not has_raw_data(cfg.paths.raw_dir):
        raise SystemExit(f"M5 files not found in {cfg.paths.raw_dir}. Run: python run.py download")
    has_built_database = False
    if cfg.paths.duckdb_file.exists() and not force:
        con = connect(cfg.paths.duckdb_file, read_only=True)
        try:
            required_tables = (
                "dim_item",
                "dim_store",
                "dim_dept",
                "dim_date",
                "dim_week",
                "stg_fiscal_weeks",
                "fact_sales_weekly",
                "fact_revenue_weekly",
            )
            n_tables = con.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema = 'main' AND table_name IN "
                f"({', '.join(repr(name) for name in required_tables)})"
            ).fetchone()[0]
            has_built_database = n_tables == len(required_tables)
        finally:
            con.close()
    if has_built_database:
        print(f"    using existing {cfg.paths.duckdb_file} (--force to rebuild)")
    else:
        if cfg.paths.duckdb_file.exists() and force:
            cfg.paths.duckdb_file.unlink()
        build(cfg)
    out = write_data_facts(cfg)
    print(f"    data facts -> {out}")


def step_backtest(
    cfg: Config,
    force: bool = False,
    model_set: str = "all",
    regressor_factory: Callable | None = None,
) -> dict[str, Any]:
    from fpa.backtest import run_backtest

    res = run_backtest(
        cfg, _panel(cfg), force=force, model_set=model_set, regressor_factory=regressor_factory
    )
    sel = res["selection"]
    print(
        f"    champion: {sel['champion']}  (base {sel['champion_base']}, "
        f"members {sel['ensemble_members']})"
    )
    return res


def load_backtest(cfg: Config) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    a = cfg.paths.artifacts_dir
    if not (a / "selection.json").exists():
        raise SystemExit("No backtest results. Run: python run.py backtest")
    return (
        io.load_table(a / "backtest_records"),
        io.load_json(a / "selection.json"),
        io.load_table(a / "interval_quantiles"),
    )


def step_reconcile(cfg: Config) -> pd.DataFrame:
    """Per-level accuracy before/after reconciliation and coherence check."""
    records, sel, _ = load_backtest(cfg)
    panel = _panel(cfg)
    base = sel["champion_base"]
    names = [base, *[f"{base}+{m}" for m in cfg.reconciliation["methods"]]]
    r = records[records["model"].isin(names) & (records["role"] != "holdout")]
    comp = level_wape(r).pivot(index="level", columns="model", values="wape").reset_index()
    coherence = {}
    for name in names:
        g = records[records["model"] == name]
        ok = True
        for (_, _), gg in g.groupby(["origin", "h"]):
            y = np.zeros(panel.h.n_nodes)
            y[gg["node"].to_numpy()] = gg["y_pred"].to_numpy()
            ok &= is_coherent(panel.h, y, rtol=float(cfg.quality["coherence_tolerance"]))
        coherence[name] = bool(ok)
    io.save_table(comp, cfg.paths.artifacts_dir / "reconciliation_comparison")
    io.save_json(coherence, cfg.paths.artifacts_dir / "coherence.json")
    for k, v in coherence.items():
        print(f"    coherent {k:<32} {v}")
    return comp


def step_plan(cfg: Config, regressor_factory: Callable | None = None) -> dict[str, Any]:
    from fpa.backtest import make_context
    from fpa.planning import run_planning

    _, sel, qtable = load_backtest(cfg)
    ctx = make_context(cfg, _panel(cfg))
    res = run_planning(cfg, ctx, sel, qtable, regressor_factory)
    tot = res["variance"]["ytd"]
    tot = tot[tot["node_id"] == "Total"].sort_values("fiscal_month").iloc[-1]
    print(
        f"    FY{int(tot['fiscal_year'])} actual {tot['actual']:,.0f} vs budget "
        f"{tot['budget']:,.0f} ({tot['var_vs_budget_pct']:+.1%})"
    )
    return res


def step_capital(cfg: Config) -> dict[str, pd.DataFrame]:
    """Working capital (SPEC §9) and scenarios (SPEC §9.1)."""
    from fpa import elasticity as el

    records, sel, _ = load_backtest(cfg)
    panel = _panel(cfg)
    w = cfg.working_capital
    champion = sel["champion"]
    methods = [champion if m == "champion" else m for m in w.compare_methods]
    bottom = records[(records["level"] == 8) & records["model"].isin(methods)]
    lt = wc.lead_time_errors(bottom, w.lead_time_weeks)
    sigma = wc.sigma_by_node(lt)
    cat_of = dict(zip(panel.nodes["node_id"], panel.nodes["cat_id"]))
    bench = [m for m in methods if m != champion]
    cap = wc.capital_table(
        sigma, cat_of, w.cogs_ratio, w.service_levels, w.annual_carrying_cost_rate, bench, champion
    )
    svc = wc.service_level_check(lt, sigma, w.service_levels, role="holdout")

    # Elasticities on bottom-level weekly data.
    nodes = panel.nodes.set_index("node_id")
    b_idx = panel.h.bottom_idx
    long = pd.DataFrame(
        {
            "node_id": np.repeat(panel.nodes["node_id"].to_numpy()[b_idx], panel.T),
            "t": np.tile(np.arange(panel.T), len(b_idx)),
            "units": panel.units[b_idx].reshape(-1),
            "price_index": panel.price_index[b_idx].reshape(-1),
        }
    )
    long["cat_id"] = long["node_id"].map(nodes["cat_id"])
    long["state_id"] = long["node_id"].map(nodes["state_id"])
    elas = el.estimate_elasticities(long, panel.weeks)

    # Price scenarios on the rolling forecast (category nodes).
    scen = pd.DataFrame()
    vpath = cfg.paths.artifacts_dir / "planning" / "forecast_values"
    if io.table_exists(vpath):
        vals = io.load_table(vpath)
        roll = vals[vals["version_id"].str.startswith("ROLL") & (vals["level"] == 4)]
        base = roll.groupby("node_id")["revenue_point"].sum()
        rows = []
        for _, e in elas.iterrows():
            b = float(base.get(e["cat_id"], np.nan))
            for dp in cfg.scenarios["price_changes"]:
                f = el.revenue_factor(dp, e["elasticity"])
                rows.append(
                    {
                        "cat_id": e["cat_id"],
                        "price_change": dp,
                        "elasticity": e["elasticity"],
                        "base_revenue_13w": b,
                        "scenario_revenue_13w": b * f,
                        "impact_usd": b * (f - 1),
                    }
                )
        scen = pd.DataFrame(rows)
    ev = el.event_uplifts(panel.y[panel.h.total_idx], panel.weeks, cfg.scenarios["major_events"])

    out = cfg.paths.artifacts_dir / "capital"
    for name, df in (
        ("working_capital", cap),
        ("service_level_check", svc),
        ("sigma_lt", sigma),
        ("elasticities", elas),
        ("price_scenarios", scen),
        ("event_uplifts", ev),
    ):
        io.save_table(df, out / name)
    released = (
        cap.dropna(subset=["working_capital_released_usd"])
        if ("working_capital_released_usd" in cap)
        else pd.DataFrame()
    )
    for _, r in released.iterrows():
        print(
            f"    SL {r['service_level']:.0%} {r['model']}: released "
            f"{r['working_capital_released_usd']:,.0f} USD, carrying cost saved "
            f"{r['carrying_cost_saved_usd']:,.0f} USD/yr"
        )
    return {
        "working_capital": cap,
        "service_level_check": svc,
        "elasticities": elas,
        "price_scenarios": scen,
        "event_uplifts": ev,
    }


def dm_test(
    records: pd.DataFrame,
    champion: str,
    benchmark: str = "seasonal_naive",
    roles: tuple[str, ...] = ("test", "holdout"),
    h_max: int = 13,
) -> dict[str, Any]:
    """DM test on bottom-level absolute-error differentials (sum over series per week)."""
    r = records[(records["level"] == 8) & records["role"].isin(roles)].dropna(subset=["y_true"])
    piv = r.assign(err=r["y_true"] - r["y_pred"]).pivot_table(
        index=["origin", "h", "node"], columns="model", values="err"
    )
    piv = piv[[champion, benchmark]].dropna()
    agg = piv.abs().groupby(level=["origin", "h"]).sum().sort_index()
    res = diebold_mariano(agg[champion].to_numpy(), agg[benchmark].to_numpy(), h=h_max, loss="abs")
    return {"champion": champion, "benchmark": benchmark, **res.__dict__}


def quality_gates(cfg: Config) -> list[str]:
    """Failures of the `check` quality thresholds (empty list = pass)."""
    records, sel, _ = load_backtest(cfg)
    fails = []
    ev = records[records["role"].isin(["test", "holdout"])]
    s = summarize(ev, ["model"], scaled=False).set_index("model")
    ratio = s.loc[sel["champion"], "wape"] / s.loc["seasonal_naive", "wape"]
    if ratio > float(cfg.quality["max_wape_ratio_vs_snaive"]):
        fails.append(f"champion WAPE / seasonal naive = {ratio:.3f} > threshold")
    for L in cfg.interval_levels:
        lo, hi = cfg.quality["coverage_band"][str(L)]
        cov = s.loc[sel["champion"], f"coverage{L}"]
        if not lo <= cov <= hi:
            fails.append(f"coverage{L} = {cov:.3f} outside [{lo}, {hi}]")
    coh_path = cfg.paths.artifacts_dir / "coherence.json"
    if coh_path.exists():
        coh = io.load_json(coh_path)
        if not all(v for k, v in coh.items() if "+" in k):
            fails.append(f"reconciliation coherence failed: {coh}")
    else:
        fails.append("coherence not checked (run reconcile)")
    return fails
