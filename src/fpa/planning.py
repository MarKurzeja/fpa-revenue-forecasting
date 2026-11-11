"""Simulated FP&A cycle: frozen budget, monthly re-forecasts, variance, PVM (SPEC §8).

- Budget: at the first week of the budget fiscal year, forecast the whole year with
  the champion pipeline and freeze it.
- Re-forecast ("latest estimate"): at the first week of each later fiscal month,
  forecast the rest of the year with actuals to date.
- Rolling: a 13-week forecast from the end of the data (used by the app).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from fpa import pvm
from fpa.commentary import month_commentary
from fpa.config import Config

log = logging.getLogger(__name__)

VALUE_COLS = ["revenue_point", "lo80", "hi80", "lo95", "hi95"]


def budget_year(weeks: pd.DataFrame, requested: int | None, min_history: int = 104) -> int:
    """Last fiscal year fully in the data with at least `min_history` weeks before it."""
    if requested:
        return int(requested)
    full = []
    for fy, g in weeks.groupby("fiscal_year"):
        nxt_present = (weeks["fiscal_year"] > fy).any()
        if g["fiscal_week"].min() == 1 and nxt_present and g["t"].min() >= min_history:
            full.append(int(fy))
    if not full:
        raise ValueError("no full fiscal year with enough history for a budget")
    return max(full)


def version_plan(weeks: pd.DataFrame, fy: int) -> pd.DataFrame:
    """Budget + one re-forecast per later fiscal month; origin = first week of the month."""
    w = weeks[weeks["fiscal_year"] == fy]
    end_t = int(w["t"].max())
    rows = []
    for fm, g in w.groupby("fiscal_month"):
        origin = int(g["t"].min())
        rows.append(
            {
                "version_id": f"FY{fy}_BUD" if fm == 1 else f"FY{fy}_RF{int(fm):02d}",
                "type": "budget" if fm == 1 else "reforecast",
                "fiscal_year": fy,
                "fiscal_month_created": int(fm),
                "origin_t": origin,
                "created_at": pd.to_datetime(g["week_start"].min()).date(),
                "horizon": end_t - origin + 1,
            }
        )
    return pd.DataFrame(rows)


def to_values(version_id: str, rec: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """forecast_values rows from champion forecast records."""
    attrs = nodes.set_index("node_id")[["store_id", "dept_id"]]
    v = rec.rename(columns={"y_pred": "revenue_point"})
    v = v[["t", "wm_yr_wk", "node_id", "level", *VALUE_COLS]].copy()
    v.insert(0, "version_id", version_id)
    return v.join(attrs, on="node_id")


def run_versions(
    cfg: Config, ctx, selection: dict, qtable: pd.DataFrame, regressor_factory=None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create budget, re-forecast and rolling versions."""
    from fpa.backtest import champion_forecast

    panel = ctx.panel
    fy = budget_year(panel.weeks, cfg.planning.get("budget_fiscal_year"))
    plan = version_plan(panel.weeks, fy)
    values = []
    for _, v in plan.iterrows():
        log.info("version %s origin %d horizon %d", v["version_id"], v["origin_t"], v["horizon"])
        rec = champion_forecast(
            cfg, ctx, selection, qtable, int(v["origin_t"]), int(v["horizon"]), regressor_factory
        )
        values.append(to_values(v["version_id"], rec, panel.nodes))
    # Rolling 13-week forecast from the end of the data.
    rec = champion_forecast(cfg, ctx, selection, qtable, panel.T, cfg.horizon, regressor_factory)
    last = panel.weeks.iloc[-1]
    roll_id = f"ROLL_{int(last['wm_yr_wk'])}"
    values.append(to_values(roll_id, rec, panel.nodes))
    plan = pd.concat(
        [
            plan,
            pd.DataFrame(
                [
                    {
                        "version_id": roll_id,
                        "type": "rolling",
                        "fiscal_year": int(last["fiscal_year"]),
                        "fiscal_month_created": int(last["fiscal_month"]),
                        "origin_t": panel.T,
                        "created_at": (
                            pd.to_datetime(last["week_end"]) + pd.Timedelta(days=1)
                        ).date(),
                        "horizon": cfg.horizon,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    plan["model"] = selection["champion"]
    return plan, pd.concat(values, ignore_index=True)


# --------------------------------------------------------------------------- variance


def actuals_long(panel) -> pd.DataFrame:
    a = panel.long("y").rename(columns={"y": "actual"})
    return a.merge(panel.weeks[["t", "fiscal_year", "fiscal_quarter", "fiscal_month"]], on="t")


def variance_tables(panel, versions: pd.DataFrame, values: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Monthly, quarterly and YTD variance per node, plus budget-gap tracking."""
    plan = versions[versions["type"].isin(["budget", "reforecast"])]
    fy = int(plan["fiscal_year"].iloc[0])
    bud_id = plan.loc[plan["type"] == "budget", "version_id"].iloc[0]
    wk = panel.weeks[panel.weeks["fiscal_year"] == fy][["t", "fiscal_month", "fiscal_quarter"]]
    act = actuals_long(panel)
    act = act[act["fiscal_year"] == fy]
    vals = values.merge(wk, on="t", how="inner")
    budget = vals[vals["version_id"] == bud_id]

    # Latest estimate for month m = version created at the start of month m.
    latest = plan[["fiscal_month_created", "version_id"]].rename(
        columns={"fiscal_month_created": "fiscal_month"}
    )
    le = vals.merge(latest, on=["fiscal_month", "version_id"], how="inner")

    def monthly(df: pd.DataFrame, col: str, name: str) -> pd.DataFrame:
        return (
            df.groupby(["node_id", "level", "fiscal_month", "fiscal_quarter"], as_index=False)[col]
            .sum()
            .rename(columns={col: name})
        )

    m = (
        monthly(act, "actual", "actual")
        .merge(
            monthly(budget, "revenue_point", "budget"),
            on=["node_id", "level", "fiscal_month", "fiscal_quarter"],
        )
        .merge(
            monthly(le, "revenue_point", "latest_forecast"),
            on=["node_id", "level", "fiscal_month", "fiscal_quarter"],
        )
    )
    m.insert(0, "fiscal_year", fy)
    m = _add_var(m)
    q = m.groupby(["fiscal_year", "node_id", "level", "fiscal_quarter"], as_index=False)[
        ["actual", "budget", "latest_forecast"]
    ].sum()
    q = _add_var(q)
    ytd = m.sort_values("fiscal_month").copy()
    for c in ("actual", "budget", "latest_forecast"):
        ytd[c] = ytd.groupby("node_id")[c].cumsum()
    ytd = _add_var(ytd.drop(columns=["fiscal_quarter"]))

    # Budget-gap tracking: full-year outlook at each version.
    fy_budget = budget.groupby("node_id")["revenue_point"].sum()
    rows = []
    for _, v in plan.iterrows():
        o = int(v["origin_t"])
        a = act[act["t"] < o].groupby("node_id")["actual"].sum()
        f = (
            vals[(vals["version_id"] == v["version_id"]) & (vals["t"] >= o)]
            .groupby("node_id")["revenue_point"]
            .sum()
        )
        outlook = a.reindex(fy_budget.index, fill_value=0) + f.reindex(
            fy_budget.index, fill_value=0
        )
        for nid in fy_budget.index:
            rows.append(
                {
                    "version_id": v["version_id"],
                    "fiscal_month": v["fiscal_month_created"],
                    "node_id": nid,
                    "fy_budget": fy_budget[nid],
                    "fy_outlook": outlook[nid],
                }
            )
    gap = pd.DataFrame(rows)
    gap["gap_usd"] = gap["fy_outlook"] - gap["fy_budget"]
    gap["gap_pct"] = gap["gap_usd"] / gap["fy_budget"]
    full_year = act.groupby("node_id")["actual"].sum().rename("fy_actual").reset_index()
    return {"monthly": m, "quarterly": q, "ytd": ytd, "gap": gap, "full_year_actual": full_year}


def _add_var(df: pd.DataFrame) -> pd.DataFrame:
    df["var_vs_budget"] = df["actual"] - df["budget"]
    df["var_vs_budget_pct"] = df["var_vs_budget"] / df["budget"]
    df["var_vs_latest"] = df["actual"] - df["latest_forecast"]
    df["var_vs_latest_pct"] = df["var_vs_latest"] / df["latest_forecast"]
    df["latest_vs_budget"] = df["latest_forecast"] - df["budget"]
    return df


# --------------------------------------------------------------------------- PVM


def load_item_weeks(db_file, wm_weeks: list[int]) -> pd.DataFrame:
    from fpa.data.build import connect

    con = connect(db_file, read_only=True)
    try:
        con.register("wk", pd.DataFrame({"wm_yr_wk": wm_weeks}))
        return con.execute(
            """
            SELECT w.wm_yr_wk, w.store_id, s.state_id, w.item_id, i.dept_id, i.cat_id,
                   w.units, w.revenue
            FROM fact_sales_weekly AS w
            JOIN wk USING (wm_yr_wk)
            JOIN dim_item AS i USING (item_id)
            JOIN dim_store AS s USING (store_id)
            """
        ).fetchdf()
    finally:
        con.close()


def budget_item_plan(
    items_trailing: pd.DataFrame, budget_bottom: pd.DataFrame, price_weeks: int = 13
) -> pd.DataFrame:
    """Planned units and prices per item and week from the bottom-level budget.

    Assumption (D-11): budget revenue of a store x department is split across items by
    their trailing-52-week revenue mix, and converted to units at trailing prices
    (unit value over the last `price_weeks` weeks with sales, else the 52-week value).
    """
    tr = items_trailing.copy()
    last_wks = sorted(tr["wm_yr_wk"].unique())[-price_weeks:]
    k = ["store_id", "dept_id", "item_id", "cat_id", "state_id"]
    agg52 = tr.groupby(k, as_index=False)[["units", "revenue"]].sum()
    agg13 = (
        tr[tr["wm_yr_wk"].isin(last_wks)]
        .groupby(k, as_index=False)[["units", "revenue"]]
        .sum()
        .rename(columns={"units": "u13", "revenue": "r13"})
    )
    it = agg52.merge(agg13, on=k, how="left")
    it = it[it["revenue"] > 0]
    it["share"] = it["revenue"] / it.groupby(["store_id", "dept_id"])["revenue"].transform("sum")
    it["price"] = np.where(it["u13"] > 0, it["r13"] / it["u13"], it["revenue"] / it["units"])
    b = budget_bottom.merge(
        it[["store_id", "dept_id", "item_id", "cat_id", "state_id", "share", "price"]],
        on=["store_id", "dept_id"],
        how="inner",
    )
    b["revenue"] = b["revenue_point"] * b["share"]
    b["units"] = b["revenue"] / b["price"]
    return b


def pvm_tables(panel, db_file, versions: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    """PVM bridges (actual vs prior year, actual vs budget) per month and full year."""
    plan = versions[versions["type"] == "budget"].iloc[0]
    fy = int(plan["fiscal_year"])
    w = panel.weeks
    cur = w[w["fiscal_year"] == fy]
    prev = w[w["fiscal_year"] == fy - 1]
    items = load_item_weeks(db_file, [*prev["wm_yr_wk"].tolist(), *cur["wm_yr_wk"].tolist()])
    items = items.merge(w[["wm_yr_wk", "fiscal_year", "fiscal_month"]], on="wm_yr_wk")
    keys = ["store_id", "state_id", "dept_id", "cat_id", "item_id"]

    bud_bottom = values[(values["version_id"] == plan["version_id"]) & (values["level"] == 8)]
    bud_bottom = bud_bottom.merge(w[["t", "fiscal_month"]], on="t")
    bud_items = budget_item_plan(items[items["fiscal_year"] == fy - 1], bud_bottom)

    out = []
    periods = [(int(m), [int(m)]) for m in sorted(cur["fiscal_month"].unique())]
    periods.append((0, sorted(int(m) for m in cur["fiscal_month"].unique())))  # 0 = full year
    for label, months in periods:
        a = items[(items["fiscal_year"] == fy) & items["fiscal_month"].isin(months)]
        a = a.groupby(keys, as_index=False)[["units", "revenue"]].sum()
        a = a.rename(columns={"units": "q1", "revenue": "r1"})
        py = items[(items["fiscal_year"] == fy - 1) & items["fiscal_month"].isin(months)]
        py = (
            py.groupby(keys, as_index=False)[["units", "revenue"]]
            .sum()
            .rename(columns={"units": "q0", "revenue": "r0"})
        )
        bd = (
            bud_items[bud_items["fiscal_month"].isin(months)]
            .groupby(keys, as_index=False)[["units", "revenue"]]
            .sum()
            .rename(columns={"units": "q0", "revenue": "r0"})
        )
        for comp, base in (("vs_prior_year", py), ("vs_budget", bd)):
            df = base.merge(a, on=keys, how="outer").fillna({"q0": 0, "r0": 0, "q1": 0, "r1": 0})
            br = pvm.bridge_by_group(df, ["store_id", "state_id", "dept_id", "cat_id"])
            br.insert(0, "comparison", comp)
            br.insert(1, "fiscal_year", fy)
            br.insert(2, "fiscal_month", label)
            out.append(br)
    return pd.concat(out, ignore_index=True)


def commentary(var: dict[str, pd.DataFrame], pvm_df: pd.DataFrame) -> pd.DataFrame:
    """Company-level commentary per fiscal month."""
    m = var["monthly"][var["monthly"]["node_id"] == "Total"].sort_values("fiscal_month")
    gap = var["gap"][var["gap"]["node_id"] == "Total"].set_index("fiscal_month")["gap_pct"]
    rows = []
    for _, r in m.iterrows():
        fm = int(r["fiscal_month"])
        p = pvm_df[(pvm_df["comparison"] == "vs_budget") & (pvm_df["fiscal_month"] == fm)]
        tot = p[["volume", "mix", "price"]].sum().to_dict()
        seg = p.groupby(["state_id", "cat_id"], as_index=False)[["volume", "mix", "price"]].sum()
        text = month_commentary(
            int(r["fiscal_year"]),
            fm,
            float(r["actual"]),
            float(r["budget"]),
            tot,
            seg,
            float(gap[fm]) if fm in gap.index else None,
            float(gap[fm - 1]) if fm - 1 in gap.index else None,
        )
        rows.append({"fiscal_year": int(r["fiscal_year"]), "fiscal_month": fm, "text": text})
    return pd.DataFrame(rows)


def run_planning(
    cfg: Config, ctx, selection: dict, qtable: pd.DataFrame, regressor_factory=None
) -> dict[str, Any]:
    """Everything in SPEC §8, written to the artifacts directory."""
    from fpa import io

    versions, values = run_versions(cfg, ctx, selection, qtable, regressor_factory)
    var = variance_tables(ctx.panel, versions, values)
    pvm_df = pvm_tables(ctx.panel, cfg.paths.duckdb_file, versions, values)
    comm = commentary(var, pvm_df)
    out = cfg.paths.artifacts_dir / "planning"
    io.save_table(versions.astype({"created_at": str}), out / "forecast_versions")
    io.save_table(values, out / "forecast_values")
    for k, df in var.items():
        io.save_table(df, out / f"variance_{k}")
    io.save_table(pvm_df, out / "pvm")
    io.save_table(comm, out / "commentary")
    return {
        "versions": versions,
        "values": values,
        "variance": var,
        "pvm": pvm_df,
        "commentary": comm,
    }
