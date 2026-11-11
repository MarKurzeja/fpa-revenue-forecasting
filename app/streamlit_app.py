"""FP&A revenue forecasting app (SPEC §11). Reads aggregated artefacts only.

Run locally with `python run.py app` after `python run.py all` (+ `export`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpa import io  # noqa: E402
from fpa import working_capital as wc  # noqa: E402
from fpa.commentary import fmt_pct, fmt_usd  # noqa: E402
from fpa.elasticity import revenue_factor  # noqa: E402
from fpa.metrics import wape  # noqa: E402

DEMO = ROOT / "artifacts" / "demo"
CAT_COLORS = {"FOODS": "#2a7ab9", "HOBBIES": "#e08a1e", "HOUSEHOLD": "#3d9a5b"}
GREEN, RED, BLUE, GREY = "#3d9a5b", "#c0392b", "#2a7ab9", "#555555"
REPO_URL = "https://github.com/MarKurzeja/fpa-revenue-forecasting"

st.set_page_config(page_title="FP&A revenue forecasting (Walmart M5)", layout="wide")


@st.cache_data
def table(name: str) -> pd.DataFrame | None:
    p = DEMO / name
    return io.load_table(p) if io.table_exists(p) else None


@st.cache_data
def js(name: str) -> dict | None:
    p = DEMO / name
    return io.load_json(p) if p.exists() else None


def usd_axis(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        yaxis_tickprefix="$",
        yaxis_tickformat="~s",
        margin=dict(t=40, b=20),
        legend=dict(orientation="h", y=1.08),
    )
    return fig


def caption(text: str) -> None:
    st.caption(text)


def style_var(v: float) -> str:
    return f"color: {GREEN if v >= 0 else RED}" if pd.notna(v) else ""


nodes, weeks, actuals = table("nodes"), table("weeks"), table("actuals")
if nodes is None:
    st.error(
        "No artefacts found in artifacts/demo. Run `python run.py all` then "
        "`python run.py export` (needs the M5 data, see README)."
    )
    st.stop()
sel = js("selection.json") or {}
champion = sel.get("champion", "")
week_date = dict(zip(weeks["t"], pd.to_datetime(weeks["week_start"])))


def pick_node(prefix: str) -> str:
    """Drop-downs for state, store, category, department -> hierarchy node id."""
    c = st.columns(4)
    states = ["All", *sorted(nodes["state_id"].dropna().unique())]
    state = c[0].selectbox("State", states, key=f"{prefix}s")
    st_nodes = nodes[nodes["level"] == 3]
    if state != "All":
        st_nodes = st_nodes[st_nodes["state_id"] == state]
    store = c[1].selectbox("Store", ["All", *sorted(st_nodes["store_id"])], key=f"{prefix}st")
    cat = c[2].selectbox(
        "Category", ["All", *sorted(nodes["cat_id"].dropna().unique())], key=f"{prefix}c"
    )
    d_nodes = nodes[nodes["level"] == 5]
    if cat != "All":
        d_nodes = d_nodes[d_nodes["cat_id"] == cat]
    dept = c[3].selectbox("Department", ["All", *sorted(d_nodes["dept_id"])], key=f"{prefix}d")
    loc = store if store != "All" else (state if state != "All" else None)
    prod = dept if dept != "All" else (cat if cat != "All" else None)
    if loc and prod:
        nid = f"{loc}/{prod}"
        if nid not in set(nodes["node_id"]):
            cat_id = nodes.set_index("node_id").at[prod, "cat_id"]
            st.info(
                f"{loc} x {prod} is not a node of the hierarchy (state x department is not "
                f"modelled); showing {loc} x {cat_id}."
            )
            nid = f"{loc}/{cat_id}"
        return nid
    return loc or prod or "Total"


page = st.sidebar.radio(
    "Page",
    [
        "Forecast explorer",
        "Accuracy",
        "Budget vs actual",
        "Working capital",
        "Scenarios",
        "About / methodology",
    ],
)
st.sidebar.markdown(f"Champion model: **{champion}**")

# ----------------------------------------------------------------------------- pages
if page == "Forecast explorer":
    st.title("Forecast explorer")
    nid = pick_node("fx")
    bt = table("backtest")
    vals = table("planning_forecast_values")
    hist = actuals[actuals["node_id"] == nid].sort_values("t")
    fig = go.Figure()
    fig.add_scatter(
        x=hist["t"].map(week_date),
        y=hist["revenue"],
        name="Actual",
        line=dict(color=GREY, width=1.5),
    )
    if vals is not None:
        roll = vals[vals["version_id"].str.startswith("ROLL") & (vals["node_id"] == nid)]
        roll = roll.sort_values("t")
        last = pd.to_datetime(weeks["week_start"]).max()
        x = [last + pd.Timedelta(weeks=int(t - weeks["t"].max())) for t in roll["t"]]
        for lv, op in ((95, 0.15), (80, 0.3)):
            fig.add_scatter(
                x=x + x[::-1],
                y=list(roll[f"hi{lv}"]) + list(roll[f"lo{lv}"])[::-1],
                fill="toself",
                fillcolor=f"rgba(42,122,185,{op})",
                line=dict(width=0),
                name=f"{lv}% interval",
            )
        fig.add_scatter(
            x=x,
            y=roll["revenue_point"],
            name="Rolling 13-week forecast",
            line=dict(color=BLUE, width=2.5),
        )
    if bt is not None:
        models = sorted(bt["model"].unique())
        default = models.index(champion) if champion in models else 0
        m = st.selectbox("Backtest model", models, index=default)
        g = bt[(bt["model"] == m) & (bt["node_id"] == nid)]
        for o, go_ in g.groupby("origin"):
            go_ = go_.sort_values("t")
            fig.add_scatter(
                x=go_["t"].map(week_date),
                y=go_["y_pred"],
                mode="lines",
                line=dict(color="#e08a1e", width=1, dash="dot"),
                name=f"backtest {m}",
                legendgroup="bt",
                showlegend=bool(o == g["origin"].min()),
            )
    st.plotly_chart(usd_axis(fig), use_container_width=True)
    caption(
        f"{nid}: weekly revenue (grey), the reconciled 13-week forecast from the end of the "
        "data with 80/95% intervals (blue), and 13-week backtest forecasts from each origin "
        "(dotted)."
    )

elif page == "Accuracy":
    st.title("Forecast accuracy")
    met = table("metrics")
    bt = table("backtest")
    role = st.radio("Origins", ["test", "holdout"], horizontal=True)
    r = bt[bt["role"] == role].dropna(subset=["y_true"])
    lvl = r.groupby(["model", "level"]).apply(lambda g: wape(g["y_true"], g["y_pred"])).unstack()
    lvl["mean"] = lvl.mean(axis=1)
    lvl = lvl.sort_values("mean")
    st.dataframe(lvl.style.format("{:.1%}"), use_container_width=True)
    caption(
        "WAPE (sum of absolute errors / sum of actuals) by model and hierarchy level "
        "(L1 company ... L8 store x department)."
    )
    fam = sel.get("families", {})
    chain = ["seasonal_naive", "run_rate"]
    stats = [m for m in lvl.index if fam.get(m) == "stats"]
    if stats:
        chain.append(min(stats, key=lambda m: lvl.at[m, "mean"]))
    chain += [m for m in ("lightgbm", "ensemble", champion) if m in lvl.index]
    vals_ = [lvl.at[m, "mean"] for m in chain]
    fig = go.Figure(
        go.Waterfall(
            x=chain,
            measure=["absolute"] + ["relative"] * (len(chain) - 1),
            y=[vals_[0]] + list(np.diff(vals_)),
            decreasing=dict(marker_color=GREEN),
            increasing=dict(marker_color=RED),
            totals=dict(marker_color=BLUE),
        )
    )
    fig.update_layout(yaxis_tickformat=".1%", margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)
    caption(
        "Forecast value added: how each step changes mean WAPE across levels "
        "(green = error reduced)."
    )
    hz = (
        r[r["model"].isin(chain)]
        .groupby(["model", "h"])
        .apply(lambda g: wape(g["y_true"], g["y_pred"]))
        .reset_index(name="wape")
    )
    fig = go.Figure()
    for m, g in hz.groupby("model"):
        fig.add_scatter(x=g["h"], y=g["wape"], name=m, mode="lines+markers")
    fig.update_layout(xaxis_title="Horizon (weeks)", yaxis_tickformat=".1%", margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)
    caption("WAPE by forecast horizon, pooled over all hierarchy levels.")
    if met is not None:
        with st.expander("All metrics"):
            st.dataframe(met[met["role"] == role], use_container_width=True)

elif page == "Budget vs actual":
    st.title("Budget vs actual vs latest forecast")
    m = table("planning_variance_monthly")
    if m is None:
        st.warning("Planning outputs not found.")
        st.stop()
    nid = pick_node("bva")
    mm = m[m["node_id"] == nid].sort_values("fiscal_month")
    ytd = table("planning_variance_ytd")
    yy = ytd[ytd["node_id"] == nid].sort_values("fiscal_month").iloc[-1]
    c = st.columns(4)
    c[0].metric("YTD actual", fmt_usd(yy["actual"]))
    c[1].metric("YTD budget", fmt_usd(yy["budget"]), fmt_pct(yy["var_vs_budget_pct"], True))
    c[2].metric(
        "YTD vs latest forecast",
        fmt_usd(yy["var_vs_latest"], True),
        fmt_pct(yy["var_vs_latest_pct"], True),
    )
    show = mm[
        [
            "fiscal_month",
            "actual",
            "budget",
            "latest_forecast",
            "var_vs_budget",
            "var_vs_budget_pct",
            "var_vs_latest",
            "var_vs_latest_pct",
        ]
    ].set_index("fiscal_month")
    st.dataframe(
        show.style.format(
            {
                "actual": "{:,.0f}",
                "budget": "{:,.0f}",
                "latest_forecast": "{:,.0f}",
                "var_vs_budget": "{:+,.0f}",
                "var_vs_budget_pct": "{:+.1%}",
                "var_vs_latest": "{:+,.0f}",
                "var_vs_latest_pct": "{:+.1%}",
            }
        ).map(style_var, subset=["var_vs_budget", "var_vs_latest"]),
        use_container_width=True,
    )
    caption(
        "Monthly actuals vs the frozen budget and vs the forecast made at the start of each month."
    )
    gap = table("planning_variance_gap")
    gg = gap[gap["node_id"] == nid].sort_values("fiscal_month")
    c[3].metric("Full-year outlook vs budget", fmt_pct(gg["gap_pct"].iloc[-1], True))
    fig = go.Figure(
        go.Scatter(x=gg["version_id"], y=gg["gap_pct"], mode="lines+markers", line=dict(color=BLUE))
    )
    fig.add_hline(y=0, line_color=GREY)
    fig.update_layout(yaxis_tickformat="+.1%", margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)
    caption("Budget-gap tracking: expected full-year revenue vs budget at each re-forecast.")
    pvm = table("planning_pvm")
    comp = st.radio("Bridge", ["vs_prior_year", "vs_budget"], horizontal=True)
    months = sorted(pvm["fiscal_month"].unique())
    fm = st.select_slider("Fiscal month (0 = full year)", months, value=0)
    node = nodes.set_index("node_id").loc[nid]
    p = pvm[(pvm["comparison"] == comp) & (pvm["fiscal_month"] == fm)]
    for col in ("state_id", "store_id", "cat_id", "dept_id"):
        if isinstance(node[col], str):
            p = p[p[col] == node[col]]
    t = p[["base", "volume", "mix", "price", "actual"]].sum()
    fig = go.Figure(
        go.Waterfall(
            x=["Base", "Volume", "Mix", "Price", "Actual"],
            measure=["absolute", "relative", "relative", "relative", "total"],
            y=[t["base"], t["volume"], t["mix"], t["price"], t["actual"]],
            increasing=dict(marker_color=GREEN),
            decreasing=dict(marker_color=RED),
            totals=dict(marker_color=BLUE),
            text=[
                fmt_usd(v, True)
                for v in (t["base"], t["volume"], t["mix"], t["price"], t["actual"])
            ],
        )
    )
    st.plotly_chart(usd_axis(fig), use_container_width=True)
    caption("Price-volume-mix bridge: the three effects add up exactly to the revenue variance.")
    comm = table("planning_commentary")
    if comm is not None:
        st.subheader("Commentary (company level, generated from the tables)")
        for txt in comm.sort_values("fiscal_month")["text"]:
            st.markdown(f"> {txt}")

elif page == "Working capital":
    st.title("What is forecast accuracy worth?")
    bt = table("backtest")
    cats = sorted(nodes["cat_id"].dropna().unique())
    c = st.columns(3)
    sl = c[0].selectbox(
        "Service level", [0.90, 0.95, 0.98], index=1, format_func=lambda x: f"{x:.0%}"
    )
    lt = c[1].slider("Lead time (weeks)", 1, 4, 2)
    rate = c[2].slider("Annual carrying cost rate", 0.05, 0.40, 0.20, 0.01)
    defaults = {"FOODS": 0.75, "HOBBIES": 0.60, "HOUSEHOLD": 0.65}
    cc = st.columns(len(cats))
    cogs = {
        k: cc[i].slider(f"COGS ratio {k}", 0.3, 0.95, defaults.get(k, 0.7), 0.01)
        for i, k in enumerate(cats)
    }
    methods = [m for m in ("seasonal_naive", "run_rate", champion) if m in set(bt["model"])]
    b = bt[(bt["level"] == 8) & bt["model"].isin(methods)]
    errs = wc.lead_time_errors(b, lt)
    sig = wc.sigma_by_node(errs)
    cat_of = dict(zip(nodes["node_id"], nodes["cat_id"]))
    cap = wc.capital_table(
        sig, cat_of, cogs, [sl], rate, [m for m in methods if m != champion], champion
    )
    base = cap[cap["model"].isin(methods)].set_index("model")["safety_stock_usd"]
    k = st.columns(len(base))
    for i, (m, v) in enumerate(base.items()):
        k[i].metric(f"Safety stock: {m}", fmt_usd(v))
    rel = cap.dropna(subset=["working_capital_released_usd"])
    st.dataframe(
        rel[["model", "working_capital_released_usd", "carrying_cost_saved_usd"]].style.format(
            {"working_capital_released_usd": "${:,.0f}", "carrying_cost_saved_usd": "${:,.0f}"}
        ),
        use_container_width=True,
    )
    caption(
        "Safety stock at cost = z(service level) x std of lead-time forecast error x COGS "
        "ratio, per store x department. Stylised: no real costs, revenue-level demand."
    )
    chk = wc.service_level_check(errs, sig, [sl], role="holdout")
    st.dataframe(chk.style.format({"achieved": "{:.1%}"}), use_container_width=True)
    caption(
        "Hold-out check: how often actual revenue over the lead time stayed within "
        "forecast + safety stock (target = the service level)."
    )

elif page == "Scenarios":
    st.title("Price and event scenarios")
    el = table("capital_elasticities")
    vals = table("planning_forecast_values")
    if el is None or vals is None:
        st.warning("Scenario inputs not found.")
        st.stop()
    dp = st.slider("Price change (%)", -10, 10, 5, 1) / 100
    roll = vals[vals["version_id"].str.startswith("ROLL") & (vals["level"] == 4)]
    base = roll.groupby("node_id")["revenue_point"].sum()
    rows = []
    for _, e in el.iterrows():
        f = revenue_factor(dp, e["elasticity"])
        b0 = float(base.get(e["cat_id"], np.nan))
        rows.append(
            {
                "category": e["cat_id"],
                "elasticity": e["elasticity"],
                "base": b0,
                "scenario": b0 * f,
                "impact": b0 * (f - 1),
            }
        )
    df = pd.DataFrame(rows)
    fig = go.Figure(
        go.Bar(
            x=df["category"],
            y=df["impact"],
            marker_color=[CAT_COLORS.get(c, BLUE) for c in df["category"]],
        )
    )
    st.plotly_chart(usd_axis(fig), use_container_width=True)
    caption(
        f"Revenue impact over the next 13 weeks of a {dp:+.0%} price change, using "
        "log-log elasticities estimated from history (observational, not causal)."
    )
    st.dataframe(
        df.style.format(
            {
                "elasticity": "{:.2f}",
                "base": "${:,.0f}",
                "scenario": "${:,.0f}",
                "impact": "${:+,.0f}",
            }
        ),
        use_container_width=True,
    )
    ev = table("capital_event_uplifts")
    if ev is not None:
        st.subheader("Historical holiday uplift (company total)")
        st.dataframe(
            ev.style.format(
                {"mean_uplift": "{:+.1%}", "min_uplift": "{:+.1%}", "max_uplift": "{:+.1%}"}
            ),
            use_container_width=True,
        )
        caption("Event week revenue vs the average of the two weeks either side (non-event weeks).")

else:
    st.title("About this project")
    st.markdown(f"""
**Business question.** How much revenue will each store and department make next quarter,
how sure are we, why are we off budget, and how much inventory cash can a better forecast
free up?

**Data.** Walmart M5 (Kaggle): daily unit sales and weekly prices for 10 stores in CA, TX and
WI, 2011-2016. Revenue = units x sell price, aggregated to Walmart weeks and to a
store x department hierarchy (8 levels). This app shows aggregated outputs only.

**Method.** Rolling-origin backtest (expanding window) with a final 13-week hold-out;
baselines (naive, seasonal naive, moving averages, run-rate), statistical models (ETS,
ARIMA, Theta), a global LightGBM, an ensemble, and hierarchical reconciliation
(bottom-up, top-down, MinT). Intervals: native for statistical models, split-conformal
otherwise. Champion: **{champion}**.

**FP&A layer.** Frozen budget, monthly re-forecasts, variance tables, price-volume-mix
bridge and template commentary. Working capital is a stylised safety-stock simulation
(assumed COGS ratios, no real inventory data). Elasticities are observational.

Code and full methodology: [{REPO_URL}]({REPO_URL})
""")
