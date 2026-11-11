"""Generate reports/latest/report.md and figures (SPEC §7.6)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fpa import io  # noqa: E402
from fpa.commentary import fmt_pct, fmt_usd  # noqa: E402
from fpa.config import Config  # noqa: E402
from fpa.evaluate import horizon_buckets, level_wape, summarize  # noqa: E402
from fpa.hierarchy import LEVELS  # noqa: E402
from fpa.metrics import wape  # noqa: E402

log = logging.getLogger(__name__)
CAT_COLORS = {"FOODS": "#2a7ab9", "HOBBIES": "#e08a1e", "HOUSEHOLD": "#3d9a5b"}


def md_table(df: pd.DataFrame, floatfmt: dict[str, str] | None = None) -> str:
    """Render a DataFrame as a GitHub markdown table without extra dependencies."""
    floatfmt = floatfmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                fmt = floatfmt.get(c, "{:.3f}")
                cells.append("" if pd.isna(v) else fmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pct(df: pd.DataFrame, cols: list[str]) -> dict[str, str]:
    return {c: "{:.1%}" for c in cols}


def _level_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [f"L{c}" if isinstance(c, (int, np.integer)) else c for c in df.columns]
    return df


def fva_steps(scores: pd.Series, sel: dict) -> list[tuple[str, float]]:
    """FVA chain: seasonal naive -> run-rate -> best stats -> LightGBM -> ensemble -> champion."""
    fam = sel["families"]
    stats = [m for m in scores.index if fam.get(m) == "stats"]
    chain = [("Seasonal naive", "seasonal_naive"), ("Run-rate (current process)", "run_rate")]
    if stats:
        best = min(stats, key=lambda m: scores[m])
        chain.append((f"Best statistical ({best})", best))
    if "lightgbm" in scores:
        chain.append(("LightGBM", "lightgbm"))
    if "ensemble" in scores:
        chain.append(("Ensemble", "ensemble"))
    chain.append((f"Reconciled ({sel['champion']})", sel["champion"]))
    return [(label, float(scores[m])) for label, m in chain if m in scores]


def fig_fva(steps: list[tuple[str, float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = [s[0] for s in steps]
    vals = [s[1] for s in steps]
    prev = vals[0]
    ax.bar(0, vals[0], color="#777")
    for i in range(1, len(vals)):
        d = vals[i] - prev
        ax.bar(i, d, bottom=prev, color="#3d9a5b" if d < 0 else "#c0392b")
        ax.text(i, max(prev, vals[i]) + 0.002, f"{-d * 100:+.1f}pp", ha="center", fontsize=8)
        prev = vals[i]
    ax.bar(len(vals), vals[-1], color="#2a7ab9")
    ax.set_xticks(range(len(vals) + 1))
    ax.set_xticklabels([*labels, "Final"], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Mean WAPE across levels")
    ax.set_title("Forecast value added (test origins); green = error reduced")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_horizon(records: pd.DataFrame, models: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    for ax, lev in zip(axes, (1, 8)):
        r = records[(records["level"] == lev) & records["model"].isin(models)]
        for m in models:
            g = r[r["model"] == m]
            w = g.groupby("h").apply(lambda x: wape(x["y_true"], x["y_pred"]))
            ax.plot(w.index, w.values, marker="o", ms=3, label=m)
        ax.set_title(f"L{lev} {LEVELS[lev][0]}")
        ax.set_xlabel("Horizon (weeks)")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_ylabel("WAPE")
    axes[1].legend(fontsize=7)
    fig.suptitle("Accuracy by horizon (test + hold-out origins)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_examples(
    panel, records: pd.DataFrame, champion: str, node_ids: list[str], path: Path
) -> None:
    fig, axes = plt.subplots(len(node_ids), 1, figsize=(10, 3 * len(node_ids)))
    axes = np.atleast_1d(axes)
    hold = records[(records["model"] == champion) & (records["role"] == "holdout")]
    for ax, nid in zip(axes, node_ids):
        i = panel.h.node_index(nid)
        o = int(hold["origin"].iloc[0])
        t0 = max(0, o - 104)
        dates = pd.to_datetime(panel.weeks["week_start"]).to_numpy()
        ax.plot(dates[t0:], panel.y[i, t0:], color="#333", lw=1, label="actual")
        g = hold[hold["node_id"] == nid].sort_values("t")
        gd = dates[g["t"].to_numpy()]
        ax.fill_between(gd, g["lo95"], g["hi95"], color="#2a7ab9", alpha=0.15, label="95%")
        ax.fill_between(gd, g["lo80"], g["hi80"], color="#2a7ab9", alpha=0.3, label="80%")
        ax.plot(gd, g["y_pred"], color="#2a7ab9", lw=1.5, label="forecast")
        ax.axvline(dates[o], color="#999", ls="--", lw=0.8)
        ax.set_title(f"{nid}: hold-out forecast ({champion})", fontsize=9)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: fmt_usd(v).replace("USD ", "$"))
        )
    axes[0].legend(fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_pvm(pvm_row: pd.Series, title: str, path: Path) -> None:
    steps = [
        ("Base", pvm_row["base"]),
        ("Volume", pvm_row["volume"]),
        ("Mix", pvm_row["mix"]),
        ("Price", pvm_row["price"]),
    ]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    run = steps[0][1]
    ax.bar(0, run, color="#777")
    for i, (_, v) in enumerate(steps[1:], start=1):
        ax.bar(i, v, bottom=run, color="#3d9a5b" if v >= 0 else "#c0392b")
        ax.text(i, run + max(v, 0), fmt_usd(v, signed=True), ha="center", va="bottom", fontsize=8)
        run += v
    ax.bar(4, pvm_row["actual"], color="#2a7ab9")
    ax.set_xticks(range(5))
    ax.set_xticklabels(["Base", "Volume", "Mix", "Price", "Actual"])
    lo = min(pvm_row["base"], pvm_row["actual"]) * 0.97
    ax.set_ylim(bottom=lo)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: fmt_usd(v).replace("USD ", "$"))
    )
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def build_report(cfg: Config) -> Path:
    from fpa.panel import load_panel
    from fpa.pipeline import dm_test, load_backtest

    records, sel, _ = load_backtest(cfg)
    panel = load_panel(cfg.paths.duckdb_file)
    fig_dir = cfg.paths.figures_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = cfg.paths.reports_dir
    rep_dir.mkdir(parents=True, exist_ok=True)

    def up(p: Path) -> str:
        return Path(os.path.relpath(p, rep_dir)).as_posix()

    champion = sel["champion"]
    fam = sel["families"]
    main = [m for m in records["model"].unique() if fam.get(m) != "reconciled" or m == champion]

    test = records[records["role"] == "test"]
    hold = records[records["role"] == "holdout"]
    evalset = records[records["role"].isin(["test", "holdout"])]
    md: list[str] = ["# Revenue forecasting report", ""]
    o = sel["origins"]
    md += [
        f"Data: {panel.h.n_bottom} store x department series, {panel.T} full Walmart weeks "
        f"({panel.weeks['week_start'].min():%Y-%m-%d} to {panel.weeks['week_end'].max():%Y-%m-%d}),"
        f" total revenue {fmt_usd(panel.y[panel.h.total_idx].sum())}.",
        f"Backtest origins (week index): validation {o['validation']}, test "
        f"{[x for x in o['backtest'] if x not in o['validation']]}, hold-out {o['holdout']}. "
        f"Horizon {cfg.horizon} weeks.",
        f"Champion (chosen on validation origins only): **{champion}**.",
        "",
    ]

    # 1. Headline tables
    lw = level_wape(test[test["model"].isin(main)]).pivot(
        index="model", columns="level", values="wape"
    )
    lw["mean"] = lw.mean(axis=1)
    lw = _level_cols(lw.sort_values("mean").reset_index())
    md += [
        "## 1. Headline: WAPE by model and hierarchy level (test origins)",
        "",
        md_table(lw, pct(lw, [c for c in lw.columns if c != "model"])),
        "",
    ]
    bot = summarize(evalset[evalset["model"].isin(main) & (evalset["level"] == 8)], ["model"])
    tot = summarize(
        evalset[evalset["model"].isin(main) & (evalset["level"] == 1)], ["model"], scaled=False
    )
    head = (
        bot[["model", "wape", "bias", "mase", "wrmsse", "coverage80", "coverage95"]]
        .merge(
            tot[["model", "wape", "bias"]].rename(
                columns={"wape": "wape_total", "bias": "bias_total"}
            ),
            on="model",
        )
        .sort_values("wape")
    )
    md += [
        "Bottom level (store x department) and company total, test + hold-out origins:",
        "",
        md_table(
            head,
            {
                **pct(
                    head, ["wape", "bias", "coverage80", "coverage95", "wape_total", "bias_total"]
                ),
                "mase": "{:.3f}",
                "wrmsse": "{:.3f}",
            },
        ),
        "",
    ]
    hsum = summarize(hold[hold["model"].isin(main)], ["model", "level"], scaled=False)
    hp = hsum.pivot(index="model", columns="level", values="wape")
    hp = _level_cols(hp.assign(mean=hp.mean(axis=1)).sort_values("mean").reset_index())
    md += [
        "### Final hold-out (last 13 weeks, never used for selection)",
        "",
        md_table(hp, pct(hp, [c for c in hp.columns if c != "model"])),
        "",
    ]

    # 2. FVA
    scores_test = level_wape(test).groupby("model")["wape"].mean()
    steps = fva_steps(scores_test, sel)
    fig_fva(steps, fig_dir / "fva_waterfall.png")
    fva = pd.DataFrame({"step": [s[0] for s in steps], "mean_wape": [s[1] for s in steps]})
    fva["fva_vs_seasonal_naive_pp"] = (fva["mean_wape"].iloc[0] - fva["mean_wape"]) * 100
    fva["fva_vs_run_rate_pp"] = (scores_test.get("run_rate", np.nan) - fva["mean_wape"]) * 100
    md += [
        "## 2. Forecast value added",
        "",
        f"![FVA]({up(fig_dir / 'fva_waterfall.png')})",
        "",
        md_table(
            fva,
            {
                "mean_wape": "{:.2%}",
                "fva_vs_seasonal_naive_pp": "{:+.2f}",
                "fva_vs_run_rate_pp": "{:+.2f}",
            },
        ),
        "",
    ]

    # 3. Horizon
    key = [
        m
        for m in ["seasonal_naive", "run_rate", "lightgbm", "ensemble", champion]
        if m in records["model"].unique()
    ]
    fig_horizon(evalset, key, fig_dir / "accuracy_by_horizon.png")
    hb = horizon_buckets(evalset[evalset["model"].isin(key)], cfg.horizon_buckets)
    hbt = (
        summarize(hb, ["model", "horizon_bucket"], scaled=False)
        .pivot(index="model", columns="horizon_bucket", values="wape")
        .reset_index()
    )
    md += [
        "## 3. Accuracy by horizon",
        "",
        f"![Horizon]({up(fig_dir / 'accuracy_by_horizon.png')})",
        "",
        "WAPE pooled over all levels:",
        "",
        md_table(hbt, pct(hbt, [c for c in hbt.columns if c != "model"])),
        "",
    ]

    # 4. Example plots
    ex = ["Total"]
    for cand in ("CA", panel.nodes.loc[panel.nodes["level"] == 2, "node_id"].iloc[0]):
        if cand in set(panel.nodes["node_id"]):
            ex.append(cand)
            break
    bottoms = panel.nodes.loc[panel.nodes["level"] == 8, "node_id"].tolist()
    ex.append("CA_1/FOODS_3" if "CA_1/FOODS_3" in bottoms else bottoms[0])
    fig_examples(panel, records, champion, ex, fig_dir / "example_forecasts.png")
    md += [
        "## 4. Example forecasts with 80% / 95% intervals (hold-out)",
        "",
        f"![Examples]({up(fig_dir / 'example_forecasts.png')})",
        "",
    ]

    # 5. Reconciliation
    base = sel["champion_base"]
    rnames = [base, *[f"{base}+{m}" for m in cfg.reconciliation["methods"]]]
    rc = (
        level_wape(evalset[evalset["model"].isin(rnames)])
        .pivot(index="level", columns="model", values="wape")
        .reset_index()
    )
    rc["level"] = rc["level"].map(lambda x: f"L{x} {LEVELS[int(x)][0]}")
    md += [
        "## 5. Reconciliation: WAPE per level before and after (test + hold-out)",
        "",
        md_table(rc, pct(rc, [c for c in rc.columns if c != "level"])),
        "",
    ]
    coh = cfg.paths.artifacts_dir / "coherence.json"
    if coh.exists():
        md += [
            "Coherence check (every parent = sum of children): "
            + ", ".join(f"{k}: {'pass' if v else 'FAIL'}" for k, v in io.load_json(coh).items()),
            "",
        ]

    # Segmentation
    seg = evalset[(evalset["model"] == champion) & (evalset["level"] == 8)].merge(
        panel.nodes[["node_id", "state_id", "cat_id"]], on="node_id"
    )
    wk = panel.weeks.set_index("t")["event_names"].fillna("")
    seg["holiday_week"] = (
        seg["t"]
        .map(wk)
        .str.contains("Christmas|Thanksgiving|Easter|SuperBowl|NewYear")
        .fillna(False)
    )
    parts = []
    for col in ("state_id", "cat_id", "holiday_week"):
        s = summarize(seg, [col], scaled=False)[[col, "wape", "bias", "coverage80", "coverage95"]]
        s.insert(0, "segment", col)
        parts.append(s.rename(columns={col: "value"}).astype({"value": str}))
    segt = pd.concat(parts)
    md += [
        f"### Segmentation of {champion} (bottom level)",
        "",
        md_table(segt, pct(segt, ["wape", "bias", "coverage80", "coverage95"])),
        "",
    ]

    # 6. Worst series / weeks
    ch = evalset[(evalset["model"] == champion) & (evalset["level"] == 8)].copy()
    ch["abs_err"] = (ch["y_true"] - ch["y_pred"]).abs()
    worst_s = (
        ch.groupby("node_id")
        .apply(
            lambda g: pd.Series(
                {"wape": wape(g["y_true"], g["y_pred"]), "revenue": g["y_true"].sum()}
            )
        )
        .sort_values("wape", ascending=False)
        .head(5)
        .reset_index()
    )
    pi = panel.price_index
    ch["event_names"] = ch["t"].map(wk)
    ch["price_chg"] = [
        pi[n, t] / pi[n, t - 4] - 1 if t >= 4 else np.nan for n, t in zip(ch["node"], ch["t"])
    ]
    worst_w = ch.sort_values("abs_err", ascending=False).head(8)[
        ["node_id", "wm_yr_wk", "h", "y_true", "y_pred", "abs_err", "event_names", "price_chg"]
    ]
    md += [
        "## 6. Worst-forecasted series and weeks",
        "",
        md_table(worst_s, {"wape": "{:.1%}", "revenue": "{:,.0f}"}),
        "",
        md_table(
            worst_w,
            {
                "y_true": "{:,.0f}",
                "y_pred": "{:,.0f}",
                "abs_err": "{:,.0f}",
                "price_chg": "{:+.1%}",
            },
        ),
        "",
        "_Event names and the 4-week change in the department price index are shown to help "
        "explain the largest misses (holiday weeks and price moves are the usual causes)._",
        "",
    ]

    # 7. DM
    dm = dm_test(records, champion)
    verdict = (
        "significantly more accurate"
        if dm["p_value"] < 0.05 and dm["statistic"] < 0
        else "not significantly different from"
    )
    md += [
        "## 7. Diebold-Mariano test (champion vs seasonal naive)",
        "",
        f"Loss: absolute error summed over bottom-level series per (origin, horizon), test + "
        f"hold-out origins, n = {dm['n']}, HLN-corrected, long-run variance with lag "
        f"{dm['h'] - 1}. Statistic = {dm['statistic']:.3f}, p = {dm['p_value']:.4f}: the "
        f"champion is {verdict} seasonal naive.",
        "_Assumes the loss differential is covariance-stationary; overlapping origins make "
        "observations dependent, which the lag window only partly absorbs._",
        "",
    ]
    io.save_json(dm, cfg.paths.artifacts_dir / "diebold_mariano.json")

    # FP&A and capital (if run)
    md += _fpa_section(cfg, fig_dir, up)
    md += _capital_section(cfg)

    # 8. Key findings
    sn = scores_test.get("seasonal_naive", np.nan)
    rr = scores_test.get("run_rate", np.nan)
    cw = scores_test.get(champion, np.nan)
    bot_c = head.set_index("model").loc[champion] if champion in set(head["model"]) else None
    find = [
        f"- The champion ({champion}) cuts mean WAPE across levels from {fmt_pct(sn)} "
        f"(seasonal naive) and {fmt_pct(rr)} (run-rate) to {fmt_pct(cw)} on test origins."
    ]
    if bot_c is not None:
        find.append(
            f"- Bottom level: WAPE {fmt_pct(bot_c['wape'])}, bias "
            f"{fmt_pct(bot_c['bias'], signed=True)}, 80% coverage "
            f"{fmt_pct(bot_c['coverage80'])}, 95% coverage {fmt_pct(bot_c['coverage95'])}."
        )
    if "ensemble" in scores_test:
        bs = sel["best_single"]
        beat = scores_test["ensemble"] < scores_test.get(bs, np.inf)
        find.append(
            f"- The ensemble of {', '.join(sel['ensemble_members'])} "
            f"{'beats' if beat else 'does not beat'} the best single model ({bs})."
        )
    find.append(f"- Diebold-Mariano vs seasonal naive: p = {dm['p_value']:.4f}.")
    md += [
        "## 8. Key findings",
        "",
        *find,
        "",
        "### Commentary",
        "",
        "_Placeholder for Maria's commentary._",
        "",
    ]
    out = rep_dir / "report.md"
    out.write_text("\n".join(md))
    return out


def _fpa_section(cfg: Config, fig_dir: Path, up) -> list[str]:
    p = cfg.paths.artifacts_dir / "planning"
    if not io.table_exists(p / "variance_monthly"):
        return []
    m = io.load_table(p / "variance_monthly")
    m = m[m["node_id"] == "Total"].sort_values("fiscal_month")
    gap = io.load_table(p / "variance_gap")
    gap = gap[gap["node_id"] == "Total"].sort_values("fiscal_month")
    pvm_df = io.load_table(p / "pvm")
    comm = io.load_table(p / "commentary")
    fy = int(m["fiscal_year"].iloc[0])
    tm = m[
        [
            "fiscal_month",
            "actual",
            "budget",
            "latest_forecast",
            "var_vs_budget",
            "var_vs_budget_pct",
            "var_vs_latest_pct",
        ]
    ]
    out = [
        f"## FP&A: FY{fy} budget vs actual vs latest forecast (company total)",
        "",
        md_table(
            tm,
            {
                "actual": "{:,.0f}",
                "budget": "{:,.0f}",
                "latest_forecast": "{:,.0f}",
                "var_vs_budget": "{:+,.0f}",
                "var_vs_budget_pct": "{:+.1%}",
                "var_vs_latest_pct": "{:+.1%}",
            },
        ),
        "",
        "Budget-gap tracking (full-year outlook vs budget at each version):",
        "",
        md_table(
            gap[["version_id", "fy_budget", "fy_outlook", "gap_usd", "gap_pct"]],
            {
                "fy_budget": "{:,.0f}",
                "fy_outlook": "{:,.0f}",
                "gap_usd": "{:+,.0f}",
                "gap_pct": "{:+.2%}",
            },
        ),
        "",
    ]
    for comp in ("vs_prior_year", "vs_budget"):
        fyp = pvm_df[(pvm_df["comparison"] == comp) & (pvm_df["fiscal_month"] == 0)]
        row = fyp[["base", "volume", "mix", "price", "actual"]].sum()
        f = fig_dir / f"pvm_{comp}.png"
        fig_pvm(row, f"FY{fy} price-volume-mix bridge, actual {comp.replace('_', ' ')}", f)
        out += [f"![PVM {comp}]({up(f)})", ""]
    out += ["Generated commentary:", ""] + [f"> {t}\n" for t in comm["text"].tolist()[:12]]
    return out


def _capital_section(cfg: Config) -> list[str]:
    p = cfg.paths.artifacts_dir / "capital"
    if not io.table_exists(p / "working_capital"):
        return []
    cap = io.load_table(p / "working_capital")
    svc = io.load_table(p / "service_level_check")
    el = io.load_table(p / "elasticities")
    out = [
        "## Working capital (stylised simulation)",
        "",
        md_table(
            cap.fillna(np.nan),
            {
                "service_level": "{:.0%}",
                "safety_stock_usd": "{:,.0f}",
                "working_capital_released_usd": "{:,.0f}",
                "carrying_cost_saved_usd": "{:,.0f}",
            },
        ),
        "",
        "Service-level check on the hold-out (share of store x department lead-time windows "
        "where actual <= forecast + safety stock):",
        "",
        md_table(svc, {"service_level": "{:.0%}", "achieved": "{:.1%}"}),
        "",
        "## Price elasticities (observational, see limitations)",
        "",
        md_table(el, {"elasticity": "{:.3f}", "se": "{:.3f}"}),
        "",
    ]
    sc = p / "price_scenarios"
    if io.table_exists(sc):
        s = io.load_table(sc)
        if not s.empty:
            out += [
                md_table(
                    s,
                    {
                        "price_change": "{:+.0%}",
                        "elasticity": "{:.3f}",
                        "base_revenue_13w": "{:,.0f}",
                        "scenario_revenue_13w": "{:,.0f}",
                        "impact_usd": "{:+,.0f}",
                    },
                ),
                "",
            ]
    ev = io.load_table(p / "event_uplifts")
    out += [
        "Historical event uplift (company total, event week vs neighbouring weeks):",
        "",
        md_table(ev, {"mean_uplift": "{:+.1%}", "min_uplift": "{:+.1%}", "max_uplift": "{:+.1%}"}),
        "",
    ]
    return out
