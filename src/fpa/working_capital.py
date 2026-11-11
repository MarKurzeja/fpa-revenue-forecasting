"""What is forecast accuracy worth? Safety-stock working capital (SPEC §9).

Stylised simulation (no real inventory or cost data):
    safety stock (cost USD) = z(service level) x sigma_LT x COGS ratio
where sigma_LT is the standard deviation of the forecast error summed over the lead
time, estimated per store x department from backtest origins. Better forecasts ->
smaller sigma_LT -> less safety stock -> working capital released.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd


def z_value(service_level: float) -> float:
    """One-sided normal quantile for a cycle service level."""
    if not 0 < service_level < 1:
        raise ValueError("service level must be in (0, 1)")
    return NormalDist().inv_cdf(service_level)


def safety_stock(
    sigma_lt: np.ndarray | float, service_level: float, cogs_ratio: np.ndarray | float
) -> np.ndarray:
    """Safety stock at cost: z * sigma_LT * COGS ratio."""
    return z_value(service_level) * np.asarray(sigma_lt) * np.asarray(cogs_ratio)


def lead_time_errors(records: pd.DataFrame, lead_time: int) -> pd.DataFrame:
    """Sum of (actual - forecast) over horizons 1..L per model, node and origin."""
    r = records[(records["h"] <= lead_time)].dropna(subset=["y_true"])
    g = (
        r.assign(err=r["y_true"] - r["y_pred"])
        .groupby(["model", "node_id", "origin", "role"], as_index=False)
        .agg(err=("err", "sum"), actual=("y_true", "sum"), forecast=("y_pred", "sum"))
    )
    return g


def sigma_by_node(
    lt_err: pd.DataFrame, roles: tuple[str, ...] = ("validation", "test")
) -> pd.DataFrame:
    """Std of lead-time errors per model and node over the given origin roles."""
    e = lt_err[lt_err["role"].isin(roles)]
    return e.groupby(["model", "node_id"], as_index=False).agg(
        sigma_lt=("err", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else np.nan),
        n_origins=("err", "size"),
    )


def capital_table(
    sigma: pd.DataFrame,
    cat_of: dict[str, str],
    cogs_ratio: dict[str, float],
    service_levels: list[float],
    carrying_rate: float,
    benchmark_methods: list[str],
    best: str,
) -> pd.DataFrame:
    """Safety stock per method and service level, and what `best` releases vs benchmarks."""
    s = sigma.copy()
    s["cat_id"] = s["node_id"].map(cat_of)
    s["cogs_ratio"] = s["cat_id"].map(cogs_ratio)
    rows = []
    for sl in service_levels:
        s["ss_cost"] = safety_stock(s["sigma_lt"], sl, s["cogs_ratio"])
        tot = s.groupby("model")["ss_cost"].sum()
        for m in tot.index:
            row = {"service_level": sl, "model": m, "safety_stock_usd": float(tot[m])}
            rows.append(row)
        for bm in benchmark_methods:
            if bm in tot.index and best in tot.index and bm != best:
                released = float(tot[bm] - tot[best])
                rows.append(
                    {
                        "service_level": sl,
                        "model": f"{best} vs {bm}",
                        "safety_stock_usd": float(tot[best]),
                        "working_capital_released_usd": released,
                        "carrying_cost_saved_usd": released * carrying_rate,
                    }
                )
    return pd.DataFrame(rows)


def service_level_check(
    lt_err: pd.DataFrame, sigma: pd.DataFrame, service_levels: list[float], role: str = "holdout"
) -> pd.DataFrame:
    """Share of (node, origin) where actual <= forecast + z * sigma_LT (revenue terms)."""
    e = lt_err[lt_err["role"] == role].merge(sigma, on=["model", "node_id"], how="inner")
    rows = []
    for sl in service_levels:
        buffer = z_value(sl) * e["sigma_lt"]
        ok = e["actual"] <= e["forecast"] + buffer
        for m, g in e.assign(ok=ok).groupby("model"):
            rows.append(
                {"service_level": sl, "model": m, "achieved": float(g["ok"].mean()), "n": len(g)}
            )
    return pd.DataFrame(rows)
