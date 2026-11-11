"""Turn backtest records into metric tables (SPEC §7.1-7.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fpa import metrics as M
from fpa.hierarchy import LEVELS


def _series_scaled(g: pd.DataFrame) -> tuple[float, float]:
    """Mean MASE over series, and revenue-weighted RMSSE (each series scaled by its history)."""
    e = g["y_true"] - g["y_pred"]
    per = (
        g.assign(abs_e=e.abs(), sq_e=e**2)
        .groupby(["node", "origin"])
        .agg(
            mae=("abs_e", "mean"),
            mse=("sq_e", "mean"),
            mase_scale=("mase_scale", "first"),
            rmsse_scale=("rmsse_scale", "first"),
            weight=("weight", "first"),
        )
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        mase = (per["mae"] / per["mase_scale"]).replace([np.inf, -np.inf], np.nan)
        rmsse = np.sqrt(per["mse"] / per["rmsse_scale"]).replace([np.inf, -np.inf], np.nan)
    return float(mase.mean()), M.weighted_mean(rmsse, per["weight"])


def metric_row(g: pd.DataFrame, levels: list[int] = (80, 95), scaled: bool = True) -> dict:
    """All point and interval metrics for one group of records."""
    a, f = g["y_true"].to_numpy(), g["y_pred"].to_numpy()
    row = {
        "n": len(g),
        "actual_usd": float(a.sum()),
        "wape": M.wape(a, f),
        "bias": M.bias(a, f),
        "mae": M.mae(a, f),
        "rmse": M.rmse(a, f),
    }
    if scaled and {"mase_scale", "rmsse_scale", "weight"} <= set(g.columns):
        row["mase"], row["wrmsse"] = _series_scaled(g)
    for L in levels:
        lo, hi = f"lo{L}", f"hi{L}"
        if lo in g and g[lo].notna().any():
            ok = g[lo].notna()
            row[f"coverage{L}"] = M.coverage(a[ok], g.loc[ok, lo], g.loc[ok, hi])
            w = M.interval_width(g.loc[ok, lo], g.loc[ok, hi])
            row[f"width{L}_usd"] = w
            row[f"width{L}_pct"] = w / np.mean(np.abs(a[ok])) if np.mean(np.abs(a[ok])) else np.nan
    if "lo80" in g and g["lo80"].notna().any():
        ok = g["lo80"].notna()
        row["pinball10"] = M.pinball(a[ok], g.loc[ok, "lo80"], 0.10)
        row["pinball50"] = M.pinball(a, f, 0.50)
        row["pinball90"] = M.pinball(a[ok], g.loc[ok, "hi80"], 0.90)
    if "lo95" in g and g["lo95"].notna().any():
        ok = g["lo95"].notna()
        row["winkler95"] = M.winkler(a[ok], g.loc[ok, "lo95"], g.loc[ok, "hi95"], 0.05)
    return row


def summarize(records: pd.DataFrame, by: list[str], scaled: bool = True) -> pd.DataFrame:
    """Metric table grouped by `by` (e.g. ['model', 'level'])."""
    r = records.dropna(subset=["y_true"])
    rows = []
    for key, g in r.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), **metric_row(g, scaled=scaled)})
    out = pd.DataFrame(rows)
    if "level" in out:
        out["level_name"] = out["level"].map(lambda x: LEVELS[int(x)][0])
    return out


def level_wape(records: pd.DataFrame) -> pd.DataFrame:
    """WAPE per model and level, pooled over nodes, origins and horizons."""
    r = records.dropna(subset=["y_true"])
    g = r.assign(abs_err=(r["y_true"] - r["y_pred"]).abs(), abs_y=r["y_true"].abs())
    t = g.groupby(["model", "level"])[["abs_err", "abs_y"]].sum()
    return (t["abs_err"] / t["abs_y"]).rename("wape").reset_index()


def selection_scores(records: pd.DataFrame, role: str = "validation") -> pd.Series:
    """Mean WAPE across hierarchy levels per model (lower is better)."""
    lw = level_wape(records[records["role"] == role])
    return lw.groupby("model")["wape"].mean().sort_values()


def horizon_buckets(records: pd.DataFrame, buckets: list[tuple[int, int]]) -> pd.DataFrame:
    r = records.copy()
    r["horizon_bucket"] = M.horizon_bucket(r["h"].to_numpy(), buckets)
    return r
