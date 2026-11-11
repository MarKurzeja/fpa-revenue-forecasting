"""Split-conformal prediction intervals (SPEC §6.5, D-08).

Scores are absolute errors scaled by the node's trailing 13-week mean revenue at the
origin: s = |y - yhat| / scale. For each (model, hierarchy level, horizon bucket) the
interval half-width multiplier is the finite-sample-corrected (1 - alpha) quantile of
the scores on the *validation* origins. Intervals are then yhat +/- q * scale (lower
bound clipped at 0). Horizons beyond the last bucket (budget forecasts) scale the last
bucket's quantile by sqrt(h / mid), a random-walk style widening.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def conformal_quantile(scores: np.ndarray, level: int) -> float:
    """ceil((n + 1)(1 - alpha)) / n empirical quantile of the scores."""
    s = np.sort(np.asarray(scores, float)[np.isfinite(scores)])
    n = len(s)
    if n == 0:
        return float("nan")
    alpha = 1 - level / 100
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(s[min(k, n) - 1])


def bucket_of(h: np.ndarray, buckets: list[tuple[int, int]]) -> np.ndarray:
    h = np.asarray(h)
    out = np.full(h.shape, len(buckets) - 1)
    for i, (a, b) in enumerate(buckets):
        out[(h >= a) & (h <= b)] = i
    return out


def calibrate(
    records: pd.DataFrame, levels: list[int], buckets: list[tuple[int, int]]
) -> pd.DataFrame:
    """Quantile table from validation records.

    `records` needs columns model, level, h, y_true, y_pred, scale.
    Returns columns model, level, bucket, q{level} for each interval level.
    """
    r = records.dropna(subset=["y_true", "y_pred"]).copy()
    r["score"] = np.abs(r["y_true"] - r["y_pred"]) / r["scale"]
    r["bucket"] = bucket_of(r["h"].to_numpy(), buckets)
    rows = []
    for (model, lev, b), g in r.groupby(["model", "level", "bucket"]):
        row = {"model": model, "level": lev, "bucket": b, "n": len(g)}
        for L in levels:
            row[f"q{L}"] = conformal_quantile(g["score"].to_numpy(), L)
        rows.append(row)
    return pd.DataFrame(rows)


def apply(
    records: pd.DataFrame,
    qtable: pd.DataFrame,
    levels: list[int],
    buckets: list[tuple[int, int]],
) -> pd.DataFrame:
    """Add lo{L}/hi{L} columns to forecast records using the calibrated quantiles."""
    r = records.copy()
    r["bucket"] = bucket_of(r["h"].to_numpy(), buckets)
    r = r.merge(
        qtable.drop(columns=["n"], errors="ignore"), on=["model", "level", "bucket"], how="left"
    )
    last_a, last_b = buckets[-1]
    mid = (last_a + last_b) / 2
    widen = np.where(r["h"] > last_b, np.sqrt(r["h"] / mid), 1.0)
    for L in levels:
        half = r[f"q{L}"] * r["scale"] * widen
        r[f"lo{L}"] = np.maximum(r["y_pred"] - half, 0.0)
        r[f"hi{L}"] = r["y_pred"] + half
        r = r.drop(columns=[f"q{L}"])
    return r.drop(columns=["bucket"])
