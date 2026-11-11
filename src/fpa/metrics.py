"""Forecast accuracy metrics (SPEC §7). Small pure functions on numpy arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

ArrayLike = np.ndarray | list[float]


def _arr(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=float)


def wape(actual: ArrayLike, forecast: ArrayLike) -> float:
    """Weighted absolute percentage error: sum|a - f| / sum|a|."""
    a, f = _arr(actual), _arr(forecast)
    denom = np.abs(a).sum()
    return float(np.abs(a - f).sum() / denom) if denom > 0 else float("nan")


def bias(actual: ArrayLike, forecast: ArrayLike) -> float:
    """Bias %: sum(f - a) / sum(a). Positive = over-forecasting."""
    a, f = _arr(actual), _arr(forecast)
    denom = a.sum()
    return float((f - a).sum() / denom) if denom != 0 else float("nan")


def mae(actual: ArrayLike, forecast: ArrayLike) -> float:
    return float(np.mean(np.abs(_arr(actual) - _arr(forecast))))


def rmse(actual: ArrayLike, forecast: ArrayLike) -> float:
    return float(np.sqrt(np.mean((_arr(actual) - _arr(forecast)) ** 2)))


def mase(actual: ArrayLike, forecast: ArrayLike, insample: ArrayLike, m: int = 52) -> float:
    """MAE scaled by the in-sample MAE of the seasonal naive (lag m)."""
    y = _arr(insample)
    if len(y) <= m:
        return float("nan")
    scale = np.mean(np.abs(y[m:] - y[:-m]))
    return mae(actual, forecast) / scale if scale > 0 else float("nan")


def rmsse(actual: ArrayLike, forecast: ArrayLike, insample: ArrayLike, m: int = 1) -> float:
    """Root mean squared scaled error; scale = in-sample MSE of the lag-m naive (M5: m=1)."""
    y = _arr(insample)
    if len(y) <= m:
        return float("nan")
    scale = np.mean((y[m:] - y[:-m]) ** 2)
    err = np.mean((_arr(actual) - _arr(forecast)) ** 2)
    return float(np.sqrt(err / scale)) if scale > 0 else float("nan")


def weighted_mean(values: ArrayLike, weights: ArrayLike) -> float:
    """Weighted mean ignoring NaN values (used for revenue-weighted RMSSE)."""
    v, w = _arr(values), _arr(weights)
    ok = np.isfinite(v) & np.isfinite(w)
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok])) if np.sum(w[ok]) > 0 else float("nan")


def coverage(actual: ArrayLike, lo: ArrayLike, hi: ArrayLike) -> float:
    """Share of actuals inside [lo, hi]."""
    a, lo_, hi_ = _arr(actual), _arr(lo), _arr(hi)
    return float(np.mean((a >= lo_) & (a <= hi_)))


def interval_width(lo: ArrayLike, hi: ArrayLike) -> float:
    return float(np.mean(_arr(hi) - _arr(lo)))


def pinball(actual: ArrayLike, quantile_forecast: ArrayLike, tau: float) -> float:
    """Mean pinball (quantile) loss at level tau."""
    diff = _arr(actual) - _arr(quantile_forecast)
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def winkler(actual: ArrayLike, lo: ArrayLike, hi: ArrayLike, alpha: float) -> float:
    """Mean Winkler (interval) score for a (1 - alpha) interval. Lower is better."""
    a, lo_, hi_ = _arr(actual), _arr(lo), _arr(hi)
    width = hi_ - lo_
    below = (2 / alpha) * np.maximum(lo_ - a, 0)
    above = (2 / alpha) * np.maximum(a - hi_, 0)
    return float(np.mean(width + below + above))


@dataclass(frozen=True)
class DMResult:
    statistic: float
    p_value: float
    n: int
    mean_loss_diff: float
    h: int
    loss: str


def diebold_mariano(
    errors_a: ArrayLike, errors_b: ArrayLike, h: int = 1, loss: str = "abs"
) -> DMResult:
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction.

    d_t = L(e_a,t) - L(e_b,t). H0: equal accuracy. A negative statistic means model A
    has lower loss. The long-run variance uses autocovariances up to lag h-1 (rectangular
    kernel, as in DM 1995); the p-value is two-sided from Student t with n-1 df.
    Assumes d_t is covariance-stationary.
    """
    ea, eb = _arr(errors_a), _arr(errors_b)
    if loss == "abs":
        d = np.abs(ea) - np.abs(eb)
    elif loss == "sq":
        d = ea**2 - eb**2
    else:
        raise ValueError("loss must be 'abs' or 'sq'")
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    gamma = [np.sum(dc[k:] * dc[: n - k]) / n for k in range(h)]
    var = gamma[0] + 2 * sum(gamma[1:])
    if var <= 0:
        var = gamma[0]
    dm = dbar / np.sqrt(var / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = float(dm * hln)
    p = float(2 * stats.t.sf(abs(stat), df=n - 1))
    return DMResult(stat, p, n, float(dbar), h, loss)


def horizon_bucket(h: np.ndarray | int, buckets: list[tuple[int, int]]) -> np.ndarray:
    """Label each horizon (1-based) with its bucket, e.g. '1-4'."""
    harr = np.atleast_1d(np.asarray(h))
    out = np.full(harr.shape, "", dtype=object)
    for a, b in buckets:
        out[(harr >= a) & (harr <= b)] = f"{a}-{b}"
    out[out == ""] = f">{buckets[-1][1]}"
    return out
