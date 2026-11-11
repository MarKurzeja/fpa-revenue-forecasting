"""Baseline forecasts (SPEC §6.1). Vectorised numpy over all nodes."""

from __future__ import annotations

import numpy as np

from fpa.models.base import ForecastContext, ForecastOutput


def naive(y: np.ndarray, h: int) -> np.ndarray:
    """Last observed week repeated."""
    return np.repeat(y[:, -1:], h, axis=1)


def seasonal_naive(y: np.ndarray, h: int, m: int = 52) -> np.ndarray:
    """Same week last year: y[o - m*k + (i-1)] with k = ceil(i/m)."""
    o = y.shape[1]
    steps = np.arange(1, h + 1)
    k = np.ceil(steps / m).astype(int)
    idx = o - m * k + (steps - 1)
    if idx.min() < 0:
        raise ValueError("not enough history for seasonal naive")
    return y[:, idx]


def moving_average(y: np.ndarray, h: int, window: int) -> np.ndarray:
    return np.repeat(y[:, -window:].mean(axis=1, keepdims=True), h, axis=1)


def seasonality_index(
    y: np.ndarray, h: int, m: int = 52, window: int = 13, max_years: int = 3
) -> np.ndarray:
    """Average over past years of (that year's week value / that year's run-rate).

    For year k back, run-rate = mean of the `window` weeks before o - k*m, and the
    index for step i is y[o - k*m + i - 1] / run-rate. Years without enough history
    are skipped; if none is available the index is 1.
    """
    n, o = y.shape
    ratios = []
    for k in range(1, max_years + 1):
        base_end = o - k * m
        if base_end - window < 0:
            break
        rr = y[:, base_end - window : base_end].mean(axis=1, keepdims=True)
        idx = base_end + np.arange(h)
        if idx.max() >= o:  # horizon longer than a year: wrap one more year back
            idx = np.where(idx >= o, idx - m, idx)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(rr > 0, y[:, idx] / rr, np.nan)
        ratios.append(r)
    if not ratios:
        return np.ones((n, h))
    si = np.nanmean(np.stack(ratios), axis=0)
    return np.where(np.isfinite(si), si, 1.0)


def run_rate(
    y: np.ndarray, h: int, m: int = 52, window: int = 13, max_years: int = 3
) -> np.ndarray:
    """Spreadsheet FP&A method: last-13-week average x seasonality index."""
    rr = y[:, -window:].mean(axis=1, keepdims=True)
    return rr * seasonality_index(y, h, m, window, max_years)


def _fitted_one_step(y: np.ndarray, fn, min_t: int) -> np.ndarray:
    """In-sample one-step-ahead fitted values by re-running `fn` at every t >= min_t."""
    n, o = y.shape
    out = np.full((n, o), np.nan)
    for t in range(min_t, o):
        out[:, t] = fn(y[:, :t], 1)[:, 0]
    return out


class Baseline:
    """Wraps one baseline function in the Forecaster interface."""

    family = "baseline"

    def __init__(self, name: str, m: int = 52, window: int = 13, max_years: int = 3):
        self.name = name
        self.m, self.window, self.max_years = m, window, max_years

    def _fn(self, y: np.ndarray, h: int) -> np.ndarray:
        if self.name == "naive":
            return naive(y, h)
        if self.name == "seasonal_naive":
            return seasonal_naive(y, h, self.m)
        if self.name.startswith("ma"):
            return moving_average(y, h, int(self.name[2:]))
        if self.name == "run_rate":
            return run_rate(y, h, self.m, self.window, self.max_years)
        raise ValueError(f"unknown baseline {self.name}")

    def min_history(self) -> int:
        if self.name == "naive":
            return 1
        if self.name.startswith("ma"):
            return int(self.name[2:])
        if self.name == "run_rate":
            return self.m + self.window
        return self.m

    def forecast(self, ctx: ForecastContext, origin: int, h: int) -> ForecastOutput:
        y = ctx.y_hist(origin)
        mean = self._fn(y, h)
        fitted = _fitted_one_step(y, self._fn, self.min_history())
        return ForecastOutput(mean=mean, fitted=fitted)


def make_baselines(names: list[str], m: int, window: int, max_years: int) -> list[Baseline]:
    return [Baseline(n, m=m, window=window, max_years=max_years) for n in names]
