"""Statistical models via statsforecast (SPEC §6.2).

- auto_ets: ETS cannot use a 52-week season (statsforecast, like R's forecast, drops
  seasonality when m > 24), so AutoETS models the trend of an STL-deseasonalised
  series (statsforecast MSTL with an AutoETS trend forecaster). See D-05.
- auto_arima: by default ARIMA with yearly Fourier terms as regressors (fast, stable);
  `arima.mode: seasonal` uses season_length=52 instead.
- auto_theta: AutoTheta with classical multiplicative decomposition at m = 52.
Native prediction intervals are returned for all three.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fpa.models.base import ForecastContext, ForecastOutput

log = logging.getLogger(__name__)

PERIOD = 365.25 / 7


def fourier_terms(t: np.ndarray, k: int, period: float = PERIOD) -> pd.DataFrame:
    """Sine/cosine pairs for yearly seasonality on integer week index t."""
    cols = {}
    for j in range(1, k + 1):
        cols[f"sin{j}"] = np.sin(2 * np.pi * j * t / period)
        cols[f"cos{j}"] = np.cos(2 * np.pi * j * t / period)
    return pd.DataFrame(cols)


def _model(name: str, m: int, arima_mode: str):
    from statsforecast.models import MSTL, AutoARIMA, AutoETS, AutoTheta

    if name == "auto_ets":
        return MSTL(season_length=m, trend_forecaster=AutoETS(model="ZZN"), alias=name)
    if name == "auto_arima":
        if arima_mode == "seasonal":
            return AutoARIMA(season_length=m, alias=name)
        return AutoARIMA(season_length=1, alias=name)
    if name == "auto_theta":
        return AutoTheta(season_length=m, decomposition_type="multiplicative", alias=name)
    raise ValueError(f"unknown statistical model {name}")


def _as_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df if "unique_id" in df.columns else df.reset_index()


class StatsModel:
    family = "stats"

    def __init__(self, name: str, m: int = 52, arima_mode: str = "fourier", fourier_k: int = 6):
        self.name = name
        self.m = m
        self.arima_mode = arima_mode
        self.fourier_k = fourier_k

    @property
    def uses_exog(self) -> bool:
        return self.name == "auto_arima" and self.arima_mode == "fourier"

    def forecast(self, ctx: ForecastContext, origin: int, h: int) -> ForecastOutput:
        from statsforecast import StatsForecast
        from statsforecast.models import SeasonalNaive

        y = ctx.y_hist(origin)
        n = y.shape[0]
        df = pd.DataFrame(
            {
                "unique_id": np.repeat(np.arange(n), origin).astype(str),
                "ds": np.tile(np.arange(origin), n),
                "y": y.reshape(-1),
            }
        )
        X_df = None
        if self.uses_exog:
            df = pd.concat([df, fourier_terms(df["ds"].to_numpy(), self.fourier_k)], axis=1)
            fut_t = np.tile(np.arange(origin, origin + h), n)
            X_df = pd.concat(
                [
                    pd.DataFrame(
                        {"unique_id": np.repeat(np.arange(n), h).astype(str), "ds": fut_t}
                    ),
                    fourier_terms(fut_t, self.fourier_k),
                ],
                axis=1,
            )
        sf = StatsForecast(
            models=[_model(self.name, self.m, self.arima_mode)],
            freq=1,
            n_jobs=-1,
            fallback_model=SeasonalNaive(season_length=self.m, alias=self.name),
        )
        fc = _as_cols(sf.forecast(df=df, h=h, level=ctx.levels, fitted=True, X_df=X_df))
        fv = _as_cols(sf.forecast_fitted_values())

        fc["node"] = fc["unique_id"].astype(int)
        fc["step"] = fc.groupby("node").cumcount()
        mean = np.zeros((n, h))
        mean[fc["node"], fc["step"]] = fc[self.name].to_numpy()
        lo, hi = {}, {}
        for lev in ctx.levels:
            lo[lev] = np.zeros((n, h))
            hi[lev] = np.zeros((n, h))
            lo[lev][fc["node"], fc["step"]] = fc[f"{self.name}-lo-{lev}"].to_numpy()
            hi[lev][fc["node"], fc["step"]] = fc[f"{self.name}-hi-{lev}"].to_numpy()
        fitted = np.full((n, origin), np.nan)
        fv["node"] = fv["unique_id"].astype(int)
        fitted[fv["node"], fv["ds"].astype(int)] = fv[self.name].to_numpy()
        return ForecastOutput(
            mean=np.maximum(mean, 0.0),
            fitted=fitted,
            lo={k: np.maximum(v, 0.0) for k, v in lo.items()},
            hi=hi,
        )
