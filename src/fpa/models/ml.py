"""Global LightGBM forecaster, direct multi-horizon strategy (SPEC §6.3).

One model is trained across all hierarchy nodes with the horizon as a feature
(D-07). Training rows at origin `o` are every past (node, o', h) whose target week
o' + h - 1 is already observed before `o`, so nothing after the origin is used.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from fpa.features import STATIC_COLS, FeatureSpec, inverse_target, make_features
from fpa.models.base import ForecastContext, ForecastOutput

log = logging.getLogger(__name__)


def default_regressor(params: dict[str, Any], seed: int):
    import lightgbm as lgb

    return lgb.LGBMRegressor(random_state=seed, n_jobs=-1, **params)


class LightGBMForecaster:
    family = "ml"

    def __init__(
        self,
        spec: FeatureSpec,
        params: dict[str, Any],
        name: str = "lightgbm",
        train_origin_stride: int = 1,
        long_horizon_stride: int = 2,
        regressor_factory: Callable[[dict[str, Any], int], Any] | None = None,
    ):
        self.name = name
        self.spec = spec
        self.params = dict(params)
        self.train_origin_stride = train_origin_stride
        self.long_horizon_stride = long_horizon_stride
        self.regressor_factory = regressor_factory or default_regressor
        self.feature_importance_: dict[str, float] | None = None

    def _fit(self, ctx: ForecastContext, origin: int, h_train: int):
        p = ctx.panel
        y_hist = ctx.y_hist(origin)
        stride = self.train_origin_stride if h_train <= 13 else self.long_horizon_stride
        train_origins = np.arange(self.spec.min_origin, origin, stride)
        if len(train_origins) == 0:
            raise ValueError("not enough history to train LightGBM")
        # Without future prices the model only ever sees price columns < origin.
        pn, pd_ = (
            (p.price_num, p.price_den)
            if self.spec.use_future_prices
            else (p.price_num[:, :origin], p.price_den[:, :origin])
        )
        X, target, _, keys = make_features(
            y_hist,
            pn,
            pd_,
            p.nodes,
            ctx.calendar,
            train_origins,
            np.arange(1, h_train + 1),
            self.spec,
            observed_only=True,
        )
        ok = np.isfinite(target)
        model = self.regressor_factory(self.params, ctx.seed)
        cat = [c for c in STATIC_COLS if c in X.columns]
        try:
            model.fit(X[ok], target[ok], categorical_feature=cat)
        except TypeError:  # regressors without categorical support (tests)
            model.fit(X[ok], target[ok])
        if hasattr(model, "feature_importances_"):
            self.feature_importance_ = dict(zip(X.columns, map(float, model.feature_importances_)))
        return model, X[ok], target[ok], keys[ok]

    def forecast(self, ctx: ForecastContext, origin: int, h: int) -> ForecastOutput:
        p = ctx.panel
        h_train = max(h, 13)
        model, X_tr, _, keys_tr = self._fit(ctx, origin, h_train)
        pn, pd_ = (
            (p.price_num, p.price_den)
            if self.spec.use_future_prices
            else (p.price_num[:, :origin], p.price_den[:, :origin])
        )
        X, _, scale, keys = make_features(
            ctx.y_hist(origin),
            pn,
            pd_,
            p.nodes,
            ctx.calendar,
            np.array([origin]),
            np.arange(1, h + 1),
            self.spec,
            observed_only=False,
        )
        pred = inverse_target(model.predict(X), scale, self.spec.transform)
        mean = np.zeros((p.h.n_nodes, h))
        mean[keys["node"].to_numpy(), keys["h"].to_numpy() - 1] = pred

        # In-sample one-step fitted values (h = 1 rows), for the MinT covariance.
        fitted = np.full((p.h.n_nodes, origin), np.nan)
        one = keys_tr["h"].to_numpy() == 1
        if one.any():
            X1 = X_tr[one]
            n_idx = keys_tr["node"].to_numpy()[one]
            t_idx = keys_tr["t"].to_numpy()[one]
            if self.spec.transform == "log1p":
                fit_val = np.maximum(np.expm1(model.predict(X1)), 0)
            else:
                sc = np.exp(X1["log_scale"].to_numpy())
                fit_val = np.maximum(model.predict(X1) * sc, 0)
            fitted[n_idx, t_idx] = fit_val
        return ForecastOutput(mean=mean, fitted=fitted)
