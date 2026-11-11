"""Smoke tests for the statsforecast and LightGBM wrappers (skipped if not installed)."""

import numpy as np
import pytest


def _ctx(panel):
    from fpa.backtest import make_context
    from fpa.config import load_config

    return make_context(load_config(), panel)


@pytest.mark.parametrize("name", ["auto_ets", "auto_arima", "auto_theta"])
def test_stats_models(small_panel, name):
    pytest.importorskip("statsforecast")
    from fpa.models.stats import StatsModel

    out = StatsModel(name).forecast(_ctx(small_panel), 150, 4)
    n = small_panel.h.n_nodes
    assert out.mean.shape == (n, 4) and np.isfinite(out.mean).all()
    assert (out.hi[95] >= out.hi[80]).all() and (out.lo[95] <= out.lo[80] + 1e-9).all()
    assert out.fitted.shape == (n, 150)


def test_lightgbm(small_panel):
    pytest.importorskip("lightgbm")
    from fpa.features import FeatureSpec
    from fpa.models.ml import LightGBMForecaster

    m = LightGBMForecaster(
        FeatureSpec(), {"n_estimators": 30, "verbose": -1}, train_origin_stride=4
    )
    out = m.forecast(_ctx(small_panel), 150, 13)
    assert out.mean.shape == (small_panel.h.n_nodes, 13) and (out.mean >= 0).all()
    assert m.feature_importance_
    assert np.isfinite(out.fitted[:, 149]).all()


def test_fourier_terms():
    from fpa.models.stats import fourier_terms

    f = fourier_terms(np.arange(10), 2)
    assert list(f.columns) == ["sin1", "cos1", "sin2", "cos2"]
    assert f["cos1"].iloc[0] == 1.0
