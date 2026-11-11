import numpy as np
import pytest

from fpa.models.baselines import (
    Baseline,
    moving_average,
    naive,
    run_rate,
    seasonal_naive,
    seasonality_index,
)


def test_simple_baselines():
    y = np.arange(1.0, 61.0)[None, :]
    assert np.all(naive(y, 3) == 60)
    assert np.all(moving_average(y, 2, 4) == np.mean([57, 58, 59, 60]))
    sn = seasonal_naive(y, 3, m=52)
    assert list(sn[0]) == [9.0, 10.0, 11.0]
    with pytest.raises(ValueError):
        seasonal_naive(y[:, :10], 3, m=52)


def test_run_rate_is_snaive_scaled_by_growth():
    # Flat 10 last year, flat 20 this year, spike week -> index 1 except the spike.
    y = np.concatenate([np.full(52, 10.0), np.full(13, 20.0)])[None, :]
    y[0, 14] = 15.0  # last year's value for step 2 (index o - 52 + 1 with o = 65 -> 14)
    si = seasonality_index(y, 3, m=52, window=13, max_years=1)
    assert si[0, 0] == pytest.approx(1.0)
    assert si[0, 1] == pytest.approx(15.0 / 10.0)
    rr = run_rate(y, 3, m=52, window=13, max_years=1)
    assert rr[0, 1] == pytest.approx(20 * 1.5)


def test_baseline_forecaster_fitted(small_panel):
    from fpa.backtest import make_context
    from fpa.config import load_config

    ctx = make_context(load_config(), small_panel)
    for name in ("naive", "seasonal_naive", "ma4", "ma13", "run_rate"):
        out = Baseline(name).forecast(ctx, 120, 13)
        assert out.mean.shape == (small_panel.h.n_nodes, 13)
        assert out.fitted.shape == (small_panel.h.n_nodes, 120)
        assert np.isfinite(out.fitted[:, -1]).all()
    with pytest.raises(ValueError):
        Baseline("nope").forecast(ctx, 120, 2)
