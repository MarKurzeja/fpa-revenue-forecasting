import math

import numpy as np
import pytest

from fpa import metrics as M


def test_wape_bias_mae_rmse():
    a = [100, 200, 300]
    f = [110, 180, 330]
    assert M.wape(a, f) == pytest.approx(60 / 600)
    assert M.bias(a, f) == pytest.approx(20 / 600)
    assert M.mae(a, f) == pytest.approx(20)
    assert M.rmse(a, f) == pytest.approx(math.sqrt((100 + 400 + 900) / 3))


def test_wape_zero_actuals_is_nan():
    assert math.isnan(M.wape([0, 0], [1, 2]))


def test_mase_and_rmsse():
    ins = [1, 2, 3, 4, 6]  # lag-2 diffs: 2, 2, 3 -> mean 7/3 ; lag-1 sq diffs: 1,1,1,4 -> 7/4
    assert M.mase([5], [4], ins, m=2) == pytest.approx(1 / (7 / 3))
    assert M.rmsse([5, 7], [4, 7], ins, m=1) == pytest.approx(math.sqrt(0.5 / (7 / 4)))
    assert math.isnan(M.mase([1], [1], [1, 2], m=52))


def test_coverage_width_pinball_winkler():
    a = np.array([1.0, 5.0, 10.0])
    lo, hi = np.array([0.0, 6.0, 8.0]), np.array([2.0, 8.0, 12.0])
    assert M.coverage(a, lo, hi) == pytest.approx(2 / 3)
    assert M.interval_width(lo, hi) == pytest.approx(8 / 3)
    # pinball tau=0.9: diffs a-q = 1-0, 5-6, 10-8 -> 0.9, 0.1, 1.8
    assert M.pinball(a, lo, 0.9) == pytest.approx((0.9 + 0.1 + 1.8) / 3)
    # winkler alpha=0.2: widths 2,2,4 + penalty (2/0.2)*(6-5)=10 on second
    assert M.winkler(a, lo, hi, 0.2) == pytest.approx((2 + 12 + 4) / 3)


def test_weighted_mean_ignores_nan():
    assert M.weighted_mean([1.0, np.nan, 3.0], [1, 5, 3]) == pytest.approx((1 + 9) / 4)


def test_diebold_mariano_matches_hand_calculation_h1():
    ea = np.array([1.0, -2.0, 0.5, 1.5, -1.0, 0.2])
    eb = np.array([2.0, -2.5, 1.0, 2.5, -1.5, 1.0])
    d = ea**2 - eb**2
    n = len(d)
    var = np.mean((d - d.mean()) ** 2)
    dm = d.mean() / math.sqrt(var / n)
    hln = math.sqrt((n + 1 - 2 + 0) / n)
    res = M.diebold_mariano(ea, eb, h=1, loss="sq")
    assert res.statistic == pytest.approx(dm * hln)
    from scipy import stats

    assert res.p_value == pytest.approx(2 * stats.t.sf(abs(dm * hln), n - 1))
    assert res.statistic < 0  # model A better


def test_diebold_mariano_reference_value():
    # Reference: d = [-1, -3, 1, -2, 0, -1] (abs loss), h = 2, computed by hand:
    # mean -1, deviations [0,-2,2,-1,1,0]; gamma0 = 10/6, gamma1 = (0-4-2-1+0)/6 = -7/6
    # var = 10/6 - 14/6 < 0 -> falls back to gamma0; DM = -1 / sqrt((10/6)/6)
    ea = np.array([1.0, 1.0, 3.0, 1.0, 2.0, 1.0])
    eb = np.array([2.0, 4.0, 2.0, 3.0, 2.0, 2.0])
    res = M.diebold_mariano(ea, eb, h=2, loss="abs")
    dm = -1 / math.sqrt((10 / 6) / 6)
    hln = math.sqrt((6 + 1 - 4 + 2 * 1 / 6) / 6)
    assert res.statistic == pytest.approx(dm * hln)


def test_dm_bad_loss():
    with pytest.raises(ValueError):
        M.diebold_mariano([1.0], [1.0], loss="x")


def test_horizon_bucket():
    out = M.horizon_bucket(np.array([1, 4, 5, 13, 20]), [(1, 4), (5, 8), (9, 13)])
    assert list(out) == ["1-4", "1-4", "5-8", "9-13", ">13"]
