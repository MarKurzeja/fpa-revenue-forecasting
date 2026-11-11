import numpy as np
import pandas as pd
import pytest

from fpa.models import intervals as conf


def test_conformal_quantile_finite_sample():
    s = np.arange(1, 11, dtype=float)  # n = 10
    assert conf.conformal_quantile(s, 80) == 9.0  # ceil(11 * 0.8) = 9th
    assert conf.conformal_quantile(s, 95) == 10.0
    assert np.isnan(conf.conformal_quantile(np.array([]), 80))


def test_calibrate_and_apply():
    rng = np.random.default_rng(0)
    n = 400
    rec = pd.DataFrame(
        {
            "model": "m",
            "level": 1,
            "h": rng.integers(1, 14, n),
            "y_true": 100 + rng.normal(0, 10, n),
            "y_pred": 100.0,
            "scale": 100.0,
        }
    )
    q = conf.calibrate(rec, [80, 95], [(1, 4), (5, 8), (9, 13)])
    assert set(q["bucket"]) == {0, 1, 2}
    out = conf.apply(rec, q, [80, 95], [(1, 4), (5, 8), (9, 13)])
    cov80 = ((out["y_true"] >= out["lo80"]) & (out["y_true"] <= out["hi80"])).mean()
    assert 0.75 <= cov80 <= 0.9
    assert (out["hi95"] - out["lo95"] >= out["hi80"] - out["lo80"]).all()
    far = conf.apply(rec.assign(h=26).head(1), q, [80], [(1, 4), (5, 8), (9, 13)])
    near = conf.apply(rec.assign(h=13).head(1), q, [80], [(1, 4), (5, 8), (9, 13)])
    assert (far["hi80"] - far["y_pred"]).iloc[0] == pytest.approx(
        (near["hi80"] - near["y_pred"]).iloc[0] * np.sqrt(26 / 11)
    )
