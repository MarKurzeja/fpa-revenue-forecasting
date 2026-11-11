import numpy as np
import pytest

from fpa.hierarchy import is_coherent
from fpa.reconcile import mint, reconcile, shrink_covariance, top_down


@pytest.mark.parametrize("method", ["bottom_up", "top_down", "mint_ols", "mint_shrink"])
def test_every_method_is_coherent(hier, method):
    rng = np.random.default_rng(1)
    yhat = hier.aggregate(rng.uniform(50, 150, size=(7, 5))) * rng.uniform(0.9, 1.1, (28, 5))
    resid = rng.normal(size=(60, 28))
    out, _ = reconcile(hier, yhat, method, resid)
    for k in range(out.shape[1]):
        assert is_coherent(hier, out[:, k])


@pytest.mark.parametrize("method", ["mint_ols", "mint_shrink"])
def test_mint_keeps_coherent_forecasts(hier, method):
    yhat = hier.aggregate(np.arange(1.0, 8.0)[:, None] * np.ones((1, 3)))
    out, _ = reconcile(hier, yhat, method, np.random.default_rng(0).normal(size=(40, 28)))
    assert np.allclose(out, yhat)


def test_top_down_keeps_total(hier):
    yhat = hier.aggregate(np.arange(1.0, 8.0)[:, None])
    yhat[hier.total_idx] = 100.0
    out = top_down(hier, yhat)
    assert out[hier.total_idx, 0] == pytest.approx(100.0)


def test_shrink_fallback_and_lambda(hier):
    _, info = mint(hier, np.ones((28, 1)), "shrink", None)
    assert info["fallback"] == "wls_struct"
    W, lam = shrink_covariance(np.random.default_rng(0).normal(size=(200, 5)))
    assert 0 <= lam <= 1 and W.shape == (5, 5)
    with pytest.raises(ValueError):
        reconcile(hier, np.ones((28, 1)), "nope")
    _, info = mint(hier, np.ones((28, 1)), "wls_struct")
    with pytest.raises(ValueError):
        mint(hier, np.ones((28, 1)), "bad")
