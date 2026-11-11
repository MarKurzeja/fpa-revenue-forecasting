import numpy as np
import pytest

from fpa.features import FeatureSpec, build_calendar, extend_weeks, inverse_target, make_features


def _feats(panel, y, origin, spec, cal):
    return make_features(
        y[:, :origin],
        panel.price_num[:, :origin],
        panel.price_den[:, :origin],
        panel.nodes,
        cal,
        np.array([origin]),
        np.arange(1, 14),
        spec,
        observed_only=False,
    )


@pytest.mark.parametrize("transform", ["log1p", "trailing_mean"])
def test_no_leakage_marker_spike_after_origin(small_panel, transform):
    """A huge spike right after the origin must not change any feature at that origin."""
    from fpa.config import load_config

    cfg = load_config()
    cal = build_calendar(small_panel.weeks, 20, cfg.fiscal)
    spec = FeatureSpec(transform=transform)
    origin = 120
    X1, _, s1, _ = _feats(small_panel, small_panel.y, origin, spec, cal)
    y2 = small_panel.y.copy()
    y2[:, origin : origin + 5] *= 1000.0
    pn2, pd2 = small_panel.price_num.copy(), small_panel.price_den.copy()
    pn2[:, origin:] *= 5
    small_panel.price_num, small_panel.price_den = pn2, pd2
    X2, _, s2, _ = _feats(small_panel, y2, origin, spec, cal)
    assert np.allclose(X1.to_numpy(float), X2.to_numpy(float), equal_nan=True)
    assert np.allclose(s1, s2)
    # training rows built at an origin never have a target on or after that origin
    Xt, yt, _, keys = make_features(
        y2[:, :origin],
        pn2[:, :origin],
        pd2[:, :origin],
        small_panel.nodes,
        cal,
        np.arange(60, origin),
        np.arange(1, 14),
        spec,
        observed_only=True,
    )
    assert (keys["t"] < origin).all()
    assert np.isfinite(yt).all()


def test_future_prices_switch_reads_target_week(small_panel):
    from fpa.config import load_config

    cal = build_calendar(small_panel.weeks, 20, load_config().fiscal)
    spec = FeatureSpec(use_future_prices=True)
    X, *_ = make_features(
        small_panel.y[:, :120],
        small_panel.price_num,
        small_panel.price_den,
        small_panel.nodes,
        cal,
        np.array([120]),
        np.arange(1, 4),
        spec,
        False,
    )
    pi = small_panel.price_index
    assert X["price_index"].iloc[0] == pytest.approx(pi[0, 120])


def test_errors_and_inverse(small_panel):
    from fpa.config import load_config

    cal = build_calendar(small_panel.weeks, 0, load_config().fiscal)
    spec = FeatureSpec()
    with pytest.raises(ValueError):
        make_features(
            small_panel.y,
            small_panel.price_num,
            small_panel.price_den,
            small_panel.nodes,
            cal,
            np.array([10]),
            np.arange(1, 3),
            spec,
            False,
        )
    with pytest.raises(ValueError):
        make_features(
            small_panel.y,
            small_panel.price_num,
            small_panel.price_den,
            small_panel.nodes,
            cal,
            np.array([small_panel.T]),
            np.arange(1, 3),
            spec,
            False,
        )
    assert inverse_target(np.log1p(np.array([5.0])), np.array([1.0]), "log1p")[0] == pytest.approx(
        5
    )
    assert inverse_target(np.array([2.0]), np.array([3.0]), "trailing_mean")[0] == 6.0


def test_extend_weeks_projects_calendar(small_panel):
    from fpa.config import load_config

    w = extend_weeks(small_panel.weeks, 10, load_config().fiscal)
    assert len(w) == small_panel.T + 10
    assert w["is_projected"].sum() == 10
    assert (np.diff(w["t"]) == 1).all()
    assert (np.diff(w["wm_yr_wk"]) > 0).all()
    cal = build_calendar(small_panel.weeks, 10, load_config().fiscal)
    assert {"wks_to_christmas", "is_holiday_week", "week_of_year"} <= set(cal.weeks.columns)
