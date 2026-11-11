import numpy as np
import pandas as pd
import pytest

from fpa import elasticity as el


def test_recovers_known_elasticity():
    from conftest import make_weeks

    w = make_weeks(150)
    rng = np.random.default_rng(3)
    rows = []
    for node, state in (("CA_1/FOODS_1", "CA"), ("TX_1/FOODS_1", "TX")):
        pi = 100 * np.exp(rng.normal(0, 0.05, len(w)))
        units = 1000 * (pi / 100) ** -1.5 * np.exp(rng.normal(0, 0.01, len(w)))
        rows.append(
            pd.DataFrame(
                {
                    "node_id": node,
                    "t": w["t"],
                    "cat_id": "FOODS",
                    "state_id": state,
                    "units": units,
                    "price_index": pi,
                }
            )
        )
    est = el.estimate_elasticities(pd.concat(rows), w)
    assert est["elasticity"].iloc[0] == pytest.approx(-1.5, abs=0.05)
    assert el.revenue_factor(0.05, -1.5) == pytest.approx(1.05**-0.5)


def test_event_uplift():
    from conftest import make_weeks

    w = make_weeks(110)
    y = np.full(110, 100.0)
    xmas = np.flatnonzero(w["event_names"].str.contains("Christmas").to_numpy())
    y[xmas] = 130.0
    out = el.event_uplifts(y, w, ["Christmas", "Easter"]).set_index("event")
    assert out.at["Christmas", "mean_uplift"] == pytest.approx(0.3)
    assert out.at["Easter", "n_occurrences"] == 0
