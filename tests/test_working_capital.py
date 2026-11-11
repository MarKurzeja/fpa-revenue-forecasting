import numpy as np
import pandas as pd
import pytest

from fpa import working_capital as wc


def test_z_values():
    assert wc.z_value(0.95) == pytest.approx(1.6448536, abs=1e-6)
    assert wc.z_value(0.90) == pytest.approx(1.2815516, abs=1e-6)
    assert wc.z_value(0.98) == pytest.approx(2.0537489, abs=1e-6)
    with pytest.raises(ValueError):
        wc.z_value(1.0)


def test_safety_stock_hand_example():
    # z(95%) * sigma 1000 * COGS 0.75 = 1.6449 * 750
    assert float(wc.safety_stock(1000.0, 0.95, 0.75)) == pytest.approx(1233.64, abs=0.01)


def _records():
    rows = []
    for model, noise in (("good", 1.0), ("bad", 3.0)):
        for o, role in ((1, "validation"), (2, "test"), (3, "test"), (4, "holdout")):
            for h in (1, 2, 3):
                err = noise * (1 if (o + h) % 2 else -1)
                rows.append(
                    {
                        "model": model,
                        "node_id": "A/X",
                        "origin": o,
                        "role": role,
                        "h": h,
                        "y_true": 100.0,
                        "y_pred": 100.0 - err,
                    }
                )
    return pd.DataFrame(rows)


def test_lead_time_sigma_and_capital_table():
    lt = wc.lead_time_errors(_records(), lead_time=2)
    assert set(lt["origin"]) == {1, 2, 3, 4}
    assert (lt[lt["model"] == "good"]["err"].abs() == 0).all()  # +1 -1 cancel over 2 weeks
    lt3 = wc.lead_time_errors(_records(), lead_time=1)
    sig = wc.sigma_by_node(lt3)
    s_good = sig.set_index("model").loc["good", "sigma_lt"]
    assert s_good == pytest.approx(np.std([1, -1, 1], ddof=1))
    cap = wc.capital_table(sig, {"A/X": "FOODS"}, {"FOODS": 0.75}, [0.95], 0.2, ["bad"], "good")
    rel = cap.dropna(subset=["working_capital_released_usd"]).iloc[0]
    ss_bad = wc.z_value(0.95) * sig.set_index("model").loc["bad", "sigma_lt"] * 0.75
    ss_good = wc.z_value(0.95) * s_good * 0.75
    assert rel["working_capital_released_usd"] == pytest.approx(ss_bad - ss_good)
    assert rel["carrying_cost_saved_usd"] == pytest.approx(0.2 * (ss_bad - ss_good))
    svc = wc.service_level_check(lt3, sig, [0.95])
    assert set(svc["model"]) == {"good", "bad"}
    assert svc["achieved"].between(0, 1).all()
