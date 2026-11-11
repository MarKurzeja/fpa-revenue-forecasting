import numpy as np
import pandas as pd
import pytest

from fpa.backtest import make_origins, run_backtest
from fpa.evaluate import summarize
from fpa.hierarchy import is_coherent
from helpers import fit_regressor, require_regressor, tiny_config


def test_make_origins():
    o = make_origins(277, 8, 4, 13, 13, 4)
    assert o.holdout == 264
    assert o.backtest[-1] == 251 and o.backtest[0] == 223
    assert o.validation == o.backtest[:4]
    assert o.role(264) == "holdout" and o.role(223) == "validation" and o.role(251) == "test"
    with pytest.raises(ValueError):
        make_origins(130, 8, 4, 13, 13, 4)


@pytest.fixture
def backtest_result(small_panel, tmp_path):
    require_regressor()
    cfg = tiny_config(tmp_path)
    return cfg, run_backtest(cfg, small_panel, regressor_factory=fit_regressor)


def test_backtest_end_to_end(backtest_result, small_panel):
    cfg, res = backtest_result
    rec, sel = res["records"], res["selection"]
    assert {"naive", "seasonal_naive", "ma4", "ma13", "run_rate", "lightgbm"} <= set(rec["model"])
    assert sel["champion"].split("+")[1] in cfg.reconciliation["methods"]
    assert set(rec["role"]) == {"validation", "test", "holdout"}
    # every model has intervals after conformal calibration
    assert rec[["lo80", "hi80", "lo95", "hi95"]].notna().all().all()
    # reconciled forecasts add up at every origin and horizon
    for name in [m for m in rec["model"].unique() if "+" in m]:
        g = rec[rec["model"] == name]
        for (_, _), gg in g.groupby(["origin", "h"]):
            y = np.zeros(small_panel.h.n_nodes)
            y[gg["node"].to_numpy()] = gg["y_pred"].to_numpy()
            assert is_coherent(small_panel.h, y)
    s = summarize(rec[rec["role"] != "validation"], ["model", "level"])
    assert s["wape"].notna().all()
    # cached second run gives identical results
    res2 = run_backtest(cfg, small_panel, regressor_factory=fit_regressor)
    pd.testing.assert_frame_equal(res2["records"], rec)


def test_champion_forecast_and_planning(backtest_result, small_panel):
    from fpa.backtest import champion_forecast, make_context
    from fpa.planning import budget_year, commentary, variance_tables, version_plan

    cfg, res = backtest_result
    ctx = make_context(cfg, small_panel)
    fc = champion_forecast(
        cfg, ctx, res["selection"], res["qtable"], small_panel.T, 13, fit_regressor
    )
    assert fc["wm_yr_wk"].notna().all() and fc["y_true"].isna().all()
    assert (fc["hi95"] >= fc["y_pred"]).all()

    fy = budget_year(small_panel.weeks, None)
    assert fy == 2014
    plan = version_plan(small_panel.weeks, fy)
    assert plan["type"].tolist()[0] == "budget" and len(plan) == 12

    from fpa.planning import run_versions

    versions, values = run_versions(cfg, ctx, res["selection"], res["qtable"], fit_regressor)
    assert versions["type"].value_counts().to_dict() == {
        "reforecast": 11,
        "budget": 1,
        "rolling": 1,
    }
    var = variance_tables(small_panel, versions, values)
    m = var["monthly"]
    tot = m[m["node_id"] == "Total"]
    assert len(tot) == 12
    assert np.allclose(tot["var_vs_budget"], tot["actual"] - tot["budget"])
    gap = var["gap"][var["gap"]["node_id"] == "Total"]
    bud = gap.iloc[0]
    assert bud["fy_outlook"] == pytest.approx(bud["fy_budget"])
    last = gap.iloc[-1]
    assert np.isfinite(last["gap_pct"])
    # commentary needs PVM rows; feed an empty-but-shaped PVM table
    pvm_df = pd.DataFrame(
        {
            "comparison": "vs_budget",
            "fiscal_month": tot["fiscal_month"],
            "state_id": "CA",
            "cat_id": "FOODS",
            "volume": 1.0,
            "mix": 0.0,
            "price": -0.5,
        }
    )
    comm = commentary(var, pvm_df)
    assert len(comm) == 12 and comm["text"].str.contains("Fiscal month").all()
