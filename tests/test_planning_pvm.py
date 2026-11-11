import numpy as np
import pandas as pd
import pytest

from conftest import make_items
from fpa import planning


def test_budget_item_plan_preserves_revenue():
    from conftest import make_weeks

    w = make_weeks(60)
    items = make_items(w[w["t"] < 52])
    bud = pd.DataFrame(
        {
            "store_id": ["CA_1", "TX_1"],
            "dept_id": "FOODS_1",
            "t": 55,
            "fiscal_month": 1,
            "revenue_point": [1000.0, 500.0],
        }
    )
    plan = planning.budget_item_plan(items, bud)
    assert plan.groupby("store_id")["revenue"].sum().to_dict() == pytest.approx(
        {"CA_1": 1000.0, "TX_1": 500.0}
    )
    assert np.allclose(plan["units"] * plan["price"], plan["revenue"])


def test_pvm_tables_sum_to_variance(small_panel, monkeypatch):
    w = small_panel.weeks
    monkeypatch.setattr(
        planning, "load_item_weeks", lambda db, wks: make_items(w[w["wm_yr_wk"].isin(wks)])
    )
    fy = planning.budget_year(w, None)
    versions = planning.version_plan(w, fy)
    bud_id = versions["version_id"].iloc[0]
    fyw = w[w["fiscal_year"] == fy]
    values = pd.DataFrame(
        [
            {
                "version_id": bud_id,
                "t": t,
                "level": 8,
                "store_id": s,
                "dept_id": "FOODS_1",
                "revenue_point": 120.0,
            }
            for t in fyw["t"]
            for s in ("CA_1", "TX_1")
        ]
    )
    out = planning.pvm_tables(small_panel, None, versions, values)
    assert set(out["comparison"]) == {"vs_prior_year", "vs_budget"}
    eff = out[["volume", "mix", "price"]].sum(axis=1)
    assert np.allclose(eff, out["variance"])
    fy_py = out[(out["comparison"] == "vs_prior_year") & (out["fiscal_month"] == 0)]
    months = out[(out["comparison"] == "vs_prior_year") & (out["fiscal_month"] > 0)]
    assert fy_py["actual"].sum() == pytest.approx(months["actual"].sum())
    assert planning.budget_year(w, 2013) == 2013
    with pytest.raises(ValueError):
        planning.budget_year(w.head(60), None)
