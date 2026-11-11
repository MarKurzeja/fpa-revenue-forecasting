"""Pipeline steps, quality gates and the markdown report on the test fixture panel."""

import pytest

import fpa.panel as panel_mod
from conftest import make_items
from fpa import pipeline, planning
from helpers import fit_regressor, require_regressor, tiny_config


def test_pipeline_steps_and_report(small_panel, tmp_path, monkeypatch):
    require_regressor()
    cfg = tiny_config(tmp_path)
    cfg.paths.ensure()
    monkeypatch.setattr(pipeline, "_panel", lambda c: small_panel)
    monkeypatch.setattr(panel_mod, "load_panel", lambda f: small_panel)
    w = small_panel.weeks
    monkeypatch.setattr(
        planning, "load_item_weeks", lambda db, wks: make_items(w[w["wm_yr_wk"].isin(wks)])
    )

    with pytest.raises(SystemExit):
        pipeline.load_backtest(cfg)
    res = pipeline.step_backtest(cfg, regressor_factory=fit_regressor)
    comp = pipeline.step_reconcile(cfg)
    assert len(comp) == 8
    plan = pipeline.step_plan(cfg, regressor_factory=fit_regressor)
    assert not plan["commentary"].empty
    cap = pipeline.step_capital(cfg)
    assert {"working_capital", "elasticities", "price_scenarios"} <= set(cap)
    assert not cap["price_scenarios"].empty
    dm = pipeline.dm_test(res["records"], res["selection"]["champion"])
    assert 0 <= dm["p_value"] <= 1
    fails = pipeline.quality_gates(cfg)
    assert isinstance(fails, list)

    from fpa.report import build_report

    out = build_report(cfg)
    text = out.read_text()
    for section in (
        "## 1. Headline",
        "## 2. Forecast value added",
        "## 5. Reconciliation",
        "## 7. Diebold-Mariano",
        "## 8. Key findings",
        "Working capital",
        "FP&A",
    ):
        assert section in text
    assert (cfg.paths.figures_dir / "fva_waterfall.png").exists()
    assert (cfg.paths.figures_dir / "pvm_vs_budget.png").exists()
