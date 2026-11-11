"""Helpers shared by tests (config for tiny runs, a sklearn stand-in for LightGBM)."""

from __future__ import annotations

import copy

import yaml

from fpa.config import ROOT, from_dict

try:
    import lightgbm  # noqa: F401

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def tiny_config(tmp_path, stats=()):
    raw = yaml.safe_load((ROOT / "config.yaml").read_text())
    raw = copy.deepcopy(raw)
    raw["backtest"].update(n_origins=3, n_validation_origins=2)
    raw["models"]["stats"] = list(stats)
    if HAS_LGBM:
        raw["models"]["lightgbm"]["grid"] = {"n_estimators": [40], "num_leaves": [15]}
        raw["models"]["lightgbm"]["fixed_params"] = {"verbose": -1}
    else:
        raw["models"]["lightgbm"]["grid"] = {"max_iter": [30]}
        raw["models"]["lightgbm"]["fixed_params"] = {}
    raw["models"]["lightgbm"]["target_transforms"] = ["log1p", "trailing_mean"]
    raw["models"]["lightgbm"]["train_origin_stride"] = 4
    raw["models"]["lightgbm"]["long_horizon_stride"] = 8
    for k in (
        "processed_dir",
        "artifacts_dir",
        "reports_dir",
        "figures_dir",
        "exports_dir",
        "demo_dir",
    ):
        raw["paths"][k] = str(tmp_path / k)
    raw["paths"]["duckdb_file"] = str(tmp_path / "fpa.duckdb")
    return from_dict(raw, root=ROOT)


def fit_regressor(params, seed):
    """LightGBM when installed (CI, your machine); sklearn's HGB as a stand-in otherwise."""
    if HAS_LGBM:
        from fpa.models.ml import default_regressor

        return default_regressor(params, seed)
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(random_state=seed, **params)


def require_regressor():
    import pytest

    if not HAS_LGBM:
        pytest.importorskip("sklearn")
