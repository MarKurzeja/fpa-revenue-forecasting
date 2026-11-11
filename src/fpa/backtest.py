"""Rolling-origin backtest, model selection, ensemble and reconciliation (SPEC §5.3, §6).

Origins (expanding window): the final hold-out origin is T - holdout_weeks; the last
backtest origin is the hold-out origin minus the horizon, so no backtest window
touches the hold-out; earlier origins step back by `origin_step`. The earliest
`n_validation_origins` are used for tuning, model selection and interval calibration;
the rest are "test" origins. Hold-out numbers are reported separately.
"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fpa import io
from fpa.config import Config
from fpa.evaluate import selection_scores
from fpa.features import FeatureSpec, build_calendar
from fpa.models import intervals as conformal
from fpa.models.base import ForecastContext, ForecastOutput
from fpa.models.baselines import make_baselines
from fpa.models.ensemble import average_outputs
from fpa.models.ml import LightGBMForecaster
from fpa.models.stats import StatsModel
from fpa.panel import Panel
from fpa.reconcile import reconcile

log = logging.getLogger(__name__)
CAL_EXTRA_WEEKS = 60  # projected calendar beyond the data (forward forecast, budget)


@dataclass(frozen=True)
class Origins:
    backtest: list[int]
    holdout: int
    validation: list[int]

    def role(self, origin: int) -> str:
        if origin == self.holdout:
            return "holdout"
        return "validation" if origin in self.validation else "test"

    @property
    def all(self) -> list[int]:
        return [*self.backtest, self.holdout]


def make_origins(
    T: int, n_origins: int, step: int, holdout_weeks: int, horizon: int, n_validation: int
) -> Origins:
    """Origins are week indices: the model sees weeks < origin and forecasts origin.. ."""
    holdout = T - holdout_weeks
    last = holdout - horizon
    bt = sorted(last - step * i for i in range(n_origins))
    if bt[0] < 104 + 14:
        raise ValueError(f"earliest origin {bt[0]} leaves too little history")
    return Origins(backtest=bt, holdout=holdout, validation=bt[:n_validation])


def make_context(cfg: Config, panel: Panel) -> ForecastContext:
    return ForecastContext(
        panel=panel,
        calendar=build_calendar(panel.weeks, CAL_EXTRA_WEEKS, cfg.fiscal),
        levels=cfg.interval_levels,
        season_length=cfg.season_length,
        seed=cfg.seed,
        use_future_prices=cfg.backtest.use_future_prices,
    )


# --------------------------------------------------------------------------- records


def node_scales(y: np.ndarray, origin: int, m: int) -> dict[str, np.ndarray]:
    """Per-node scales computed from history before the origin."""
    hist = y[:, :origin]
    scale = np.maximum(hist[:, -13:].mean(axis=1), 1.0)
    mase_scale = np.mean(np.abs(hist[:, m:] - hist[:, :-m]), axis=1)
    rmsse_scale = np.mean(np.diff(hist, axis=1) ** 2, axis=1)
    weight = hist[:, -52:].sum(axis=1)
    return {"scale": scale, "mase_scale": mase_scale, "rmsse_scale": rmsse_scale, "weight": weight}


def to_records(
    model: str, origin: int, role: str, out: ForecastOutput, panel: Panel, levels: list[int], m: int
) -> pd.DataFrame:
    n, h = out.mean.shape
    t = origin + np.arange(h)
    y_true = np.full((n, h), np.nan)
    obs = t < panel.T
    y_true[:, obs] = panel.y[:, t[obs]]
    sc = node_scales(panel.y, origin, m)
    nodes = panel.nodes
    rec = pd.DataFrame(
        {
            "model": model,
            "origin": origin,
            "role": role,
            "node": np.repeat(np.arange(n), h),
            "node_id": np.repeat(nodes["node_id"].to_numpy(), h),
            "level": np.repeat(nodes["level"].to_numpy(), h),
            "h": np.tile(np.arange(1, h + 1), n),
            "t": np.tile(t, n),
            "y_true": y_true.reshape(-1),
            "y_pred": out.mean.reshape(-1),
            **{k: np.repeat(v, h) for k, v in sc.items()},
        }
    )
    wk = panel.weeks.set_index("t")["wm_yr_wk"]
    rec["wm_yr_wk"] = rec["t"].map(wk)
    for L in levels:
        rec[f"lo{L}"] = out.lo[L].reshape(-1) if L in out.lo else np.nan
        rec[f"hi{L}"] = out.hi[L].reshape(-1) if L in out.hi else np.nan
    return rec


# --------------------------------------------------------------------------- models


def build_models(cfg: Config) -> list[Any]:
    """Baseline and statistical forecasters from config (LightGBM is tuned separately)."""
    mcfg = cfg.models
    rr = mcfg.get("run_rate", {})
    models: list[Any] = make_baselines(
        mcfg["baselines"], cfg.season_length, int(rr.get("window", 13)), int(rr.get("max_years", 3))
    )
    ar = mcfg.get("arima", {})
    for name in mcfg.get("stats", []):
        models.append(
            StatsModel(
                name,
                m=cfg.season_length,
                arima_mode=ar.get("mode", "fourier"),
                fourier_k=int(ar.get("fourier_k", 6)),
            )
        )
    return models


def lightgbm_from(
    cfg: Config, transform: str, params: dict, regressor_factory=None, name: str = "lightgbm"
) -> LightGBMForecaster:
    lg = cfg.models["lightgbm"]
    spec = FeatureSpec(
        lags=tuple(lg["lags"]),
        rolling_windows=tuple(lg["rolling_windows"]),
        transform=transform,
        use_future_prices=cfg.backtest.use_future_prices,
    )
    return LightGBMForecaster(
        spec,
        {**lg.get("fixed_params", {}), **params},
        name=name,
        train_origin_stride=int(lg.get("train_origin_stride", 1)),
        long_horizon_stride=int(lg.get("long_horizon_stride", 2)),
        regressor_factory=regressor_factory,
    )


def grid(params: dict[str, list]) -> list[dict]:
    keys = sorted(params)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(params[k] for k in keys))]


class BacktestRunner:
    """Runs forecasters over origins with a per-model cache."""

    def __init__(
        self, cfg: Config, panel: Panel, force: bool = False, cache_dir: Path | None = None
    ):
        self.cfg = cfg
        self.panel = panel
        self.ctx = make_context(cfg, panel)
        self.force = force
        self.cache_dir = cache_dir or (cfg.paths.artifacts_dir / "backtest")
        bt = cfg.backtest
        self.origins = make_origins(
            panel.T,
            bt.n_origins,
            bt.origin_step,
            bt.holdout_weeks,
            cfg.horizon,
            bt.n_validation_origins,
        )
        self.outputs: dict[str, dict[int, ForecastOutput]] = {}
        self.timings: dict[str, float] = {}

    def run(self, model: Any, origins: list[int] | None = None) -> dict[int, ForecastOutput]:
        origins = origins or self.origins.all
        path = self.cache_dir / f"outputs_{model.name}.pkl"
        cached: dict[int, ForecastOutput] = {}
        if path.exists() and not self.force:
            cached = io.load_pickle(path)
        todo = [o for o in origins if o not in cached]
        t0 = time.perf_counter()
        for o in todo:
            log.info("%-26s origin %d (%s)", model.name, o, self.origins.role(o))
            cached[o] = model.forecast(self.ctx, o, self.cfg.horizon)
        if todo:
            io.save_pickle(cached, path)
            self.timings[model.name] = time.perf_counter() - t0
        self.outputs[model.name] = cached
        return {o: cached[o] for o in origins}

    def records(self, name: str, outputs: dict[int, ForecastOutput] | None = None) -> pd.DataFrame:
        outputs = outputs if outputs is not None else self.outputs[name]
        return pd.concat(
            [
                to_records(
                    name,
                    o,
                    self.origins.role(o),
                    out,
                    self.panel,
                    self.cfg.interval_levels,
                    self.cfg.season_length,
                )
                for o, out in sorted(outputs.items())
            ],
            ignore_index=True,
        )


def tune_lightgbm(runner: BacktestRunner, regressor_factory=None) -> tuple[dict, pd.DataFrame]:
    """Pick transform and hyperparameters on validation origins only."""
    lg = runner.cfg.models["lightgbm"]
    rows = []
    best = None
    for transform in lg["target_transforms"]:
        for i, params in enumerate(grid(lg["grid"])):
            name = f"lgbm_tune_{transform}_{i}"
            model = lightgbm_from(runner.cfg, transform, params, regressor_factory, name=name)
            runner.run(model, runner.origins.validation)
            rec = runner.records(
                name, {o: runner.outputs[name][o] for o in runner.origins.validation}
            )
            score = float(selection_scores(rec).iloc[0])
            rows.append({"candidate": name, "transform": transform, **params, "score": score})
            if best is None or score < best[0]:
                best = (score, transform, params, name)
    assert best is not None
    choice = {"transform": best[1], "params": best[2], "score": best[0], "candidate": best[3]}
    return choice, pd.DataFrame(rows).sort_values("score")


def reconcile_outputs(
    panel: Panel, outputs: dict[int, ForecastOutput], method: str
) -> tuple[dict[int, ForecastOutput], list[dict]]:
    out, infos = {}, []
    for o, fo in outputs.items():
        resid = None
        if fo.fitted is not None:
            resid = (panel.y[:, :o] - fo.fitted).T  # (n_obs, n_nodes)
        rec_mean, info = reconcile(panel.h, fo.mean, method, resid)
        out[o] = ForecastOutput(mean=rec_mean, fitted=fo.fitted)
        infos.append({"origin": o, **info})
    return out, infos


def run_backtest(
    cfg: Config, panel: Panel, force: bool = False, model_set: str = "all", regressor_factory=None
) -> dict[str, Any]:
    """Full backtest. Returns records (all models), selection and calibration tables."""
    runner = BacktestRunner(cfg, panel, force=force)
    out_dir = cfg.paths.artifacts_dir
    levels, buckets = cfg.interval_levels, cfg.horizon_buckets

    models = build_models(cfg)
    if model_set == "baselines":
        models = [m for m in models if m.family == "baseline"]
    elif model_set == "stats":
        models = [m for m in models if m.family in ("baseline", "stats")]
    elif model_set == "ml":
        models = [m for m in models if m.family == "baseline"]
    all_records = []
    families: dict[str, str] = {}
    for m in models:
        runner.run(m)
        all_records.append(runner.records(m.name))
        families[m.name] = m.family

    selection: dict[str, Any] = {"origins": runner.origins.__dict__}
    if model_set in ("all", "ml") and cfg.models.get("ml"):
        choice, tuning = tune_lightgbm(runner, regressor_factory)
        io.save_table(tuning, out_dir / "lightgbm_tuning")
        selection["lightgbm"] = choice
        lgbm = lightgbm_from(cfg, choice["transform"], choice["params"], regressor_factory)
        # reuse the tuned candidate's validation outputs
        runner.outputs.setdefault("lightgbm", {})
        for o in runner.origins.validation:
            runner.outputs["lightgbm"][o] = runner.outputs[choice["candidate"]][o]
        cache = runner.cache_dir / "outputs_lightgbm.pkl"
        if cache.exists() and not force:
            prev = io.load_pickle(cache)
            if prev.get("_choice") == choice["candidate"]:
                runner.outputs["lightgbm"].update({k: v for k, v in prev.items() if k != "_choice"})
        todo = [o for o in runner.origins.all if o not in runner.outputs["lightgbm"]]
        for o in todo:
            log.info("lightgbm origin %d (%s)", o, runner.origins.role(o))
            runner.outputs["lightgbm"][o] = lgbm.forecast(runner.ctx, o, cfg.horizon)
        io.save_pickle({**runner.outputs["lightgbm"], "_choice": choice["candidate"]}, cache)
        all_records.append(runner.records("lightgbm", runner.outputs["lightgbm"]))
        families["lightgbm"] = "ml"
        if lgbm.feature_importance_:
            selection["lightgbm_feature_importance"] = dict(
                sorted(lgbm.feature_importance_.items(), key=lambda kv: -kv[1])[:20]
            )

    records = pd.concat(all_records, ignore_index=True)
    scores = selection_scores(records)
    singles = [s for s in scores.index if s in families]

    # Ensemble of the top-k single models (chosen on validation).
    k = int(cfg.models.get("ensemble_top_k", 3))
    members = [s for s in singles if s != "naive"][:k]
    if len(members) >= 2:
        ens = {
            o: average_outputs([runner.outputs[m][o] for m in members]) for o in runner.origins.all
        }
        runner.outputs["ensemble"] = ens
        records = pd.concat([records, runner.records("ensemble", ens)], ignore_index=True)
        families["ensemble"] = "ensemble"
    scores = selection_scores(records)
    best_single = singles[0]
    champion_base = best_single
    if "ensemble" in scores and scores["ensemble"] < scores[best_single]:
        champion_base = "ensemble"
    selection.update(best_single=best_single, ensemble_members=members, champion_base=champion_base)

    # Reconciliation of the champion base forecasts.
    recon_info = []
    for method in cfg.reconciliation["methods"]:
        name = f"{champion_base}+{method}"
        rec_out, infos = reconcile_outputs(panel, runner.outputs[champion_base], method)
        runner.outputs[name] = rec_out
        records = pd.concat([records, runner.records(name, rec_out)], ignore_index=True)
        families[name] = "reconciled"
        recon_info += [{"model": name, **i} for i in infos]

    # Conformal intervals for every model without native intervals.
    no_native = [m for m, f in families.items() if f != "stats"]
    val = records[(records["role"] == "validation") & records["model"].isin(no_native)]
    qtable = conformal.calibrate(val, levels, buckets)
    parts = []
    for name, g in records.groupby("model", sort=False):
        if name in no_native:
            g = conformal.apply(
                g.drop(columns=[f"lo{L}" for L in levels] + [f"hi{L}" for L in levels]),
                qtable,
                levels,
                buckets,
            )
        parts.append(g)
    records = pd.concat(parts, ignore_index=True)
    records["family"] = records["model"].map(families)

    scores = selection_scores(records)
    recon_names = [n for n, f in families.items() if f == "reconciled"]
    champion = min(recon_names, key=lambda n: scores[n])
    selection.update(
        champion=champion,
        reconciliation_method=champion.split("+", 1)[1],
        validation_scores=scores.to_dict(),
        families=families,
        timings_sec=runner.timings,
    )
    io.save_table(records, out_dir / "backtest_records")
    io.save_table(qtable, out_dir / "interval_quantiles")
    io.save_table(pd.DataFrame(recon_info), out_dir / "reconciliation_info")
    io.save_json(selection, out_dir / "selection.json")
    return {"records": records, "selection": selection, "qtable": qtable, "runner": runner}


# --------------------------------------------------------------------------- champion


def champion_forecast(
    cfg: Config,
    ctx: ForecastContext,
    selection: dict,
    qtable: pd.DataFrame,
    origin: int,
    h: int,
    regressor_factory=None,
) -> pd.DataFrame:
    """Forecast every node with the selected champion pipeline (base -> reconcile -> PI)."""
    base = selection["champion_base"]
    names = selection["ensemble_members"] if base == "ensemble" else [base]
    singles = {m.name: m for m in build_models(cfg)}
    outs = []
    for nm in names:
        if nm == "lightgbm":
            ch = selection["lightgbm"]
            model = lightgbm_from(cfg, ch["transform"], ch["params"], regressor_factory)
        else:
            model = singles[nm]
        outs.append(model.forecast(ctx, origin, h))
    fo = average_outputs(outs) if len(outs) > 1 else outs[0]
    resid = (ctx.panel.y[:, :origin] - fo.fitted).T if fo.fitted is not None else None
    mean, _ = reconcile(ctx.panel.h, fo.mean, selection["reconciliation_method"], resid)
    rec = to_records(
        selection["champion"],
        origin,
        "forecast",
        ForecastOutput(mean=mean),
        ctx.panel,
        cfg.interval_levels,
        cfg.season_length,
    )
    rec = rec.drop(columns=[c for c in rec.columns if c[:2] in ("lo", "hi")])
    rec = conformal.apply(rec, qtable, cfg.interval_levels, cfg.horizon_buckets)
    cal = ctx.calendar.weeks.set_index("t")["wm_yr_wk"]
    rec["wm_yr_wk"] = rec["t"].map(cal)
    return rec
