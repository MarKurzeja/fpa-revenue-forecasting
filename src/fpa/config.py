"""Load `config.yaml` into typed, read-only dataclasses."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path
    raw_dir: Path
    processed_dir: Path
    duckdb_file: Path
    artifacts_dir: Path
    reports_dir: Path
    figures_dir: Path
    exports_dir: Path
    demo_dir: Path

    def ensure(self) -> None:
        """Create every output directory."""
        for p in (
            self.processed_dir,
            self.artifacts_dir,
            self.reports_dir,
            self.figures_dir,
            self.exports_dir,
            self.demo_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class BacktestConfig:
    n_origins: int
    origin_step: int
    holdout_weeks: int
    n_validation_origins: int
    use_future_prices: bool
    selection_metric: str = "mean_wape_levels"


@dataclass(frozen=True)
class WorkingCapitalConfig:
    cogs_ratio: dict[str, float]
    lead_time_weeks: int
    service_levels: list[float]
    annual_carrying_cost_rate: float
    compare_methods: list[str]


@dataclass(frozen=True)
class Config:
    seed: int
    paths: Paths
    data: dict[str, Any]
    fiscal: dict[str, Any]
    horizon: int
    season_length: int
    backtest: BacktestConfig
    models: dict[str, Any]
    intervals: dict[str, Any]
    reconciliation: dict[str, Any]
    planning: dict[str, Any]
    working_capital: WorkingCapitalConfig
    scenarios: dict[str, Any]
    quality: dict[str, Any]
    check: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def interval_levels(self) -> list[int]:
        return [int(x) for x in self.intervals["levels"]]

    @property
    def horizon_buckets(self) -> list[tuple[int, int]]:
        return [(int(a), int(b)) for a, b in self.intervals["horizon_buckets"]]


def _paths(raw: dict[str, Any], root: Path) -> Paths:
    p = raw["paths"]
    return Paths(root=root, **{k: (root / v) for k, v in p.items()})


def from_dict(raw: dict[str, Any], root: Path = ROOT) -> Config:
    """Build a Config from a parsed YAML dict."""
    return Config(
        seed=int(raw["seed"]),
        paths=_paths(raw, root),
        data=raw["data"],
        fiscal=raw["fiscal"],
        horizon=int(raw["forecast"]["horizon"]),
        season_length=int(raw["forecast"]["season_length"]),
        backtest=BacktestConfig(**raw["backtest"]),
        models=raw["models"],
        intervals=raw["intervals"],
        reconciliation=raw["reconciliation"],
        planning=raw["planning"],
        working_capital=WorkingCapitalConfig(**raw["working_capital"]),
        scenarios=raw["scenarios"],
        quality=raw["quality"],
        check=raw.get("check", {}),
        raw=raw,
    )


def load_config(path: str | Path | None = None, quick: bool = False) -> Config:
    """Read config.yaml. `quick=True` applies the `check` overrides (fast real-data run)."""
    path = Path(path) if path else ROOT / "config.yaml"
    raw = yaml.safe_load(path.read_text())
    if quick:
        raw = apply_quick(raw)
    return from_dict(raw, root=path.resolve().parent)


def apply_quick(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the config with the fast `check` settings applied."""
    raw = copy.deepcopy(raw)
    chk = raw.get("check", {})
    raw["data"]["stores"] = chk.get("stores")
    raw["backtest"]["n_origins"] = chk.get("n_origins", raw["backtest"]["n_origins"])
    raw["backtest"]["n_validation_origins"] = chk.get(
        "n_validation_origins", raw["backtest"]["n_validation_origins"]
    )
    raw["models"]["stats"] = chk.get("stats", raw["models"]["stats"])
    if "lightgbm_grid" in chk:
        raw["models"]["lightgbm"]["grid"] = chk["lightgbm_grid"]
    for key in ("processed_dir", "artifacts_dir", "duckdb_file"):
        raw["paths"][key] = raw["paths"][key].replace("data/processed", "data/processed/check")
    raw["paths"]["reports_dir"] = "reports/check"
    raw["paths"]["figures_dir"] = "reports/check/figures"
    raw["paths"]["exports_dir"] = "exports/check"
    raw["paths"]["demo_dir"] = "data/processed/check/demo"
    return raw
