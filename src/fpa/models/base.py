"""Common forecaster interface.

Every model forecasts all hierarchy nodes at once from history `y[:, :origin]` and
returns point forecasts for horizons 1..h, optional native intervals, and in-sample
one-step fitted values (used to estimate the MinT covariance).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from fpa.features import Calendar
from fpa.panel import Panel


@dataclass
class ForecastOutput:
    mean: np.ndarray  # (n_nodes, h)
    fitted: np.ndarray | None = None  # (n_nodes, origin), NaN where unavailable
    lo: dict[int, np.ndarray] = field(default_factory=dict)  # level -> (n_nodes, h)
    hi: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def has_native_intervals(self) -> bool:
        return bool(self.lo)


@dataclass
class ForecastContext:
    """What a model may see at an origin. `history` truncates every observed series."""

    panel: Panel
    calendar: Calendar  # known-in-advance calendar, extended beyond the data
    levels: list[int]
    season_length: int
    seed: int
    use_future_prices: bool = False

    def y_hist(self, origin: int) -> np.ndarray:
        return self.panel.y[:, :origin]


class Forecaster(Protocol):
    name: str
    family: str

    def forecast(self, ctx: ForecastContext, origin: int, h: int) -> ForecastOutput: ...
