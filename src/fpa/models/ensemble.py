"""Simple-average ensemble of the top models chosen on validation (SPEC §6.4)."""

from __future__ import annotations

import numpy as np

from fpa.models.base import ForecastOutput


def average_outputs(outputs: list[ForecastOutput]) -> ForecastOutput:
    """Average point forecasts and fitted values. Intervals come from conformal calibration."""
    mean = np.mean([o.mean for o in outputs], axis=0)
    fitted_list = [o.fitted for o in outputs if o.fitted is not None]
    fitted = None
    if fitted_list:
        with np.errstate(invalid="ignore"):
            fitted = np.mean(np.stack(fitted_list), axis=0)  # NaN where any member is NaN
    return ForecastOutput(mean=mean, fitted=fitted)
