"""Price-volume-mix revenue bridge (SPEC §8.3).

For a group (store x department) with items i, base period 0 (prior year or budget)
and actual period 1, units q and unit prices p (p = revenue / units):

    Q = sum_i q_i,  m_i = q_i / Q,  P0_bar = sum_i m0_i p0_i = R0 / Q0
    volume = (Q1 - Q0) * P0_bar                 total units at base prices and base mix
    mix    = Q1 * sum_i (m1_i - m0_i) * p0_i    mix shift within the group at base prices
    price  = sum_i q1_i * (p1_i - p0_i)         price change on actual units

volume + mix + price = R1 - R0 exactly. Items new in period 1 get p0 := p1 (no price
effect; their revenue shows up as volume/mix). If Q0 = 0 the whole revenue is mix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Bridge:
    base: float
    volume: float
    mix: float
    price: float
    actual: float

    @property
    def total_variance(self) -> float:
        return self.actual - self.base


def bridge(q0: np.ndarray, r0: np.ndarray, q1: np.ndarray, r1: np.ndarray) -> Bridge:
    """PVM bridge for one group. Arrays are aligned by item."""
    q0, r0, q1, r1 = (np.asarray(x, dtype=float) for x in (q0, r0, q1, r1))
    with np.errstate(invalid="ignore", divide="ignore"):
        p0 = np.where(q0 > 0, r0 / q0, np.nan)
        p1 = np.where(q1 > 0, r1 / q1, np.nan)
    p0 = np.where(np.isnan(p0), np.where(np.isnan(p1), 0.0, p1), p0)
    p1 = np.where(np.isnan(p1), p0, p1)
    Q0, Q1 = q0.sum(), q1.sum()
    R0 = float(np.sum(q0 * p0))
    R1 = float(np.sum(q1 * p1))
    m0 = q0 / Q0 if Q0 > 0 else np.zeros_like(q0)
    m1 = q1 / Q1 if Q1 > 0 else np.zeros_like(q1)
    p0_bar = R0 / Q0 if Q0 > 0 else 0.0
    volume = (Q1 - Q0) * p0_bar
    mix = Q1 * float(np.sum((m1 - m0) * p0))
    price = float(np.sum(q1 * (p1 - p0)))
    return Bridge(base=R0, volume=float(volume), mix=mix, price=price, actual=R1)


def bridge_by_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Bridge per group. df: one row per (group, item) with q0, r0, q1, r1."""
    rows = []
    for key, g in df.groupby(group_cols, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        b = bridge(g["q0"], g["r0"], g["q1"], g["r1"])
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "base": b.base,
                "volume": b.volume,
                "mix": b.mix,
                "price": b.price,
                "actual": b.actual,
                "variance": b.total_variance,
            }
        )
    return pd.DataFrame(rows)


def roll_up(bridges: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Sum group bridges to a coarser level (mix stays 'within department')."""
    cols = ["base", "volume", "mix", "price", "actual", "variance"]
    if not by:
        return bridges[cols].sum().to_frame().T
    return bridges.groupby(by, as_index=False)[cols].sum()
