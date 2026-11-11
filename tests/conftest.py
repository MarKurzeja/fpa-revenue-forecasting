"""Shared test fixtures.

These are tiny hand-built arrays for unit tests of the maths (not a dataset: the
pipeline itself only ever runs on the real M5 data).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpa.hierarchy import build_hierarchy
from fpa.panel import panel_from_frames

KEYS = pd.DataFrame(
    [
        ("CA", "CA_1", "FOODS", "FOODS_1"),
        ("CA", "CA_1", "HOBBIES", "HOBBIES_1"),
        ("CA", "CA_2", "FOODS", "FOODS_1"),
        ("CA", "CA_2", "HOBBIES", "HOBBIES_1"),
        ("TX", "TX_1", "FOODS", "FOODS_1"),
        ("TX", "TX_1", "FOODS", "FOODS_2"),
        ("TX", "TX_1", "HOBBIES", "HOBBIES_1"),
    ],
    columns=["state_id", "store_id", "cat_id", "dept_id"],
)


@pytest.fixture
def hier():
    return build_hierarchy(KEYS)


def make_weeks(T: int, start: str = "2011-01-29") -> pd.DataFrame:
    ws = pd.date_range(start, periods=T, freq="7D")
    from fpa.fiscal import fiscal_year_month

    fym = fiscal_year_month(ws + pd.Timedelta(days=3))
    w = pd.DataFrame(
        {
            "t": np.arange(T),
            "wm_yr_wk": 11101 + np.arange(T),
            "week_start": ws,
            "week_end": ws + pd.Timedelta(days=6),
        }
    )
    w["fiscal_year"] = fym["fiscal_year"].to_numpy()
    w["fiscal_month"] = fym["fiscal_month"].to_numpy()
    w["fiscal_quarter"] = (w["fiscal_month"] - 1) // 3 + 1
    w["fiscal_week"] = w.groupby("fiscal_year").cumcount() + 1
    w["fiscal_period"] = (
        "FY" + w["fiscal_year"].astype(str) + "-M" + w["fiscal_month"].astype(str).str.zfill(2)
    )
    for c in ("n_sporting", "n_cultural", "n_national", "n_religious"):
        w[c] = 0
    woy = (w["week_start"] + pd.Timedelta(days=3)).dt.isocalendar().week.to_numpy()
    w["event_names"] = np.where(woy == 52, "Christmas", np.where(woy == 47, "Thanksgiving", ""))
    w.loc[woy == 52, "n_religious"] = 1
    w.loc[woy == 47, "n_national"] = 1
    w["snap_ca_days"] = 10
    w["snap_tx_days"] = 9
    w["snap_wi_days"] = 10
    w["has_closure"] = (woy == 52).astype(int)
    return w


def make_bottom(T: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    t = np.arange(T)
    for j, k in KEYS.iterrows():
        level = 1000 * (j + 1)
        season = 1 + 0.2 * np.sin(2 * np.pi * t / 52.18) + 0.3 * ((t % 52) == 47)
        rev = level * (1 + 0.002 * t) * season * rng.lognormal(0, 0.05, T)
        price = 100 * (1 + 0.0005 * t)
        rows.append(
            pd.DataFrame(
                {
                    "wm_yr_wk": 11101 + t,
                    **k.to_dict(),
                    "revenue": rev,
                    "units": rev / 3.0,
                    "laspeyres_num": price * 10.0,
                    "laspeyres_den": 1000.0,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def make_items(weeks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, w in weeks.iterrows():
        for store, state in (("CA_1", "CA"), ("TX_1", "TX")):
            for k in range(3):
                units = 10 + k + (w["t"] % 5)
                price = 2.0 + k + 0.01 * w["t"]
                rows.append(
                    {
                        "wm_yr_wk": w["wm_yr_wk"],
                        "store_id": store,
                        "state_id": state,
                        "item_id": f"FOODS_1_00{k}",
                        "dept_id": "FOODS_1",
                        "cat_id": "FOODS",
                        "units": units,
                        "revenue": units * price,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def small_panel():
    T = 170
    return panel_from_frames(make_bottom(T), make_weeks(T))
