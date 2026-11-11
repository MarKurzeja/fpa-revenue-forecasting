"""Weekly revenue panel for every hierarchy node, loaded from DuckDB."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fpa.hierarchy import Hierarchy, build_hierarchy


@dataclass
class Panel:
    """Revenue (and units, price-index parts) for all nodes x full weeks.

    Arrays are (n_nodes, T); column t matches row t of `weeks` (dim_week).
    """

    h: Hierarchy
    y: np.ndarray
    units: np.ndarray
    price_num: np.ndarray
    price_den: np.ndarray
    weeks: pd.DataFrame

    @property
    def T(self) -> int:
        return self.y.shape[1]

    @property
    def nodes(self) -> pd.DataFrame:
        return self.h.nodes

    @property
    def price_index(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return 100.0 * self.price_num / self.price_den

    def long(self, value: str = "y") -> pd.DataFrame:
        """Long frame (node_id, level, t, wm_yr_wk, value)."""
        arr = getattr(self, value)
        n, T = arr.shape
        return pd.DataFrame(
            {
                "node_id": np.repeat(self.nodes["node_id"].to_numpy(), T),
                "level": np.repeat(self.nodes["level"].to_numpy(), T),
                "t": np.tile(np.arange(T), n),
                "wm_yr_wk": np.tile(self.weeks["wm_yr_wk"].to_numpy(), n),
                value: arr.reshape(-1),
            }
        )


def panel_from_frames(bottom: pd.DataFrame, weeks: pd.DataFrame) -> Panel:
    """Build a Panel from a bottom-level long frame and the weekly dimension.

    `bottom` columns: wm_yr_wk, store_id, state_id, dept_id, cat_id, revenue, units,
    and optionally laspeyres_num, laspeyres_den.
    """
    weeks = weeks.sort_values("wm_yr_wk").reset_index(drop=True)
    if "t" not in weeks:
        weeks["t"] = np.arange(len(weeks))
    h = build_hierarchy(bottom)
    t_of = dict(zip(weeks["wm_yr_wk"], weeks["t"]))
    b = bottom[bottom["wm_yr_wk"].isin(t_of)].copy()
    b["t"] = b["wm_yr_wk"].map(t_of).astype(int)
    key = h.bottom[["store_id", "dept_id"]].reset_index().rename(columns={"index": "j"})
    b = b.merge(key, on=["store_id", "dept_id"], how="inner")
    T, nb = len(weeks), h.n_bottom

    def mat(col: str) -> np.ndarray:
        m = np.zeros((nb, T))
        if col in b:
            m[b["j"].to_numpy(), b["t"].to_numpy()] = b[col].fillna(0.0).to_numpy(float)
        return m

    return Panel(
        h=h,
        y=h.aggregate(mat("revenue")),
        units=h.aggregate(mat("units")),
        price_num=h.aggregate(mat("laspeyres_num")),
        price_den=h.aggregate(mat("laspeyres_den")),
        weeks=weeks,
    )


def load_panel(db_file: Path | str) -> Panel:
    """Load the panel from the DuckDB build."""
    from fpa.data.build import connect

    con = connect(db_file, read_only=True)
    try:
        weeks = con.execute("SELECT * FROM dim_week ORDER BY t").fetchdf()
        bottom = con.execute(
            """
            SELECT f.wm_yr_wk, f.store_id, s.state_id, f.dept_id, d.cat_id,
                   f.revenue, f.units, f.laspeyres_num, f.laspeyres_den
            FROM fact_revenue_weekly AS f
            JOIN dim_store AS s USING (store_id)
            JOIN dim_dept AS d USING (dept_id)
            """
        ).fetchdf()
    finally:
        con.close()
    return panel_from_frames(bottom, weeks)
