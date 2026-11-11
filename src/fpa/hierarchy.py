"""Grouped revenue hierarchy and its summing matrix (SPEC §5.1).

Levels: L1 total, L2 state, L3 store, L4 category, L5 department, L6 state x category,
L7 store x category, L8 store x department (bottom). Store and product dimensions
cross, so this is a grouped (not strictly nested) hierarchy. The summing matrix S
maps bottom series to every node: y_all = S @ y_bottom.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LEVELS: dict[int, tuple[str, tuple[str, ...]]] = {
    1: ("Total", ()),
    2: ("State", ("state_id",)),
    3: ("Store", ("store_id",)),
    4: ("Category", ("cat_id",)),
    5: ("Department", ("dept_id",)),
    6: ("State x Category", ("state_id", "cat_id")),
    7: ("Store x Category", ("store_id", "cat_id")),
    8: ("Store x Department", ("store_id", "dept_id")),
}
BOTTOM_LEVEL = 8
ATTRS = ("state_id", "store_id", "cat_id", "dept_id")


@dataclass(frozen=True)
class Hierarchy:
    nodes: pd.DataFrame  # one row per node: node_id, level, level_name, state_id..dept_id
    S: np.ndarray  # (n_nodes, n_bottom)
    bottom: pd.DataFrame  # bottom keys in column order of S

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_bottom(self) -> int:
        return len(self.bottom)

    @property
    def bottom_idx(self) -> np.ndarray:
        return np.flatnonzero(self.nodes["level"].to_numpy() == BOTTOM_LEVEL)

    @property
    def total_idx(self) -> int:
        return int(np.flatnonzero(self.nodes["level"].to_numpy() == 1)[0])

    def level_idx(self, level: int) -> np.ndarray:
        return np.flatnonzero(self.nodes["level"].to_numpy() == level)

    def aggregate(self, y_bottom: np.ndarray) -> np.ndarray:
        """Sum bottom-level values (n_bottom, ...) to all nodes (n_nodes, ...)."""
        return np.tensordot(self.S, y_bottom, axes=(1, 0))

    def node_index(self, node_id: str) -> int:
        return int(self.nodes.index[self.nodes["node_id"] == node_id][0])


def node_id(level: int, row: pd.Series | dict) -> str:
    """Readable node id, e.g. 'Total', 'CA', 'CA_1/FOODS_1'."""
    cols = LEVELS[level][1]
    if not cols:
        return "Total"
    return "/".join(str(row[c]) for c in cols)


def build_hierarchy(bottom: pd.DataFrame) -> Hierarchy:
    """Build nodes and S from bottom keys (columns state_id, store_id, cat_id, dept_id)."""
    missing = set(ATTRS) - set(bottom.columns)
    if missing:
        raise ValueError(f"bottom keys missing columns: {sorted(missing)}")
    b = (
        bottom[list(ATTRS)]
        .drop_duplicates()
        .sort_values(["store_id", "dept_id"])
        .reset_index(drop=True)
    )
    rows, s_rows = [], []
    for level, (name, cols) in LEVELS.items():
        groups = [((), b)] if not cols else list(b.groupby(list(cols), sort=True))
        for key, g in groups:
            key = key if isinstance(key, tuple) else (key,)
            attrs = dict.fromkeys(ATTRS)
            attrs.update(dict(zip(cols, key)))
            if "store_id" in cols:  # a store implies its state
                attrs["state_id"] = g["state_id"].iloc[0]
            if "dept_id" in cols:  # a department implies its category
                attrs["cat_id"] = g["cat_id"].iloc[0]
            rows.append(
                {"node_id": node_id(level, attrs), "level": level, "level_name": name, **attrs}
            )
            s = np.zeros(len(b))
            s[g.index.to_numpy()] = 1.0
            s_rows.append(s)
    nodes = pd.DataFrame(rows)
    return Hierarchy(nodes=nodes, S=np.vstack(s_rows), bottom=b)


def is_coherent(h: Hierarchy, y_all: np.ndarray, rtol: float = 1e-6) -> bool:
    """True if every node equals the sum of its bottom series (to rounding)."""
    y_bottom = y_all[h.bottom_idx]
    implied = h.aggregate(y_bottom)
    scale = np.maximum(1.0, np.abs(implied))
    return bool(np.all(np.abs(implied - y_all) <= rtol * scale))


def store_path_parents(h: Hierarchy) -> dict[int, list[int]]:
    """For each bottom node, its ancestors on the store path: total -> state -> store.

    Used by top-down forecast proportions, which need a strict tree.
    """
    nodes = h.nodes
    idx = {nid: i for i, nid in enumerate(nodes["node_id"])}
    out = {}
    for j in h.bottom_idx:
        r = nodes.iloc[j]
        out[int(j)] = [idx["Total"], idx[r["state_id"]], idx[r["store_id"]]]
    return out
