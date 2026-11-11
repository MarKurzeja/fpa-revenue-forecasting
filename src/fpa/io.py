"""Small persistence helpers for the Parquet / pickle cache."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


def _has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True


def save_table(df: pd.DataFrame, path: Path) -> Path:
    """Write Parquet (or pickle if pyarrow is unavailable). Returns the written path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if _has_pyarrow():
        p = path.with_suffix(".parquet")
        df.to_parquet(p, index=False)
    else:
        p = path.with_suffix(".pkl")
        df.to_pickle(p)
    return p


def load_table(path: Path) -> pd.DataFrame:
    for suffix in (".parquet", ".pkl"):
        p = path.with_suffix(suffix)
        if p.exists():
            return pd.read_parquet(p) if suffix == ".parquet" else pd.read_pickle(p)
    raise FileNotFoundError(path)


def table_exists(path: Path) -> bool:
    return any(path.with_suffix(s).exists() for s in (".parquet", ".pkl"))


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())
