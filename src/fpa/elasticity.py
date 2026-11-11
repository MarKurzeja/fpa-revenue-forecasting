"""Price elasticity per category and simple scenarios (SPEC §9.1).

log(units) = beta * log(price_index / 100) + node fixed effects + node trends
             + yearly Fourier terms + event counts + SNAP days + closure flag + e
on store x department weekly data, estimated by OLS separately per category.
Observational data: prices are not randomised (promotions coincide with demand
shocks), so beta is a descriptive association, not a causal elasticity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fpa.features import EVENT_COLS


def _fourier(t: np.ndarray, k: int = 6, period: float = 365.25 / 7) -> np.ndarray:
    return np.column_stack(
        [f(2 * np.pi * j * t / period) for j in range(1, k + 1) for f in (np.sin, np.cos)]
    )


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and classical standard errors."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    return beta, np.sqrt(np.clip(np.diag(cov), 0, None))


def estimate_elasticities(bottom: pd.DataFrame, weeks: pd.DataFrame) -> pd.DataFrame:
    """bottom: long store x dept x week with t, node_id, cat_id, state_id, units, price_index."""
    w = weeks.set_index("t")
    rows = []
    for cat, g in bottom.groupby("cat_id"):
        g = g[(g["units"] > 0) & (g["price_index"] > 0)].copy()
        t = g["t"].to_numpy()
        snap_col = g["state_id"].str.lower().map(lambda s: f"snap_{s}_days")
        snap = np.array([w.at[ti, c] for ti, c in zip(t, snap_col)], dtype=float)
        nodes = pd.get_dummies(g["node_id"]).to_numpy(float)
        trends = nodes * ((t - t.mean()) / 52.0)[:, None]
        events = np.column_stack([w.loc[t, c].to_numpy(float) for c in EVENT_COLS])
        closure = w.loc[t, "has_closure"].to_numpy(float)[:, None]
        X = np.column_stack(
            [
                np.log(g["price_index"].to_numpy() / 100.0),
                nodes,
                trends,
                _fourier(t),
                events,
                snap[:, None],
                closure,
            ]
        )
        y = np.log(g["units"].to_numpy(float))
        beta, se = ols(X, y)
        rows.append(
            {"cat_id": cat, "elasticity": float(beta[0]), "se": float(se[0]), "n_obs": len(y)}
        )
    return pd.DataFrame(rows)


def revenue_factor(price_change: float, elasticity: float) -> float:
    """Revenue multiplier for a price change: (1 + dp)^(1 + elasticity)."""
    return float((1 + price_change) ** (1 + elasticity))


def event_uplifts(
    y_total: np.ndarray, weeks: pd.DataFrame, events: list[str], window: int = 2
) -> pd.DataFrame:
    """Average revenue uplift of event weeks vs the surrounding non-event weeks."""
    names = weeks["event_names"].fillna("").astype(str).str.split("|")
    has_any = names.apply(lambda xs: any(x for x in xs)).to_numpy()
    rows = []
    for ev in events:
        idx = np.flatnonzero(names.apply(lambda xs, ev=ev: ev in xs).to_numpy())
        ups = []
        for i in idx:
            if i >= len(y_total):
                continue
            nb = [
                j
                for j in range(i - window, i + window + 1)
                if j != i and 0 <= j < len(y_total) and not has_any[j]
            ]
            if nb:
                ups.append(y_total[i] / np.mean(y_total[nb]) - 1)
        rows.append(
            {
                "event": ev,
                "n_occurrences": len(ups),
                "mean_uplift": float(np.mean(ups)) if ups else np.nan,
                "min_uplift": float(np.min(ups)) if ups else np.nan,
                "max_uplift": float(np.max(ups)) if ups else np.nan,
            }
        )
    return pd.DataFrame(rows)
