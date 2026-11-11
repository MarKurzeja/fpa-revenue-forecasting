"""Hierarchical reconciliation (SPEC §6.6): BottomUp, TopDown, MinTrace.

All methods return coherent forecasts y_tilde = S b for every node. Implemented in
numpy (D-09) so the maths is explicit and unit-tested:
- bottom_up: b = base bottom forecasts.
- top_down: forecast proportions along the store path (total -> state -> store ->
  store x department), the strict tree inside the grouped hierarchy.
- mint_ols: G = (S'S)^-1 S'.
- mint_shrink: W = Schafer-Strimmer shrinkage of the in-sample one-step residual
  covariance towards its diagonal (Wickramasuriya et al. 2019). Falls back to
  structural scaling (W = diag(S 1)) when residuals are too few.
"""

from __future__ import annotations

import numpy as np

from fpa.hierarchy import Hierarchy, store_path_parents


def bottom_up(h: Hierarchy, yhat: np.ndarray) -> np.ndarray:
    """yhat: (n_nodes, H) base forecasts -> coherent (n_nodes, H)."""
    return h.S @ yhat[h.bottom_idx]


def top_down(h: Hierarchy, yhat: np.ndarray) -> np.ndarray:
    """Top-down with forecast proportions along the store path."""
    y = np.maximum(yhat, 0.0)
    nodes = h.nodes
    paths = store_path_parents(h)
    bottom = np.zeros((h.n_bottom, y.shape[1]))
    # children sums at each step of the path
    state_of = nodes["state_id"].to_numpy()
    store_of = nodes["store_id"].to_numpy()
    lev = nodes["level"].to_numpy()
    state_nodes = np.flatnonzero(lev == 2)
    store_nodes = np.flatnonzero(lev == 3)
    sum_states = y[state_nodes].sum(axis=0)
    for k, j in enumerate(h.bottom_idx):
        tot, st, so = paths[int(j)]
        sum_stores = y[store_nodes[state_of[store_nodes] == state_of[st]]].sum(axis=0)
        sib = h.bottom_idx[store_of[h.bottom_idx] == store_of[so]]
        sum_bottom = y[sib].sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            p = (
                np.where(sum_states > 0, y[st] / sum_states, 0.0)
                * np.where(sum_stores > 0, y[so] / sum_stores, 0.0)
                * np.where(sum_bottom > 0, y[j] / sum_bottom, 0.0)
            )
        bottom[k] = p * y[tot]
    return h.S @ bottom


def mint_matrix(S: np.ndarray, W: np.ndarray) -> np.ndarray:
    """G = (S' W^-1 S)^-1 S' W^-1, so that y_tilde = S G yhat."""
    Winv = np.linalg.pinv(W)
    StW = S.T @ Winv
    return np.linalg.solve(StW @ S, StW)


def shrink_covariance(residuals: np.ndarray) -> tuple[np.ndarray, float]:
    """Schafer-Strimmer shrinkage of the covariance towards its diagonal.

    residuals: (n_obs, n_series). Returns (W, lambda).
    """
    E = residuals - residuals.mean(axis=0)
    n = E.shape[0]
    cov = E.T @ E / n
    sd = np.sqrt(np.diag(cov))
    sd = np.where(sd > 0, sd, 1.0)
    X = E / sd
    w_bar = X.T @ X / n
    corr = w_bar * n / (n - 1)
    sum_w2 = (X**2).T @ (X**2)
    var_r = n / (n - 1) ** 3 * (sum_w2 - n * w_bar**2)
    off = ~np.eye(cov.shape[0], dtype=bool)
    denom = np.sum(corr[off] ** 2)
    lam = float(np.clip(np.sum(var_r[off]) / denom, 0.0, 1.0)) if denom > 0 else 1.0
    W = (1 - lam) * cov
    W[np.diag_indices_from(W)] = np.diag(cov)
    W += np.eye(len(W)) * 1e-9 * max(1.0, float(np.mean(np.diag(cov))))
    return W, lam


def mint(
    h: Hierarchy,
    yhat: np.ndarray,
    method: str = "ols",
    residuals: np.ndarray | None = None,
    min_obs: int = 20,
) -> tuple[np.ndarray, dict]:
    """MinTrace reconciliation. residuals: (n_obs, n_nodes) in-sample one-step errors."""
    info: dict = {"method": method}
    if method == "ols":
        W = np.eye(h.n_nodes)
    elif method == "shrink":
        ok = None if residuals is None else np.all(np.isfinite(residuals), axis=1)
        if residuals is None or ok.sum() < min_obs:
            W = np.diag(h.S.sum(axis=1))
            info["fallback"] = "wls_struct"
        else:
            W, lam = shrink_covariance(residuals[ok])
            info["lambda"] = lam
            info["n_obs"] = int(ok.sum())
    elif method == "wls_struct":
        W = np.diag(h.S.sum(axis=1))
    else:
        raise ValueError(f"unknown MinT method {method}")
    G = mint_matrix(h.S, W)
    return h.S @ (G @ yhat), info


def reconcile(
    h: Hierarchy, yhat: np.ndarray, method: str, residuals: np.ndarray | None = None
) -> tuple[np.ndarray, dict]:
    """Dispatch by method name: bottom_up | top_down | mint_ols | mint_shrink."""
    if method == "bottom_up":
        return bottom_up(h, yhat), {"method": method}
    if method == "top_down":
        return top_down(h, yhat), {"method": method}
    if method.startswith("mint_"):
        return mint(h, yhat, method.removeprefix("mint_"), residuals)
    raise ValueError(f"unknown reconciliation method {method}")
