"""Known-in-advance calendar and the LightGBM feature matrix (SPEC §6.3).

Direct multi-horizon strategy: one row per (node, origin o, horizon h). Every
history-based feature is computed from y[:, :o] only; the target is y[o + h - 1].
Calendar, event and SNAP features describe the *target* week (known in advance).
Price features use the last week before the origin unless `use_future_prices`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fpa.fiscal import weekly_fiscal_calendar

HOLIDAYS = ("Christmas", "Thanksgiving", "Easter", "SuperBowl")
EVENT_COLS = ("n_sporting", "n_cultural", "n_national", "n_religious")
STATES = ("CA", "TX", "WI")
CAP = 26  # weeks-to/from holiday features are capped at half a year


@dataclass
class Calendar:
    """Weekly calendar over observed weeks (t < T) and projected future weeks."""

    weeks: pd.DataFrame  # one row per t, including projected rows (is_projected)
    T_obs: int

    def col(self, name: str) -> np.ndarray:
        return self.weeks[name].to_numpy()


def extend_weeks(weeks: pd.DataFrame, n_extra: int, fiscal: dict) -> pd.DataFrame:
    """Append `n_extra` projected weeks after the data.

    Dates and the fiscal calendar are exact. Events and SNAP days are copied from the
    same week one year earlier (52 weeks back): an approximation for moving holidays.
    """
    w = weeks.sort_values("t").reset_index(drop=True).copy()
    w["is_projected"] = False
    if n_extra <= 0:
        return w
    last = w.iloc[-1]
    starts = pd.to_datetime(last["week_start"]) + pd.to_timedelta(
        7 * np.arange(1, n_extra + 1), unit="D"
    )
    daily = pd.DataFrame(
        {
            "date": np.concatenate([pd.date_range(s, periods=7, freq="D") for s in starts]),
            "wm_yr_wk": np.repeat(_next_wm_weeks(int(last["wm_yr_wk"]), n_extra), 7),
        }
    )
    fw = weekly_fiscal_calendar(
        daily,
        int(fiscal["year_end_month"]),
        str(fiscal["month_mode"]),
        int(fiscal["week_anchor_day"]),
    )
    ext = fw[
        [
            "wm_yr_wk",
            "week_start",
            "week_end",
            "fiscal_year",
            "fiscal_quarter",
            "fiscal_month",
            "fiscal_week",
            "fiscal_period",
        ]
    ].copy()
    ext["t"] = int(last["t"]) + 1 + np.arange(n_extra)
    same_fy = ext["fiscal_year"] == int(last["fiscal_year"])
    ext.loc[same_fy, "fiscal_week"] += int(last["fiscal_week"])
    if str(fiscal["month_mode"]) == "445":
        from fpa.fiscal import _month_445

        ext["fiscal_month"] = ext["fiscal_week"].map(_month_445)
        ext["fiscal_quarter"] = (ext["fiscal_month"] - 1) // 3 + 1
        ext["fiscal_period"] = (
            "FY"
            + ext["fiscal_year"].astype(str)
            + "-M"
            + ext["fiscal_month"].astype(str).str.zfill(2)
        )
    copy_cols = [
        *EVENT_COLS,
        "event_names",
        "snap_ca_days",
        "snap_tx_days",
        "snap_wi_days",
        "has_closure",
    ]
    known = {int(r["t"]): {c: r[c] for c in copy_cols} for _, r in w.iterrows()}
    rows = []
    for t in ext["t"]:
        row = known[int(t) - 52] if int(t) - 52 in known else known[max(known)]
        known[int(t)] = row
        rows.append(row)
    for c in copy_cols:
        ext[c] = [r[c] for r in rows]
    ext["is_projected"] = True
    return pd.concat([w, ext], ignore_index=True)


def _next_wm_weeks(last_wm: int, n: int) -> list[int]:
    year, wk = divmod(last_wm, 100)
    out = []
    for _ in range(n):
        wk += 1
        if wk > 52:
            year, wk = year + 1, 1
        out.append(year * 100 + wk)
    return out


def build_calendar(weeks: pd.DataFrame, n_extra: int, fiscal: dict) -> Calendar:
    """Calendar features for observed + projected weeks."""
    w = extend_weeks(weeks, n_extra, fiscal)
    ws = pd.to_datetime(w["week_start"])
    w["week_of_year"] = (ws + pd.Timedelta(days=3)).dt.isocalendar().week.astype(int).to_numpy()
    w["month"] = (ws + pd.Timedelta(days=3)).dt.month.to_numpy()
    names = w["event_names"].fillna("").astype(str)
    t = w["t"].to_numpy()
    for hol in HOLIDAYS:
        is_h = names.str.split("|").apply(lambda xs, hol=hol: hol in xs).to_numpy()
        w[f"is_{hol.lower()}"] = is_h.astype(int)
        w[f"wks_to_{hol.lower()}"], w[f"wks_from_{hol.lower()}"] = _distance_to_events(t, is_h)
    w["is_holiday_week"] = w[[f"is_{h.lower()}" for h in HOLIDAYS]].max(axis=1)
    return Calendar(weeks=w, T_obs=int((~w["is_projected"]).sum()))


def _distance_to_events(t: np.ndarray, is_event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ev = t[is_event]
    to = np.full(len(t), CAP)
    frm = np.full(len(t), CAP)
    for i, ti in enumerate(t):
        nxt = ev[ev >= ti]
        prv = ev[ev <= ti]
        if len(nxt):
            to[i] = min(CAP, nxt[0] - ti)
        if len(prv):
            frm[i] = min(CAP, ti - prv[-1])
    return to, frm


# --------------------------------------------------------------------------- features

STATIC_COLS = ("level", "state_code", "store_code", "cat_code", "dept_code")
CAL_COLS = (
    "week_of_year",
    "month",
    "fiscal_month",
    "fiscal_quarter",
    *EVENT_COLS,
    "has_closure",
    "is_holiday_week",
    *[f"wks_to_{h.lower()}" for h in HOLIDAYS],
    *[f"wks_from_{h.lower()}" for h in HOLIDAYS],
)


def node_static(nodes: pd.DataFrame) -> pd.DataFrame:
    """Integer codes for the static categorical features (-1 = aggregated / not set)."""
    out = pd.DataFrame({"level": nodes["level"].astype(int).to_numpy()})
    for col, code in (
        ("state_id", "state_code"),
        ("store_id", "store_code"),
        ("cat_id", "cat_code"),
        ("dept_id", "dept_code"),
    ):
        vals = nodes[col]
        cats = sorted(v for v in vals.dropna().unique())
        mapping = {v: i for i, v in enumerate(cats)}
        out[code] = vals.map(mapping).fillna(-1).astype(int).to_numpy()
    return out


def snap_weights(nodes: pd.DataFrame) -> np.ndarray:
    """(n_nodes, 3) share of each state's SNAP calendar that applies to a node."""
    states_present = sorted(nodes["state_id"].dropna().unique())
    w = np.zeros((len(nodes), len(STATES)))
    for i, s in enumerate(nodes["state_id"]):
        if isinstance(s, str) and s in STATES:
            w[i, STATES.index(s)] = 1.0
        else:
            for sp in states_present:
                if sp in STATES:
                    w[i, STATES.index(sp)] = 1.0 / len(states_present)
    return w


def price_reference(nodes: pd.DataFrame) -> np.ndarray:
    """Index of the node a node's price is compared with (same product, all stores)."""
    ids = {nid: i for i, nid in enumerate(nodes["node_id"])}
    ref = np.empty(len(nodes), dtype=int)
    for i, r in nodes.reset_index(drop=True).iterrows():
        if isinstance(r["dept_id"], str):
            ref[i] = ids[r["dept_id"]]
        elif isinstance(r["cat_id"], str):
            ref[i] = ids[r["cat_id"]]
        else:
            ref[i] = ids["Total"]
    return ref


@dataclass
class FeatureSpec:
    lags: tuple[int, ...] = (1, 2, 3, 4, 8, 13, 26, 52)
    rolling_windows: tuple[int, ...] = (4, 13, 52)
    transform: str = "log1p"  # log1p | trailing_mean
    scale_window: int = 13
    use_future_prices: bool = False

    @property
    def min_origin(self) -> int:
        return max(max(self.lags), max(self.rolling_windows), self.scale_window) + 1


def make_features(
    y_hist: np.ndarray,
    price_num: np.ndarray,
    price_den: np.ndarray,
    nodes: pd.DataFrame,
    cal: Calendar,
    origins: np.ndarray,
    horizons: np.ndarray,
    spec: FeatureSpec,
    observed_only: bool,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """Build features for all nodes x origins x horizons.

    Parameters
    ----------
    y_hist : (n, n_obs) revenue history. Only columns < n_obs are ever read, so passing
        y[:, :origin] guarantees no leakage.
    price_num, price_den : Laspeyres parts (n, T_any). Without future prices only
        columns < origin of each row are read.
    observed_only : keep only rows whose target week is inside y_hist (training).

    Returns X, target (transformed; NaN when unobserved), scale (to invert), keys.
    """
    n, n_obs = y_hist.shape
    origins = np.asarray(origins, dtype=int)
    horizons = np.asarray(horizons, dtype=int)
    if origins.min() < spec.min_origin:
        raise ValueError(f"origin {origins.min()} < min_origin {spec.min_origin}")
    if origins.max() > n_obs:
        raise ValueError("origin beyond history")

    yt = np.log1p(np.maximum(y_hist, 0)) if spec.transform == "log1p" else y_hist
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(yt, axis=1)], axis=1)
    cs2 = np.concatenate([np.zeros((n, 1)), np.cumsum(yt**2, axis=1)], axis=1)
    csr = np.concatenate([np.zeros((n, 1)), np.cumsum(y_hist, axis=1)], axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        pidx = 100.0 * price_num / price_den
    ref = price_reference(nodes)
    static = node_static(nodes)
    snapw = snap_weights(nodes)
    snap = np.vstack([cal.col("snap_ca_days"), cal.col("snap_tx_days"), cal.col("snap_wi_days")])

    # Grid of (origin, horizon, node), node fastest.
    oo, hh = np.meshgrid(origins, horizons, indexing="ij")
    oo, hh = oo.reshape(-1), hh.reshape(-1)
    tgt = oo + hh - 1
    if observed_only:
        keep = tgt < n_obs
        oo, hh, tgt = oo[keep], hh[keep], tgt[keep]
    if tgt.max() >= len(cal.weeks):
        raise ValueError("calendar too short for requested horizons")
    m = len(oo)
    node = np.tile(np.arange(n), m)
    o_r = np.repeat(oo, n)
    h_r = np.repeat(hh, n)
    t_r = np.repeat(tgt, n)

    # Scale = trailing mean of raw revenue over `scale_window` weeks before the origin.
    scale = (csr[node, o_r] - csr[node, o_r - spec.scale_window]) / spec.scale_window
    scale = np.maximum(scale, 1.0)

    feats: dict[str, np.ndarray] = {"h": h_r, "log_scale": np.log(scale)}
    norm = scale if spec.transform == "trailing_mean" else np.ones_like(scale)
    for k in spec.lags:
        feats[f"lag_{k}"] = yt[node, o_r - k] / norm
    for w in spec.rolling_windows:
        s = cs[node, o_r] - cs[node, o_r - w]
        s2 = cs2[node, o_r] - cs2[node, o_r - w]
        mu = s / w
        sd = np.sqrt(np.maximum(s2 / w - mu**2, 0.0))
        feats[f"roll_mean_{w}"] = mu / norm
        feats[f"roll_std_{w}"] = sd / norm
    # Same week last year / two years ago relative to the target (always before origin
    # for h <= 52; NaN if unavailable).
    for back in (52, 104):
        src = t_r - back
        ok = (src >= 0) & (src < o_r)
        v = np.full(len(src), np.nan)
        v[ok] = yt[node[ok], src[ok]] / norm[ok]
        feats[f"y_tgt_minus_{back}"] = v

    # Price features.
    p_last = pidx[node, o_r - 1]
    p_prev = pidx[node, np.maximum(o_r - 5, 0)]
    p_ref = pidx[ref[node], o_r - 1]
    if spec.use_future_prices:
        p_t = pidx[node, np.minimum(t_r, pidx.shape[1] - 1)]
        feats["price_index"] = p_t
        feats["price_chg_4w"] = p_t / p_last - 1
        feats["price_rel_dept"] = p_t / pidx[ref[node], np.minimum(t_r, pidx.shape[1] - 1)]
    else:
        feats["price_index"] = p_last
        feats["price_chg_4w"] = p_last / p_prev - 1
        feats["price_rel_dept"] = p_last / p_ref

    for c in CAL_COLS:
        feats[c] = cal.col(c)[t_r].astype(float)
    feats["snap_days"] = np.einsum("ij,ji->i", snapw[node], snap[:, t_r])
    for c in STATIC_COLS:
        feats[c] = static[c].to_numpy()[node]

    X = pd.DataFrame(feats)
    for c in STATIC_COLS:
        X[c] = X[c].astype(int)
    target = np.full(len(X), np.nan)
    obs = t_r < n_obs
    raw_t = y_hist[node[obs], t_r[obs]]
    if spec.transform == "log1p":
        target[obs] = np.log1p(np.maximum(raw_t, 0))
    else:
        target[obs] = raw_t / scale[obs]
    keys = pd.DataFrame({"node": node, "origin": o_r, "h": h_r, "t": t_r})
    return X, target, scale, keys


def inverse_target(pred: np.ndarray, scale: np.ndarray, transform: str) -> np.ndarray:
    """Map model output back to revenue (USD)."""
    if transform == "log1p":
        return np.maximum(np.expm1(pred), 0.0)
    return np.maximum(pred * scale, 0.0)
