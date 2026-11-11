"""Fiscal calendar on Walmart weeks (SPEC §3.2).

Walmart's fiscal year ends on 31 January; `fiscal_year` is the calendar year in which
it ends (FY2016 = Feb 2015 .. Jan 2016). Whole Walmart weeks (Saturday..Friday) are
assigned to a fiscal year and month by one anchor day of the week (default Tuesday,
the middle day, so a week goes where most of its days are). Daily rows inherit the
fiscal attributes of their week, so daily and weekly reporting always agree.
"""

from __future__ import annotations

import pandas as pd

PATTERN_445 = [4, 4, 5] * 4


def fiscal_year_month(dates: pd.Series, year_end_month: int = 1) -> pd.DataFrame:
    """Calendar-month fiscal year and month for each date."""
    d = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    m = d.dt.month
    fy = d.dt.year + (m > year_end_month).astype(int)
    fm = (m - (year_end_month + 1)) % 12 + 1
    return pd.DataFrame({"fiscal_year": fy.astype(int), "fiscal_month": fm.astype(int)})


def weekly_fiscal_calendar(
    calendar: pd.DataFrame,
    year_end_month: int = 1,
    month_mode: str = "calendar",
    week_anchor_day: int = 3,
) -> pd.DataFrame:
    """One row per `wm_yr_wk` with week bounds and fiscal attributes.

    `calendar` needs columns `date` and `wm_yr_wk` (daily). Weeks with fewer than
    7 days are kept here but flagged `is_full_week=False`.
    """
    cal = calendar[["date", "wm_yr_wk"]].copy()
    cal["date"] = pd.to_datetime(cal["date"])
    wk = (
        cal.groupby("wm_yr_wk", as_index=False)
        .agg(week_start=("date", "min"), week_end=("date", "max"), n_days=("date", "count"))
        .sort_values("wm_yr_wk")
        .reset_index(drop=True)
    )
    # Anchor from the theoretical Saturday start, so partial weeks anchor consistently.
    theoretical_start = wk["week_end"] - pd.to_timedelta(6, unit="D")
    starts = wk["week_start"].where(wk["n_days"] == 7, theoretical_start)
    wk.loc[wk["n_days"] < 7, "week_start"] = starts
    anchor = starts + pd.to_timedelta(week_anchor_day, unit="D")
    wk["anchor_date"] = anchor
    fym = fiscal_year_month(anchor, year_end_month)
    wk["fiscal_year"] = fym["fiscal_year"].to_numpy()
    wk["fiscal_week"] = wk.groupby("fiscal_year").cumcount() + 1
    if month_mode == "calendar":
        wk["fiscal_month"] = fym["fiscal_month"].to_numpy()
    elif month_mode == "445":
        wk["fiscal_month"] = wk["fiscal_week"].map(_month_445)
    else:
        raise ValueError(f"unknown fiscal.month_mode: {month_mode}")
    wk["fiscal_quarter"] = (wk["fiscal_month"] - 1) // 3 + 1
    wk["is_full_week"] = wk["n_days"] == 7
    wk["fiscal_period"] = (
        "FY" + wk["fiscal_year"].astype(str) + "-M" + wk["fiscal_month"].astype(str).str.zfill(2)
    )
    return wk


def _month_445(fiscal_week: int) -> int:
    """Fiscal month for a week number under a 4-4-5 pattern (week 53 joins month 12)."""
    cum = 0
    for month, n in enumerate(PATTERN_445, start=1):
        cum += n
        if fiscal_week <= cum:
            return month
    return 12


def full_fiscal_years(weeks: pd.DataFrame, year_end_month: int = 1) -> list[int]:
    """Fiscal years fully covered by full weeks in the data, in ascending order."""
    out = []
    for fy, w in weeks.groupby("fiscal_year"):
        fy_end = pd.Timestamp(year=int(fy), month=year_end_month, day=1) + pd.offsets.MonthEnd(0)
        fy_start = fy_end - pd.DateOffset(years=1) + pd.Timedelta(days=1)
        first_gap = (w["anchor_date"].min() - fy_start).days
        last_gap = (fy_end - w["anchor_date"].max()).days
        if w["is_full_week"].all() and len(w) >= 52 and first_gap < 7 and last_gap < 7:
            out.append(int(fy))
    return sorted(out)
