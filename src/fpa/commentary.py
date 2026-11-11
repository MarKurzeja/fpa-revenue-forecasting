"""Deterministic, template-based management commentary (SPEC §8.4).

Every number comes straight from the variance and PVM tables; rules pick which
drivers to mention (largest absolute effects first).
"""

from __future__ import annotations

import pandas as pd


def fmt_usd(x: float, signed: bool = False) -> str:
    """USD with M/K suffix, e.g. 'USD 41.2M', '-USD 1.6M', '+USD 350K'."""
    sign = "-" if x < 0 else ("+" if signed and x > 0 else "")
    a = abs(x)
    if a >= 1e9:
        body = f"{a / 1e9:.2f}bn"
    elif a >= 1e6:
        body = f"{a / 1e6:.1f}M"
    elif a >= 1e4:
        body = f"{a / 1e3:.0f}K"
    elif a >= 1e3:
        body = f"{a / 1e3:.1f}K"
    else:
        body = f"{a:.0f}"
    return f"{sign}USD {body}"


def fmt_pct(x: float, signed: bool = False) -> str:
    s = f"{abs(x) * 100:.1f}%"
    if signed:
        return ("+" if x >= 0 else "-") + s
    return s


def _direction(x: float, up: str = "above", down: str = "below") -> str:
    return up if x >= 0 else down


def month_commentary(
    fiscal_year: int,
    fiscal_month: int,
    actual: float,
    budget: float,
    pvm_total: dict[str, float],
    pvm_segments: pd.DataFrame,
    outlook_gap_pct: float | None,
    prev_outlook_gap_pct: float | None,
) -> str:
    """One paragraph for one fiscal month at company level.

    pvm_segments: rows per (state_id, cat_id) with volume, mix, price (actual vs budget).
    """
    var = actual - budget
    var_pct = var / budget if budget else 0.0
    parts = [
        f"Fiscal month {fiscal_month} (FY{fiscal_year}) revenue was {fmt_usd(actual)}, "
        f"{fmt_pct(var_pct)} {_direction(var)} budget ({fmt_usd(var, signed=True)})."
    ]
    effects = sorted(("volume", "mix", "price"), key=lambda k: -abs(pvm_total.get(k, 0.0)))
    for k in effects[:2]:
        v = pvm_total.get(k, 0.0)
        if abs(v) < 0.05 * max(abs(var), 1.0) and k != effects[0]:
            continue
        seg = ""
        if not pvm_segments.empty and k in pvm_segments:
            same = pvm_segments[pvm_segments[k] * v > 0]
            if not same.empty:
                top = same.loc[same[k].abs().idxmax()]
                seg = f", mostly {top['cat_id']} in {top['state_id']}"
        if k == "price":
            verb = "added" if v >= 0 else "took away"
            parts.append(f"Price {verb} {fmt_usd(abs(v))}{seg}.")
        else:
            parts.append(f"{k.capitalize()} explains {fmt_usd(v, signed=True)}{seg}.")
    if outlook_gap_pct is not None:
        s = (
            f"The full-year outlook is now {fmt_pct(outlook_gap_pct)} "
            f"{_direction(outlook_gap_pct)} budget"
        )
        if prev_outlook_gap_pct is not None:
            change = outlook_gap_pct - prev_outlook_gap_pct
            word = "up" if change >= 0 else "down"
            prev = f"{fmt_pct(prev_outlook_gap_pct)} {_direction(prev_outlook_gap_pct)}"
            s += f", {word} from {prev} last month"
        parts.append(s + ".")
    return " ".join(parts)
