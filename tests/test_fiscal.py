import pandas as pd
import pytest

from fpa.fiscal import fiscal_year_month, full_fiscal_years, weekly_fiscal_calendar


@pytest.mark.parametrize(
    "date, fy, fm",
    [
        ("2015-02-01", 2016, 1),  # first day of FY2016
        ("2016-01-31", 2016, 12),  # last day of FY2016
        ("2015-08-15", 2016, 7),
        ("2011-01-29", 2011, 12),
        ("2015-12-31", 2016, 11),
    ],
)
def test_calendar_month_mapping(date, fy, fm):
    out = fiscal_year_month(pd.Series([pd.Timestamp(date)]))
    assert out["fiscal_year"].iloc[0] == fy
    assert out["fiscal_month"].iloc[0] == fm


def _daily(start="2011-01-29", days=7 * 160 + 2):
    d = pd.date_range(start, periods=days, freq="D")
    wk = [11101 + i // 7 for i in range(days)]
    return pd.DataFrame({"date": d, "wm_yr_wk": wk})


def test_weekly_calendar_quarters_and_partial_week():
    w = weekly_fiscal_calendar(_daily())
    assert w["n_days"].iloc[-1] == 2 and not w["is_full_week"].iloc[-1]
    first = w.iloc[0]  # Sat 2011-01-29, anchor Tue 2011-02-01
    assert (first["fiscal_year"], first["fiscal_month"], first["fiscal_quarter"]) == (2012, 1, 1)
    aug = w[w["week_start"] == pd.Timestamp("2011-08-13")].iloc[0]  # anchor 2011-08-16
    assert (aug["fiscal_year"], aug["fiscal_month"], aug["fiscal_quarter"]) == (2012, 7, 3)
    assert w.groupby("fiscal_year")["fiscal_week"].min().eq(1).all()


def test_445_pattern():
    w = weekly_fiscal_calendar(_daily(), month_mode="445")
    fy = w[w["fiscal_year"] == 2012]
    counts = fy.groupby("fiscal_month").size().tolist()
    assert counts[:11] == [4, 4, 5, 4, 4, 5, 4, 4, 5, 4, 4]
    assert counts[11] in (5, 6)
    with pytest.raises(ValueError):
        weekly_fiscal_calendar(_daily(), month_mode="bad")


def test_full_fiscal_years():
    w = weekly_fiscal_calendar(_daily(days=7 * 120))
    assert full_fiscal_years(w) == [2012, 2013]  # FY2014 is cut off in May 2013
