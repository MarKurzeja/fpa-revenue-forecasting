"""DuckDB build + data-quality checks on a tiny hand-written M5-format fixture.

Skipped when duckdb is not installed. The fixture is a few rows in the M5 file
layout, written to a temp folder; it exists only to test the SQL.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from helpers import tiny_config

duckdb = pytest.importorskip("duckdb")

ITEMS = [
    ("FOODS_1_001", "FOODS_1", "FOODS"),
    ("FOODS_1_002", "FOODS_1", "FOODS"),
    ("HOBBIES_1_001", "HOBBIES_1", "HOBBIES"),
]
STORES = [("CA_1", "CA"), ("TX_1", "TX")]
N_WEEKS = 60
N_DAYS = 7 * N_WEEKS + 2  # last week is partial and must be dropped
PRICE_UP_WEEK = 53  # first week of FY2013 in this fixture: +10% on every price


def write_fixture(raw, break_with: str | None = None) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    days = pd.date_range("2011-01-29", periods=N_DAYS + 28, freq="D")
    cal = pd.DataFrame(
        {
            "date": days.strftime("%Y-%m-%d"),
            "wm_yr_wk": [11101 + i // 7 for i in range(len(days))],
            "weekday": days.day_name(),
            "wday": [(i % 7) + 1 for i in range(len(days))],
            "month": days.month,
            "year": days.year,
            "d": [f"d_{i + 1}" for i in range(len(days))],
            "event_name_1": "",
            "event_type_1": "",
            "event_name_2": "",
            "event_type_2": "",
            "snap_CA": (days.day <= 10).astype(int),
            "snap_TX": 0,
            "snap_WI": 0,
        }
    )
    xmas = (days.month == 12) & (days.day == 25)
    cal.loc[xmas, ["event_name_1", "event_type_1"]] = ["Christmas", "National"]
    cal.to_csv(raw / "calendar.csv", index=False)

    rng = np.random.default_rng(0)
    sales, prices = [], []
    for store, state in STORES:
        for k, (item, dept, cat) in enumerate(ITEMS):
            units = rng.integers(0, 5, N_DAYS)
            first_week = 3 if item.startswith("HOBBIES") else 0  # not listed before week 3
            units[: first_week * 7] = 0
            sales.append(
                {
                    "id": f"{item}_{store}_evaluation",
                    "item_id": item,
                    "dept_id": dept,
                    "cat_id": cat,
                    "store_id": store,
                    "state_id": state,
                    **{f"d_{i + 1}": int(u) for i, u in enumerate(units)},
                }
            )
            base = 2.0 + k
            for w in range(first_week, N_WEEKS + 1):
                p = base * (1.1 if w >= PRICE_UP_WEEK else 1.0)
                prices.append(
                    {
                        "store_id": store,
                        "item_id": item,
                        "wm_yr_wk": 11101 + w,
                        "sell_price": round(p, 4),
                    }
                )
    sales_df, prices_df = pd.DataFrame(sales), pd.DataFrame(prices)
    if break_with == "duplicate":
        prices_df = pd.concat([prices_df, prices_df.iloc[[5]]])
    if break_with == "missing_price":
        prices_df = prices_df[
            ~((prices_df["item_id"] == "FOODS_1_001") & (prices_df["wm_yr_wk"] == 11110))
        ]
        sales_df.loc[0, [f"d_{i}" for i in range(64, 71)]] = 3
    sales_df.to_csv(raw / "sales_train_evaluation.csv", index=False)
    prices_df.to_csv(raw / "sell_prices.csv", index=False)


def _cfg(tmp_path):
    cfg = tiny_config(tmp_path)
    cfg.data["expected"] = {**cfg.data["expected"], "n_items": 3, "n_stores": 2}
    return cfg


def test_build_passes_checks_and_facts(tmp_path):
    from fpa.data.build import build, connect, data_facts

    cfg = _cfg(tmp_path)
    write_fixture(tmp_path / "raw")
    db = build(cfg, raw_dir=tmp_path / "raw", db_file=tmp_path / "ok.duckdb")
    con = connect(db)
    con.execute("DROP TABLE stg_sales_wide")
    facts = data_facts(con)
    assert facts["n_series"] == 6 and facts["n_stores"] == 2 and facts["n_depts"] == 2
    assert facts["n_days"] == N_DAYS
    assert facts["n_full_weeks"] == N_WEEKS and facts["dropped_partial_weeks"] >= 1

    # Revenue = units x price, with not-yet-listed days at zero revenue
    d = con.execute(
        "SELECT SUM(revenue), SUM(units) FROM fact_sales_daily WHERE NOT is_listed"
    ).fetchone()
    assert d == (0.0, 0)
    assert con.execute("SELECT COUNT(*) FROM dim_date WHERE is_closure").fetchone()[0] == 1

    # Laspeyres index: 100 in the base fiscal year (constant prices), 110 after +10%
    pi = con.execute(
        "SELECT w.t, p.price_index FROM price_index_weekly p JOIN dim_week w USING (wm_yr_wk) "
        "WHERE p.store_id = 'CA_1' AND p.dept_id = 'FOODS_1' ORDER BY w.t"
    ).fetchdf()
    assert np.allclose(pi.loc[pi["t"] < PRICE_UP_WEEK, "price_index"], 100.0)
    assert np.allclose(pi.loc[pi["t"] >= PRICE_UP_WEEK, "price_index"], 110.0)
    fy = con.execute("SELECT fiscal_year, fiscal_month FROM dim_week WHERE t = 0").fetchone()
    assert fy == (2012, 1)
    con.close()


def test_step_build_rebuilds_empty_existing_database(tmp_path):
    from dataclasses import replace

    import fpa.pipeline as pipeline
    from fpa.data.build import connect

    cfg = _cfg(tmp_path)
    raw = tmp_path / "raw"
    write_fixture(raw)
    db = tmp_path / "empty.duckdb"
    duckdb.connect(str(db)).close()
    cfg = replace(
        cfg,
        paths=replace(cfg.paths, root=tmp_path, raw_dir=raw, duckdb_file=db),
    )

    pipeline.step_build(cfg)

    con = connect(db, read_only=True)
    try:
        assert con.execute("SELECT COUNT(*) FROM fact_revenue_weekly").fetchone()[0] > 0
    finally:
        con.close()
    assert (tmp_path / "docs" / "DATA_FACTS.md").exists()


def test_build_calendar_without_day_id_column(tmp_path):
    from fpa.data.build import build, connect

    cfg = _cfg(tmp_path)
    raw = tmp_path / "raw"
    write_fixture(raw)
    calendar_path = raw / "calendar.csv"
    pd.read_csv(calendar_path).drop(columns="d").to_csv(calendar_path, index=False)
    sales_path = raw / "sales_train_evaluation.csv"
    pd.read_csv(sales_path).drop(columns="id").to_csv(sales_path, index=False)

    db = build(cfg, raw_dir=raw, db_file=tmp_path / "no-calendar-day-id.duckdb")
    con = connect(db, read_only=True)
    try:
        assert con.execute(
            "SELECT d FROM dim_date WHERE d_num IN (1, 2, 3) ORDER BY d_num"
        ).fetchall() == [("d_1",), ("d_2",), ("d_3",)]
    finally:
        con.close()


@pytest.mark.parametrize("broken", ["duplicate", "missing_price"])
def test_broken_fixture_fails_checks(tmp_path, broken):
    from fpa.data.build import DataQualityError, build

    cfg = _cfg(tmp_path)
    write_fixture(tmp_path / "raw", break_with=broken)
    with pytest.raises(DataQualityError):
        build(cfg, raw_dir=tmp_path / "raw", db_file=tmp_path / f"{broken}.duckdb")


def test_panel_from_duckdb(tmp_path):
    from fpa.data.build import build
    from fpa.panel import load_panel

    cfg = _cfg(tmp_path)
    write_fixture(tmp_path / "raw")
    db = build(cfg, raw_dir=tmp_path / "raw", db_file=tmp_path / "p.duckdb")
    panel = load_panel(db)
    assert panel.T == N_WEEKS
    assert panel.h.n_bottom == 4
    assert np.allclose(panel.y[panel.h.total_idx], panel.y[panel.h.bottom_idx].sum(axis=0))
