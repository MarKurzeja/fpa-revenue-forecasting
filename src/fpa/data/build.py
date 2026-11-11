"""Run the versioned SQL build and the data-quality checks (SPEC §3)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fpa.config import ROOT, Config
from fpa.fiscal import weekly_fiscal_calendar

log = logging.getLogger(__name__)

SQL_DIR = ROOT / "sql"
CHECK_DIR = SQL_DIR / "checks"


class DataQualityError(RuntimeError):
    """Raised when a data-quality check returns violating rows."""


@dataclass
class CheckResult:
    name: str
    n_violations: int
    sample: pd.DataFrame


def connect(db_file: Path | str, read_only: bool = False):
    """Open a DuckDB connection (imported lazily so pure modules do not need duckdb)."""
    import duckdb

    if str(db_file) != ":memory:":
        Path(db_file).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_file), read_only=read_only)


def split_statements(sql: str) -> list[str]:
    """Split a SQL script on semicolons at line ends, dropping comments-only chunks."""
    parts = re.split(r";\s*(?:\n|$)", sql)
    out = []
    for p in parts:
        body = "\n".join(line for line in p.splitlines() if not line.strip().startswith("--"))
        if body.strip():
            out.append(p.strip())
    return out


def run_sql_file(con: Any, path: Path, params: dict[str, Any]) -> None:
    """Execute every statement of a templated SQL file."""
    sql = path.read_text().format(**params)
    for stmt in split_statements(sql):
        con.execute(stmt)


def closure_expr(closures: list[list[int]]) -> str:
    """SQL boolean expression flagging closure days (e.g. Christmas)."""
    if not closures:
        return "FALSE"
    parts = [f"(c.month = {m} AND EXTRACT(day FROM c.date) = {d})" for m, d in closures]
    return " OR ".join(parts)


def store_filter(stores: list[str] | None) -> str:
    if not stores:
        return "TRUE"
    quoted = ", ".join("'" + s.replace("'", "''") + "'" for s in stores)
    return f"store_id IN ({quoted})"


def build(cfg: Config, raw_dir: Path | None = None, db_file: Path | str | None = None) -> Path:
    """Build the DuckDB star schema from the M5 CSVs and run all checks."""
    raw_dir = Path(raw_dir or cfg.paths.raw_dir)
    db_file = db_file or cfg.paths.duckdb_file
    con = connect(db_file)
    try:
        stores = cfg.data.get("stores")
        params = {"raw_dir": raw_dir.as_posix(), "store_filter": store_filter(stores)}
        log.info("01_staging")
        run_sql_file(con, SQL_DIR / "01_staging.sql", params)

        n_day_cols = sum(
            1
            for c in con.execute("DESCRIBE stg_sales_wide").fetchdf()["column_name"]
            if re.fullmatch(r"d_\d+", c)
        )
        cal = con.execute(
            "SELECT date, wm_yr_wk FROM stg_calendar WHERE d_num <= ?", [n_day_cols]
        ).fetchdf()
        weeks = weekly_fiscal_calendar(
            cal,
            year_end_month=int(cfg.fiscal["year_end_month"]),
            month_mode=str(cfg.fiscal["month_mode"]),
            week_anchor_day=int(cfg.fiscal["week_anchor_day"]),
        )
        con.register("weeks_df", weeks)
        con.execute("CREATE OR REPLACE TABLE stg_fiscal_weeks AS SELECT * FROM weeks_df")
        con.unregister("weeks_df")

        params.update(
            closure_expr=closure_expr(cfg.data.get("closure_events", [])),
            last_sales_day=n_day_cols,
        )
        for f in sorted(SQL_DIR.glob("0[2-7]_*.sql")):
            log.info(f.stem)
            run_sql_file(con, f, params)
        con.execute("DROP TABLE IF EXISTS stg_sales_long")  # large and no longer needed
        results = run_checks(con, cfg)
        failed = [r for r in results if r.n_violations > 0]
        if failed:
            msg = "; ".join(f"{r.name}: {r.n_violations} violation(s)" for r in failed)
            raise DataQualityError(msg + "\n" + "\n".join(r.sample.to_string() for r in failed))
        log.info("all %d data-quality checks passed", len(results))
    finally:
        con.close()
    return Path(db_file)


def expected_series(cfg: Config) -> int:
    exp = cfg.data["expected"]
    stores = cfg.data.get("stores")
    n_stores = len(stores) if stores else int(exp["n_stores"])
    return int(exp["n_items"]) * n_stores


def run_checks(con: Any, cfg: Config) -> list[CheckResult]:
    """Run every query in sql/checks; each returns the rows that violate it."""
    params = {
        "expected_series": expected_series(cfg),
        "tolerance": float(cfg.data["row_count_tolerance"]),
    }
    results = []
    for f in sorted(CHECK_DIR.glob("*.sql")):
        sql = f.read_text().format(**params)
        df = con.execute(sql).fetchdf()
        results.append(CheckResult(f.stem, len(df), df.head(10)))
        log.info("check %-28s %s", f.stem, "ok" if df.empty else f"FAILED ({len(df)})")
    return results


def data_facts(con: Any) -> dict[str, Any]:
    """Facts about the loaded data, to verify SPEC §2.1."""
    q = lambda s: con.execute(s).fetchone()[0]  # noqa: E731
    return {
        "n_series": q(
            "SELECT COUNT(*) FROM (SELECT DISTINCT store_id, item_id FROM fact_sales_weekly)"
        ),
        "n_items": q("SELECT COUNT(DISTINCT item_id) FROM dim_item"),
        "n_stores": q("SELECT COUNT(*) FROM dim_store"),
        "n_states": q("SELECT COUNT(DISTINCT state_id) FROM dim_store"),
        "n_cats": q("SELECT COUNT(DISTINCT cat_id) FROM dim_dept"),
        "n_depts": q("SELECT COUNT(*) FROM dim_dept"),
        "first_date": str(q("SELECT MIN(date) FROM dim_date")),
        "last_date": str(q("SELECT MAX(date) FROM dim_date")),
        "n_days": q("SELECT COUNT(*) FROM dim_date"),
        "n_full_weeks": q("SELECT COUNT(*) FROM dim_week"),
        "first_full_week": str(q("SELECT MIN(week_start) FROM dim_week")),
        "last_full_week_end": str(q("SELECT MAX(week_end) FROM dim_week")),
        "dropped_partial_weeks": q("SELECT COUNT(*) FROM stg_fiscal_weeks WHERE NOT is_full_week"),
        "total_revenue_usd": float(q("SELECT SUM(revenue) FROM fact_revenue_weekly")),
        "total_units": int(q("SELECT SUM(units) FROM fact_revenue_weekly")),
        "stores": ", ".join(
            r[0] for r in con.execute("SELECT store_id FROM dim_store ORDER BY 1").fetchall()
        ),
        "depts": ", ".join(
            r[0] for r in con.execute("SELECT dept_id FROM dim_dept ORDER BY 1").fetchall()
        ),
    }


def write_data_facts(cfg: Config, db_file: Path | None = None, out: Path | None = None) -> Path:
    """Compare loaded data with the facts in SPEC §2.1 and write docs/DATA_FACTS.md."""
    con = connect(db_file or cfg.paths.duckdb_file, read_only=True)
    try:
        facts = data_facts(con)
    finally:
        con.close()
    exp = cfg.data["expected"]
    lines = [
        "# Data facts (verified from the loaded M5 data)",
        "",
        "Generated by `python run.py build`. Compares SPEC §2.1 (from memory) with the data.",
        "",
        "| Fact | SPEC §2.1 | Loaded data | Match |",
        "|---|---|---|---|",
    ]
    for k, v in facts.items():
        e = exp.get(k, "")
        match = "" if e == "" else ("yes" if str(e) == str(v) else "NO")
        if k == "total_revenue_usd":
            v = f"{v:,.0f}"
        lines.append(f"| {k} | {e} | {v} | {match} |")
    if cfg.data.get("stores"):
        lines += ["", f"_Subset run: stores = {cfg.data['stores']}_"]
    out = out or (cfg.paths.root / "docs" / "DATA_FACTS.md")
    if cfg.data.get("stores"):
        out = cfg.paths.reports_dir / "DATA_FACTS.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out
