"""SQL files render with their template parameters (no duckdb needed)."""

from fpa.data.build import CHECK_DIR, SQL_DIR, closure_expr, split_statements, store_filter


def test_templates_render_and_split():
    params = {
        "raw_dir": "data/raw",
        "store_filter": store_filter(["CA_1", "O'Brien"]),
        "closure_expr": closure_expr([[12, 25]]),
        "last_sales_day": 1941,
        "expected_series": 30490,
        "tolerance": 0.01,
    }
    for f in [*sorted(SQL_DIR.glob("*.sql")), *sorted(CHECK_DIR.glob("*.sql"))]:
        sql = f.read_text().format(**params)
        stmts = split_statements(sql)
        assert stmts, f.name
        assert "{" not in sql.replace("{'", "").replace("'}", ""), f.name
    assert store_filter(None) == "TRUE"
    assert "O''Brien" in params["store_filter"]
    assert closure_expr([]) == "FALSE"
