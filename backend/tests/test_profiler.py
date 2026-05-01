from __future__ import annotations

from pathlib import Path

import duckdb

from backend.app.duckdb_tools.profiler import DuckDBProfiler


def test_profiler_detects_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "test.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("CREATE TABLE customers (customer_id INTEGER, name VARCHAR, signup_date DATE)")
    conn.execute("INSERT INTO customers VALUES (1, 'A', '2024-01-01'), (2, 'B', '2024-01-03')")
    conn.close()

    profiler = DuckDBProfiler()
    bundle = profiler.profile_database(str(db_path))

    assert bundle.tables
    assert bundle.tables[0]["table_name"] == "customers"

