from __future__ import annotations

import pytest

from backend.app.duckdb_tools.sql_safety import InvalidSQLError, SQLSafetyLayer


def test_rejects_mutating_sql() -> None:
    layer = SQLSafetyLayer()
    with pytest.raises(InvalidSQLError):
        layer.validate_readonly_sql("DROP TABLE customers")


def test_applies_row_limit() -> None:
    layer = SQLSafetyLayer()
    sql = layer.apply_row_limit("SELECT * FROM customers", 100)
    assert "LIMIT 100" in sql.upper()

