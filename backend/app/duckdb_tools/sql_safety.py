from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, TimeoutError
from dataclasses import dataclass

import duckdb
import sqlglot
from sqlglot import exp

FORBIDDEN_TOKENS = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "COPY",
    "EXPORT",
    "ATTACH",
    "INSTALL",
    "LOAD",
    "PRAGMA",
    "CALL",
}


class InvalidSQLError(ValueError):
    pass


class QueryTimeoutError(TimeoutError):
    pass


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[dict]
    row_count: int


def _execute_query(db_path: str, sql: str) -> QueryResult:
    conn = duckdb.connect(database=db_path, read_only=True)
    try:
        df = conn.execute(sql).fetchdf()
    finally:
        conn.close()
    return QueryResult(columns=list(df.columns), rows=df.to_dict("records"), row_count=len(df.index))


class SQLSafetyLayer:
    def validate_readonly_sql(self, sql: str) -> None:
        upper_sql = sql.upper()
        for token in FORBIDDEN_TOKENS:
            if token in upper_sql:
                raise InvalidSQLError(f"Forbidden SQL token detected: {token}")

        parsed = sqlglot.parse(sql, read="duckdb")
        if len(parsed) != 1:
            raise InvalidSQLError("Multiple SQL statements are not allowed.")
        statement = parsed[0]
        if not isinstance(statement, (exp.Select, exp.Union, exp.With, exp.Subquery, exp.CTE)):
            root = statement.this if isinstance(statement, exp.With) else statement
            if not isinstance(root, (exp.Select, exp.Union)):
                raise InvalidSQLError("Only SELECT or WITH queries are allowed.")

    def apply_row_limit(self, sql: str, max_rows: int) -> str:
        parsed = sqlglot.parse_one(sql, read="duckdb")
        if not parsed.args.get("limit"):
            parsed = parsed.limit(max_rows)
        return parsed.sql(dialect="duckdb")

    def execute_safe_query(self, db_path: str, sql: str, timeout_seconds: int) -> QueryResult:
        with ProcessPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_execute_query, db_path, sql)
            try:
                return future.result(timeout=timeout_seconds)
            except TimeoutError as exc:
                future.cancel()
                raise QueryTimeoutError("Query execution exceeded timeout.") from exc

