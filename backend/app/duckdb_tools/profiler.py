from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from backend.app.duckdb_tools.connector import DuckDBConnector


@dataclass
class ProfileBundle:
    tables: list[dict]
    relationships: list[dict]
    summary_input: dict


class DuckDBProfiler:
    def __init__(self, connector: Optional[DuckDBConnector] = None) -> None:
        self.connector = connector or DuckDBConnector()

    def profile_database(self, db_path: str) -> ProfileBundle:
        conn = self.connector.connect_read_only(db_path)
        try:
            tables = [
                row[0]
                for row in conn.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY 1"
                ).fetchall()
            ]
            table_profiles = [self.profile_table(conn, table_name) for table_name in tables]
            relationships = self.infer_relationships(table_profiles)
            summary_input = {"tables": table_profiles, "relationships": relationships}
            return ProfileBundle(
                tables=table_profiles,
                relationships=relationships,
                summary_input=summary_input,
            )
        finally:
            conn.close()

    def profile_table(self, conn, table_name: str) -> dict:
        columns = conn.execute(f"DESCRIBE {self.quote_ident(table_name)}").fetchall()
        row_count = conn.execute(f"SELECT COUNT(*) FROM {self.quote_ident(table_name)}").fetchone()[0]
        profile_columns = []
        for name, data_type, *_rest in columns:
            stats = self.profile_column(conn, table_name, name, data_type)
            profile_columns.append(stats)

        sample_rows = conn.execute(
            f"SELECT * FROM {self.quote_ident(table_name)} LIMIT 5"
        ).fetchdf().to_dict("records")

        return {
            "table_name": table_name,
            "row_count": int(row_count),
            "primary_key_guess": self.guess_primary_key(profile_columns, row_count),
            "columns": profile_columns,
            "sample_rows": sample_rows,
        }

    def profile_column(self, conn, table_name: str, column_name: str, data_type: str) -> dict:
        quoted_table = self.quote_ident(table_name)
        quoted_column = self.quote_ident(column_name)
        null_count, distinct_count = conn.execute(
            f"""
            SELECT
              SUM(CASE WHEN {quoted_column} IS NULL THEN 1 ELSE 0 END) AS null_count,
              COUNT(DISTINCT {quoted_column}) AS distinct_count
            FROM {quoted_table}
            """
        ).fetchone()
        row_count = conn.execute(f"SELECT COUNT(*) FROM {quoted_table}").fetchone()[0] or 1
        stats = {"kind": "generic"}
        if any(token in data_type.upper() for token in ["INT", "DECIMAL", "DOUBLE", "FLOAT"]):
            min_val, max_val, avg_val, stddev_val = conn.execute(
                f"SELECT MIN({quoted_column}), MAX({quoted_column}), AVG({quoted_column}), STDDEV_SAMP({quoted_column}) FROM {quoted_table}"
            ).fetchone()
            stats = {
                "kind": "numeric",
                "min": min_val,
                "max": max_val,
                "avg": avg_val,
                "stddev": stddev_val,
            }
        elif "DATE" in data_type.upper() or "TIME" in data_type.upper():
            min_val, max_val = conn.execute(
                f"SELECT MIN({quoted_column}), MAX({quoted_column}) FROM {quoted_table}"
            ).fetchone()
            stats = {"kind": "temporal", "min": str(min_val), "max": str(max_val)}
        else:
            top_values = conn.execute(
                f"""
                SELECT CAST({quoted_column} AS VARCHAR) AS value, COUNT(*) AS freq
                FROM {quoted_table}
                WHERE {quoted_column} IS NOT NULL
                GROUP BY 1
                ORDER BY 2 DESC
                LIMIT 5
                """
            ).fetchall()
            stats = {"kind": "categorical", "top_values": top_values}
        return {
            "column_name": column_name,
            "data_type": data_type,
            "null_rate": float(null_count or 0) / float(row_count),
            "distinct_count": int(distinct_count or 0),
            "stats": stats,
        }

    def infer_relationships(self, table_profiles: list[dict]) -> list[dict]:
        table_columns = {
            table["table_name"]: {column["column_name"] for column in table["columns"]}
            for table in table_profiles
        }
        relationships = []
        for from_table, columns in table_columns.items():
            for column in columns:
                if not column.endswith("_id"):
                    continue
                target_name = column[:-3]
                for to_table, target_columns in table_columns.items():
                    if to_table == from_table:
                        continue
                    if column in target_columns or f"{target_name}_id" in target_columns:
                        relationships.append(
                            {
                                "from_table": from_table,
                                "from_column": column,
                                "to_table": to_table,
                                "to_column": column if column in target_columns else f"{target_name}_id",
                                "confidence": 0.6,
                                "evidence": {"rule": "shared_id_name"},
                            }
                        )
        return relationships

    def guess_primary_key(self, column_profiles: list[dict], row_count: int) -> Optional[str]:
        for column in column_profiles:
            if column["distinct_count"] == row_count and column["null_rate"] == 0.0:
                return column["column_name"]
        return None

    def quote_ident(self, identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'
