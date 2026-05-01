from __future__ import annotations

import hashlib
from pathlib import Path

import duckdb


class DuckDBConnector:
    def connect_read_only(self, db_path: str) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(database=db_path, read_only=True)

    def compute_fingerprint(self, db_path: str) -> str:
        path = Path(db_path)
        stat = path.stat()
        raw = f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode()
        return hashlib.sha256(raw).hexdigest()

