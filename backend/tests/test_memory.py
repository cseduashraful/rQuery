from __future__ import annotations

from backend.app.db.metadata import SessionLocal, init_metadata_db
from backend.app.memory.store import MemoryStore


def test_memory_is_database_scoped() -> None:
    init_metadata_db()
    session = SessionLocal()
    try:
        store = MemoryStore(session)
        store.replace_memory("db_one", "database_summary", "customers and orders")
        store.replace_memory("db_two", "database_summary", "invoices and payments")
        session.commit()

        results = store.search_memory("db_one", "customers")
        assert results
        assert all("invoices" not in item["content"] for item in results)
    finally:
        session.close()

