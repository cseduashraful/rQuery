from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from backend.app.config import get_settings

Base = declarative_base()

settings = get_settings()
engine = create_engine(settings.metadata_database_url, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_db_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_metadata_db() -> None:
    from backend.app.db import models

    Base.metadata.create_all(bind=engine)

