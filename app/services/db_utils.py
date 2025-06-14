"""Database utility functions for the game service."""
from contextlib import contextmanager
from sqlalchemy.orm import Session
from app.db.database import SessionLocal

@contextmanager
def get_db_session() -> Session:
    """Get a database session with automatic cleanup.
    
    Yields:
        Session: A database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
