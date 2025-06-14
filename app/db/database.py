from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session, Session
from sqlalchemy.pool import NullPool
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from typing import Generator, AsyncGenerator

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/mbti_game")

# Async engine for async database operations
engine = create_async_engine(
    DATABASE_URL,
    echo=True,
    future=True,
    pool_pre_ping=True,
    poolclass=NullPool
)

# Alias for backward compatibility
async_engine = engine

# Sync engine for synchronous database operations
SYNC_DATABASE_URL = DATABASE_URL.replace('+asyncpg', '')
sync_engine = create_engine(SYNC_DATABASE_URL)

# Session factory for async operations
async_session_factory = async_sessionmaker(
    bind=engine, 
    class_=AsyncSession, 
    expire_on_commit=False,
    autoflush=False
)

# Session factory for sync operations
SyncSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=sync_engine
)

# Alias for backward compatibility
SessionLocal = async_session_factory

# Dependency to get async DB session
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session.
    
    Yields:
        AsyncSession: An async database session
    """
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()

# Dependency to get sync DB session
def get_sync_db() -> Generator[Session, None, None]:
    """Get a synchronous database session.
    
    Yields:
        Session: A database session
    """
    db = SyncSessionLocal()
    try:
        yield db
    finally:
        db.close()

Base = declarative_base()

async def init_db():
    """Initialize the database by creating all tables.
    
    This should be called on application startup.
    """
    async with async_engine.begin() as conn:
        # Create all tables
        from app.db.models import GameSession, PlayerScore  # Import models to ensure they're registered with Base
        await conn.run_sync(Base.metadata.create_all)

    # Also create tables in sync database for synchronous operations
    Base.metadata.create_all(bind=sync_engine)

# Export commonly used names for backward compatibility
__all__ = [
    'Base',
    'engine',
    'sync_engine',
    'SessionLocal',
    'SyncSessionLocal',
    'get_db',
    'get_sync_db',
    'init_db'
]
