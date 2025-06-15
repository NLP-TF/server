from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker, AsyncEngine
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session, Session
from sqlalchemy.pool import NullPool
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
from typing import Generator, AsyncGenerator, Optional

load_dotenv()

# Ensure DATABASE_URL has the correct asyncpg scheme
def get_database_url() -> str:
    url = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/mbti_game")
    # Ensure the URL uses asyncpg
    if 'postgresql://' in url and '+asyncpg' not in url:
        url = url.replace('postgresql://', 'postgresql+asyncpg://')
    return url

DATABASE_URL = get_database_url()

# Async engine for async database operations
engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    echo=True,
    future=True,
    pool_pre_ping=True,
    poolclass=NullPool
)

# Alias for backward compatibility
async_engine = engine

def get_sync_engine():
    """Get the sync engine, creating it if it doesn't exist."""
    global sync_engine
    if sync_engine is None:
        sync_url = DATABASE_URL.replace('+asyncpg', '')
        sync_engine = create_engine(sync_url)
    return sync_engine

# Initialize sync_engine as None - will be created on demand
sync_engine = None

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
    db = None
    try:
        db = Session(get_sync_engine())
        yield db
    finally:
        if db:
            db.close()

Base = declarative_base()

async def init_db():
    """Initialize the database by creating all tables.
    
    This should be called on application startup.
    """
    # Import models to ensure they are registered with SQLAlchemy
    from app.models.user import User
    from app.models.game import GameSession, GameRound
    
    # Create all tables using async engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Also create tables in sync database for synchronous operations
    Base.metadata.create_all(bind=get_sync_engine())
    print("Database tables created successfully")

# Export commonly used names for backward compatibility
__all__ = [
    'Base',
    'engine',
    'sync_engine',
    'async_engine',
    'get_db',
    'get_sync_db',
    'get_sync_engine',
    'init_db',
    'async_session_factory',
    'SessionLocal',
    'SyncSessionLocal',
    'AsyncSession',
    'Session'
]
