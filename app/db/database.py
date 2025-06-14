from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import NullPool
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/mbti_game")

engine = create_async_engine(
    DATABASE_URL,
    echo=True,
    future=True,
    pool_pre_ping=True,
    poolclass=NullPool
)

async def get_db() -> AsyncSession:
    async_session = sessionmaker(
        engine, 
        class_=AsyncSession, 
        expire_on_commit=False,
        autoflush=False
    )
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()

Base = declarative_base()

async def init_db():
    async with engine.begin() as conn:
        # Create all tables
        from app.db.models import GameSession, PlayerScore  # Import models to ensure they're registered with Base
        await conn.run_sync(Base.metadata.create_all)
