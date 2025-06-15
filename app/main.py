# app/main.py

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, AsyncGenerator
import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Database
from app.db.database import Base, engine, init_db, get_sync_engine

# Routers
from app.routers import game as game_router

# Load environment variables
load_dotenv()


# Create database tables on startup
async def create_tables():
    # Create tables using async engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Also create tables in sync database
    Base.metadata.create_all(bind=get_sync_engine())


# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)


def initialize_stats() -> None:
    """Initialize stats.json with empty leaderboard if it doesn't exist."""
    stats_file = Path("data/stats.json")
    if not stats_file.exists():
        with open(stats_file, "w") as f:
            json.dump({"leaderboard": [], "total_games": 0}, f)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Create database tables
    await create_tables()
    print("✅ Database tables created")

    # Initialize stats
    initialize_stats()

    # Load initial data or perform startup events
    # For example, load ML models, connect to databases, etc.

    yield

    # Clean up resources on shutdown
    # For example, close database connections, etc.
    await engine.dispose()


# Initialize the FastAPI application with lifespan
app = FastAPI(
    title="MBTI Game API",
    description="MBTI Game API Server",
    version="1.0.0",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS middleware
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "https://your-production-domain.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.models.load_model import load_model_and_tokenizer
import logging

logger = logging.getLogger(__name__)


@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting server... (Model load begins)")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_model_and_tokenizer)
    logger.info("✅ Model and tokenizer loaded successfully")


# Include routers
app.include_router(game_router.router)


# Example route
@app.get("/")
async def read_root():
    return {"message": "Welcome to the MBTI Game API"}


# Sample data model
class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None


# Sample routes (can be removed if not needed)
items_db = []


@app.get("/items/")
async def read_items():
    return items_db


@app.post("/items/")
async def create_item(item: Item):
    items_db.append(item)
    return item


@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id < 0 or item_id >= len(items_db):
        raise HTTPException(status_code=404, detail="Item not found")
    return items_db[item_id]


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
