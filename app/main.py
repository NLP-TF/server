# app/main.py

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

def initialize_stats() -> None:
    """Initialize stats.json with empty leaderboard if it doesn't exist."""
    stats_file = Path("data/stats.json")
    if not stats_file.exists():
        with open(stats_file, "w") as f:
            json.dump({"leaderboard": [], "total_games": 0}, f)

# Database initialization
from app.db.database import init_db, Base, engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    # Clean up on shutdown
    await engine.dispose()

# Initialize the FastAPI application with lifespan
app = FastAPI(
    title="MBTI Game API",
    description="API for MBTI-based T/F style classification game",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS middleware
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "https://your-production-domain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from app.routers import game, predict, ranking

app.include_router(predict.router)
app.include_router(game.router)
app.include_router(ranking.router)


# 기본 예제 라우트
class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None


items_db = []


@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI server!"}


@app.get("/items/", response_model=List[Item])
async def read_items():
    return items_db


@app.post("/items/", response_model=Item)
async def create_item(item: Item):
    items_db.append(item)
    return item


@app.get("/items/{item_id}", response_model=Item)
async def read_item(item_id: int):
    if item_id < 0 or item_id >= len(items_db):
        raise HTTPException(status_code=404, detail="Item not found")
    return items_db[item_id]


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
