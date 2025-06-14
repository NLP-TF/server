# app/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

def initialize_stats() -> None:
    """Initialize stats.json with empty leaderboard if it doesn't exist."""
    stats_file = Path("data/stats.json")
    if not stats_file.exists():
        with open(stats_file, "w") as f:
            json.dump({"leaderboard": [], "total_games": 0}, f)

# Initialize the FastAPI application
app = FastAPI(
    title="MBTI Game API",
    description="API for MBTI-based T/F style classification game",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from app.routers import game, predict

app.include_router(predict.router)
app.include_router(game.router)


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
