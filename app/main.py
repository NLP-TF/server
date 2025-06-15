from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
import os
import json
from pathlib import Path
from typing import Optional
import logging

from dotenv import load_dotenv
from contextlib import asynccontextmanager
from typing import AsyncGenerator

# Database
from app.db.database import Base, engine, init_db, get_sync_engine

# Routers
from app.routers import game as game_router

# Model utils (Lazy Loading)
from app.models.load_model import load_models_if_needed, predict_tf_style

# Logging setup
logger = logging.getLogger(__name__)

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


async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Base.metadata.create_all(bind=get_sync_engine())


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Create database tables
    await create_tables()
    print("✅ Database tables created")
    # Initialize stats
    initialize_stats()
    yield
    await engine.dispose()


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="MBTI Game API",
    description="MBTI Game API Server",
    version="1.0.0",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
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


# Healthcheck endpoint
@app.get("/health")
async def health():
    return {"status": "ok"}


# Lazy model loading: 실제 예측 등에서 최초 호출시만 모델 로딩
class PredictRequest(BaseModel):
    text: str
    situation: Optional[str] = "친구_갈등"


@app.post("/predict")
async def predict(req: PredictRequest):
    try:
        load_models_if_needed()
        result = predict_tf_style(req.text, req.situation)
        return {"result": result}
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Include game router
app.include_router(game_router.router)


@app.get("/")
async def read_root():
    return {"message": "Welcome to the MBTI Game API"}


# Sample Item model and routes for demonstration
class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None


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
    import uvicorn
    import os

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        reload=False,
    )
