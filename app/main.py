# app/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from app.routes import router as predict_router

app = FastAPI(
    title="FastAPI Server",
    description="A FastAPI server for your application",
    version="1.0.0",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 운영 환경에서는 꼭 제한할 것
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ predict 라우터 등록
app.include_router(predict_router)


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
