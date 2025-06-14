from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any
from ..services.ranking_service import ranking_service

router = APIRouter(
    prefix="/api/v1/rankings",
    tags=["rankings"],
    responses={404: {"description": "Not found"}},
)

@router.get("")
async def get_rankings(limit: int = 10) -> Dict[str, Any]:
    """
    Get top player rankings.
    
    - **limit**: Number of top players to return (default: 10)
    """
    try:
        if limit < 1:
            raise HTTPException(status_code=400, detail="Limit must be greater than 0")
        return ranking_service.get_rankings(limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
