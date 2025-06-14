from fastapi import APIRouter, HTTPException, Depends, status
from typing import List, Dict, Any
import uuid
from app.services.game_service import game_service
from app.schemas import (
    GameStartRequest, 
    RoundResponse, 
    ScoreRequest, 
    GameSummary,
    ErrorResponse
)

router = APIRouter(
    prefix="/api/v1/game",
    tags=["game"],
    responses={404: {"description": "Not found"}},
)



@router.post("/start", response_model=dict)
async def start_game(request: GameStartRequest):
    """
    Start a new game session.
    
    - **nickname**: Player's nickname (1-20 characters)
    - **user_type**: Player's MBTI type ("T" or "F")
    
    Returns a session ID to be used for subsequent requests.
    """
    try:
        session_id = game_service.start_game(
            nickname=request.nickname,
            user_type=request.user_type
        )
        return {"session_id": session_id, "message": "Game started successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/round/{round_number}", response_model=RoundResponse)
async def get_round(round_number: int):
    """
    Get the situation and example response for a specific round.
    
    - **round_number**: The round number (1-5)
    """
    try:
        round_info = game_service.get_round(round_number)
        if not round_info:
            raise HTTPException(status_code=404, detail=f"Round {round_number} not found")
        return round_info
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/score", response_model=dict)
async def submit_score(request: ScoreRequest):
    """
    Submit a response for a round and get the score.
    
    - **session_id**: The game session ID
    - **user_response**: The user's response text
    - **round_number**: The round number (1-5)
    """
    try:
        score = game_service.submit_response(
            session_id=request.session_id,
            user_response=request.user_response,
            round_number=request.round_number
        )
        return {"score": score, "message": "Response scored successfully"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/summary/{session_id}", response_model=GameSummary)
async def get_summary(session_id: str):
    """
    Get the game summary including scores and leaderboard position.
    
    - **session_id**: The game session ID
    """
    try:
        return game_service.get_summary(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
