import logging
from fastapi import APIRouter, HTTPException, Depends, status

# Configure logging
logger = logging.getLogger(__name__)
from typing import List, Dict, Any
import uuid
from app.services.game_service import game_service
from app.schemas import (
    GameStartRequest, 
    RoundResponse, 
    ScoreRequest, 
    GameSummary,
    ErrorResponse,
    UserType
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
        # Pass the raw string value to the service
        session_id = await game_service.start_game(
            nickname=request.nickname,
            user_type=request.user_type.value  # Pass the raw string value ("T" or "F")
        )
        return {"session_id": session_id, "message": "Game started successfully"}
    except ValueError as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid user_type. Must be one of: {[t.value for t in UserType]}"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/round/{session_id}/{round_number}", response_model=RoundResponse)
async def get_round(session_id: str, round_number: int):
    """
    Get the situation and example response for a specific round for a session.
    - **session_id**: The game session ID
    - **round_number**: The round number (1-5)
    Returns the round details including the situation and example response.
    """
    try:
        round_info = game_service.get_round(session_id, round_number)
        if not round_info:
            raise HTTPException(status_code=404, detail=f"Round {round_number} not found for session {session_id}")
        return round_info
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/submit", response_model=dict)
async def submit_score(request: ScoreRequest):
    """
    Submit a response for a round and get the score.
    
    - **session_id**: The game session ID
    - **user_response**: The user's response text
    - **round_number**: The round number (1-5)
    
    Returns the score and round information.
    """
    try:
        result = await game_service.submit_response(
            session_id=request.session_id,
            user_response=request.user_response,
            round_number=request.round_number
        )
        if not result:
            raise HTTPException(status_code=404, detail="Session not found or round already completed")
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/summary/{session_id}", response_model=GameSummary)
async def get_summary(session_id: str):
    """
    Get the game summary including scores and leaderboard position.
    
    - **session_id**: The game session ID
    
    Returns the game summary with scores and rankings.
    """
    try:
        # Await the async get_summary method
        summary = await game_service.get_summary(session_id)
        if not summary:
            raise HTTPException(status_code=404, detail="Session not found or game not completed")
        return summary
    except Exception as e:
        logger.error(f"Error getting game summary: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
