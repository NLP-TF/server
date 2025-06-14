from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum

class UserType(str, Enum):
    THINKING = "T"
    FEELING = "F"

class GameStartRequest(BaseModel):
    nickname: str = Field(..., min_length=1, max_length=20)
    user_type: UserType

class RoundResponse(BaseModel):
    round_number: int
    situation: str
    example_response: str

class ScoreRequest(BaseModel):
    session_id: str
    user_response: str
    round_number: int

class RoundScore(BaseModel):
    round_number: int
    score: float
    user_response: str
    is_correct_style: bool

class GameSummary(BaseModel):
    session_id: str
    nickname: str
    user_type: UserType
    total_score: float
    round_scores: List[RoundScore]
    percentile: float
    rank: int
    top_players: List[Dict[str, str]]
    feedback: str

class ErrorResponse(BaseModel):
    detail: str
