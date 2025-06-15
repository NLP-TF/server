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
    situation_detail: str
    friend_message: str
    example_response: str

class ScoreRequest(BaseModel):
    session_id: str
    user_response: str
    round_number: int
    situation: str  # e.g., "연인_갈등", "친구_갈등"

class RoundScore(BaseModel):
    round_number: int
    situation: str  # e.g., "연인_갈등", "친구_갈등"
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

class PlayerRanking(BaseModel):
    nickname: str
    score: float
    average: float
    rank: int

class RankingResponse(BaseModel):
    rankings: List[PlayerRanking]
    total_players: int

class PlayerScore(BaseModel):
    nickname: str
    total_score: float
    game_count: int = 1
    user_type: str = 'U'  # 'T' for Thinking, 'F' for Feeling, 'U' for Unknown
    
    @property
    def average_score(self) -> float:
        return self.total_score / self.game_count if self.game_count > 0 else 0.0
