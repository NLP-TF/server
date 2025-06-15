"""
Game service for managing MBTI game sessions, scoring, and leaderboards.

This module provides the GameService class which handles the core game logic,
including session management, scoring, and leaderboard functionality.
"""

import aiofiles
import json
import logging
import os
import random
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast, TypedDict

# Database
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import select, func, update, delete, and_, or_, text
from app.db.models import GameSession as DBGameSession, PlayerScore as DBPlayerScore
from .db_utils import get_db_session

from app.models.predict import predict_tf_style
from app.schemas import UserType, RoundScore, GameSummary
from .ranking_service import ranking_service


# Type aliases for better code readability
class LeaderboardEntry(TypedDict, total=False):
    """Type definition for leaderboard entries."""

    session_id: str
    nickname: str
    user_type: str
    total_score: float
    timestamp: str


class SessionResponse(TypedDict, total=False):
    """Type definition for session responses."""

    round: int
    response: str
    score: float
    timestamp: str


class GameSessionDict(TypedDict, total=False):
    """Type definition for game session dictionaries."""

    session_id: str
    nickname: str
    user_type: UserType
    scenarios: List[Dict[str, str]]
    current_round: int
    scores: List[float]
    responses: List[SessionResponse]
    start_time: datetime
    end_time: Optional[datetime]
    completed: bool


# Configure logging with a more detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_DATA_DIR = Path("data")
SCENARIOS_FILE = DEFAULT_DATA_DIR / "scenarios.json"
STATS_FILE = DEFAULT_DATA_DIR / "stats.json"

# Type aliases for backward compatibility
GameSessions = Dict[str, GameSessionDict]
Leaderboard = List[Dict[str, Any]]  # Using Dict for backward compatibility

# In-memory storage for active game sessions (temporary, will be replaced with DB)
sessions: Dict[str, GameSessionDict] = {}

def ensure_data_directory() -> None:
    """
    Ensure the data directory exists.

    Creates the default data directory if it doesn't exist.

    Raises:
        OSError: If directory creation fails
    """
    try:
        DEFAULT_DATA_DIR.mkdir(exist_ok=True, parents=True)
        logger.debug("Ensured data directory exists at: %s", DEFAULT_DATA_DIR)
    except OSError as e:
        logger.error("Failed to create data directory: %s", e)
        raise


# Initialize data directory
ensure_data_directory()

def load_scenarios() -> List[Dict[str, str]]:
    """
    Load game scenarios from JSON file or return default scenarios.

    Attempts to load scenarios from the configured JSON file. If the file
    doesn't exist or is invalid, falls back to default scenarios.

    Returns:
        List[Dict[str, str]]: List of scenario dictionaries with situation and examples
    """
    def validate_scenario(scenario: Dict[str, Any]) -> Dict[str, str]:
        """Ensure each scenario has the required fields with proper defaults."""
        validated = {
            'situation': '친구_갈등',  # Default situation
            'situation_detail': '',
            'friend_message': '',
        }
        
        # Update with provided values
        validated.update(scenario)
        
        # If situation_detail is missing but situation exists, use it as detail
        if not validated['situation_detail'] and validated['situation'] != '친구_갈등':
            validated['situation_detail'] = validated['situation']
            validated['situation'] = '친구_갈등'
            
        return validated
    
    # First try to load from scenarios.json
    if SCENARIOS_FILE.exists():
        try:
            logger.info("Attempting to load scenarios from %s", SCENARIOS_FILE)
            with open(SCENARIOS_FILE, "r", encoding="utf-8") as f:
                scenarios = json.load(f)
                
                if not isinstance(scenarios, list):
                    logger.error("Scenarios file does not contain a list. Using default scenarios.")
                    return DEFAULT_SCENARIOS
                
                if not scenarios:
                    logger.warning("Scenarios file is empty. Using default scenarios.")
                    return DEFAULT_SCENARIOS
                
                # Validate each scenario
                validated_scenarios = [validate_scenario(s) for s in scenarios]
                logger.info(
                    "Successfully loaded %d scenarios from %s", 
                    len(validated_scenarios), 
                    SCENARIOS_FILE
                )
                return validated_scenarios
                
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON from %s: %s", SCENARIOS_FILE, e)
        except Exception as e:
            logger.error("Unexpected error loading scenarios from %s: %s", SCENARIOS_FILE, e)
    else:
        logger.warning("Scenarios file not found at %s", SCENARIOS_FILE)
    
    # Fall back to default scenarios if file loading fails
    logger.info("Using %d default scenarios", len(DEFAULT_SCENARIOS))
    return DEFAULT_SCENARIOS


# Default scenarios used when no scenarios file is found
DEFAULT_SCENARIOS = [
    {
        "situation": "친구_갈등",
        "situation_detail": "친구와의 다툼 후 화해하고 싶을 때",
        "friend_message": "나 우리 사이에 뭔가 이상한 거 같아..."
    },
    {
        "situation": "연인_갈등",
        "situation_detail": "데이트 약속을 잊어버렸을 때",
        "friend_message": "너 진짜 오늘 약속 있었는지도 몰랐어?"
    },
    {
        "situation": "실수_자책",
        "situation_detail": "중요한 발표에서 실수했을 때",
        "friend_message": "다 망쳤어..."
    },
    {
        "situation": "친구_갈등",
        "situation_detail": "친구가 나를 오해했을 때",
        "friend_message": "너 진짜 그런 말 했다며?"
    },
    {
        "situation": "연인_갈등",
        "situation_detail": "선물이 마음에 들지 않았을 때",
        "friend_message": "고마운데... 솔직히 마음에 안 들어"
    }
]

def save_stats() -> Dict[str, Any]:
    """
    Save game statistics to file.

    Creates or updates the stats file with current game statistics.

    Returns:
        Dict[str, Any]: The saved statistics

    Raises:
        IOError: If file operations fail
    """
    stats = {
        "total_games_played": 0,  # This would be updated with actual stats
        "average_score": 0,  # This would be calculated from game data
        "last_updated": datetime.now().isoformat(),
        "total_players": 0,  # This would be updated based on actual data
    }
    
    try:
        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info("Successfully saved game statistics to %s", STATS_FILE)
        return stats
    except (IOError, OSError) as e:
        logger.error("Failed to save game statistics to %s: %s", STATS_FILE, e)
        raise


def load_stats() -> Dict[str, Any]:
    """Load game statistics from file."""
    try:
        if STATS_FILE.exists():
            with open(STATS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logger.error("Failed to load stats from %s: %s", STATS_FILE, e)
    return {}


class GameService:
    """Service for managing MBTI game sessions, scoring, and leaderboards.

    This class handles the core game logic including:
    - Game session management
    - Response scoring
    - Leaderboard updates
    - Game summary generation
    """

    def __init__(self):
        """Initialize the GameService with scenarios and stats."""
        self.scenarios = load_scenarios()
        try:
            self.stats = load_stats() or {
                "total_games": 0,
                "total_players": 0,
                "avg_score": 0.0,
                "last_updated": datetime.now().isoformat(),
            }
            logger.info(
                "GameService initialized with %d scenarios and stats",
                len(self.scenarios),
            )
        except Exception as e:
            logger.error("Failed to load stats: %s", str(e))
            self.stats = {
                "total_games": 0,
                "total_players": 0,
                "avg_score": 0.0,
                "last_updated": datetime.now().isoformat(),
            }
            logger.info("GameService initialized with default stats")

    async def start_game(self, nickname: str, user_type: UserType) -> str:
        """
        Start a new game session.

        Args:
            nickname: Player's nickname
            user_type: Player's MBTI type (T/F)

        Returns:
            str: Session ID for the new game
        """
        if not nickname or not user_type:
            raise ValueError("Nickname and user_type are required")

        session_id = str(uuid.uuid4())
        scenarios = random.sample(self.scenarios, min(5, len(self.scenarios)))  # Select 5 random scenarios
        
        session_data: GameSessionDict = {
            'session_id': session_id,
            'nickname': nickname,
            'user_type': user_type,
            'scenarios': scenarios,
            'current_round': 0,
            'scores': [],
            'responses': [],
            'start_time': datetime.now(),
            'end_time': None,
            'completed': False
        }
        
        sessions[session_id] = session_data
        logger.info("Started new game session %s for %s", session_id, nickname)
        return session_id

    def get_round(self, session_id: str, round_number: int) -> Optional[Dict[str, Any]]:
        """
        Get round information for a session.

        Args:
            session_id: The session ID
            round_number: The round number (1-based)

        Returns:
            Optional[Dict]: Round information or None if invalid
        """
        if session_id not in sessions:
            return None
            
        session = sessions[session_id]
        if not 1 <= round_number <= len(session['scenarios']):
            return None
            
        scenario = session['scenarios'][round_number - 1]
        return {
            'round_number': round_number,
            'situation': scenario['situation'],
            'situation_detail': scenario['situation_detail'],
            'friend_message': scenario['friend_message']
        }

    def submit_response(
        self, 
        session_id: str, 
        user_response: str, 
        round_number: int
    ) -> Optional[Dict[str, Any]]:
        """
        Submit a response for a round.

        Args:
            session_id: The session ID
            user_response: The player's response
            round_number: The round number

        Returns:
            Optional[Dict]: Score and feedback or None if invalid
        """
        if session_id not in sessions:
            return None
            
        session = sessions[session_id]
        if not 1 <= round_number <= len(session['scenarios']):
            return None
            
        # Calculate score (simplified - implement your actual scoring logic)
        score = self._calculate_score(user_response, session['user_type'])
        
        # Store response and score
        response: SessionResponse = {
            'round': round_number,
            'response': user_response,
            'score': score,
            'timestamp': datetime.now().isoformat()
        }
        
        session['responses'].append(response)
        session['scores'].append(score)
        session['current_round'] = round_number
        
        # Check if game is complete
        if round_number == len(session['scenarios']):
            session['completed'] = True
            session['end_time'] = datetime.now()
            self._update_stats(score)
        
        return {
            'score': score,
            'round_number': round_number,
            'is_complete': session['completed']
        }

    def get_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get game summary for a session.

        Args:
            session_id: The session ID

        Returns:
            Optional[Dict]: Game summary or None if invalid
        """
        if session_id not in sessions:
            return None
            
        session = sessions[session_id]
        total_score = sum(session['scores']) if session['scores'] else 0
        avg_score = total_score / len(session['scores']) if session['scores'] else 0
        
        return {
            'session_id': session_id,
            'nickname': session['nickname'],
            'user_type': session['user_type'],
            'total_score': total_score,
            'average_score': avg_score,
            'rounds_played': len(session['responses']),
            'completed': session['completed'],
            'start_time': session['start_time'].isoformat(),
            'end_time': session['end_time'].isoformat() if session['end_time'] else None
        }

    def _calculate_score(self, response: str, user_type: UserType) -> float:
        """
        Calculate score for a response.
        
        Args:
            response: The user's response
            user_type: The user's MBTI type (T/F)
            
        Returns:
            float: Score between 0 and 1
        """
        # TODO: Implement actual scoring logic using your prediction model
        # This is a placeholder that returns a random score
        return random.uniform(0.5, 1.0)
        
    def _update_stats(self, score: float) -> None:
        """
        Update game statistics.
        
        Args:
            score: The score to add to statistics
        """
        try:
            self.stats['total_games'] += 1
            self.stats['total_players'] = len(sessions)
            
            # Update average score
            total_score = self.stats['avg_score'] * (self.stats['total_games'] - 1) + score
            self.stats['avg_score'] = total_score / self.stats['total_games']
            self.stats['last_updated'] = datetime.now().isoformat()
            
            # Save updated stats
            save_stats()
        except Exception as e:
            logger.error("Failed to update stats: %s", str(e))


# Create singleton instance of GameService
game_service = GameService()

# Load initial stats on module import
try:
    load_stats()
    logger.info("Successfully loaded initial game stats")
except Exception as e:
    logger.error("Failed to load initial game stats: %s", e, exc_info=True)
    # Continue with default stats if loading fails

# Module-level exports
__all__ = ["game_service", "GameService", "GameSessionDict", "LeaderboardEntry"]

# This ensures the module can be imported directly
if __name__ == "__main__":
    print(f"{__name__} module loaded successfully")
