"""
Game service for managing MBTI game sessions, scoring, and leaderboards.

This module provides the GameService class which handles the core game logic,
including session management, scoring, and leaderboard functionality.
"""

import json
import logging
import os
import random
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast, TypedDict

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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_DATA_DIR = Path("data")
SCENARIOS_FILE = DEFAULT_DATA_DIR / "scenarios.json"
STATS_FILE = DEFAULT_DATA_DIR / "stats.json"
DEFAULT_SCENARIOS = [
    {
        "situation": "친구가 시험에 떨어졌을 때",
        "example_T": "다음에 더 잘하면 돼. 공부 방법을 바꿔봐.",
        "example_F": "너무 속상하겠다. 괜찮아? 기분이 어때?"
    },
    {
        "situation": "동료가 실수로 커피를 쏟았을 때",
        "example_T": "다음엔 조심해. 휴지 좀 가져올게.",
        "example_F": "괜찮아? 다치진 않았어? 너무 놀랐겠다."
    },
    {
        "situation": "팀 프로젝트에서 의견이 엇갈릴 때",
        "example_T": "각각의 장단점을 분석해보고 결정하자.",
        "example_F": "모두의 의견을 들어보는 게 좋을 것 같아."
    }
]

# Type aliases for backward compatibility
GameSessions = Dict[str, GameSessionDict]
Leaderboard = List[Dict[str, Any]]  # Using Dict for backward compatibility

# In-memory storage for game state
sessions: Dict[str, GameSessionDict] = {}
leaderboard: List[Dict[str, Any]] = []  # Global leaderboard storage

# Initialize leaderboard if it doesn't exist
if not isinstance(globals().get('leaderboard'), list):
    leaderboard = []


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
        
    Raises:
        json.JSONDecodeError: If the scenarios file contains invalid JSON
    """
    try:
        if SCENARIOS_FILE.exists():
            logger.debug("Loading scenarios from %s", SCENARIOS_FILE)
            with open(SCENARIOS_FILE, 'r', encoding='utf-8') as f:
                scenarios = json.load(f)
                if not isinstance(scenarios, list):
                    error_msg = f"Scenarios must be a list, got {type(scenarios).__name__}"
                    logger.error(error_msg)
                    raise json.JSONDecodeError(error_msg, doc=str(SCENARIOS_FILE), pos=0)
                logger.info("Loaded %d scenarios from %s", len(scenarios), SCENARIOS_FILE)
                return scenarios
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning("Using default scenarios due to: %s", e)
    
    logger.info("Using %d default scenarios", len(DEFAULT_SCENARIOS))
    return DEFAULT_SCENARIOS

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
        "average_score": 0,       # This would be calculated from game data
        "last_updated": datetime.now().isoformat(),
        "total_players": len(leaderboard)
    }
    
    try:
        with open(STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info("Successfully saved game statistics to %s", STATS_FILE)
        return stats
    except (IOError, OSError) as e:
        logger.error("Failed to save game statistics to %s: %s", STATS_FILE, e)
        raise
    except IOError:
        print("Failed to save stats file")

def load_stats():
    """Load game statistics from file."""
    global leaderboard
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                leaderboard = data.get("leaderboard", [])
    except (json.JSONDecodeError, IOError):
        leaderboard = []

# Initialize stats on module load
try:
    load_stats()
    logger.info("Successfully loaded initial game stats")
except Exception as e:
    logger.error("Failed to load initial game stats: %s", e, exc_info=True)
    # Continue with default stats if loading fails

class GameService:
    """
    Service for managing MBTI game sessions, scoring, and leaderboards.
    
    This class handles the core game logic including:
    - Game session management
    - Response scoring
    - Leaderboard updates
    - Game summary generation
    """
    
    def __init__(self):
        """
        Initialize the GameService with scenarios and stats.
        
        Loads game scenarios and statistics, and initializes the service.
        """
        self.scenarios = load_scenarios()
        self.stats = load_stats()
        logger.info("GameService initialized with %d scenarios", len(self.scenarios))
        
    def start_game(self, nickname: str, user_type: UserType) -> str:
        """Start a new game session.
        
        Args:
            nickname: Player's nickname
            user_type: Player's MBTI type (T/F)
            
        Returns:
            str: Unique session ID for the new game session
            
        Raises:
            ValueError: If nickname is empty or user_type is invalid
        """
        if not nickname or not isinstance(nickname, str):
            raise ValueError("Nickname is required and must be a string")
            
        if not isinstance(user_type, UserType):
            raise ValueError("Invalid user type")
        
        try:
            session_id = str(uuid.uuid4())
            scenarios = self.scenarios.copy()
            random.shuffle(scenarios)
            
            # Create new game session
            sessions[session_id] = {
                "session_id": session_id,
                "nickname": nickname,
                "user_type": user_type,
                "scenarios": scenarios,
                "current_round": 0,
                "scores": [],
                "start_time": datetime.now(),
                "end_time": None,
                "responses": []
            }
            
            logger.info("Started new game session: %s for %s", session_id, nickname)
            return session_id
            
        except Exception as e:
            logger.error("Failed to start game: %s", e, exc_info=True)
            raise
    
    def get_round(self, round_number: int) -> Optional[Dict[str, Any]]:
        """Get round information by number.
        
        Args:
            round_number: The round number (1-based index)
            
        Returns:
            Optional[Dict[str, Any]]: Round information including situation,
            or None if round number is invalid
        """
        try:
            # Load scenarios from the JSON file
            scenarios_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'scenarios.json')
            try:
                with open(scenarios_path, 'r') as f:
                    scenarios = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.error("Failed to load scenarios: %s", e)
                # Return default scenarios if the file is not found or invalid
                scenarios = DEFAULT_SCENARIOS
            
            # Get the scenario for the requested round (1-based to 0-based index)
            if not 1 <= round_number <= len(scenarios):
                logger.warning("Invalid round number: %d", round_number)
                return None
                
            scenario = scenarios[round_number - 1]
            
            # Return the round information
            return {
                "round_number": round_number,
                "situation": scenario["situation"],
                "example_response": scenario.get("example_response", ""),
                "user_type": scenario.get("user_type", "T")  # Default to Thinking type if not specified
            }
        
            # Prepare response with all required fields
            round_data = {
                "round_number": round_number,
                "situation": scenario["situation"],
                "example_T": scenario["example_T"],
                "example_F": scenario["example_F"],
                "example_response": ""  # Add empty string as default example response
            }
            
            return round_data
            
        except (IndexError, KeyError) as e:
            logger.error("Error getting round %d: %s", round_number, e)
            return None
    
    def submit_response(
        self, 
        session_id: str, 
        user_response: str, 
        round_number: int
    ) -> Optional[Dict[str, Any]]:
        """Submit a user response for a round and return the score.
        
        Args:
            session_id: The game session ID
            user_response: The player's response text
            round_number: The round number being responded to
            
        Returns:
            Optional[Dict[str, Any]]: Score and round information, or None if invalid
            
        Raises:
            ValueError: If input validation fails
        """
        if not session_id or not isinstance(session_id, str):
            raise ValueError("Session ID is required and must be a string")
            
        if not user_response or not isinstance(user_response, str):
            raise ValueError("Response text is required and must be a string")
            
        if not isinstance(round_number, int) or round_number < 1:
            raise ValueError("Round number must be a positive integer")
        
        if session_id not in sessions:
            logger.warning("Session not found: %s", session_id)
            return None
            
        session = sessions[session_id]
        
        # Validate round number
        expected_round = session["current_round"] + 1
        if round_number != expected_round:
            logger.warning("Invalid round number: expected %d, got %d", 
                         expected_round, round_number)
            return None
            
        if round_number > len(session["scenarios"]):
            logger.warning("Round number %d exceeds total rounds", round_number)
            return None
        
        try:
            # Calculate score
            score = self._calculate_score(user_response, session["user_type"])
            
            # Update session
            session["current_round"] = round_number
            session["scores"].append(score)
            session["responses"].append({
                "round": round_number,
                "response": user_response,
                "score": score,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.debug("Round %d score for session %s: %.2f", 
                        round_number, session_id, score)
            
            # If game is complete, finalize it
            is_complete = session["current_round"] >= len(session["scenarios"])
            if is_complete:
                logger.info("Game complete for session %s", session_id)
                self._complete_game(session_id)
            
            return {
                "score": score,
                "round_number": round_number,
                "is_complete": is_complete,
                "total_score": sum(session["scores"]),
                "rounds_remaining": len(session["scenarios"]) - round_number
            }
            
        except Exception as e:
            logger.error("Error processing response for session %s: %s", 
                        session_id, e, exc_info=True)
            raise
    
    def get_summary(self, session_id: str) -> Dict[str, Any]:
        """Get game summary including scores and leaderboard position.
        
        Args:
            session_id: The game session ID
            
        Returns:
            Dict[str, Any]: Game summary including scores, rankings, and feedback
            
        Raises:
            ValueError: If session_id is invalid
        """
        global leaderboard, sessions
        
        if not session_id or not isinstance(session_id, str):
            raise ValueError("Session ID is required and must be a string")
            
        if session_id not in sessions:
            logger.warning("Session not found: %s", session_id)
            raise ValueError("Invalid session ID")
            
        try:
            session = sessions[session_id]
            round_scores = session.get("scores", [])
            total_score = sum(round_scores) if round_scores else 0.0
            
            logger.debug("Generating summary for session %s with score %.2f", 
                        session_id, total_score)
            
            # Complete the game if not already completed
            if not session.get("completed", False):
                logger.debug("Game not marked as completed, completing now")
                self._complete_game(session_id)
            
            # Calculate percentile
            percentile = self._calculate_percentile(total_score)
            
            # Get top players for the summary
            top_players = self._get_top_players(3)
            
            # Prepare round scores with additional metadata
            round_details = []
            for i, score in enumerate(round_scores):
                try:
                    response = session["responses"][i] if i < len(session.get("responses", [])) else {}
                    round_details.append({
                        "round_number": i + 1,
                        "score": score,
                        "user_response": response.get("response", ""),
                        "is_correct_style": score > 50,  # Score > 50 means correct style
                        "timestamp": response.get("timestamp", datetime.now().isoformat())
                    })
                except (IndexError, KeyError) as e:
                    logger.warning("Error processing round %d details: %s", i + 1, e)
                    continue
            
            # Create summary dictionary
            summary = {
                "session_id": session_id,
                "nickname": session.get("nickname", ""),
                "user_type": session.get("user_type", UserType.THINKING).value,
                "start_time": session.get("start_time", datetime.now()).isoformat(),
                "end_time": session.get("end_time", datetime.now()).isoformat(),
                "total_rounds": len(session.get("scenarios", [])),
                "completed_rounds": len(round_scores),
                "total_score": total_score,
                "round_scores": round_details,
                "percentile": round(percentile, 1),
                "rank": self._calculate_rank(total_score, leaderboard),
                "feedback": self._generate_feedback(session, total_score),
                "top_players": top_players
            }
            
            # Validate against the Pydantic model
            try:
                game_summary = GameSummary(**summary)
                logger.debug("Successfully validated game summary")
                return game_summary.dict()
            except Exception as e:
                logger.error("GameSummary validation error: %s", e, exc_info=True)
                # Return the summary even if validation fails, but log the error
                return summary
                
        except Exception as e:
            logger.error("Error generating game summary: %s", e, exc_info=True)
            raise
    
    def _calculate_percentile(self, score: float) -> float:
        """Calculate the percentile rank of a score relative to the leaderboard.
        
        Args:
            score: The score to calculate percentile for
            
        Returns:
            float: Percentile rank (0-100)
        """
        global leaderboard
        
        # Ensure leaderboard is a list
        current_leaderboard = leaderboard if isinstance(leaderboard, list) else []
        
        if not current_leaderboard:
            return 100.0
            
        # Count how many players have a lower or equal score
        better_scores = [
            float(s.get("total_score", 0)) 
            for s in current_leaderboard 
            if float(s.get("total_score", 0)) >= score
        ]
        
        # Calculate percentile (0-100)
        percentile = (len(better_scores) / len(current_leaderboard)) * 100 if current_leaderboard else 100
        return min(100.0, max(0.0, percentile))  # Ensure within 0-100 range
        
    def _calculate_rank(self, score: float, leaderboard_data: List[Dict[str, Any]]) -> int:
        """Calculate the rank of a score in the leaderboard.
        
        Args:
            score: The score to find rank for
            leaderboard_data: The leaderboard data
            
        Returns:
            int: 1-based rank (1 is best)
        """
        if not leaderboard_data:
            return 1
            
        # Count how many players have a higher score and add 1 for 1-based ranking
        higher_scores = [s for s in leaderboard_data 
                        if s.get("total_score", 0) > score]
        return len(higher_scores) + 1
        
    def _get_top_players(self, count: int = 3) -> List[Dict[str, str]]:
        """Get the top N players from the leaderboard.
        
        Args:
            count: Number of top players to return
            
        Returns:
            List[Dict[str, str]]: List of top players with nickname and score as string
        """
        global leaderboard
        
        try:
            # Ensure leaderboard is a list
            current_leaderboard = leaderboard if isinstance(leaderboard, list) else []
            
            if not current_leaderboard:
                return []
                
            # Sort by total_score descending and take top N
            sorted_players = sorted(
                current_leaderboard, 
                key=lambda x: float(x.get("total_score", 0)), 
                reverse=True
            )[:max(1, count)]  # Ensure at least 1 player is returned
            
            return [
                {
                    "nickname": str(p.get("nickname", "")),
                    "score": str(p.get("total_score", 0))
                }
                for p in sorted_players
            ]
            
        except Exception as e:
            logger.error("Error getting top players: %s", e, exc_info=True)
            return []
            
    def _complete_game(self, session_id: str) -> None:
        """Mark a game as completed and update leaderboard and stats.
        
        Args:
            session_id: The ID of the session to complete
            
        Raises:
            ValueError: If session_id is invalid
        """
        global leaderboard, sessions
        
        if not session_id or not isinstance(session_id, str):
            raise ValueError("Session ID is required and must be a string")
            
        if session_id not in sessions:
            logger.warning("Session %s not found in _complete_game", session_id)
            return
            
        try:
            session = sessions[session_id]
            session["completed"] = True
            session["end_time"] = datetime.now()
            
            # Calculate total score
            round_scores = session.get("scores", [])
            total_score = sum(round_scores) if round_scores else 0.0
            
            logger.info("Completing game for %s (session: %s) with score: %.2f", 
                       session.get("nickname", "Unknown"), session_id, total_score)
            
            # Update ranking service
            self._update_ranking_service(session, total_score)
            
            # Save stats
            self._save_game_stats()
            
            logger.info("Successfully completed game for session %s", session_id)
            
        except Exception as e:
            logger.error("Error completing game: %s", e, exc_info=True)
            raise
            
    def _update_ranking_service(self, session: Dict[str, Any], total_score: float) -> None:
        """
        Update the ranking service with the player's score.
        
        Args:
            session: The game session data
            total_score: The player's total score
        """
        global leaderboard
        
        try:
            # Ensure leaderboard is properly initialized
            current_leaderboard = leaderboard if isinstance(leaderboard, list) else []
            
            # Update the ranking service
            ranking_service.update_ranking(
                nickname=session.get("nickname", "Anonymous"),
                score=total_score,
                user_type=session.get("user_type", UserType.THINKING).value,
                session_id=session.get("session_id")
            )
            
            # Create leaderboard entry
            leaderboard_entry = {
                "session_id": session.get("session_id", ""),
                "nickname": session.get("nickname", ""),
                "user_type": session.get("user_type", UserType.THINKING).value,
                "total_score": total_score,
                "timestamp": session.get("end_time", datetime.now()).isoformat()
            }
            
            # Create a new list to avoid modifying the global during iteration
            updated_leaderboard = list(current_leaderboard)
            
            # Check if this session is already in the leaderboard
            session_id = leaderboard_entry.get("session_id")
            if session_id:  # Only proceed if we have a valid session_id
                existing_index = next(
                    (i for i, e in enumerate(updated_leaderboard) 
                     if e.get("session_id") == session_id),
                    None
                )
                
                if existing_index is not None:
                    # Update existing entry
                    updated_leaderboard[existing_index] = leaderboard_entry
                else:
                    # Add new entry
                    updated_leaderboard.append(leaderboard_entry)
                
                # Sort and keep top 100 entries
                updated_leaderboard.sort(key=lambda x: float(x.get("total_score", 0)), reverse=True)
                
                # Update the global leaderboard
                leaderboard = updated_leaderboard[:100]
                
                logger.info("Updated ranking service and leaderboard for player: %s", 
                           session.get("nickname", "Anonymous"))
            else:
                logger.warning("Cannot update leaderboard: missing session_id in leaderboard entry")
                       
        except Exception as e:
            logger.error("Error updating ranking: %s", e, exc_info=True)
            raise
            
    def _update_leaderboard(self, entry: Dict[str, Any]) -> None:
        """Update the leaderboard with a new or updated entry.
        
        Args:
            entry: The leaderboard entry to add or update
        """
        global leaderboard
        
        try:
            # Ensure leaderboard is properly initialized
            current_leaderboard = leaderboard if isinstance(leaderboard, list) else []
            
            session_id = entry.get("session_id")
            if not session_id:
                logger.warning("Cannot update leaderboard: missing session_id")
                return
                
            # Create a new list to avoid modifying the global during iteration
            updated_leaderboard = list(current_leaderboard)
            
            # Check if this session is already in the leaderboard
            existing_index = next(
                (i for i, e in enumerate(updated_leaderboard) 
                 if e.get("session_id") == session_id),
                None
            )
            
            if existing_index is not None:
                # Update existing entry
                updated_leaderboard[existing_index] = entry
            else:
                # Add new entry
                updated_leaderboard.append(entry)
                
            # Sort and keep top 100 entries
            updated_leaderboard.sort(key=lambda x: float(x.get("total_score", 0)), reverse=True)
            
            # Update the global leaderboard
            leaderboard = updated_leaderboard[:100]
            
        except Exception as e:
            logger.error("Error updating leaderboard: %s", e, exc_info=True)
            raise
            
    def _update_ranking_service(self, session: Dict[str, Any], score: float) -> None:
        """Update the ranking service with the final score.
        
        Args:
            session: The game session data
            score: The final score to record
        """
        try:
            ranking_service.update_ranking(
                session_id=session.get("session_id", ""),
                nickname=session.get("nickname", ""),
                score=score,
                user_type=session.get("user_type", UserType.THINKING).value
            )
            logger.debug("Successfully updated ranking service")
            # Try to get rankings directly from the file
            try:
                ranking_file = os.path.join(ranking_service.data_dir, 'rankings.json')
                print(f"Checking ranking file at: {ranking_file}")
                if os.path.exists(ranking_file):
                    with open(ranking_file, 'r') as f:
                        file_content = f.read()
                        print(f"Ranking file content: {file_content}")
                else:
                    print("Ranking file does not exist")
            except Exception as e:
                print(f"Error reading ranking file: {e}")
                
        except Exception as e:
            print(f"Error updating ranking: {e}")
            import traceback
            traceback.print_exc()
        
        global leaderboard
        try:
            if isinstance(leaderboard, list):
                # Create a new sorted list to avoid modifying the global during sorting
                sorted_leaderboard = sorted(leaderboard, key=lambda x: float(x.get("total_score", 0)), reverse=True)
                leaderboard = sorted_leaderboard[:100]  # Keep only the top 100 entries
            else:
                leaderboard = []
                
            # Save stats
            save_stats()
            print("Game completion and stats saved")
        except Exception as e:
            print(f"Error updating leaderboard: {e}")
            import traceback
            traceback.print_exc()
    
    def _calculate_score(self, response: str, user_type: UserType) -> float:
        """Calculate score based on how well the response matches the opposite style.
        
        Args:
            response: The player's response text
            user_type: The player's MBTI type (T/F)
            
        Returns:
            float: Score between 0 and 100, where higher is better
            
        Raises:
            ValueError: If response is empty or invalid
        """
        if not response or not isinstance(response, str):
            raise ValueError("Response must be a non-empty string")
            
        try:
            # Get prediction from ML model
            prediction = predict_tf_style(response)
            
            # Validate prediction format
            if not isinstance(prediction, dict) or not all(
                f"{t}_prob" in prediction for t in ["T", "F"]
            ):
                logger.error("Invalid prediction format: %s", prediction)
                return 0.0
            
            # Convert user type to the style we want to match (opposite of user's type)
            target_style = self._get_opposite_style(user_type)
            
            # Get the probability of the target style
            score = float(prediction[f"{target_style}_prob"])
            
            # Ensure score is within valid range
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.error("Error calculating score: %s", e, exc_info=True)
            return 0.0
    
    def _get_opposite_style(self, user_type: UserType) -> str:
        """Get the opposite style as a string for dictionary access.
        
        Args:
            user_type: The user's MBTI type (T/F)
            
        Returns:
            str: The opposite type ('T' or 'F')
            
        Raises:
            ValueError: If user_type is invalid
        """
        if not isinstance(user_type, UserType):
            raise ValueError("Invalid user type")
            
        return "F" if user_type == UserType.THINKING else "T"
    
    def _generate_feedback(self, session: Dict[str, Any], total_score: float) -> str:
        """Generate personalized feedback based on the game results.
        
        Args:
            session: The game session data
            total_score: The player's total score
            
        Returns:
            str: Personalized feedback message in Korean
        """
        try:
            user_type = session.get("user_type")
            if not isinstance(user_type, UserType):
                logger.warning("Invalid user type in session: %s", user_type)
                return "게임 결과를 분석 중입니다. 나중에 다시 시도해주세요."
                
            target_style = self._get_opposite_style(user_type)
            
            # Define feedback tiers
            feedback_tiers = [
                (80, f"훌륭해요! {target_style} 스타일을 매우 잘 이해하고 계시네요!"),
                (60, f"잘 하셨습니다! {target_style} 스타일을 이해하는 데 도움이 될 거예요."),
                (0, f"조금 더 연습이 필요해요. {target_style} 스타일을 이해하는 데 집중해보세요!")
            ]
            
            # Find the appropriate feedback message
            for threshold, message in feedback_tiers:
                if total_score >= threshold:
                    return message
                    
            return "게임 결과를 분석 중입니다."
            
        except Exception as e:
            logger.error("Error generating feedback: %s", e, exc_info=True)
            return "피드백을 생성하는 중에 오류가 발생했습니다."

    def _save_game_stats(self) -> None:
        """
        Save game statistics to disk.
        
        This method handles the saving of game statistics and includes
        error handling to prevent game disruption if saving fails.
        """
        try:
            save_stats()
            logger.debug("Successfully saved game stats")
        except Exception as e:
            logger.error("Error saving game stats: %s", e, exc_info=True)
            # Continue even if stats save fails to prevent game disruption

# Create singleton instance of GameService
game_service = GameService()

# Load initial stats on module import
try:
    load_stats()
    logger.info("Successfully loaded initial game stats")
except Exception as e:
    logger.error("Failed to load initial game stats: %s", e, exc_info=True)
    # Continue with default stats if loading fails

# Add module-level documentation
__all__ = ['game_service', 'GameService', 'GameSessionDict', 'LeaderboardEntry']

# This ensures the module can be imported directly
if __name__ == "__main__":
    print(f"{__name__} module loaded successfully")
