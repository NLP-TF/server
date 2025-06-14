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
from sqlalchemy import select
from app.db.database import sync_engine
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
        "friend_message": "시험에 떨어졌어... 너무 속상해"
    },
    {
        "situation": "동료가 실수로 커피를 쏟았을 때",
        "friend_message": "어머나, 미안해요! 제가 커피를 쏟아버렸어요..."
    },
    {
        "situation": "팀 프로젝트에서 의견이 엇갈릴 때",
        "friend_message": "우리 팀원들끼리 의견이 너무 안 맞는 것 같아..."
    },
    {
        "situation": "소중한 물건을 잃어버렸을 때",
        "friend_message": "남자친구가 사준 반지를 잃어버렸어 어떡하지"
    }
]

# Type aliases for backward compatibility
GameSessions = Dict[str, GameSessionDict]
Leaderboard = List[Dict[str, Any]]  # Using Dict for backward compatibility

# Database session dependency
def get_db():
    with get_db_session() as db:
        yield db

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
        try:
            self.stats = load_stats() or {
                'total_games': 0,
                'total_players': 0,
                'avg_score': 0.0,
                'last_updated': datetime.now().isoformat()
            }
            logger.info("GameService initialized with %d scenarios and stats", len(self.scenarios))
        except Exception as e:
            logger.error("Failed to load stats: %s", str(e))
            self.stats = {
                'total_games': 0,
                'total_players': 0,
                'avg_score': 0.0,
                'last_updated': datetime.now().isoformat()
            }
            logger.info("GameService initialized with default stats")
        
    async def start_game(self, nickname: str, user_type: UserType) -> str:
        """Start a new game session.
        
        Args:
            nickname: Player's nickname
            user_type: Player's MBTI type (T/F)
            
        Returns:
            str: Unique session ID for the new game session
            
        Raises:
            ValueError: If nickname is empty or user_type is invalid
        """
        if not nickname or not nickname.strip():
            raise ValueError("Nickname cannot be empty")
            
        # Convert string to UserType if needed
        if isinstance(user_type, str):
            try:
                user_type = UserType(user_type.upper())
            except ValueError:
                raise ValueError("Invalid user type. Must be 'T' or 'F'")
        
        if user_type not in (UserType.THINKING, UserType.FEELING):
            raise ValueError("Invalid user type. Must be 'T' or 'F'")
            
        db = next(get_db())
        try:
            # Create new game session in database
            db_session = DBGameSession(
                id=str(uuid.uuid4()),
                nickname=nickname,
                user_type=user_type.value,
                total_score=0.0
            )
            
            async with db.begin():
                db.add(db_session)
                await db.flush()
                
                # Also store in memory for active sessions
                sessions[db_session.id] = {
                    'session_id': db_session.id,
                    'nickname': nickname,
                    'user_type': user_type,
                    'scenarios': self.scenarios.copy(),
                    'current_round': 1,
                    'scores': [],
                    'responses': [],
                    'start_time': datetime.now(),
                    'end_time': None,
                    'completed': False
                }
                
                logger.info("Started new game session %s for %s (type: %s)", 
                          db_session.id, nickname, user_type)
                return db_session.id
                
        except Exception as e:
            await db.rollback()
            logger.error("Failed to start game: %s", str(e))
            raise ValueError(f"Failed to start game: {str(e)}")
        finally:
            await db.close()
            
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
            
            # Prepare response with required fields
            round_data = {
                "round_number": round_number,
                "situation": scenario["situation"],
                "friend_message": scenario["friend_message"],
                "example_response": ""
            }
            
            return round_data
            
        except (IndexError, KeyError) as e:
            logger.error("Error getting round %d: %s", round_number, e)
            return None
    
    async def submit_response(
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
        if not session_id or not user_response or not round_number:
            raise ValueError("Missing required fields")
            
        if session_id not in sessions:
            raise ValueError("Invalid session ID")
            
        session = sessions[session_id]
        
        # Validate round number
        if round_number != session['current_round']:
            raise ValueError(f"Invalid round number. Expected {session['current_round']}, got {round_number}")
        
        # Validate response
        user_response = user_response.strip()
        if not user_response:
            raise ValueError("Response cannot be empty")
        
        # Calculate score and determine if response matches the opposite style
        score = self._calculate_score(user_response, session['user_type'])
        is_correct_style = score >= 50.0  # Assuming 50% is the threshold
        
        # Get a new async database session
        db = next(get_db())
        try:
            # Start a transaction
            async with db.begin():
                # Get the game session from database
                stmt = select(DBGameSession).where(DBGameSession.id == session_id)
                result = await db.execute(stmt)
                db_session = result.scalar_one_or_none()
                if not db_session:
                    raise ValueError("Game session not found in database")
                
                # Save player score to database
                player_score = DBPlayerScore(
                    session_id=session_id,
                    round_number=round_number,
                    user_response=user_response,
                    is_correct_style=is_correct_style,
                    score=score
                )
                
                # Add player score
                db.add(player_score)
                
                # Update total score in game session
                db_session.total_score = (db_session.total_score or 0.0) + score
                
                # Store response in memory for active session
                session['scores'].append(score)
                session['responses'].append({
                    'round': round_number,
                    'response': user_response,
                    'score': score,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Update current round
                session['current_round'] += 1
                
                # Check if game is complete (after 5 rounds)
                is_complete = round_number >= 5  # Game ends after 5 rounds
                if is_complete:
                    db_session.completed = True
                    db_session.completed_at = datetime.now()
                    logger.info("Marking game as complete. Round: %d/5", round_number)
                
                # Commit the transaction
                await db.commit()
                
            # Complete the game outside the transaction if needed
            if is_complete:
                # Use a new session for game completion to avoid transaction conflicts
                await self._complete_game(session_id)
                
            return {
                'score': score,
                'round_number': round_number,
                'is_complete': is_complete,
                'total_score': db_session.total_score,
                'rounds_remaining': 5 - round_number  # Fixed 5 rounds total
            }
                
        except Exception as e:
            await db.rollback()
            logger.error("Error processing response: %s", str(e), exc_info=True)
            raise ValueError(f"Failed to process response: {str(e)}")
        finally:
            await db.close()
            
    async def get_summary(self, session_id: str) -> Dict[str, Any]:
        """Get game summary including scores and leaderboard position.
        
        Args:
            session_id: The game session ID
            
        Returns:
            Dict[str, Any]: Game summary including scores, rankings, and feedback
            
        Raises:
            ValueError: If session_id is invalid
        """
        db = next(get_db())
        try:
            # Start a transaction
            async with db.begin():
                # Get game session from database using SQLAlchemy 2.0 async syntax
                stmt = select(DBGameSession).where(DBGameSession.id == session_id)
                result = await db.execute(stmt)
                db_session = result.scalar_one_or_none()
                if not db_session:
                    raise ValueError("Game session not found")
                    
                # Get all scores for this session
                stmt = select(DBPlayerScore).where(
                    DBPlayerScore.session_id == session_id
                ).order_by(DBPlayerScore.round_number)
                result = await db.execute(stmt)
                scores = result.scalars().all()
                
                if not scores:
                    raise ValueError("No scores found for this session")
                    
                # Calculate total score if not already set
                if not db_session.total_score:
                    db_session.total_score = sum(score.score for score in scores)
                    await db.flush()
                
                total_score = db_session.total_score
                
                # Get leaderboard data - filter for completed games (where completed_at is not null)
                stmt = select(
                    DBGameSession.id,
                    DBGameSession.nickname,
                    DBGameSession.user_type,
                    DBGameSession.total_score,
                    DBGameSession.completed_at
                ).where(
                    DBGameSession.completed_at.isnot(None)
                ).order_by(
                    DBGameSession.total_score.desc()
                )
                
                result = await db.execute(stmt)
                leaderboard_data = result.all()
                
                # Convert to list of dicts for compatibility
                leaderboard = [{
                    'id': str(session.id),
                    'nickname': session.nickname,
                    'user_type': session.user_type,
                    'total_score': float(session.total_score) if session.total_score else 0.0,
                    'completed_at': session.completed_at.isoformat() if session.completed_at else None
                } for session in leaderboard_data]
                
                # Calculate rank and percentile
                rank = await self._calculate_rank(total_score, leaderboard)
                percentile = await self._calculate_percentile(total_score)
                
                # Get top players (excluding current player)
                top_players = [
                    {
                        'nickname': str(p['nickname']),
                        'score': str(float(p['total_score'])),  # Convert to float then string
                        'rank': str(i + 1)  # Convert rank to string
                    }
                    for i, p in enumerate(leaderboard)
                    if p['id'] != session_id
                ][:3]  # Take top 3
                
                # Prepare round scores
                round_scores = [{
                    'round_number': score.round_number,
                    'score': score.score,
                    'user_response': score.user_response,
                    'is_correct_style': score.is_correct_style
                } for score in scores]
                
                # Generate feedback
                feedback = self._generate_feedback({
                    'user_type': db_session.user_type,
                    'scores': [s.score for s in scores]
                }, total_score)
                
                # Create summary dictionary
                summary = {
                    "session_id": session_id,
                    "nickname": db_session.nickname,
                    "user_type": db_session.user_type,
                    "start_time": db_session.created_at.isoformat(),
                    "end_time": db_session.completed_at.isoformat() if db_session.completed_at else None,
                    "total_rounds": len(self.scenarios),
                    "completed_rounds": len(round_scores),
                    "total_score": float(total_score) if total_score else 0.0,
                    "round_scores": round_scores,
                    "percentile": percentile,
                    "rank": rank,
                    "feedback": feedback,
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
        finally:
            await db.close()
    
    async def _calculate_percentile(self, score: float) -> float:
        """Calculate the percentile rank of a score relative to the leaderboard.
        
        Args:
            score: The score to calculate percentile for
            
        Returns:
            float: Percentile rank (0-100)
        """
        db = next(get_db())
        try:
            # Get total number of completed games
            stmt = select(func.count()).select_from(DBGameSession).where(
                DBGameSession.completed_at.isnot(None)
            )
            result = await db.execute(stmt)
            total_players = result.scalar()
            
            if total_players == 0:
                return 100.0
                
            # Count how many players have a lower score
            stmt = select(func.count()).select_from(DBGameSession).where(
                DBGameSession.completed_at.isnot(None),
                DBGameSession.total_score >= score
            )
            result = await db.execute(stmt)
            better_players = result.scalar()
            
            # Calculate percentile (0-100)
            percentile = (better_players / total_players) * 100
            return min(100.0, max(0.0, percentile))  # Ensure within 0-100 range
            
        except Exception as e:
            logger.error("Error calculating percentile: %s", str(e), exc_info=True)
            return 100.0  # Return 100% on error to avoid breaking the game
        finally:
            await db.close()
        
    async def _calculate_rank(self, score: float, leaderboard_data: List[Dict[str, Any]] = None) -> int:
        """Calculate the rank of a score in the leaderboard.
        
        Args:
            score: The score to find rank for
            leaderboard_data: Optional leaderboard data (if not provided, will query the database)
            
        Returns:
            int: 1-based rank (1 is best)
        """
        if leaderboard_data is not None:
            # Use provided leaderboard data
            if not leaderboard_data:
                return 1
            
            # Count how many players have a higher score and add 1 for 1-based ranking
            higher_scores = [s for s in leaderboard_data 
                          if s.get("total_score", 0) > score]
            return len(higher_scores) + 1
            
        # If leaderboard data not provided, query the database
        db = next(get_db())
        try:
            stmt = select(func.count()).select_from(DBGameSession).where(
                DBGameSession.completed_at.isnot(None),
                DBGameSession.total_score > score
            )
            result = await db.execute(stmt)
            rank = result.scalar() + 1  # Add 1 for 1-based ranking
            return rank
                
        except Exception as e:
            logger.error("Error calculating rank: %s", str(e), exc_info=True)
            return 1  # Return rank 1 on error to avoid breaking the game
        finally:
            await db.close()
        
    async def _get_top_players(self, count: int = 3) -> List[Dict[str, str]]:
        """Get the top N players from the leaderboard.
        
        Args:
            count: Number of top players to return (default: 3)
            
        Returns:
            List[Dict[str, str]]: List of top players with nickname, score and rank
        """
        db = next(get_db())
        try:
            # Query completed game sessions ordered by total_score (descending)
            stmt = select(DBGameSession).where(
                DBGameSession.completed_at.isnot(None)
            ).order_by(
                DBGameSession.total_score.desc()
            ).limit(count)
            
            result = await db.execute(stmt)
            top_sessions = result.scalars().all()
            
            # Format the top players
            top_players = []
            for i, session in enumerate(top_sessions, 1):
                top_players.append({
                    "nickname": session.nickname or "Anonymous",
                    "score": str(session.total_score or 0),
                    "rank": str(i)
                })
                
            return top_players
            
        except Exception as e:
            logger.error("Error getting top players: %s", str(e), exc_info=True)
            return []
        finally:
            await db.close()
            
    async def _complete_game(self, session_id: str) -> None:
        """Mark a game as completed and update leaderboard and stats.
        
        Args:
            session_id: The ID of the session to complete
            
        Raises:
            ValueError: If session_id is invalid
        """
        db = next(get_db())
        try:
            # Start a new transaction
            async with db.begin():
                # Get the game session from database using SQLAlchemy 2.0 syntax with async
                stmt = select(DBGameSession).where(DBGameSession.id == session_id)
                result = await db.execute(stmt)
                db_session = result.scalar_one_or_none()
                if not db_session:
                    raise ValueError("Game session not found")
                    
                # Get the session from memory
                session = sessions.get(session_id)
                if not session:
                    raise ValueError("Session data not found")
                    
                # Calculate total score
                total_score = sum(session.get('scores', []))
                
                # Update the database session
                db_session.completed_at = datetime.now()
                db_session.total_score = total_score
                
                # Flush changes to the database
                await db.flush()
                
                # Update stats
                self.stats["total_games"] = self.stats.get("total_games", 0) + 1
                self.stats["total_players"] = len(sessions)
                
                # Save updated stats
                self._save_game_stats()
                
                logger.info(f"Game {session_id} completed with score {total_score}")
                
                # Update the ranking service asynchronously
                await self._update_ranking_service({
                    'id': session_id,
                    'nickname': session.get('nickname', 'Anonymous'),
                    'user_type': session.get('user_type', 'T'),
                    'total_score': total_score,
                    'timestamp': datetime.now().isoformat()
                }, total_score)
                
        except Exception as e:
            await db.rollback()
            logger.error(f"Error completing game {session_id}: {str(e)}", exc_info=True)
            raise
        finally:
            await db.close()
            
    async def _update_ranking_service(self, session: Dict[str, Any], total_score: float) -> None:
        """Update the ranking service with the player's score.
        
        Args:
            session: The game session data
            total_score: The player's total score
            
        Note:
            This method is called within a database transaction, so it should not
            perform any database operations that would conflict with the transaction.
        """
        try:
            logger.debug("Starting ranking update for session %s", session.get("id", "unknown"))
            
            # Ensure we have a valid session ID
            session_id = session.get("id")
            if not session_id:
                logger.warning("Cannot update ranking: missing session ID")
                return
                
            # Update the ranking service asynchronously
            await ranking_service.update_ranking(
                nickname=session.get("nickname", "Anonymous"),
                score=total_score,
                user_type=session.get("user_type", "T"),  # Default to 'T' if not specified
                session_id=session_id
            )
            
            logger.info("Successfully updated ranking for session %s with score %.2f", 
                       session_id, total_score)
            
            # Try to log current rankings for debugging
            try:
                rankings_file = ranking_service.rankings_file
                if rankings_file.exists():
                    async with aiofiles.open(rankings_file, 'r') as f:
                        content = await f.read()
                        logger.debug("Current rankings: %s", content)
            except Exception as e:
                logger.debug("Could not read rankings file: %s", str(e))
            
        except Exception as e:
            logger.error("Error updating ranking service: %s", str(e), exc_info=True)
            # Don't re-raise to avoid disrupting game flow
            pass
            
    def _update_leaderboard(self, entry: Dict[str, Any]) -> None:
        """Update the leaderboard with a new or updated entry.
        
        Note: This method is kept for backward compatibility but is no longer needed
        as leaderboard is now managed by the database.
        
        Args:
            entry: The leaderboard entry to add or update
        """
        try:
            logger.debug("Leaderboard update requested for session %s", entry.get("session_id"))
            # No action needed as leaderboard is now managed by the database
            pass
            
        except Exception as e:
            logger.error("Error in leaderboard update: %s", str(e), exc_info=True)
            # Don't raise to avoid disrupting game flow
            pass

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
            session: The game session data (can be a dict or DB model)
            total_score: The player's total score
            
        Returns:
            str: Personalized feedback message in Korean
        """
        try:
            # Handle both dict and SQLAlchemy model
            if hasattr(session, 'user_type'):
                # It's a SQLAlchemy model
                user_type = session.user_type
                scores = session.scores if hasattr(session, 'scores') else []
            else:
                # It's a dict
                user_type = session.get("user_type", "T")  # Default to 'T' if not specified
                scores = session.get("scores", [])
            
            # Calculate average score per round
            num_rounds = len(scores)
            avg_score = total_score / num_rounds if num_rounds > 0 else 0
            
            # Determine feedback based on average score
            if avg_score >= 75:
                if user_type == "T" or user_type == UserType.THINKING:
                    return ("당신은 T 유형으로서 F 유형의 감정을 정말 잘 이해하고 있어요! "
                            "상대방의 감정에 공감하는 능력이 뛰어나네요.")
                else:
                    return ("당신은 F 유형으로서 T 유형의 사고 방식을 정말 잘 이해하고 있어요! "
                            "논리적인 사고에 대한 이해도가 높으시네요.")
                            
            elif avg_score >= 50:
                if user_type == "T" or user_type == UserType.THINKING:
                    return ("T 유형으로서 F 유형의 감정을 잘 이해하고 계세요. "
                            "약간의 연습만 더 하면 더 나은 결과를 얻을 수 있을 거예요!")
                else:
                    return ("F 유형으로서 T 유형의 사고 방식을 잘 이해하고 계세요. "
                            "조금 더 연습하면 더 나은 결과를 얻을 수 있을 거예요!")
                            
            else:
                if user_type == "T" or user_type == UserType.THINKING:
                    return ("T 유형으로서 F 유형의 감정을 이해하는 데 조금 어려움이 있으신 것 같아요. "
                            "상대방의 감정에 더 공감하려는 노력이 필요해 보여요.")
                else:
                    return ("F 유형으로서 T 유형의 사고 방식을 이해하는 데 조금 어려움이 있으신 것 같아요. "
                            "논리적인 사고에 대한 이해를 높이기 위해 노력해보세요!")
                            
        except Exception as e:
            logger.error("Error generating feedback: %s", str(e), exc_info=True)
            return "게임 결과에 대한 피드백을 생성하는 중 오류가 발생했습니다."

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
