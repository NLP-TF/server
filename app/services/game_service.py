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


# Default scenarios to use if scenarios.json is not found or invalid
DEFAULT_SCENARIOS = [
    {
        "situation": "친구_갈등",
        "situation_detail": "중요한 발표에서 실수했을 때",
        "friend_message": "발표 중에 머리가 하얘져서 아무 말도 못했어… 너무 창피해",
    },
    {
        "situation": "연인_갈등",
        "situation_detail": "기념일을 깜빡했을 때",
        "friend_message": "오늘 무슨 날인지 기억나? 나는 너한테 실망이야.",
    },
    {
        "situation": "가족_갈등",
        "situation_detail": "집안일을 도와주지 않을 때",
        "friend_message": "너는 왜 항상 집안일을 도와주지 않니? 나만 바쁜 줄 아니?",
    },
    {
        "situation": "직장_갈등",
        "situation_detail": "팀 프로젝트에서 실수했을 때",
        "friend_message": "이번 프로젝트에서 네 태도가 많이 아쉬웠어. 팀원들에게 피해를 주고 있어.",
    },
    {
        "situation": "친구_갈등",
        "situation_detail": "약속을 자주 어길 때",
        "friend_message": "너는 왜 항상 약속을 안 지켜? 나한테 관심이 없는 거야?",
    },
]


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
            with open(SCENARIOS_FILE, "r", encoding="utf-8") as f:
                scenarios = json.load(f)
                if not isinstance(scenarios, list):
                    error_msg = (
                        f"Scenarios must be a list, got {type(scenarios).__name__}"
                    )
                    logger.error(error_msg)
                    raise json.JSONDecodeError(
                        error_msg, doc=str(SCENARIOS_FILE), pos=0
                    )
                logger.info(
                    "Loaded %d scenarios from %s", len(scenarios), SCENARIOS_FILE
                )
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
        "average_score": 0,  # This would be calculated from game data
        "last_updated": datetime.now().isoformat(),
        "total_players": len(leaderboard),
    }

    try:
        with open(STATS_FILE, "w", encoding="utf-8") as f:
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
            with open(STATS_FILE, "r", encoding="utf-8") as f:
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
            # Ensure user_type is a string 'T' or 'F'
            user_type_str = (
                user_type.value
                if hasattr(user_type, "value")
                else str(user_type).upper()
            )
            user_type_str = (
                "T" if user_type_str.upper() == "T" else "F"
            )  # Force to 'T' or 'F'

            logger.debug(f"Creating session with user_type: {user_type_str}")

            # Create new game session in database
            db_session = DBGameSession(
                id=str(uuid.uuid4()),
                nickname=nickname,
                user_type=user_type_str,
                total_score=0.0,
            )

            async with db.begin():
                db.add(db_session)
                await db.flush()

                # 랜덤 시나리오 5개 추출
                all_scenarios = self.scenarios.copy()
                if len(all_scenarios) < 5:
                    raise ValueError("시나리오가 5개 이상 필요합니다.")
                selected_scenarios = random.sample(all_scenarios, 5)

                # Also store in memory for active sessions
                sessions[db_session.id] = {
                    "session_id": db_session.id,
                    "nickname": nickname,
                    "user_type": user_type,
                    "scenarios": selected_scenarios,
                    "current_round": 1,
                    "scores": [],
                    "responses": [],
                    "start_time": datetime.now(),
                    "end_time": None,
                    "completed": False,
                }

                logger.info(
                    "Started new game session %s for %s (type: %s)",
                    db_session.id,
                    nickname,
                    user_type,
                )
                return db_session.id

        except Exception as e:
            await db.rollback()
            logger.error("Failed to start game: %s", str(e))
            raise ValueError(f"Failed to start game: {str(e)}")
        finally:
            await db.close()

    def get_round(self, session_id: str, round_number: int) -> Optional[Dict[str, Any]]:
        """
        Get round information for the given session and round number.

        Args:
            session_id: The session ID
            round_number: The round number (0 or 1-based index)

        Returns:
            Optional[Dict[str, Any]]: Round information including situation, or None if invalid
        """
        try:
            session = sessions.get(session_id)
            if not session:
                logger.warning(f"Session not found: {session_id}")
                return None

            scenarios = session.get("scenarios", [])

            # Store the original round number for the response
            original_round_number = round_number

            # Convert to 0-based for array access
            array_index = round_number - 1 if round_number > 0 else 0

            if not 0 <= array_index < len(scenarios):
                logger.warning("Invalid round number: %d", round_number)
                return None

            scenario = scenarios[array_index]

            # Extract situation and situation_detail from the scenario
            situation = scenario.get("situation", "친구_갈등")
            situation_detail = scenario.get("situation_detail", "")

            round_data = {
                "round_number": original_round_number,  # Return the original 1-based round number
                "situation": situation,  # General situation type (e.g., "친구_갈등")
                "situation_detail": situation_detail,  # Specific situation description
                "friend_message": scenario["friend_message"],
                "example_response": "",
            }
            return round_data
        except (IndexError, KeyError) as e:
            logger.error("Error getting round %d: %s", round_number, e)
            return None

    async def submit_response(
        self,
        session_id: str,
        user_response: str,
        round_number: int,
        situation: str = "친구_갈등",
    ) -> Optional[Dict[str, Any]]:
        """Submit a user response for a round and return the score.

        Args:
            session_id: The game session ID
            user_response: The player's response text
            round_number: The round number being responded to (0 or 1-based)
            situation: The situation label (e.g., "연인_갈등", "친구_갈등")

        Returns:
            Optional[Dict[str, Any]]: Score and round information, or None if invalid

        Raises:
            ValueError: If input validation fails
        """
        if not session_id or not user_response or round_number is None:
            raise ValueError("Missing required fields")

        if session_id not in sessions:
            raise ValueError("Invalid session ID")

        session = sessions[session_id]

        # Get the current round number (1-based)
        current_round = session["current_round"]

        # Convert 0-based round_number to 1-based for comparison if needed
        effective_round = round_number + 1 if round_number == 0 else round_number

        if effective_round != current_round:
            raise ValueError(
                f"Invalid round number. Expected {current_round}, got {round_number}"
            )

        # Get the current round's situation and situation_detail
        round_info = self.get_round(session_id, round_number)
        if not round_info:
            raise ValueError(
                f"Could not find round {round_number} in session {session_id}"
            )

        situation = round_info.get("situation", "친구_갈등")
        situation_detail = round_info.get("situation_detail", "")

        # Calculate score and determine if response matches the opposite style
        score = self._calculate_score(user_response, session["user_type"], situation)
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

                # Save player score to database with situation and situation_detail
                player_score = DBPlayerScore(
                    session_id=session_id,
                    round_number=round_number,
                    situation=situation,
                    situation_detail=situation_detail,  # Include the situation detail
                    user_response=user_response,
                    is_correct_style=is_correct_style,
                    score=score,
                )

                # Add player score
                db.add(player_score)

                # Update total score in game session
                db_session.total_score = (db_session.total_score or 0.0) + score

                # Store response in memory for active session
                session["scores"].append(score)
                session["responses"].append(
                    {
                        "round": round_number,
                        "response": user_response,
                        "score": score,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                # Update current round
                session["current_round"] += 1

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
                "score": score,
                "round_number": round_number,
                "is_complete": is_complete,
                "total_score": db_session.total_score,
                "rounds_remaining": 5 - round_number,  # Fixed 5 rounds total
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
        logger.info(f"Getting summary for session: {session_id}")
        db = next(get_db())
        try:
            # Start a transaction
            async with db.begin():
                # Get game session from database using SQLAlchemy 2.0 async syntax
                stmt = select(DBGameSession).where(DBGameSession.id == session_id)
                logger.debug(f"Executing query: {stmt}")
                result = await db.execute(stmt)
                db_session = result.scalar_one_or_none()
                if not db_session:
                    raise ValueError("Game session not found")

                # Log the user_type from database
                logger.info(
                    f"Database user_type: {db_session.user_type} (type: {type(db_session.user_type)})"
                )

                # Get all scores for this session
                stmt = (
                    select(DBPlayerScore)
                    .where(DBPlayerScore.session_id == session_id)
                    .order_by(DBPlayerScore.round_number)
                )
                result = await db.execute(stmt)
                scores = result.scalars().all()

                if not scores:
                    raise ValueError("No scores found for this session")

                # Ensure the game is marked as completed
                if not db_session.completed_at:
                    db_session.completed_at = datetime.now()

                # Calculate total score if not already set
                if not db_session.total_score:
                    db_session.total_score = sum(score.score for score in scores)
                    await db.flush()

                total_score = db_session.total_score

                # Get leaderboard data - filter for completed games (where completed_at is not null)
                stmt = (
                    select(
                        DBGameSession.id,
                        DBGameSession.nickname,
                        DBGameSession.user_type,
                        DBGameSession.total_score,
                        DBGameSession.completed_at,
                    )
                    .where(DBGameSession.completed_at.isnot(None))
                    .order_by(DBGameSession.total_score.desc())
                )

                result = await db.execute(stmt)
                leaderboard_data = result.all()

                # Convert to list of dicts for compatibility
                leaderboard = [
                    {
                        "id": str(session.id),
                        "nickname": session.nickname,
                        "user_type": session.user_type,
                        "total_score": (
                            float(session.total_score) if session.total_score else 0.0
                        ),
                        "completed_at": (
                            session.completed_at.isoformat()
                            if session.completed_at
                            else None
                        ),
                    }
                    for session in leaderboard_data
                ]

                # Calculate rank and percentile
                rank = await self._calculate_rank(total_score, leaderboard)
                percentile = await self._calculate_percentile(total_score)

                # Get all players with scores (excluding current player)
                top_players = [
                    {
                        "nickname": str(p["nickname"]),
                        "score": str(
                            float(p["total_score"])
                        ),  # Convert to float then string
                        "rank": str(i + 1),  # Convert rank to string
                        "user_type": str(p["user_type"]),  # Add user type (T/F)
                    }
                    for i, p in enumerate(leaderboard)
                    if p["id"] != session_id
                ]  # Include all players with scores

                # Prepare round scores with situation and situation_detail
                round_scores = []
                for score in scores:
                    round_score = {
                        "round_number": score.round_number,
                        "score": score.score,
                        "user_response": score.user_response,
                        "is_correct_style": score.is_correct_style,
                        "situation": score.situation
                        or "친구_갈등",  # Default to 친구_갈등 if not set
                        "situation_detail": score.situation_detail
                        or "",  # Include situation_detail if available
                    }
                    round_scores.append(round_score)

                # Log the user_type for debugging
                logger.info(
                    f"[get_summary] Raw user_type from DB: {db_session.user_type} (type: {type(db_session.user_type)})"
                )

                # Get user_type from the session and ensure it's properly formatted
                user_type = (
                    str(db_session.user_type).strip().upper()
                    if db_session.user_type
                    else "T"
                )
                user_type = user_type[
                    0
                ]  # Take first character in case it's a string like 'THINKING'
                user_type = (
                    "T" if user_type == "T" else "F"
                )  # Ensure it's either T or F

                logger.info(f"[get_summary] Processed user_type: {user_type}")

                # Generate feedback based on user type and scores
                feedback = self._generate_feedback(
                    session={
                        "user_type": user_type,
                        "scores": [
                            {
                                "score": score.score,
                                "is_correct_style": score.is_correct_style,
                            }
                            for score in scores
                        ],
                    },
                    total_score=total_score,
                )

                logger.info(
                    f"[get_summary] Generated feedback: {feedback[:50]}..."
                )  # Log first 50 chars of feedback

                # Create summary dictionary
                summary = {
                    "session_id": session_id,
                    "nickname": db_session.nickname,
                    "user_type": db_session.user_type,
                    "start_time": db_session.created_at.isoformat(),
                    "end_time": (
                        db_session.completed_at.isoformat()
                        if db_session.completed_at
                        else None
                    ),
                    "total_rounds": len(self.scenarios),
                    "completed_rounds": len(round_scores),
                    "total_score": float(total_score) if total_score else 0.0,
                    "round_scores": round_scores,
                    "percentile": percentile,
                    "rank": rank,
                    "feedback": feedback,
                    "top_players": top_players,
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
            stmt = (
                select(func.count())
                .select_from(DBGameSession)
                .where(DBGameSession.completed_at.isnot(None))
            )
            result = await db.execute(stmt)
            total_players = result.scalar()

            if total_players == 0:
                return 100.0

            # Count how many players have a lower score
            stmt = (
                select(func.count())
                .select_from(DBGameSession)
                .where(
                    DBGameSession.completed_at.isnot(None),
                    DBGameSession.total_score >= score,
                )
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

    async def _calculate_rank(
        self, score: float, leaderboard_data: List[Dict[str, Any]] = None
    ) -> int:
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
            higher_scores = [
                s for s in leaderboard_data if s.get("total_score", 0) > score
            ]
            return len(higher_scores) + 1

        # If leaderboard data not provided, query the database
        db = next(get_db())
        try:
            stmt = (
                select(func.count())
                .select_from(DBGameSession)
                .where(
                    DBGameSession.completed_at.isnot(None),
                    DBGameSession.total_score > score,
                )
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
            stmt = (
                select(DBGameSession)
                .where(DBGameSession.completed_at.isnot(None))
                .order_by(DBGameSession.total_score.desc())
                .limit(count)
            )

            result = await db.execute(stmt)
            top_sessions = result.scalars().all()

            # Format the top players
            top_players = []
            for i, session in enumerate(top_sessions, 1):
                top_players.append(
                    {
                        "nickname": session.nickname or "Anonymous",
                        "score": str(session.total_score or 0),
                        "rank": str(i),
                    }
                )

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
                total_score = sum(session.get("scores", []))

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
                await self._update_ranking_service(
                    {
                        "id": session_id,
                        "nickname": session.get("nickname", "Anonymous"),
                        "user_type": session.get("user_type", "T"),
                        "total_score": total_score,
                        "timestamp": datetime.now().isoformat(),
                    },
                    total_score,
                )

        except Exception as e:
            await db.rollback()
            logger.error(f"Error completing game {session_id}: {str(e)}", exc_info=True)
            raise
        finally:
            await db.close()

    async def _update_ranking_service(
        self, session: Dict[str, Any], total_score: float
    ) -> None:
        """Update the ranking service with the player's score.

        Args:
            session: The game session data
            total_score: The player's total score

        Note:
            This method is called within a database transaction, so it should not
            perform any database operations that would conflict with the transaction.
        """
        session_id = session.get("id")
        if not session_id:
            logger.warning("Cannot update ranking: missing session ID")
            return

        # Get user_type from session and ensure it's a string
        user_type = str(session.get("user_type", "T")).upper()
        user_type = "T" if user_type == "T" else "F"  # Ensure it's either T or F

        logger.info(
            f"Updating ranking - Session: {session_id}, Nickname: {session.get('nickname')}, "
            f"Score: {total_score}, User Type: {user_type}"
        )

        try:
            await ranking_service.update_ranking(
                nickname=session.get("nickname", "Anonymous"),
                score=total_score,
                user_type=user_type,
                session_id=session_id,
            )

            logger.info(
                "Successfully updated ranking for session %s with score %.2f",
                session_id,
                total_score,
            )

            # Try to log current rankings for debugging
            try:
                rankings_file = ranking_service.rankings_file
                if rankings_file.exists():
                    async with aiofiles.open(rankings_file, "r") as f:
                        content = await f.read()
                        logger.debug("Current rankings: %s", content)
            except Exception as e:
                logger.warning("Failed to log current rankings: %s", str(e))

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
            logger.debug(
                "Leaderboard update requested for session %s", entry.get("session_id")
            )
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
                sorted_leaderboard = sorted(
                    leaderboard,
                    key=lambda x: float(x.get("total_score", 0)),
                    reverse=True,
                )
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

    def _calculate_score(
        self, response: str, user_type: UserType, situation: str = "친구_갈등"
    ) -> float:
        """Calculate score based on how well the response matches the user's chosen style.

        Args:
            response: The player's response text
            user_type: The player's MBTI type (T/F) - this is the style they want to practice
            situation: The situation label (e.g., "연인_갈등", "친구_갈등")

        Returns:
            float: Score between 0 and 100, where higher is better

        Raises:
            ValueError: If response is empty or invalid
        """
        if not response or not response.strip():
            raise ValueError("Response cannot be empty")

        try:
            # Get prediction from the model with the situation
            result = predict_tf_style(response, situation=situation)

            # Use the user's chosen style for scoring
            chosen_style = user_type.value  # Get 'T' or 'F' from the UserType enum
            # Use the correct key from predict_tf_style's return value
            score = result[f"{chosen_style}_prob"]

            # Debug log for verification
            logger.debug("=" * 50)
            logger.debug(f"User type (chosen style): {chosen_style}")
            logger.debug(f"Raw model result: {result}")
            logger.debug(f"Selected score: {score} (from {chosen_style}_prob)")

            logger.debug(
                "Scored response '%s' for type %s in situation %s: %.2f",
                response[:50] + ("..." if len(response) > 50 else ""),
                chosen_style,
                situation,
                score,
            )

            return score

        except Exception as e:
            logger.error("Error calculating score: %s", str(e), exc_info=True)
            # Return a neutral score in case of errors
            return 50.0

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
            logger.info("[_generate_feedback] Starting feedback generation")

            # Handle both dict and SQLAlchemy model
            if hasattr(session, "user_type"):
                # It's a SQLAlchemy model
                user_type = session.user_type
                scores = session.scores if hasattr(session, "scores") else []
                logger.info(
                    f"[Feedback] SQLAlchemy model - user_type: {user_type}, type: {type(user_type)}"
                )
            else:
                # It's a dict
                user_type = session.get(
                    "user_type", "T"
                )  # Default to 'T' if not specified
                scores = session.get("scores", [])
                logger.info(
                    f"[Feedback] Dict input - user_type: {user_type}, type: {type(user_type)}"
                )

            # Debug logging of the entire session for troubleshooting
            logger.debug(f"[Feedback] Full session data: {session}")

            # Convert to string and ensure it's 'T' or 'F'
            if hasattr(user_type, "value"):
                user_type = user_type.value
                logger.info(
                    f"[Feedback] After getting enum value: {user_type} (type: {type(user_type)})"
                )

            # Convert to string and clean up
            user_type_str = str(user_type).strip().upper()
            logger.info(
                f"[Feedback] After string conversion: {user_type_str} (type: {type(user_type_str)})"
            )

            # Extract first character if it's a string (e.g., 'THINKING' -> 'T')
            if len(user_type_str) > 1:
                user_type_str = user_type_str[0]
                logger.info(f"[Feedback] Extracted first character: {user_type_str}")

            # Force to 'T' or 'F' (case-insensitive)
            user_type_str = "T" if user_type_str.upper() == "T" else "F"

            logger.info(f"[Feedback] Final user_type: {user_type_str}")
            is_thinking = user_type_str == "T"
            logger.info(f"[Feedback] is_thinking: {is_thinking}")
            logger.info(f"[Feedback] Scores: {scores}")
            logger.info(f"[Feedback] Total score: {total_score}")

            # Calculate average score per round
            num_rounds = len(scores)
            avg_score = total_score / num_rounds if num_rounds > 0 else 0

            # Log the scores and calculations for debugging
            logger.info(
                f"[Feedback] Total score: {total_score}, Num rounds: {num_rounds}, Avg score: {avg_score}"
            )
            logger.info(
                f"[Feedback] User type: {user_type_str}, Is thinking: {is_thinking}"
            )

            # Determine feedback based on average score and user type
            if is_thinking:
                if avg_score >= 75:
                    return (
                        "T 유형의 사고 방식을 정말 잘 활용하고 계시네요! "
                        "논리적이고 분석적인 접근이 돋보입니다."
                    )
                elif avg_score >= 50:
                    return (
                        "T 유형의 사고 방식을 잘 활용하고 계시네요. "
                        "조금 더 논리적으로 접근해보는 건 어떨까요?"
                    )
                else:  # avg_score < 50
                    return (
                        "T 유형의 사고 방식을 이해하는 데 조금 어려움이 있으신 것 같아요. "
                        "논리적인 사고에 대한 이해를 높이기 위해 노력해보세요!"
                    )
            else:  # Feeling type
                if avg_score >= 75:
                    return (
                        "F 유형의 감정을 정말 잘 이해하고 있어요! "
                        "상대방의 감정에 공감하는 능력이 뛰어나네요."
                    )
                elif avg_score >= 50:
                    return (
                        "F 유형의 감정을 잘 이해하고 있어요. "
                        "조금 더 감정적인 표현을 연습해보는 건 어떨까요?"
                    )
                else:  # avg_score < 50
                    return (
                        "F 유형의 감정을 이해하는 데 조금 어려움이 있으신 것 같아요. "
                        "상대방의 감정에 더 공감하려는 노력이 필요해 보여요."
                    )

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
__all__ = ["game_service", "GameService", "GameSessionDict", "LeaderboardEntry"]

# This ensures the module can be imported directly
if __name__ == "__main__":
    print(f"{__name__} module loaded successfully")
