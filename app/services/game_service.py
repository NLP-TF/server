import json
import os
import random
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from app.models.predict import predict_tf_style
from app.schemas import UserType, RoundScore, GameSummary

# In-memory storage for game state
sessions: Dict[str, dict] = {}
leaderboard: List[dict] = []

# Paths
DATA_DIR = Path("data")
SCENARIOS_FILE = DATA_DIR / "scenarios.json"
STATS_FILE = DATA_DIR / "stats.json"

def load_scenarios() -> List[dict]:
    """Load game scenarios from JSON file."""
    try:
        with open(SCENARIOS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Return default scenarios if file doesn't exist or is invalid
        return [
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
            },
            {
                "situation": "친구가 이별 통보를 했을 때",
                "example_T": "이유가 뭐야? 앞으로 계획은 어떻게 세울 거야?",
                "example_F": "너무 힘들었겠다. 지금 기분이 어떤지 말해줄 수 있어?"
            },
            {
                "situation": "가족과의 갈등 상황에서",
                "example_T": "논리적으로 문제를 해결할 방법을 찾아보자.",
                "example_F": "서로의 감정을 이해하는 게 먼저인 것 같아."
            }
        ]

def save_stats():
    """Save game statistics to file."""
    global leaderboard
    try:
        with open(STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump({"leaderboard": leaderboard}, f, ensure_ascii=False, indent=2)
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
load_stats()

class GameService:
    def __init__(self):
        self.scenarios = load_scenarios()
        
    def start_game(self, nickname: str, user_type: UserType) -> str:
        """Start a new game session."""
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "session_id": session_id,
            "nickname": nickname,
            "user_type": user_type,
            "start_time": datetime.now().isoformat(),
            "round_scores": [],
            "current_round": 0,
            "completed": False
        }
        return session_id
    
    def get_round(self, round_number: int) -> dict:
        """Get round information by number."""
        if round_number < 1 or round_number > len(self.scenarios):
            return None
            
        scenario = self.scenarios[round_number - 1]
        return {
            "round_number": round_number,
            "situation": scenario["situation"],
            "example_response": scenario[f"example_{self._get_opposite_style(UserType.THINKING)}"]
        }
    
    def submit_response(self, session_id: str, user_response: str, round_number: int) -> float:
        """Submit a user response for a round and return the score."""
        if session_id not in sessions:
            raise ValueError("Invalid session ID")
            
        session = sessions[session_id]
        
        # Prevent submitting for the same round multiple times
        if any(score["round_number"] == round_number for score in session["round_scores"]):
            raise ValueError("Response already submitted for this round")
            
        # Get the scenario
        scenario = self.scenarios[round_number - 1]
        example_response = scenario[f"example_{self._get_opposite_style(session['user_type'])}"]
        
        # Score the response
        score = self._calculate_score(user_response, session['user_type'])
        
        # Store the score
        round_score = {
            "round_number": round_number,
            "score": score,
            "user_response": user_response,
            "is_correct_style": score > 50,  # Assuming score > 50 means correct style
            "timestamp": datetime.now().isoformat()
        }
        session["round_scores"].append(round_score)
        
        # Update current round
        session["current_round"] = round_number
        
        # Check if game is completed
        if len(session["round_scores"]) >= len(self.scenarios):
            self._complete_game(session_id)
            
        return score
    
    def get_summary(self, session_id: str) -> dict:
        """Get game summary including scores and leaderboard position."""
        if session_id not in sessions:
            raise ValueError("Invalid session ID")
            
        session = sessions[session_id]
        round_scores = session["round_scores"]
        total_score = sum(score["score"] for score in round_scores)
        
        # Calculate percentile (simplified)
        global leaderboard
        if not leaderboard:
            percentile = 100.0
        else:
            better_scores = sum(1 for entry in leaderboard if entry["total_score"] <= total_score)
            percentile = (better_scores / len(leaderboard)) * 100
        
        # Get top 3 players
        top_players = sorted(leaderboard, key=lambda x: x["total_score"], reverse=True)[:3]
        
        # Generate feedback
        feedback = self._generate_feedback(session, total_score)
        
        # Create summary
        summary = {
            "session_id": session_id,
            "nickname": session["nickname"],
            "user_type": session["user_type"],
            "total_score": total_score,
            "round_scores": round_scores,
            "percentile": round(percentile, 1),
            "rank": len([s for s in leaderboard if s["total_score"] > total_score]) + 1,
            "top_players": top_players,
            "feedback": feedback
        }
        
        return summary
    
    def _complete_game(self, session_id: str):
        """Mark game as completed and update leaderboard."""
        session = sessions[session_id]
        if session["completed"]:
            return
            
        session["completed"] = True
        session["end_time"] = datetime.now().isoformat()
        
        # Calculate total score
        total_score = sum(score["score"] for score in session["round_scores"])
        
        # Add to leaderboard
        global leaderboard
        leaderboard.append({
            "nickname": session["nickname"],
            "user_type": session["user_type"].value,
            "total_score": total_score,
            "timestamp": session["end_time"]
        })
        
        # Sort leaderboard
        leaderboard.sort(key=lambda x: x["total_score"], reverse=True)
        
        # Save stats
        save_stats()
    
    def _calculate_score(self, response: str, user_type: UserType) -> float:
        """Calculate score based on how well the response matches the opposite style."""
        # Get prediction from model
        prediction = predict_tf_style(response)
        
        # For T users, higher F score is better (and vice versa)
        if user_type == UserType.THINKING:
            return prediction["F_prob"]
        else:
            return prediction["T_prob"]
    
    def _get_opposite_style(self, user_type: UserType) -> str:
        """Get the opposite style as a string for dictionary access."""
        return "F" if user_type == UserType.THINKING else "T"
    
    def _generate_feedback(self, session: dict, total_score: float) -> str:
        """Generate personalized feedback based on the game results."""
        user_type = session["user_type"]
        target_style = "감정형(F)" if user_type == UserType.THINKING else "사고형(T)"
        
        avg_score = total_score / len(session["round_scores"])
        
        if avg_score >= 80:
            feedback = f"훌륭해요! {target_style} 스타일을 매우 잘 이해하고 계시네요!"
        elif avg_score >= 60:
            feedback = f"잘하고 있어요! {target_style} 스타일을 잘 따라하고 계시네요."
        elif avg_score >= 40:
            feedback = f"좋은 시도였어요! {target_style} 스타일을 더 연습해보세요."
        else:
            feedback = f"조금 더 노력이 필요해요. {target_style} 스타일을 이해하는 데 집중해보세요."
            
        return feedback

# Singleton instance
game_service = GameService()
