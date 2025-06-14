import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from ..schemas import PlayerScore, PlayerRanking, RankingResponse

class RankingService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RankingService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the ranking service with data file."""
        self.data_dir = Path(os.environ.get("RANKING_DATA_DIR", "data"))
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.rankings_file = self.data_dir / "rankings.json"
        self.rankings: Dict[str, PlayerScore] = {}
        self._load_rankings()
    
    def _load_rankings(self):
        """Load rankings from JSON file."""
        if self.rankings_file.exists():
            try:
                with open(self.rankings_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.rankings = {
                        nickname: PlayerScore(**player_data)
                        for nickname, player_data in data.items()
                    }
            except Exception as e:
                print(f"Error loading rankings: {e}")
                self.rankings = {}
    
    async def _save_rankings(self):
        """Save rankings to JSON file asynchronously."""
        try:
            rankings_data = {}
            print(f"\n=== _save_rankings called ===")
            print(f"Current rankings: {[p.dict() for p in self.rankings.values()]}")
            print(f"Saving to: {self.rankings_file.absolute()}")
            print(f"Directory exists: {self.rankings_file.parent.exists()}")
            
            # Ensure the directory exists
            self.rankings_file.parent.mkdir(exist_ok=True, parents=True)
            
            # Prepare the data to be saved
            for nickname, player in self.rankings.items():
                print(f"Preparing to save player {nickname}: {player.dict()}")
                rankings_data[nickname] = player.dict()
            
            # Use aiofiles for async file operations
            import aiofiles
            import asyncio
            
            # Write to a temporary file first, then rename to handle filesystem atomicity
            temp_file = self.rankings_file.with_suffix('.tmp')
            
            # Write to temporary file asynchronously
            async with aiofiles.open(temp_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(rankings_data, ensure_ascii=False, indent=2))
            
            # On Unix systems, rename is atomic
            temp_file.replace(self.rankings_file)
            
            print(f"Successfully wrote to {self.rankings_file.absolute()}")
            print(f"File exists after save: {self.rankings_file.exists()}")
            print("=== End of _save_rankings ===\n")
            return True
            
        except Exception as e:
            print(f"Error saving rankings: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def update_ranking(self, nickname: str, score: float, user_type: str = None, session_id: str = None):
        """Update player's ranking with new score.
        
        Args:
            nickname: Player's nickname
            score: Score to add to player's total
            user_type: The MBTI user type (e.g., 'T' for Thinking, 'F' for Feeling)
            session_id: Optional session ID for tracking
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            print("\n=== update_ranking ===")
            print(f"Nickname: {nickname}")
            print(f"Score: {score}")
            print(f"User Type: {user_type}")
            print(f"Session ID: {session_id}")
            print(f"RankingService instance: {id(self)}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Data directory: {self.data_dir.absolute()}")
            print(f"Rankings file: {self.rankings_file.absolute()}")
            
            # Print current rankings before update
            print(f"Current rankings before update: {[p.dict() for p in self.rankings.values()]}")
            
            # Update player's score
            if nickname in self.rankings:
                player = self.rankings[nickname]
                player.total_score += score
                player.game_count += 1
                if user_type:
                    player.user_type = user_type
                print(f"Updated existing player {nickname}: total_score={player.total_score}, game_count={player.game_count}")
            else:
                self.rankings[nickname] = PlayerScore(
                    nickname=nickname,
                    total_score=score,
                    game_count=1,
                    user_type=user_type or 'U'  # Default to 'U' for Unknown if not provided
                )
                print(f"Added new player {nickname}: total_score={score}, game_count=1, user_type={user_type or 'U'}")
            
            # Save the updated rankings asynchronously
            print("Saving rankings...")
            save_success = await self._save_rankings()
            print(f"Save successful: {save_success}")
            
            # Verify the rankings were saved correctly
            if save_success and self.rankings_file.exists():
                try:
                    import aiofiles
                    async with aiofiles.open(self.rankings_file, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        saved_data = json.loads(content)
                        print(f"Saved data from disk: {saved_data}")
                except Exception as e:
                    print(f"Error reading saved rankings: {e}")
            
            print(f"Current rankings state: {[p.dict() for p in self.rankings.values()]}")
            print("=== End of update_ranking ===\n")
            
            return save_success
            
        except Exception as e:
            print(f"Error in update_ranking: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_rankings(self, limit: int = 10) -> dict:
        """Get top N rankings.
        
        Args:
            limit: Maximum number of rankings to return
            
        Returns:
            dict: Contains 'rankings' (list of player dicts) and 'total_players' (int)
        """
        # Convert to list and sort by total_score in descending order
        players = list(self.rankings.values())
        players.sort(key=lambda x: x.total_score, reverse=True)
        
        # Apply limit and format results
        limited_players = players[:limit] if limit is not None else players
        
        # Create rankings with proper rank (1-based index)
        rankings = []
        for i, player in enumerate(limited_players, 1):
            rankings.append({
                "nickname": player.nickname,
                "score": player.total_score,
                "average": player.average_score,
                "rank": i
            })
        
        return {
            "rankings": rankings,
            "total_players": len(self.rankings)
        }

# Singleton instance
ranking_service = RankingService()
