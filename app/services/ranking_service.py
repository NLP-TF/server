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
        # Change to store by session_id instead of nickname
        self.rankings: Dict[str, PlayerScore] = {}  # key: session_id
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
            
            # --- DEBUG: Write to file for test diagnosis ---
            with open("rankings_backend_debug.txt", "w") as dbg:
                dbg.write(f"rankings_file: {self.rankings_file}\n")
                dbg.write(f"rankings_file.exists: {self.rankings_file.exists()}\n")
                dbg.write(f"rankings_file.parent: {self.rankings_file.parent}\n")
                dbg.write(f"rankings_file.parent.exists: {self.rankings_file.parent.exists()}\n")
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
            session_id: Session ID for tracking individual players (required)
            
        Returns:
            bool: True if update was successful, False otherwise
            
        Raises:
            ValueError: If session_id is not provided
        """
        if not session_id:
            raise ValueError("session_id is required for tracking player scores")
            
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
            
            # Normalize user_type
            user_type = (user_type or 'U').upper()
            
            # Create a unique key that combines nickname and user_type to prevent duplicates
            player_key = f"{nickname}_{user_type}"
            
            # Check if player with same nickname and user_type already exists
            existing_session_id = None
            for sid, player in self.rankings.items():
                if player.nickname == nickname and player.user_type == user_type:
                    existing_session_id = sid
                    break
            
            if existing_session_id:
                # Update existing player's score
                player = self.rankings[existing_session_id]
                player.total_score = score  # Update to new score
                player.game_count += 1
                print(f"Updated existing player {player.nickname} (type: {player.user_type}): total_score={player.total_score}, game_count={player.game_count}")
            else:
                # Add new player
                self.rankings[session_id] = PlayerScore(
                    nickname=nickname,
                    total_score=score,
                    game_count=1,
                    user_type=user_type
                )
                print(f"Added new player {nickname} (type: {user_type}): total_score={score}")
            
            # Save the updated rankings
            print("Saving rankings...")
            save_success = await self._save_rankings()
            
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
    
    def get_rankings(self, limit: int = 10, user_type: str = None) -> dict:
        """Get top N rankings, optionally filtered by user_type.
        
        Args:
            limit: Maximum number of rankings to return
            user_type: Optional filter to get rankings for specific user type ('T' or 'F')
            
        Returns:
            dict: Contains 'rankings' (list of player dicts) and 'total_players' (int)
        """
        # Reload rankings from disk to ensure we have the latest data
        self._load_rankings()
        
        # Filter players by user_type if specified
        players = list(self.rankings.values())
        if user_type and user_type.upper() in ['T', 'F']:
            players = [p for p in players if p.user_type == user_type.upper()]
        
        # Sort by total_score in descending order
        players.sort(key=lambda x: x.total_score, reverse=True)
        
        # Apply limit
        limited_players = players[:limit] if limit is not None else players
        
        # Create rankings with proper rank (1-based index)
        rankings = []
        for i, player in enumerate(limited_players, 1):
            rankings.append({
                "nickname": player.nickname,
                "score": player.total_score,
                "average": player.average_score,
                "rank": i,
                "user_type": player.user_type,
                "game_count": player.game_count
            })
        
        return {
            "rankings": rankings,
            "total_players": len(players)  # Return count of filtered players
        }

# Singleton instance
ranking_service = RankingService()
