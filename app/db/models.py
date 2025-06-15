from sqlalchemy import Boolean, Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from .database import Base

class GameSession(Base):
    __tablename__ = "game_sessions"
    
    id = Column(String, primary_key=True, index=True)
    nickname = Column(String, nullable=False)
    user_type = Column(String(1), nullable=False)  # 'T' or 'F'
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    total_score = Column(Float, default=0.0)
    
    # Relationship
    scores = relationship("PlayerScore", back_populates="game_session")

class PlayerScore(Base):
    __tablename__ = "player_scores"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("game_sessions.id", ondelete="CASCADE"))
    round_number = Column(Integer, nullable=False)
    situation = Column(String, nullable=False)  # e.g., "연인_갈등", "친구_갈등"
    situation_detail = Column(String, nullable=True)  # Detailed description of the situation
    user_response = Column(String, nullable=False)
    is_correct_style = Column(Boolean, nullable=False)
    score = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship
    game_session = relationship("GameSession", back_populates="scores")
