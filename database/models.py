from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, Float
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class ContentItem(Base):
    __tablename__ = "content_items"

    id = Column(Integer, primary_key=True)
    persona_id = Column(String(50), nullable=False)
    trend_id = Column(String(100))
    trend_type = Column(String(50))
    pipeline = Column(String(1))
    status = Column(String(20), default="pending")
    video_path = Column(String(500))
    caption = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    approved_at = Column(DateTime)
    image_paths = Column(Text)
    score = Column(Float)

class TrendCache(Base):
    __tablename__ = "trend_cache"

    id = Column(Integer, primary_key=True)
    reel_id = Column(String(100), unique=True)
    dance_type = Column(String(200))
    score = Column(Float)
    used = Column(Boolean, default=False)
    scraped_at = Column(DateTime, default=datetime.utcnow)
    raw_data = Column(Text)

class PersonaStats(Base):
    __tablename__ = "persona_stats"

    id = Column(Integer, primary_key=True)
    persona_id = Column(String(50))
    date = Column(String(20))
    posts_generated = Column(Integer, default=0)
    posts_approved = Column(Integer, default=0)
    posts_rejected = Column(Integer, default=0)
