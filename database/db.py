from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base

DATABASE_URL = "sqlite:///./ai_influencer.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Create all tables."""
    import os
    os.makedirs("output/approved/mia", exist_ok=True)
    os.makedirs("output/approved/zara", exist_ok=True)
    os.makedirs("output/approved/luna", exist_ok=True)
    os.makedirs("output/pending/mia", exist_ok=True)
    os.makedirs("output/pending/zara", exist_ok=True)
    os.makedirs("output/pending/luna", exist_ok=True)
    os.makedirs("output/rejected", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
