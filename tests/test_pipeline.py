"""Basic import and unit tests for the pipeline modules."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_personas_load():
    """Test that all 3 persona files load correctly."""
    import json
    from pathlib import Path

    persona_dir = Path("config/personas")
    assert persona_dir.exists(), "config/personas directory missing"

    personas = list(persona_dir.glob("*.json"))
    assert len(personas) == 3, f"Expected 3 personas, found {len(personas)}"

    for f in personas:
        data = json.loads(f.read_text())
        assert "id" in data
        assert "display_name" in data
        assert "caption_style" in data


def test_db_init():
    """Test database initialization."""
    import os
    # Use test DB
    os.environ["DATABASE_URL"] = "sqlite:///./test_ai_influencer.db"

    from database.db import init_db, SessionLocal
    from database.models import ContentItem

    init_db()

    db = SessionLocal()
    count = db.query(ContentItem).count()
    assert count == 0
    db.close()

    # Cleanup
    if os.path.exists("test_ai_influencer.db"):
        os.remove("test_ai_influencer.db")


def test_settings_load():
    """Test settings load without errors."""
    from config.settings import settings
    assert settings.DASHBOARD_PORT == 8080
    assert settings.PIPELINE_DEFAULT in ("A", "C")


def test_caption_prompt_build():
    """Test caption prompt builds without API call."""
    from caption.caption_generator import CaptionGenerator

    gen = CaptionGenerator(api_key="test")
    persona = {
        "id": "mia",
        "display_name": "Mia",
        "personality": "playful, sweet",
        "caption_style": {
            "tone": "fun, flirty",
            "emojis": True,
            "hashtag_count": 15,
            "hashtags": ["#dance", "#aesthetic"]
        }
    }
    trend = {"dance_type": "viral shuffle"}
    prompt = gen._build_prompt(persona, trend)
    assert "Mia" in prompt
    assert "viral shuffle" in prompt


if __name__ == "__main__":
    test_personas_load()
    test_db_init()
    test_settings_load()
    test_caption_prompt_build()
    print("All tests passed!")
