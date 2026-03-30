"""Central configuration: API keys, character settings, and content themes."""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ───────────────────────────────────────────────────────────────────
LEONARDO_API_KEY = os.getenv("LEONARDO_API_KEY", "")
KLING_API_KEY = os.getenv("KLING_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN", "")
INSTAGRAM_USER_ID = os.getenv("INSTAGRAM_USER_ID", "")
TIKTOK_ACCESS_TOKEN = os.getenv("TIKTOK_ACCESS_TOKEN", "")

# ── Character ──────────────────────────────────────────────────────────────────
CHARACTER_NAME = "Sofia"
CHARACTER_STYLE = (
    "beautiful young woman, 25 years old, curvy hourglass figure, "
    "long dark brown wavy hair, warm brown eyes, olive skin, natural makeup, "
    "lifestyle photography, candid shot, soft natural lighting"
)
LEONARDO_MODEL_ID = "aa77f04e-3eec-4034-9c07-d0f619684628"  # Leonardo Diffusion XL

# ── Content Themes ─────────────────────────────────────────────────────────────
CONTENT_THEMES = [
    "morning routine / coffee",
    "gym workout",
    "fashion / outfit of the day",
    "travel / outdoor",
    "food & lifestyle",
]

# ── Scheduler ──────────────────────────────────────────────────────────────────
SCHEDULE_TIMES = ["12:00", "19:00"]

# ── Storage ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "output"
POST_LOG_DB = "post_log.db"
LOG_DIR = "logs"
