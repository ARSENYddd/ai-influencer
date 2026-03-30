"""SQLite post log: stores every posted piece of content."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from config import POST_LOG_DB


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(POST_LOG_DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                theme TEXT,
                image_path TEXT,
                video_path TEXT,
                caption TEXT,
                instagram_id TEXT,
                tiktok_id TEXT,
                success INTEGER DEFAULT 1,
                error TEXT
            )
        """)


def log_post(result: dict) -> int:
    """Insert a pipeline result into the log. Returns row id."""
    init_db()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO posts
               (created_at, theme, image_path, video_path, caption, instagram_id, tiktok_id, success, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                now,
                result.get("theme"),
                result.get("image_path"),
                result.get("video_path"),
                result.get("caption"),
                result.get("instagram_id"),
                result.get("tiktok_id"),
                1 if result.get("success", True) else 0,
                result.get("error"),
            ),
        )
        return cur.lastrowid


def get_posts(limit: int = 50) -> list[dict]:
    """Return the most recent posts."""
    init_db()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM posts ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
