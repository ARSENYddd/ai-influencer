"""
ffmpeg video prep/cleanup for Pipeline A.
"""
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

def prepare_video(input_path: str, output_path: str) -> str | None:
    """Normalize video to 9:16 1080x1920 for Reels."""
    try:
        cmd = [
            "ffmpeg", "-i", input_path,
            "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
            "-c:v", "libx264", "-c:a", "aac",
            "-y", output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Video prepared: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg failed: {e.stderr.decode()}")
        return None
    except FileNotFoundError:
        logger.error("ffmpeg not found — install ffmpeg")
        return None
