"""Main pipeline: generate image → generate caption → post to Instagram."""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configure logging
log_dir = Path("output/logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"),
    ],
)
logger = logging.getLogger(__name__)


def run_pipeline(dry_run: bool = False) -> dict:
    """
    Execute the full AI influencer pipeline.

    Args:
        dry_run: If True, skip real API calls.

    Returns:
        Dict with results: image_path, caption, post_id, timestamp.
    """
    logger.info("=" * 60)
    logger.info(f"AI Influencer Pipeline starting {'[DRY RUN]' if dry_run else '[LIVE]'}")
    logger.info("=" * 60)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Step 1: Generate image ──────────────────────────────────
    logger.info("STEP 1/3: Generating AI image...")
    if dry_run:
        from generator import generate_image_dry_run
        image_path = generate_image_dry_run()
    else:
        _require_env("REPLICATE_API_TOKEN")
        from generator import generate_image
        image_path = generate_image()

    logger.info(f"Image ready: {image_path}")

    # ── Step 2: Generate caption ────────────────────────────────
    logger.info("STEP 2/3: Generating caption...")
    if dry_run:
        from caption_generator import generate_caption_dry_run
        caption_text, full_caption = generate_caption_dry_run()
    else:
        _require_env("ANTHROPIC_API_KEY")
        from caption_generator import generate_caption
        caption_text, full_caption = generate_caption()

    logger.info(f"Caption ready ({len(full_caption)} chars)")

    # ── Step 3: Post to Instagram ───────────────────────────────
    logger.info("STEP 3/3: Posting to Instagram...")
    if dry_run:
        from poster import post_dry_run
        post_id = post_dry_run(image_path, full_caption)
    else:
        _require_env("INSTAGRAM_USERNAME")
        _require_env("INSTAGRAM_PASSWORD")
        from poster import post_to_instagram
        post_id = post_to_instagram(
            image_path=image_path,
            caption=full_caption,
            username=os.environ["INSTAGRAM_USERNAME"],
            password=os.environ["INSTAGRAM_PASSWORD"],
        )

    result = {
        "timestamp": timestamp,
        "image_path": image_path,
        "caption": caption_text,
        "post_id": post_id,
        "dry_run": dry_run,
    }

    logger.info("=" * 60)
    logger.info(f"Pipeline complete! Post ID: {post_id}")
    logger.info("=" * 60)

    return result


def _require_env(key: str) -> None:
    """Raise an error if a required environment variable is missing."""
    if not os.environ.get(key):
        raise EnvironmentError(
            f"Missing required environment variable: {key}\n"
            f"Copy .env.template to .env and fill in your credentials."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Influencer Pipeline")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without real API calls (for testing)",
    )
    args = parser.parse_args()

    try:
        result = run_pipeline(dry_run=args.dry_run)
        print("\nResult summary:")
        for k, v in result.items():
            print(f"  {k}: {v}")
    except EnvironmentError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)
