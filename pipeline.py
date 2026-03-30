"""Main pipeline: generate image → generate caption → post to Instagram."""

import argparse
import json
import logging
import os
import shutil
import sys
import traceback
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

MIN_DISK_FREE_MB = 200  # require at least 200 MB free before starting


# ── Pre-flight ─────────────────────────────────────────────────────────────────

def _require_env(key: str) -> str:
    """Return env var value or raise a descriptive EnvironmentError."""
    value = os.environ.get(key, "").strip()
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set.\n"
            f"  → Copy .env.template to .env and fill in your credentials."
        )
    return value


def _check_env_file() -> None:
    """Warn if .env file is missing (user may have forgotten to create it)."""
    if not Path(".env").exists():
        logger.warning(
            "[pipeline] .env file not found. "
            "Copy .env.template to .env and fill in your API keys."
        )


def _check_disk_space() -> None:
    """Abort early if disk space is critically low."""
    free_mb = shutil.disk_usage(".").free / (1024 * 1024)
    if free_mb < MIN_DISK_FREE_MB:
        raise OSError(
            f"Insufficient disk space: {free_mb:.1f} MB free "
            f"(minimum required: {MIN_DISK_FREE_MB} MB). "
            "Free up space and retry."
        )
    logger.debug(f"[pipeline] Disk space OK: {free_mb:.0f} MB free")


def _write_audit_log(result: dict) -> None:
    """Append pipeline result to a JSON-lines audit log."""
    audit_path = log_dir / "audit.jsonl"
    try:
        with open(audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        logger.debug(f"[pipeline] Audit entry written to {audit_path}")
    except OSError as exc:
        logger.warning(f"[pipeline] Could not write audit log: {exc}")


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_pipeline(dry_run: bool = False) -> dict:
    """
    Execute the full AI influencer pipeline.

    Steps:
      1. Generate AI image (Replicate Flux Dev)
      2. Generate caption (Claude Haiku)
      3. Post to Instagram (instagrapi)

    Args:
        dry_run: If True, skip real API calls (test mode).

    Returns:
        Dict with: timestamp, image_path, caption, post_id, dry_run, success, error.

    Raises:
        EnvironmentError: A required env variable is missing.
        OSError: Disk space too low.
        Any module-specific exception if all retries are exhausted.
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info(f"AI Influencer Pipeline  {'[DRY RUN]' if dry_run else '[LIVE]'}")
    logger.info(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    result = {
        "timestamp": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "dry_run": dry_run,
        "success": False,
        "error": None,
        "image_path": None,
        "caption": None,
        "post_id": None,
    }

    # Pre-flight
    _check_env_file()
    _check_disk_space()

    # ── STEP 1: Generate image ─────────────────────────────────────────────────
    logger.info("─" * 40)
    logger.info("STEP 1/3  Generate AI image")
    logger.info("─" * 40)

    image_path = None
    try:
        if dry_run:
            from generator import generate_image_dry_run
            image_path = generate_image_dry_run()
        else:
            _require_env("REPLICATE_API_TOKEN")
            from generator import generate_image
            image_path = generate_image()

        result["image_path"] = image_path
        logger.info(f"[pipeline] ✓ Image ready: {image_path}")

    except EnvironmentError:
        raise  # propagate missing-env errors immediately
    except Exception as exc:
        logger.error(f"[pipeline] ✗ Image generation failed: {type(exc).__name__}: {exc}")
        logger.debug(traceback.format_exc())
        result["error"] = f"image_generation: {type(exc).__name__}: {exc}"
        _write_audit_log(result)
        raise

    # ── STEP 2: Generate caption ───────────────────────────────────────────────
    logger.info("─" * 40)
    logger.info("STEP 2/3  Generate caption")
    logger.info("─" * 40)

    caption_text = None
    full_caption = None
    try:
        if dry_run:
            from caption_generator import generate_caption_dry_run
            caption_text, full_caption = generate_caption_dry_run()
        else:
            _require_env("ANTHROPIC_API_KEY")
            from caption_generator import generate_caption
            caption_text, full_caption = generate_caption()

        result["caption"] = caption_text
        logger.info(f"[pipeline] ✓ Caption ready ({len(full_caption)} chars)")

    except EnvironmentError:
        raise
    except Exception as exc:
        logger.error(f"[pipeline] ✗ Caption generation failed: {type(exc).__name__}: {exc}")
        logger.debug(traceback.format_exc())
        result["error"] = f"caption_generation: {type(exc).__name__}: {exc}"
        # Image was already generated — log its path so it can be reused
        logger.info(f"[pipeline] Image was saved at {image_path} — you can post it manually.")
        _write_audit_log(result)
        raise

    # ── STEP 3: Post to Instagram ──────────────────────────────────────────────
    logger.info("─" * 40)
    logger.info("STEP 3/3  Post to Instagram")
    logger.info("─" * 40)

    post_id = None
    try:
        if dry_run:
            from poster import post_dry_run
            post_id = post_dry_run(image_path, full_caption)
        else:
            username = _require_env("INSTAGRAM_USERNAME")
            _require_env("INSTAGRAM_PASSWORD")
            from poster import post_to_instagram
            post_id = post_to_instagram(
                image_path=image_path,
                caption=full_caption,
                username=username,
                password=os.environ["INSTAGRAM_PASSWORD"],
            )

        result["post_id"] = post_id
        logger.info(f"[pipeline] ✓ Posted! Media ID: {post_id}")

    except EnvironmentError:
        raise
    except Exception as exc:
        logger.error(f"[pipeline] ✗ Instagram posting failed: {type(exc).__name__}: {exc}")
        logger.debug(traceback.format_exc())
        result["error"] = f"instagram_post: {type(exc).__name__}: {exc}"
        logger.info(
            f"[pipeline] Image and caption were prepared:\n"
            f"  image:   {image_path}\n"
            f"  caption: {caption_text[:80]}..."
        )
        _write_audit_log(result)
        raise

    # ── Done ───────────────────────────────────────────────────────────────────
    duration = (datetime.now() - start_time).total_seconds()
    result["success"] = True
    result["duration_seconds"] = round(duration, 1)

    logger.info("=" * 60)
    logger.info(f"Pipeline complete! Post ID: {post_id}  ({duration:.1f}s)")
    logger.info("=" * 60)

    _write_audit_log(result)
    return result


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI Influencer Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --dry-run     # Test without API calls
  python pipeline.py               # Live run
""",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without real API calls (for testing)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("[pipeline] DEBUG logging enabled")

    try:
        result = run_pipeline(dry_run=args.dry_run)
        print("\n── Result ───────────────────────────────")
        for k, v in result.items():
            print(f"  {k:20s}: {v}")
        print("─────────────────────────────────────────")
        sys.exit(0)

    except EnvironmentError as exc:
        logger.error(f"Configuration error: {exc}")
        sys.exit(2)   # exit code 2 = config error

    except OSError as exc:
        logger.error(f"System error: {exc}")
        sys.exit(3)   # exit code 3 = system/disk error

    except Exception as exc:
        logger.error(f"Pipeline failed: {type(exc).__name__}: {exc}")
        logger.debug(traceback.format_exc())
        sys.exit(1)   # exit code 1 = pipeline error
