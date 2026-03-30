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

def run_pipeline(dry_run: bool = False, style_index: int = None) -> dict:
    """
    Execute the full AI influencer pipeline.

    Steps:
      1. Resolve style index (Qdrant rotation or time-based fallback)
      2. Generate AI image (Replicate Flux Dev)
      3. Generate caption (Claude Haiku)
      4. Duplicate caption check (Qdrant cosine similarity)
      5. Post to Instagram (instagrapi)
      6. Store result in Qdrant

    Args:
        dry_run: If True, skip real API calls (test mode).
        style_index: Override style selection (0-7). None = auto from Qdrant.

    Returns:
        Dict with: timestamp, image_path, caption, post_id, style_index,
                   vector_id, dry_run, success, error, duration_seconds.

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
        "style_index": style_index,
        "vector_id": None,
    }

    # Pre-flight
    _check_env_file()
    _check_disk_space()

    # ── STEP 0: Resolve style index via Qdrant rotation ────────────────────────
    if style_index is None:
        try:
            from vector_store import get_next_style_index
            from generator import STYLES
            style_index = get_next_style_index(num_styles=len(STYLES))
            logger.info(f"[pipeline] Style index from Qdrant rotation: {style_index}")
        except Exception as exc:
            import time as _time
            from generator import STYLES
            style_index = int(_time.time() / 3600) % len(STYLES)
            logger.warning(
                f"[pipeline] Qdrant style rotation failed ({exc}), "
                f"using time-based fallback: style_index={style_index}"
            )
    result["style_index"] = style_index

    # ── STEP 1: Generate image ─────────────────────────────────────────────────
    logger.info("─" * 40)
    logger.info("STEP 1/4  Generate AI image")
    logger.info("─" * 40)

    image_path = None
    try:
        if dry_run:
            from generator import generate_image_dry_run
            image_path = generate_image_dry_run()
        else:
            _require_env("REPLICATE_API_TOKEN")
            from generator import generate_image
            image_path = generate_image(style_index=style_index)

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
    logger.info("STEP 2/4  Generate caption")
    logger.info("─" * 40)

    caption_text = None
    full_caption = None
    try:
        if dry_run:
            from caption_generator import generate_caption_dry_run
            caption_text, full_caption = generate_caption_dry_run()
        else:
            _require_env("ANTHROPIC_API_KEY")
            from caption_generator import generate_caption, CAPTION_STYLES
            caption_text, full_caption = generate_caption(
                style_hint=CAPTION_STYLES[style_index % len(CAPTION_STYLES)]
            )

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

    # ── STEP 3 (pre): Duplicate caption check via Qdrant ──────────────────────
    try:
        from vector_store import check_duplicate_caption, DuplicateCaptionError
        check_duplicate_caption(caption_text)
        logger.info("[pipeline] ✓ Caption uniqueness check passed.")
    except DuplicateCaptionError as exc:
        logger.warning(f"[pipeline] Near-duplicate caption detected ({exc.score:.3f}). Regenerating...")
        try:
            from caption_generator import generate_caption, CAPTION_STYLES
            alt_style_index = (style_index + 1) % len(CAPTION_STYLES)
            caption_text, full_caption = generate_caption(
                style_hint=CAPTION_STYLES[alt_style_index]
            )
            result["caption"] = caption_text
            logger.info(f"[pipeline] ✓ Regenerated caption (style #{alt_style_index}): {caption_text[:60]}...")
        except Exception as regen_exc:
            logger.warning(f"[pipeline] Caption regeneration failed: {regen_exc}. Proceeding with original.")
    except Exception as exc:
        # Qdrant unavailable — non-fatal, proceed without duplicate check
        logger.warning(f"[pipeline] Qdrant duplicate check skipped (non-fatal): {exc}")

    # ── STEP 3: Post to Instagram ──────────────────────────────────────────────
    logger.info("─" * 40)
    logger.info("STEP 3/4  Post to Instagram")
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

    # ── STEP 4: Store result in Qdrant ────────────────────────────────────────
    logger.info("─" * 40)
    logger.info("STEP 4/4  Store in vector DB")
    logger.info("─" * 40)

    try:
        from vector_store import store_post
        vector_id = store_post(
            caption_text=caption_text,
            style_index=style_index,
            post_id=str(post_id),
            image_path=image_path,
            timestamp=result["timestamp"],
            is_dry_run=dry_run,
        )
        result["vector_id"] = vector_id
        logger.info(f"[pipeline] ✓ Stored in Qdrant: {vector_id}")
    except Exception as exc:
        # Qdrant failure must NEVER abort a successful Instagram post
        logger.warning(
            f"[pipeline] Could not store in Qdrant (non-fatal, post already succeeded): "
            f"{type(exc).__name__}: {exc}"
        )

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
