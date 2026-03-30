"""Entry point: CLI for running the AI influencer pipeline."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from loguru import logger

from config import LOG_DIR, OUTPUT_DIR


# ── Logging setup ──────────────────────────────────────────────────────────────

def _setup_logging(debug: bool = False) -> None:
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    logger.remove()
    level = "DEBUG" if debug else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        colorize=True,
    )
    logger.add(
        log_dir / "app.log",
        rotation="10 MB",
        retention="30 days",
        level="DEBUG",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}",
    )


# ── Pipeline ───────────────────────────────────────────────────────────────────

async def run_once(theme: str | None = None) -> dict:
    """
    Run the full pipeline once and return result dict.

    Steps:
      1. Generate image  (Leonardo AI)
      2. Generate video  (Kling AI)
      3. Upload to S3    (public URLs for Instagram/TikTok)
      4. Generate caption (OpenAI GPT-4o)
      5. Publish to Instagram Reels + TikTok
      6. Log to SQLite

    All errors are caught, logged with context, and re-raised so the caller
    (scheduler / CLI) can decide what to do.
    """
    import random
    from config import CONTENT_THEMES
    from db import log_post

    theme = theme or random.choice(CONTENT_THEMES)
    logger.info(f"{'─'*50}")
    logger.info(f"Pipeline start | theme: '{theme}'")
    logger.info(f"{'─'*50}")

    result: dict = {
        "theme": theme,
        "image_path": None,
        "video_path": None,
        "image_url": None,
        "video_url": None,
        "caption": None,
        "instagram_id": None,
        "tiktok_id": None,
        "success": False,
        "error": None,
    }

    # ── Step 1: Generate image ─────────────────────────────────────────────────
    logger.info("Step 1/5 → Generating image (Leonardo AI)...")
    try:
        from generators.image import generate_image
        result["image_path"] = await generate_image(theme)
        logger.success(f"  Image: {result['image_path']}")
    except EnvironmentError as exc:
        _fail(result, "image_generation", exc, fatal=True)
    except Exception as exc:
        _fail(result, "image_generation", exc, fatal=True)

    # ── Step 2: Generate video ─────────────────────────────────────────────────
    logger.info("Step 2/5 → Generating video (Kling AI)...")
    try:
        from generators.video import generate_video
        result["video_path"] = await generate_video(result["image_path"], theme)
        logger.success(f"  Video: {result['video_path']}")
    except EnvironmentError as exc:
        _fail(result, "video_generation", exc, fatal=True)
    except Exception as exc:
        _fail(result, "video_generation", exc, fatal=True)

    # ── Step 3: Upload to S3 ───────────────────────────────────────────────────
    logger.info("Step 3/5 → Uploading media to S3...")
    try:
        from storage.s3 import upload_file
        result["image_url"] = upload_file(result["image_path"])
        result["video_url"] = upload_file(result["video_path"])
        logger.success(f"  Image URL: {result['image_url']}")
        logger.success(f"  Video URL: {result['video_url']}")
    except EnvironmentError as exc:
        _fail(result, "s3_upload", exc, fatal=True)
    except Exception as exc:
        _fail(result, "s3_upload", exc, fatal=True)

    # ── Step 4: Generate caption ───────────────────────────────────────────────
    logger.info("Step 4/5 → Generating caption (OpenAI GPT-4o)...")
    try:
        from generators.caption import generate_caption
        result["caption"] = await generate_caption(theme)
        logger.success(f"  Caption: {result['caption'][:80]}...")
    except EnvironmentError as exc:
        _fail(result, "caption_generation", exc, fatal=True)
    except Exception as exc:
        _fail(result, "caption_generation", exc, fatal=True)

    # ── Step 5: Publish ────────────────────────────────────────────────────────
    logger.info("Step 5/5 → Publishing...")

    # Instagram (non-fatal if it fails — TikTok can still succeed)
    try:
        from publishers.instagram import post_reel_from_url
        result["instagram_id"] = await post_reel_from_url(result["video_url"], result["caption"])
        logger.success(f"  Instagram Reel: {result['instagram_id']}")
    except EnvironmentError as exc:
        logger.warning(f"  Instagram skipped — config missing: {exc}")
        result["error"] = f"instagram: {exc}"
    except Exception as exc:
        logger.error(f"  Instagram FAILED: {_format_error(exc)}")
        result["error"] = f"instagram: {exc}"

    # TikTok (non-fatal)
    try:
        from publishers.tiktok import post_tiktok_from_url
        result["tiktok_id"] = await post_tiktok_from_url(result["video_url"], result["caption"])
        logger.success(f"  TikTok: {result['tiktok_id']}")
    except EnvironmentError as exc:
        logger.warning(f"  TikTok skipped — config missing: {exc}")
    except Exception as exc:
        logger.error(f"  TikTok FAILED: {_format_error(exc)}")
        if result["error"]:
            result["error"] += f" | tiktok: {exc}"
        else:
            result["error"] = f"tiktok: {exc}"

    # ── Done ───────────────────────────────────────────────────────────────────
    published = bool(result["instagram_id"] or result["tiktok_id"])
    result["success"] = published
    log_post(result)

    if published:
        logger.success(f"Pipeline complete! IG={result['instagram_id']} TT={result['tiktok_id']}")
    else:
        logger.error("Pipeline finished but nothing was published — check errors above.")

    return result


def _fail(result: dict, step: str, exc: Exception, *, fatal: bool) -> None:
    """Log a structured error and raise if fatal."""
    msg = _format_error(exc)
    logger.error(f"  [{step}] FAILED: {msg}")
    result["error"] = f"{step}: {exc}"
    from db import log_post
    log_post(result)
    if fatal:
        raise exc


def _format_error(exc: Exception) -> str:
    """Return a concise one-line error string with type."""
    return f"{type(exc).__name__}: {exc}"


# ── Test mode ──────────────────────────────────────────────────────────────────

async def _test_mode() -> None:
    """Check all env vars and S3 connectivity. No API calls, no posting."""
    from config import (
        LEONARDO_API_KEY, KLING_API_KEY, OPENAI_API_KEY,
        INSTAGRAM_ACCESS_TOKEN, INSTAGRAM_USER_ID,
        TIKTOK_ACCESS_TOKEN,
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET, AWS_REGION,
    )

    checks = {
        "LEONARDO_API_KEY":        LEONARDO_API_KEY,
        "KLING_API_KEY":           KLING_API_KEY,
        "OPENAI_API_KEY":          OPENAI_API_KEY,
        "INSTAGRAM_ACCESS_TOKEN":  INSTAGRAM_ACCESS_TOKEN,
        "INSTAGRAM_USER_ID":       INSTAGRAM_USER_ID,
        "TIKTOK_ACCESS_TOKEN":     TIKTOK_ACCESS_TOKEN,
        "AWS_ACCESS_KEY_ID":       AWS_ACCESS_KEY_ID,
        "AWS_SECRET_ACCESS_KEY":   AWS_SECRET_ACCESS_KEY,
        "S3_BUCKET":               S3_BUCKET,
        "AWS_REGION":              AWS_REGION,
    }

    ok = True
    logger.info("Checking environment variables:")
    for key, val in checks.items():
        if val:
            logger.info(f"  ✓ {key}")
        else:
            logger.warning(f"  ✗ {key}  ← MISSING")
            ok = False

    # Check S3 connectivity if credentials are set
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and S3_BUCKET:
        logger.info("Checking S3 bucket access...")
        try:
            from storage.s3 import check_s3_access
            check_s3_access()
        except Exception as exc:
            logger.error(f"  S3 check failed: {_format_error(exc)}")
            ok = False
    else:
        logger.warning("  S3 check skipped — credentials not set.")

    if ok:
        logger.success("All checks passed.")
    else:
        logger.error("Some checks failed — fix your .env file.")
        sys.exit(1)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Influencer Automation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --test                          # check all API keys + S3
  python main.py --run-now                       # run pipeline immediately
  python main.py --run-now --theme "gym workout" # force a specific theme
  python main.py --schedule                      # start cron (12:00 + 19:00)
  python main.py --run-now --debug               # verbose debug output
""",
    )
    parser.add_argument("--run-now",  action="store_true", help="Run pipeline immediately")
    parser.add_argument("--schedule", action="store_true", help="Start cron scheduler")
    parser.add_argument("--test",     action="store_true", help="Check config + S3 access")
    parser.add_argument("--theme",    type=str, default=None, help="Force specific theme")
    parser.add_argument("--debug",    action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    _setup_logging(debug=args.debug)

    if args.test:
        asyncio.run(_test_mode())
        return

    if args.run_now:
        try:
            result = asyncio.run(run_once(args.theme))
        except Exception as exc:
            logger.critical(f"Pipeline aborted: {_format_error(exc)}")
            sys.exit(1)
        if not result.get("success"):
            sys.exit(1)
        return

    if args.schedule:
        from scheduler import start_scheduler
        start_scheduler()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
