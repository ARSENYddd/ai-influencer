"""Entry point: CLI for running the AI influencer pipeline."""

import argparse
import asyncio
import sys
from pathlib import Path

from loguru import logger

from config import LOG_DIR, OUTPUT_DIR


def _setup_logging() -> None:
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    logger.add(log_dir / "app.log", rotation="10 MB", retention="30 days", level="DEBUG")


async def run_once(theme: str | None = None) -> dict:
    """Run the full pipeline once and return result dict."""
    from generators.image import generate_image
    from generators.video import generate_video
    from generators.caption import generate_caption
    from publishers.instagram import post_to_instagram
    from publishers.tiktok import post_to_tiktok
    from db import log_post

    import random
    from config import CONTENT_THEMES

    theme = theme or random.choice(CONTENT_THEMES)
    logger.info(f"Theme: {theme}")

    # Step 1 – generate image
    logger.info("Step 1/4 → Generating image (Leonardo AI)...")
    image_path = await generate_image(theme)
    logger.success(f"Image saved: {image_path}")

    # Step 2 – generate video
    logger.info("Step 2/4 → Generating video (Kling AI)...")
    video_path = await generate_video(image_path, theme)
    logger.success(f"Video saved: {video_path}")

    # Step 3 – generate caption
    logger.info("Step 3/4 → Generating caption (OpenAI GPT-4o)...")
    caption = await generate_caption(theme)
    logger.success(f"Caption: {caption[:80]}...")

    # Step 4 – publish
    logger.info("Step 4/4 → Publishing...")
    ig_id = await post_to_instagram(video_path, caption)
    tt_id = await post_to_tiktok(video_path, caption)
    logger.success(f"Instagram: {ig_id} | TikTok: {tt_id}")

    result = {
        "theme": theme,
        "image_path": image_path,
        "video_path": video_path,
        "caption": caption,
        "instagram_id": ig_id,
        "tiktok_id": tt_id,
    }
    log_post(result)
    return result


async def _test_mode() -> None:
    """Validate env vars and API connectivity without posting."""
    from config import LEONARDO_API_KEY, KLING_API_KEY, OPENAI_API_KEY
    from config import INSTAGRAM_ACCESS_TOKEN, TIKTOK_ACCESS_TOKEN

    checks = {
        "LEONARDO_API_KEY": LEONARDO_API_KEY,
        "KLING_API_KEY": KLING_API_KEY,
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "INSTAGRAM_ACCESS_TOKEN": INSTAGRAM_ACCESS_TOKEN,
        "TIKTOK_ACCESS_TOKEN": TIKTOK_ACCESS_TOKEN,
    }
    ok = True
    for key, val in checks.items():
        status = "✓" if val else "✗ MISSING"
        logger.info(f"  {key}: {status}")
        if not val:
            ok = False

    if ok:
        logger.success("All environment variables are set.")
    else:
        logger.error("Some variables are missing — check your .env file.")
        sys.exit(1)


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="AI Influencer Automation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --run-now              # Run pipeline immediately
  python main.py --schedule             # Start cron scheduler (12:00 + 19:00)
  python main.py --test                 # Check API keys without posting
  python main.py --run-now --theme "gym workout"
""",
    )
    parser.add_argument("--run-now", action="store_true", help="Run pipeline immediately")
    parser.add_argument("--schedule", action="store_true", help="Start scheduler")
    parser.add_argument("--test", action="store_true", help="Check config without posting")
    parser.add_argument("--theme", type=str, default=None, help="Force a specific theme")
    args = parser.parse_args()

    if args.test:
        asyncio.run(_test_mode())
        return

    if args.run_now:
        result = asyncio.run(run_once(args.theme))
        logger.info("Done.")
        for k, v in result.items():
            logger.info(f"  {k}: {v}")
        return

    if args.schedule:
        from scheduler import start_scheduler
        start_scheduler()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
