"""APScheduler cron scheduler: posts at 12:00 and 19:00 daily."""

import asyncio
import sys
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from config import SCHEDULE_TIMES, LOG_DIR


def _setup_logging() -> None:
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    logger.add(Path(LOG_DIR) / "app.log", rotation="10 MB", retention="30 days", level="DEBUG")


def _run_pipeline() -> None:
    """Synchronous wrapper for the async pipeline (called by APScheduler)."""
    try:
        from main import run_once
        result = asyncio.run(run_once())
        logger.success(f"Scheduled post done: {result.get('instagram_id')}")
        try:
            from plyer import notification
            notification.notify(
                title="AI Influencer — Posted!",
                message=f"Instagram: {result.get('instagram_id')}",
                timeout=8,
            )
        except Exception:
            pass
    except Exception as exc:
        logger.error(f"Scheduled post FAILED: {exc}")
        try:
            from plyer import notification
            notification.notify(
                title="AI Influencer — FAILED",
                message=str(exc)[:200],
                timeout=10,
            )
        except Exception:
            pass


def start_scheduler() -> None:
    _setup_logging()
    scheduler = BlockingScheduler(timezone="local")

    for time_str in SCHEDULE_TIMES:
        hour, minute = map(int, time_str.split(":"))
        scheduler.add_job(
            _run_pipeline,
            trigger=CronTrigger(hour=hour, minute=minute),
            id=f"post_{time_str.replace(':', '')}",
            name=f"Daily post at {time_str}",
            max_instances=1,
            misfire_grace_time=300,
        )
        logger.info(f"Scheduled: daily post at {time_str}")

    logger.info("Scheduler started. Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    start_scheduler()
