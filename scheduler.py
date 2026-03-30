"""Daily scheduler: runs the pipeline at 09:00 and 18:00."""

import logging
import sys
import time
import signal
from datetime import datetime
from pathlib import Path

import schedule
from dotenv import load_dotenv

load_dotenv()

# Configure logging
log_dir = Path("output/logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "scheduler.log"),
    ],
)
logger = logging.getLogger(__name__)


def job(dry_run: bool = False) -> None:
    """Scheduled job that runs the pipeline."""
    logger.info(f"Scheduled job triggered at {datetime.now().strftime('%H:%M:%S')}")
    try:
        from pipeline import run_pipeline
        result = run_pipeline(dry_run=dry_run)
        logger.info(f"Scheduled post complete: {result['post_id']}")
    except Exception as e:
        logger.exception(f"Scheduled job failed: {e}")


def run_scheduler(dry_run: bool = False) -> None:
    """Start the scheduler with jobs at 09:00 and 18:00."""
    logger.info("Starting AI Influencer Scheduler")
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")

    # Schedule posts at 09:00 and 18:00 daily
    schedule.every().day.at("09:00").do(job, dry_run=dry_run)
    schedule.every().day.at("18:00").do(job, dry_run=dry_run)

    logger.info("Scheduled: daily posts at 09:00 and 18:00")
    logger.info(f"Next run: {schedule.next_run()}")

    # Graceful shutdown on SIGINT/SIGTERM
    def _shutdown(signum, frame):
        logger.info("Scheduler stopping (received signal).")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while True:
        schedule.run_pending()
        time.sleep(30)  # Check every 30 seconds


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI Influencer Scheduler")
    parser.add_argument("--dry-run", action="store_true", help="Run jobs without real API calls")
    parser.add_argument("--run-now", action="store_true", help="Run one job immediately and keep scheduling")
    args = parser.parse_args()

    if args.run_now:
        logger.info("Running immediate job before starting scheduler...")
        job(dry_run=args.dry_run)

    run_scheduler(dry_run=args.dry_run)
