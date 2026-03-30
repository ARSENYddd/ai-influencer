"""Daily scheduler: runs the pipeline at 09:00 and 18:00."""

import logging
import sys
import time
import signal
import threading
import traceback
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

# ── State tracking ─────────────────────────────────────────────────────────────
_job_lock = threading.Lock()     # prevent overlapping executions
_consecutive_failures = 0
MAX_CONSECUTIVE_FAILURES = 5     # stop scheduler after N failures in a row


# ── Job ────────────────────────────────────────────────────────────────────────

def job(dry_run: bool = False) -> None:
    """
    Scheduled job that runs the full pipeline.

    - Uses a lock to prevent overlapping runs (e.g. if a post takes longer than expected).
    - Tracks consecutive failures and shuts down the scheduler if threshold is reached.
    """
    global _consecutive_failures

    # Prevent overlapping runs
    acquired = _job_lock.acquire(blocking=False)
    if not acquired:
        logger.warning(
            "[scheduler] Previous job is still running. "
            "Skipping this scheduled execution to avoid overlap."
        )
        return

    trigger_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[scheduler] Job triggered at {trigger_time}")

    try:
        from pipeline import run_pipeline
        result = run_pipeline(dry_run=dry_run)

        if result.get("success"):
            _consecutive_failures = 0
            logger.info(
                f"[scheduler] Job succeeded. "
                f"Post ID: {result.get('post_id')}  "
                f"Duration: {result.get('duration_seconds', '?')}s"
            )
        else:
            # run_pipeline returned without raising but marked success=False
            _consecutive_failures += 1
            logger.error(
                f"[scheduler] Job reported failure (success=False). "
                f"Error: {result.get('error')}. "
                f"Consecutive failures: {_consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}"
            )

    except EnvironmentError as exc:
        # Missing API key / credentials — don't retry, alert loudly
        _consecutive_failures += 1
        logger.critical(
            f"[scheduler] Configuration error — job cannot run: {exc}\n"
            f"Fix your .env file and restart the scheduler."
        )
    except OSError as exc:
        _consecutive_failures += 1
        logger.error(
            f"[scheduler] System/disk error during job: {exc}. "
            f"Consecutive failures: {_consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}"
        )
    except Exception as exc:
        _consecutive_failures += 1
        logger.error(
            f"[scheduler] Job failed with {type(exc).__name__}: {exc}. "
            f"Consecutive failures: {_consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}"
        )
        logger.debug(traceback.format_exc())
    finally:
        _job_lock.release()

    # Shut down after too many consecutive failures to avoid spamming APIs
    if _consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
        logger.critical(
            f"[scheduler] {_consecutive_failures} consecutive failures reached. "
            "Stopping scheduler to prevent further damage. "
            "Fix the underlying issue and restart manually."
        )
        sys.exit(1)


# ── Scheduler loop ─────────────────────────────────────────────────────────────

def run_scheduler(dry_run: bool = False) -> None:
    """
    Start the scheduler with daily jobs at 09:00 and 18:00.

    Handles:
    - SIGINT / SIGTERM for graceful shutdown
    - schedule library exceptions (logged, not fatal)
    - Heartbeat log every 30 minutes
    """
    logger.info("=" * 50)
    logger.info("AI Influencer Scheduler starting")
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    logger.info("=" * 50)

    # Register scheduled jobs
    schedule.every().day.at("09:00").do(job, dry_run=dry_run)
    schedule.every().day.at("18:00").do(job, dry_run=dry_run)

    next_run = schedule.next_run()
    logger.info(f"[scheduler] Jobs scheduled: 09:00 and 18:00 daily")
    logger.info(f"[scheduler] Next run: {next_run}")

    # Graceful shutdown handlers
    def _shutdown(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info(f"[scheduler] Received {sig_name}. Shutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Main loop
    heartbeat_interval = 30 * 60  # 30 minutes
    last_heartbeat = time.time()

    while True:
        try:
            schedule.run_pending()
        except Exception as exc:
            # schedule library itself threw — log and continue
            logger.error(
                f"[scheduler] Unexpected error in schedule.run_pending(): "
                f"{type(exc).__name__}: {exc}"
            )
            logger.debug(traceback.format_exc())

        # Heartbeat log
        now = time.time()
        if now - last_heartbeat >= heartbeat_interval:
            next_run = schedule.next_run()
            logger.info(
                f"[scheduler] Heartbeat — still running. "
                f"Consecutive failures: {_consecutive_failures}. "
                f"Next run: {next_run}"
            )
            last_heartbeat = now

        time.sleep(30)  # check every 30 seconds


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AI Influencer Scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scheduler.py                  # Start scheduler (09:00 + 18:00)
  python scheduler.py --dry-run        # Dry-run mode (no real API calls)
  python scheduler.py --run-now        # Run one job now, then keep scheduling
  python scheduler.py --debug          # Enable DEBUG logging
""",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run jobs without real API calls")
    parser.add_argument("--run-now", action="store_true", help="Run one job immediately and keep scheduling")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG-level logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("[scheduler] DEBUG logging enabled")

    if args.run_now:
        logger.info("[scheduler] --run-now flag set: running immediate job before scheduling...")
        job(dry_run=args.dry_run)

    run_scheduler(dry_run=args.dry_run)
