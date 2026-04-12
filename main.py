"""
Main entry point.
Starts:
1. FastAPI dashboard on port 8080
2. Scheduler that runs content pipeline at configured times
"""
import uvicorn
import schedule
import time
import threading
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)


def start_scheduler():
    from config.settings import settings
    from pipeline_runner import run_pipeline

    times = settings.CONTENT_SCHEDULE_TIMES.split(",")
    for t in times:
        schedule.every().day.at(t.strip()).do(run_pipeline)
        logging.info(f"Scheduled pipeline at {t.strip()}")

    while True:
        schedule.run_pending()
        time.sleep(60)


def start_dashboard():
    from config.settings import settings
    from dashboard.app import app
    uvicorn.run(app, host="0.0.0.0", port=settings.DASHBOARD_PORT)


if __name__ == "__main__":
    # Ensure dirs and DB exist
    from database.db import init_db
    init_db()

    logging.info("Starting AI Influencer Dashboard + Scheduler")

    # Start scheduler in background thread
    scheduler_thread = threading.Thread(target=start_scheduler, daemon=True)
    scheduler_thread.start()

    # Start dashboard (blocking)
    start_dashboard()
