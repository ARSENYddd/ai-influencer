"""Instagram posting module using instagrapi."""

import logging
import time
import random
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from instagrapi import Client
from instagrapi.exceptions import LoginRequired, ChallengeRequired, TwoFactorRequired

logger = logging.getLogger(__name__)

SESSION_FILE = "session_instagram.json"


def _get_client(username: str, password: str) -> Client:
    """Initialize and authenticate Instagram client with session caching."""
    cl = Client()
    cl.delay_range = [2, 5]  # Human-like delays between requests

    session_path = Path(SESSION_FILE)
    if session_path.exists():
        try:
            cl.load_settings(str(session_path))
            cl.login(username, password)
            logger.info("Logged in using saved session.")
            return cl
        except (LoginRequired, Exception) as e:
            logger.warning(f"Session expired or invalid: {e}. Re-logging in...")
            session_path.unlink(missing_ok=True)

    try:
        cl.login(username, password)
        cl.dump_settings(SESSION_FILE)
        logger.info("Fresh login successful. Session saved.")
    except ChallengeRequired:
        logger.error("Instagram challenge required (email/phone verification). Complete it manually first.")
        raise
    except TwoFactorRequired:
        logger.error("Two-factor authentication required. Disable 2FA or handle it manually.")
        raise

    return cl


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=3, min=10, max=60),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: logger.warning(
        f"Post attempt {retry_state.attempt_number} failed, retrying..."
    ),
)
def post_to_instagram(image_path: str, caption: str, username: str, password: str) -> str:
    """
    Post an image to Instagram.

    Args:
        image_path: Local path to the image file.
        caption: Caption text with hashtags.
        username: Instagram username.
        password: Instagram password.

    Returns:
        Post media ID.
    """
    logger.info(f"Connecting to Instagram as @{username}...")
    cl = _get_client(username, password)

    # Small random delay to appear more human
    time.sleep(random.uniform(3, 8))

    logger.info(f"Uploading photo: {image_path}")
    media = cl.photo_upload(
        path=image_path,
        caption=caption,
    )

    post_id = str(media.pk)
    logger.info(f"Posted successfully! Media ID: {post_id}")
    return post_id


def post_dry_run(image_path: str, caption: str) -> str:
    """Dry-run: simulate posting without hitting Instagram API."""
    logger.info(f"[DRY RUN] Would post image: {image_path}")
    logger.info(f"[DRY RUN] Caption preview:\n{caption[:200]}")
    fake_id = f"dry_run_{int(time.time())}"
    logger.info(f"[DRY RUN] Fake media ID: {fake_id}")
    return fake_id
