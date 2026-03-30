"""Instagram posting module using instagrapi."""

import logging
import os
import time
import random
import traceback
from pathlib import Path
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)

from instagrapi import Client
from instagrapi.exceptions import (
    LoginRequired,
    ChallengeRequired,
    TwoFactorRequired,
    BadPassword,
    InvalidTargetUser,
    UserNotFound,
    MediaNotFound,
    ClientError,
    ClientLoginRequired,
    ClientForbiddenError,
    ClientThrottledError,
    ReloginAttemptExceeded,
    PleaseWaitFewMinutes,
    FeedbackRequired,
)

logger = logging.getLogger(__name__)

SESSION_FILE = "session_instagram.json"
MAX_CAPTION_LENGTH = 2200   # Instagram hard limit
MAX_IMAGE_SIZE_MB = 8       # Instagram upload limit


# ── Custom exceptions ──────────────────────────────────────────────────────────

class PosterError(Exception):
    """Base class for poster errors."""

class PosterAuthError(PosterError):
    """Wrong credentials or account banned."""

class PosterChallengeError(PosterError):
    """Instagram requires email/phone verification."""

class Poster2FAError(PosterError):
    """Two-factor authentication required."""

class PosterRateLimitError(PosterError):
    """Too many requests — Instagram throttle."""

class PosterImageError(PosterError):
    """Image file is missing, too large, or invalid format."""

class PosterNetworkError(PosterError):
    """Network connectivity issue."""

class PosterAccountError(PosterError):
    """Account suspended or permanently restricted."""


# ── Pre-flight validation ──────────────────────────────────────────────────────

def _validate_image(image_path: str) -> None:
    """Validate image file before upload attempt."""
    path = Path(image_path)

    if not path.exists():
        raise PosterImageError(f"Image file does not exist: {image_path}")

    if not path.is_file():
        raise PosterImageError(f"Image path is not a file: {image_path}")

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise PosterImageError(
            f"Image too large: {size_mb:.1f} MB (Instagram limit: {MAX_IMAGE_SIZE_MB} MB). "
            f"Path: {image_path}"
        )

    if path.stat().st_size < 1000:
        raise PosterImageError(
            f"Image file is suspiciously small ({path.stat().st_size} bytes). "
            f"Likely corrupted: {image_path}"
        )

    suffix = path.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise PosterImageError(
            f"Unsupported image format: {suffix!r}. Instagram accepts .jpg / .png"
        )


def _validate_caption(caption: str) -> str:
    """Truncate caption if over Instagram's 2200-char limit."""
    if len(caption) > MAX_CAPTION_LENGTH:
        logger.warning(
            f"[poster] Caption too long ({len(caption)} chars > {MAX_CAPTION_LENGTH}). "
            "Truncating to fit Instagram limit."
        )
        caption = caption[:MAX_CAPTION_LENGTH - 3] + "..."
    return caption


def _validate_credentials(username: str, password: str) -> None:
    """Basic sanity-check on credentials before hitting the API."""
    if not username or not username.strip():
        raise PosterAuthError("INSTAGRAM_USERNAME is empty.")
    if not password or not password.strip():
        raise PosterAuthError("INSTAGRAM_PASSWORD is empty.")
    if " " in username:
        raise PosterAuthError(
            f"INSTAGRAM_USERNAME contains spaces: {username!r}. "
            "Instagram usernames cannot have spaces."
        )


# ── Session management ─────────────────────────────────────────────────────────

def _get_client(username: str, password: str) -> Client:
    """
    Initialize and authenticate Instagram client with session caching.

    Session is loaded from SESSION_FILE on subsequent runs to avoid
    repeated logins (which Instagram flags as suspicious).
    """
    cl = Client()
    cl.delay_range = [2, 5]  # Human-like delays between requests

    session_path = Path(SESSION_FILE)

    # Try loading cached session first
    if session_path.exists():
        logger.info(f"[poster] Loading cached session from {SESSION_FILE}")
        try:
            cl.load_settings(str(session_path))
            cl.login(username, password)
            logger.info("[poster] Logged in using saved session.")
            return cl
        except LoginRequired as exc:
            logger.warning(f"[poster] Cached session expired (LoginRequired): {exc}")
            _delete_session(session_path)
        except BadPassword as exc:
            logger.error(f"[poster] Wrong password detected while restoring session: {exc}")
            _delete_session(session_path)
            raise PosterAuthError(
                f"Instagram rejected password for @{username}. "
                "Check INSTAGRAM_PASSWORD in your .env file."
            ) from exc
        except ReloginAttemptExceeded as exc:
            logger.error("[poster] Too many relogin attempts — Instagram has flagged this account.")
            raise PosterAccountError(
                "Instagram is blocking repeated login attempts. "
                "Wait several hours before retrying."
            ) from exc
        except ChallengeRequired as exc:
            logger.error("[poster] Instagram challenge triggered during session restore.")
            raise PosterChallengeError(
                "Instagram requires email/phone verification. "
                "Log in manually once to complete the challenge."
            ) from exc
        except Exception as exc:
            logger.warning(f"[poster] Session restore failed ({type(exc).__name__}: {exc}). Retrying fresh login.")
            logger.debug(traceback.format_exc())
            _delete_session(session_path)

    # Fresh login
    logger.info(f"[poster] Attempting fresh login for @{username}...")
    try:
        cl.login(username, password)
        cl.dump_settings(SESSION_FILE)
        logger.info(f"[poster] Fresh login successful. Session cached to {SESSION_FILE}")
    except BadPassword as exc:
        raise PosterAuthError(
            f"Wrong password for @{username}. "
            "Check INSTAGRAM_PASSWORD in your .env file."
        ) from exc
    except ChallengeRequired as exc:
        raise PosterChallengeError(
            "Instagram requires email/phone verification before posting. "
            "Log in manually via the Instagram app to complete the challenge, "
            "then re-run this script."
        ) from exc
    except TwoFactorRequired as exc:
        raise Poster2FAError(
            "Two-factor authentication is enabled on this account. "
            "Disable 2FA in Instagram settings, or handle the 2FA code manually."
        ) from exc
    except FeedbackRequired as exc:
        raise PosterAccountError(
            f"Instagram returned FeedbackRequired — account may be restricted or suspended. "
            f"Details: {exc}"
        ) from exc
    except ReloginAttemptExceeded as exc:
        raise PosterAccountError(
            "Instagram is blocking login attempts (too many in a short time). "
            "Wait a few hours and try again."
        ) from exc
    except PleaseWaitFewMinutes as exc:
        raise PosterRateLimitError(
            f"Instagram says 'please wait a few minutes': {exc}"
        ) from exc
    except ClientForbiddenError as exc:
        raise PosterAuthError(
            f"Instagram account @{username} is forbidden (suspended or disabled): {exc}"
        ) from exc
    except ClientError as exc:
        msg = str(exc).lower()
        if "checkpoint" in msg:
            raise PosterChallengeError(
                f"Instagram checkpoint required (suspicious activity detected): {exc}"
            ) from exc
        logger.error(f"[poster] Instagram ClientError during login: {exc}")
        logger.debug(traceback.format_exc())
        raise PosterNetworkError(f"Instagram API error during login: {exc}") from exc
    except Exception as exc:
        logger.error(f"[poster] Unexpected error during Instagram login: {type(exc).__name__}: {exc}")
        logger.debug(traceback.format_exc())
        raise

    return cl


def _delete_session(path: Path) -> None:
    """Safely delete a stale session file."""
    try:
        path.unlink(missing_ok=True)
        logger.debug(f"[poster] Deleted stale session file: {path}")
    except OSError as exc:
        logger.warning(f"[poster] Could not delete session file {path}: {exc}")


# ── Posting ────────────────────────────────────────────────────────────────────

def _should_retry_post(exc: Exception) -> bool:
    """Return True for transient errors only."""
    if isinstance(exc, (
        PosterAuthError, Poster2FAError, PosterChallengeError,
        PosterImageError, PosterAccountError,
    )):
        return False   # permanent — don't retry
    return True


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=3, min=10, max=90),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda rs: logger.warning(
        f"[poster] Post attempt {rs.attempt_number} failed "
        f"({type(rs.outcome.exception()).__name__}: {rs.outcome.exception()}), "
        f"retrying in {rs.next_action.sleep:.1f}s..."
    ),
    reraise=True,
)
def _upload(cl: Client, image_path: str, caption: str) -> str:
    """Perform the actual photo upload. Raises classified exceptions."""
    try:
        media = cl.photo_upload(path=image_path, caption=caption)
    except ClientThrottledError as exc:
        raise PosterRateLimitError(
            f"Instagram throttled the upload request: {exc}"
        ) from exc
    except PleaseWaitFewMinutes as exc:
        raise PosterRateLimitError(
            f"Instagram says 'please wait a few minutes' during upload: {exc}"
        ) from exc
    except FeedbackRequired as exc:
        raise PosterAccountError(
            f"Instagram FeedbackRequired during upload — account may be restricted: {exc}"
        ) from exc
    except MediaNotFound as exc:
        raise PosterImageError(f"Instagram could not process the media file: {exc}") from exc
    except ClientLoginRequired as exc:
        raise PosterAuthError(f"Instagram requires re-login during upload: {exc}") from exc
    except ClientForbiddenError as exc:
        raise PosterAuthError(
            f"Instagram upload forbidden — account may be suspended: {exc}"
        ) from exc
    except ClientError as exc:
        msg = str(exc).lower()
        if "checkpoint" in msg or "challenge" in msg:
            raise PosterChallengeError(
                f"Instagram challenge triggered during upload: {exc}"
            ) from exc
        if "spam" in msg or "block" in msg:
            raise PosterAccountError(
                f"Post blocked by Instagram spam filter: {exc}"
            ) from exc
        logger.error(f"[poster] Instagram ClientError during upload: {exc}")
        logger.debug(traceback.format_exc())
        raise PosterNetworkError(f"Instagram API error during upload: {exc}") from exc
    except Exception as exc:
        logger.error(f"[poster] Unexpected error during upload: {type(exc).__name__}: {exc}")
        logger.debug(traceback.format_exc())
        raise

    return str(media.pk)


def post_to_instagram(image_path: str, caption: str, username: str, password: str) -> str:
    """
    Post an image to Instagram.

    Args:
        image_path: Local path to the image file.
        caption: Caption text with hashtags.
        username: Instagram username.
        password: Instagram password.

    Returns:
        Post media ID string.

    Raises:
        PosterAuthError: Wrong credentials or account banned.
        PosterChallengeError: Instagram challenge (verification) required.
        Poster2FAError: Two-factor authentication required.
        PosterRateLimitError: Instagram throttle.
        PosterImageError: Image file invalid, too large, or missing.
        PosterAccountError: Account suspended or restricted.
        PosterNetworkError: Network connectivity error.
    """
    logger.info(f"[poster] Starting Instagram post for @{username}")

    # Pre-flight checks
    _validate_credentials(username, password)
    _validate_image(image_path)
    caption = _validate_caption(caption)

    # Authenticate
    cl = _get_client(username, password)

    # Human-like delay before uploading
    delay = random.uniform(3, 8)
    logger.debug(f"[poster] Waiting {delay:.1f}s before upload (human simulation)...")
    time.sleep(delay)

    logger.info(f"[poster] Uploading {image_path} ({Path(image_path).stat().st_size / 1024:.1f} KB)...")

    try:
        post_id = _upload(cl, image_path, caption)
    except RetryError as exc:
        original = exc.last_attempt.exception()
        logger.error(f"[poster] All 3 upload attempts failed. Last error: {original}")
        raise original from exc

    logger.info(f"[poster] Posted successfully! Media ID: {post_id}")
    return post_id


# ── Dry-run ────────────────────────────────────────────────────────────────────

def post_dry_run(image_path: str, caption: str) -> str:
    """Dry-run: simulate posting without hitting Instagram API."""
    # Still validate the image file exists
    try:
        _validate_image(image_path)
        logger.info(f"[poster] [DRY RUN] Image validated OK: {image_path}")
    except PosterImageError as exc:
        logger.warning(f"[poster] [DRY RUN] Image validation warning: {exc}")

    caption = _validate_caption(caption)
    logger.info(f"[poster] [DRY RUN] Would post image: {image_path}")
    logger.info(f"[poster] [DRY RUN] Caption ({len(caption)} chars):\n{caption[:300]}")

    fake_id = f"dry_run_{int(time.time())}"
    logger.info(f"[poster] [DRY RUN] Fake media ID: {fake_id}")
    return fake_id
