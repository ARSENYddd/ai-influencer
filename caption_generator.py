"""Caption generation module using Anthropic Claude Haiku."""

import os
import logging
import random
import traceback
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)

import anthropic

logger = logging.getLogger(__name__)

CAPTION_STYLES = [
    "motivational and inspiring",
    "fun and playful with emojis",
    "mysterious and aesthetic",
    "lifestyle and wellness focused",
    "travel and adventure themed",
    "fashion and beauty focused",
    "self-love and confidence boosting",
]

HASHTAG_SETS = [
    "#aesthetic #lifestyle #vibes #instagood #photooftheday",
    "#fashion #ootd #style #trendy #instafashion",
    "#travel #wanderlust #explore #adventure #travelgram",
    "#fitness #wellness #health #motivation #selfcare",
    "#beauty #makeup #glow #skincare #naturalbeauty",
    "#coffee #cozy #morningvibes #dailylife #homestyle",
    "#sunset #goldenhour #nature #photography #beautiful",
]

SYSTEM_PROMPT = """You are a social media expert writing Instagram captions for a popular AI lifestyle influencer.
Write engaging, authentic-sounding captions that:
- Feel personal and relatable
- Use appropriate emojis naturally (not excessively)
- Include a subtle call-to-action or question to boost engagement
- Sound like a real 20-something woman posting her daily life
- Are 2-4 sentences maximum
- Never mention AI, algorithms, or automation
Output ONLY the caption text, nothing else."""


# ── Custom exceptions ──────────────────────────────────────────────────────────

class CaptionError(Exception):
    """Base class for caption generator errors."""

class CaptionAuthError(CaptionError):
    """ANTHROPIC_API_KEY is missing or invalid."""

class CaptionRateLimitError(CaptionError):
    """Anthropic rate limit exceeded."""

class CaptionOverloadError(CaptionError):
    """Anthropic API is overloaded (529)."""

class CaptionEmptyResponseError(CaptionError):
    """Claude returned an empty or malformed response."""

class CaptionModelError(CaptionError):
    """Model not found or bad request."""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _check_api_token() -> None:
    """Raise CaptionAuthError if ANTHROPIC_API_KEY is not set."""
    token = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not token:
        raise CaptionAuthError(
            "ANTHROPIC_API_KEY is not set. "
            "Get your key at https://console.anthropic.com/"
        )
    if not token.startswith("sk-"):
        raise CaptionAuthError(
            f"ANTHROPIC_API_KEY looks invalid (should start with 'sk-'). "
            f"Got: {token[:8]}..."
        )


def _classify_anthropic_error(exc: anthropic.APIError) -> Exception:
    """Map Anthropic API errors to domain-specific exceptions."""
    status = getattr(exc, "status_code", None)
    msg = str(exc).lower()

    if status == 401 or isinstance(exc, anthropic.AuthenticationError):
        return CaptionAuthError(f"Anthropic authentication failed: {exc}")
    if status == 429 or isinstance(exc, anthropic.RateLimitError):
        return CaptionRateLimitError(f"Anthropic rate limit exceeded: {exc}")
    if status == 529 or (isinstance(exc, anthropic.APIStatusError) and status == 529):
        return CaptionOverloadError(f"Anthropic API overloaded (529): {exc}")
    if status == 400 or isinstance(exc, anthropic.BadRequestError):
        return CaptionModelError(f"Bad request to Anthropic API: {exc}")
    if status == 404:
        return CaptionModelError(f"Anthropic model not found: {exc}")

    return exc


def _should_retry(exc: Exception) -> bool:
    """Permanent errors should not be retried."""
    if isinstance(exc, (CaptionAuthError, CaptionModelError)):
        return False
    return True


# ── API call with retry ────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=5, max=40),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda rs: logger.warning(
        f"[caption] Attempt {rs.attempt_number} failed "
        f"({type(rs.outcome.exception()).__name__}: {rs.outcome.exception()}), "
        f"retrying in {rs.next_action.sleep:.1f}s..."
    ),
    reraise=True,
)
def _call_claude(client: anthropic.Anthropic, style: str) -> str:
    """Make the Anthropic API call. Raises classified exceptions."""
    user_prompt = (
        f"Write an Instagram caption in {style} style for a lifestyle photo. "
        "Keep it short and engaging."
    )

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except anthropic.AuthenticationError as exc:
        raise CaptionAuthError(f"Anthropic authentication failed: {exc}") from exc
    except anthropic.RateLimitError as exc:
        logger.warning(f"[caption] Rate limit hit, will retry: {exc}")
        raise CaptionRateLimitError(str(exc)) from exc
    except anthropic.APIStatusError as exc:
        classified = _classify_anthropic_error(exc)
        if not _should_retry(classified):
            logger.error(f"[caption] Permanent Anthropic error: {classified}")
            raise classified from exc
        raise classified from exc
    except anthropic.APIConnectionError as exc:
        logger.warning(f"[caption] Network error connecting to Anthropic: {exc}")
        raise
    except anthropic.APITimeoutError as exc:
        logger.warning(f"[caption] Anthropic request timed out: {exc}")
        raise
    except anthropic.APIError as exc:
        logger.error(f"[caption] Unknown Anthropic API error: {type(exc).__name__}: {exc}")
        logger.debug(traceback.format_exc())
        raise
    except Exception as exc:
        logger.error(f"[caption] Unexpected error: {type(exc).__name__}: {exc}")
        logger.debug(traceback.format_exc())
        raise

    # Validate response structure
    if not message.content:
        raise CaptionEmptyResponseError(
            "Claude returned a message with no content blocks."
        )

    first_block = message.content[0]
    if not hasattr(first_block, "text"):
        raise CaptionEmptyResponseError(
            f"Claude response content[0] has no 'text' attribute. "
            f"Got type: {type(first_block).__name__}"
        )

    text = first_block.text.strip()
    if not text:
        raise CaptionEmptyResponseError(
            "Claude returned an empty string as caption."
        )
    if len(text) < 10:
        raise CaptionEmptyResponseError(
            f"Claude response suspiciously short ({len(text)} chars): {text!r}"
        )

    logger.debug(f"[caption] Stop reason: {message.stop_reason}, tokens used: {message.usage}")
    return text


# ── Public function ────────────────────────────────────────────────────────────

def generate_caption(style_hint: str = None) -> tuple[str, str]:
    """
    Generate an Instagram caption using Claude Haiku.

    Args:
        style_hint: Optional style context (e.g., 'beach', 'coffee shop').

    Returns:
        Tuple of (caption_text, full_caption_with_hashtags).

    Raises:
        CaptionAuthError: API key missing or invalid.
        CaptionRateLimitError: Rate limit exceeded after retries.
        CaptionOverloadError: Anthropic API overloaded.
        CaptionModelError: Bad request or model not found.
        CaptionEmptyResponseError: Claude returned empty/invalid text.
    """
    logger.info("[caption] Starting caption generation")

    _check_api_token()

    style = style_hint or random.choice(CAPTION_STYLES)
    hashtags = random.choice(HASHTAG_SETS)

    logger.info(f"[caption] Style: {style}")
    logger.debug(f"[caption] Hashtag set: {hashtags}")

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    try:
        caption = _call_claude(client, style)
    except RetryError as exc:
        original = exc.last_attempt.exception()
        logger.error(f"[caption] All 3 Anthropic attempts failed. Last error: {original}")
        raise original from exc

    full_caption = f"{caption}\n\n{hashtags}"
    logger.info(f"[caption] Caption generated ({len(caption)} chars): {caption[:60]}...")
    return caption, full_caption


# ── Dry-run ────────────────────────────────────────────────────────────────────

def generate_caption_dry_run() -> tuple[str, str]:
    """Dry-run: return a pre-written sample caption without API call."""
    samples = [
        (
            "Living for these golden hour moments ✨ Life is too short not to chase the light. What's your favorite time of day?",
            "#aesthetic #lifestyle #vibes #instagood #photooftheday",
        ),
        (
            "Coffee first, everything else second ☕ Starting the morning right — what's your morning ritual?",
            "#coffee #cozy #morningvibes #dailylife #homestyle",
        ),
        (
            "New day, new energy 🌿 Grateful for every little moment that makes life beautiful. Drop a ❤️ if you feel this!",
            "#fitness #wellness #health #motivation #selfcare",
        ),
    ]
    caption_text, hashtags = random.choice(samples)
    full = f"{caption_text}\n\n{hashtags}"
    logger.info(f"[caption] Dry-run caption: {caption_text[:60]}...")
    return caption_text, full
