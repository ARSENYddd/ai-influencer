"""Image generation module using Replicate Flux Dev model."""

import os
import shutil
import time
import logging
import requests
import traceback
from pathlib import Path
from datetime import datetime
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)

import replicate
import replicate.exceptions

logger = logging.getLogger(__name__)

FLUX_MODEL = "black-forest-labs/flux-dev"
MIN_IMAGE_SIZE_BYTES = 10_000   # <10 KB → treat as corrupted
MIN_DISK_FREE_MB = 100          # require at least 100 MB free

PROMPT_TEMPLATES = [
    "beautiful AI girl, {style}, natural lighting, Instagram photo, ultra realistic, 8k",
    "gorgeous young woman, {style}, golden hour light, lifestyle photography, photorealistic",
    "stunning AI influencer, {style}, aesthetic background, professional photography, highly detailed",
    "attractive girl, {style}, soft bokeh background, fashion shoot, hyperrealistic",
    "modern AI girl, {style}, candid street photography style, warm tones, cinematic",
]

STYLES = [
    "casual chic outfit, coffee shop background",
    "beach sunset, summer dress, ocean view",
    "urban fashion, city streets, modern architecture",
    "cozy home aesthetic, warm interior, morning light",
    "fitness lifestyle, activewear, gym or outdoor",
    "travel blogger, exotic location, adventurous",
    "luxury fashion, elegant dress, upscale setting",
    "bohemian style, nature background, golden hour",
]


# ── Custom exceptions ──────────────────────────────────────────────────────────

class GeneratorError(Exception):
    """Base class for generator errors."""

class GeneratorAuthError(GeneratorError):
    """REPLICATE_API_TOKEN is missing or invalid."""

class GeneratorRateLimitError(GeneratorError):
    """Replicate rate limit exceeded."""

class GeneratorModelError(GeneratorError):
    """Model not found or refused to run."""

class GeneratorEmptyOutputError(GeneratorError):
    """Replicate returned no output."""

class GeneratorCorruptImageError(GeneratorError):
    """Downloaded image is too small / corrupted."""

class GeneratorDiskError(GeneratorError):
    """Not enough disk space to save the image."""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _check_disk_space(path: str = ".") -> None:
    """Raise GeneratorDiskError if free disk space is below threshold."""
    free_mb = shutil.disk_usage(path).free / (1024 * 1024)
    if free_mb < MIN_DISK_FREE_MB:
        raise GeneratorDiskError(
            f"Not enough disk space: {free_mb:.1f} MB free, need {MIN_DISK_FREE_MB} MB"
        )


def _check_api_token() -> None:
    """Raise GeneratorAuthError if REPLICATE_API_TOKEN is not set."""
    token = os.environ.get("REPLICATE_API_TOKEN", "").strip()
    if not token:
        raise GeneratorAuthError(
            "REPLICATE_API_TOKEN is not set. "
            "Get your token at https://replicate.com/account/api-tokens"
        )
    if len(token) < 10:
        raise GeneratorAuthError(
            f"REPLICATE_API_TOKEN looks invalid (too short: {len(token)} chars)."
        )


def _classify_replicate_error(exc: Exception) -> Exception:
    """Map Replicate exceptions to domain-specific errors."""
    msg = str(exc).lower()

    if isinstance(exc, replicate.exceptions.ReplicateError):
        if "authentication" in msg or "unauthorized" in msg or "forbidden" in msg or "401" in msg:
            return GeneratorAuthError(f"Replicate authentication failed: {exc}")
        if "rate limit" in msg or "too many requests" in msg or "429" in msg:
            return GeneratorRateLimitError(f"Replicate rate limit hit: {exc}")
        if "model" in msg and ("not found" in msg or "404" in msg):
            return GeneratorModelError(f"Replicate model not found: {exc}")
        if "nsfw" in msg or "safety" in msg or "content policy" in msg:
            return GeneratorModelError(f"Prompt blocked by Replicate safety filter: {exc}")

    return exc  # return as-is, let tenacity retry it


# ── Main generation function ───────────────────────────────────────────────────

def _should_retry(exc: Exception) -> bool:
    """Return True for transient errors, False for permanent ones."""
    if isinstance(exc, (GeneratorAuthError, GeneratorModelError, GeneratorDiskError)):
        return False   # permanent — don't retry
    return True        # network, rate limit, unknown → retry


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=5, max=40),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda rs: logger.warning(
        f"[generator] Attempt {rs.attempt_number} failed "
        f"({type(rs.outcome.exception()).__name__}: {rs.outcome.exception()}), "
        f"retrying in {rs.next_action.sleep:.1f}s..."
    ),
    reraise=True,
)
def _run_replicate(prompt: str) -> list:
    """Call Replicate API with full error classification."""
    try:
        output = replicate.run(
            FLUX_MODEL,
            input={
                "prompt": prompt,
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 28,
                "guidance": 3.5,
                "output_format": "jpg",
                "output_quality": 90,
            },
        )
    except replicate.exceptions.ReplicateError as exc:
        classified = _classify_replicate_error(exc)
        if not _should_retry(classified):
            logger.error(f"[generator] Permanent Replicate error: {classified}")
            raise classified from exc
        raise classified from exc
    except requests.exceptions.ConnectionError as exc:
        logger.warning(f"[generator] Network connection error to Replicate: {exc}")
        raise
    except requests.exceptions.Timeout as exc:
        logger.warning(f"[generator] Request to Replicate timed out: {exc}")
        raise
    except Exception as exc:
        logger.error(f"[generator] Unexpected error calling Replicate: {type(exc).__name__}: {exc}")
        logger.debug(traceback.format_exc())
        raise

    return output


def generate_image(prompt: str = None, style_index: int = None) -> str:
    """
    Generate an AI image using Replicate Flux Dev.

    Args:
        prompt: Custom prompt override. If None, auto-selects.
        style_index: Style index override. If None, uses time-based rotation.

    Returns:
        Path to the saved image file.

    Raises:
        GeneratorAuthError: Token missing or invalid.
        GeneratorRateLimitError: Replicate rate limit hit.
        GeneratorModelError: Model not found or blocked by safety filter.
        GeneratorEmptyOutputError: Replicate returned no image.
        GeneratorCorruptImageError: Downloaded image is corrupted.
        GeneratorDiskError: Not enough disk space.
    """
    logger.info("[generator] Starting image generation")

    # Pre-flight checks
    _check_api_token()
    _check_disk_space()

    # Build prompt
    if style_index is None:
        style_index = int(time.time() / 3600) % len(STYLES)
    style = STYLES[style_index]

    if prompt is None:
        template_index = style_index % len(PROMPT_TEMPLATES)
        prompt = PROMPT_TEMPLATES[template_index].format(style=style)

    logger.info(f"[generator] Prompt: {prompt[:100]}...")
    logger.debug(f"[generator] Full prompt: {prompt}")

    # Call Replicate (with retry)
    try:
        output = _run_replicate(prompt)
    except RetryError as exc:
        original = exc.last_attempt.exception()
        logger.error(f"[generator] All 3 Replicate attempts failed. Last error: {original}")
        raise original from exc

    # Validate output
    if not output:
        raise GeneratorEmptyOutputError(
            "Replicate returned empty output. The model may have failed silently."
        )

    image_url = str(output[0]) if isinstance(output, list) else str(output)
    if not image_url.startswith("http"):
        raise GeneratorEmptyOutputError(
            f"Replicate output is not a valid URL: {image_url!r}"
        )

    logger.info(f"[generator] Image URL received, downloading...")
    logger.debug(f"[generator] Image URL: {image_url}")

    return _download_image(image_url)


def _download_image(url: str) -> str:
    """Download image from URL and save to output/images/."""
    output_dir = Path("output/images")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise GeneratorDiskError(f"Cannot create output directory: {exc}") from exc

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"ai_girl_{timestamp}.jpg"

    try:
        response = requests.get(url, timeout=90, stream=True)
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        logger.error(f"[generator] HTTP {status} downloading image from Replicate")
        raise
    except requests.exceptions.ConnectionError as exc:
        logger.error(f"[generator] Network error downloading image: {exc}")
        raise
    except requests.exceptions.Timeout:
        logger.error("[generator] Timeout downloading image (>90s)")
        raise

    content = response.content
    size_kb = len(content) / 1024

    if len(content) < MIN_IMAGE_SIZE_BYTES:
        raise GeneratorCorruptImageError(
            f"Downloaded image is too small ({size_kb:.1f} KB < "
            f"{MIN_IMAGE_SIZE_BYTES / 1024:.0f} KB minimum). "
            f"Likely corrupted or invalid URL: {url}"
        )

    try:
        with open(filepath, "wb") as f:
            f.write(content)
    except OSError as exc:
        raise GeneratorDiskError(f"Cannot write image to disk: {exc}") from exc

    logger.info(f"[generator] Image saved: {filepath} ({size_kb:.1f} KB)")
    return str(filepath)


# ── Dry-run ────────────────────────────────────────────────────────────────────

def generate_image_dry_run() -> str:
    """Dry-run: skip API call, return a placeholder path."""
    output_dir = Path("output/images")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise GeneratorDiskError(f"Cannot create output directory: {exc}") from exc

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"dry_run_{timestamp}.jpg"

    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (512, 512), color=(180, 120, 200))
        draw = ImageDraw.Draw(img)
        draw.text((120, 230), "DRY RUN\nAI Girl Placeholder", fill=(255, 255, 255))
        img.save(str(filepath), "JPEG")
        logger.info(f"[generator] Dry-run placeholder created: {filepath}")
    except ImportError:
        logger.warning("[generator] Pillow not installed, creating empty placeholder file")
        filepath.touch()
    except Exception as exc:
        logger.warning(f"[generator] Could not create placeholder image: {exc}")
        filepath.touch()

    return str(filepath)
