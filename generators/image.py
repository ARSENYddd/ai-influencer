"""Image generation via Leonardo AI API."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

import httpx
from loguru import logger

from config import LEONARDO_API_KEY, CHARACTER_STYLE, LEONARDO_MODEL_ID, OUTPUT_DIR

_BASE = "https://cloud.leonardo.ai/api/rest/v1"


# ── Custom exceptions ──────────────────────────────────────────────────────────

class LeonardoError(Exception):
    """Base Leonardo error."""

class LeonardoAuthError(LeonardoError):
    """API key invalid or missing."""

class LeonardoRateLimitError(LeonardoError):
    """Daily generation limit reached."""

class LeonardoSafetyError(LeonardoError):
    """Prompt blocked by safety filter."""

class LeonardoGenerationError(LeonardoError):
    """Generation failed or returned no images."""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _check_config() -> None:
    if not LEONARDO_API_KEY:
        raise EnvironmentError(
            "LEONARDO_API_KEY is not set.\n"
            "  Get your key at https://app.leonardo.ai/api-access\n"
            "  Free tier: 150 generations/day"
        )


def _headers() -> dict:
    return {"Authorization": f"Bearer {LEONARDO_API_KEY}", "Content-Type": "application/json"}


def _classify(exc: httpx.HTTPStatusError, context: str) -> LeonardoError:
    status = exc.response.status_code
    try:
        body = exc.response.json()
        message = body.get("error") or body.get("message") or str(body)
    except Exception:
        message = exc.response.text[:200]

    if status == 401:
        return LeonardoAuthError(
            f"[{context}] Leonardo API key invalid or expired.\n"
            "  Regenerate at https://app.leonardo.ai/api-access\n"
            f"  Response: {message}"
        )
    if status == 429:
        return LeonardoRateLimitError(
            f"[{context}] Leonardo rate limit reached.\n"
            "  Free tier: 150 generations/day. Resets at midnight UTC.\n"
            "  Consider upgrading at https://leonardo.ai/pricing"
        )
    if status == 400 and ("nsfw" in message.lower() or "safety" in message.lower()):
        return LeonardoSafetyError(
            f"[{context}] Prompt blocked by Leonardo safety filter.\n"
            "  Modify the prompt to avoid restricted content.\n"
            f"  Details: {message}"
        )
    return LeonardoError(f"[{context}] HTTP {status}: {message}")


def _build_prompt(theme: str) -> str:
    return (
        f"{CHARACTER_STYLE}, {theme}, "
        "ultra realistic, 8k, Instagram lifestyle photo, cinematic lighting"
    )


# ── Generation polling ─────────────────────────────────────────────────────────

async def _poll_generation(
    client: httpx.AsyncClient,
    generation_id: str,
    max_wait: int = 120,
) -> list[str]:
    """Poll Leonardo until images are ready. Returns list of image URLs."""
    elapsed = 0
    while elapsed < max_wait:
        await asyncio.sleep(5)
        elapsed += 5
        try:
            r = await client.get(
                f"{_BASE}/generations/{generation_id}",
                headers=_headers(),
            )
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise _classify(exc, "poll_generation") from exc

        data = r.json().get("generations_by_pk", {})
        status = data.get("status")
        logger.debug(f"[leonardo] Generation {generation_id}: {status} ({elapsed}s)")

        if status == "COMPLETE":
            urls = [img["url"] for img in data.get("generated_images", [])]
            if not urls:
                raise LeonardoGenerationError(
                    f"Leonardo generation {generation_id} completed but returned no images.\n"
                    "  The model may have filtered all outputs. Try a different prompt."
                )
            return urls
        if status == "FAILED":
            raise LeonardoGenerationError(
                f"Leonardo generation {generation_id} failed.\n"
                f"  Response: {data}\n"
                "  Try again or check https://status.leonardo.ai"
            )

    raise LeonardoGenerationError(
        f"Leonardo generation {generation_id} timed out after {max_wait}s.\n"
        "  Leonardo may be under heavy load. Try again in a few minutes."
    )


# ── Main ───────────────────────────────────────────────────────────────────────

async def generate_image(theme: str) -> str:
    """
    Generate a character image via Leonardo Diffusion XL.

    Args:
        theme: Content theme string (e.g. "morning routine / coffee").

    Returns:
        Local path to the saved image file.

    Raises:
        EnvironmentError: API key not set.
        LeonardoAuthError: Key invalid/expired.
        LeonardoRateLimitError: Daily limit exceeded.
        LeonardoSafetyError: Prompt blocked.
        LeonardoGenerationError: Generation failed or timed out.
    """
    _check_config()
    prompt = _build_prompt(theme)
    logger.debug(f"[leonardo] Prompt: {prompt[:120]}...")

    payload = {
        "modelId": LEONARDO_MODEL_ID,
        "prompt": prompt,
        "width": 1024,
        "height": 1024,
        "num_images": 1,
        "guidance_scale": 7,
        "photoReal": False,
        "alchemy": True,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        for attempt in range(1, 4):
            try:
                r = await client.post(f"{_BASE}/generations", json=payload, headers=_headers())
                r.raise_for_status()
                generation_id = r.json()["sdGenerationJob"]["generationId"]
                logger.debug(f"[leonardo] Generation started: {generation_id}")
                break
            except httpx.HTTPStatusError as exc:
                typed = _classify(exc, "start_generation")
                # Auth and safety errors are permanent — don't retry
                if isinstance(typed, (LeonardoAuthError, LeonardoSafetyError)):
                    raise typed from exc
                if isinstance(typed, LeonardoRateLimitError):
                    raise typed from exc  # no point retrying rate limit
                if attempt == 3:
                    raise typed from exc
                wait = 2 ** attempt * 5
                logger.warning(f"[leonardo] Attempt {attempt} failed: {typed}. Retrying in {wait}s")
                await asyncio.sleep(wait)
            except httpx.ConnectError as exc:
                if attempt == 3:
                    raise LeonardoError(
                        "Cannot connect to Leonardo AI.\n"
                        "  Check internet connection or https://status.leonardo.ai"
                    ) from exc
                await asyncio.sleep(2 ** attempt * 5)

        urls = await _poll_generation(client, generation_id)

        # Download first image
        out_dir = Path(OUTPUT_DIR) / "images"
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = out_dir / f"sofia_{timestamp}.jpg"

        try:
            r = await client.get(urls[0], timeout=60)
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LeonardoGenerationError(
                f"Failed to download image from Leonardo CDN: HTTP {exc.response.status_code}\n"
                f"  URL: {urls[0]}"
            ) from exc
        except httpx.ConnectError as exc:
            raise LeonardoGenerationError(
                f"Network error downloading image from Leonardo CDN.\n"
                f"  URL: {urls[0]}\n"
                f"  Error: {exc}"
            ) from exc

        if len(r.content) < 10_000:
            raise LeonardoGenerationError(
                f"Downloaded image is suspiciously small ({len(r.content)} bytes).\n"
                "  Likely a CDN error or corrupt file. Retrying may help."
            )

        filepath.write_bytes(r.content)
        logger.info(f"[leonardo] Image saved: {filepath} ({len(r.content) / 1024:.1f} KB)")
        return str(filepath)
