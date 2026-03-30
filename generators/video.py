"""Video generation via Kling AI API."""

from __future__ import annotations

import asyncio
import base64
from datetime import datetime
from pathlib import Path

import httpx
from loguru import logger

from config import KLING_API_KEY, CHARACTER_STYLE, OUTPUT_DIR

_BASE = "https://api.klingai.com/v1"


# ── Custom exceptions ──────────────────────────────────────────────────────────

class KlingError(Exception):
    """Base Kling error."""

class KlingAuthError(KlingError):
    """API key invalid or missing."""

class KlingRateLimitError(KlingError):
    """Too many requests or quota exceeded."""

class KlingGenerationError(KlingError):
    """Video generation task failed or timed out."""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _check_config() -> None:
    if not KLING_API_KEY:
        raise EnvironmentError(
            "KLING_API_KEY is not set.\n"
            "  Get your key at https://platform.klingai.com\n"
            "  Paid service — check pricing at https://klingai.com/pricing"
        )


def _headers() -> dict:
    return {"Authorization": f"Bearer {KLING_API_KEY}", "Content-Type": "application/json"}


def _classify(exc: httpx.HTTPStatusError, context: str) -> KlingError:
    status = exc.response.status_code
    try:
        body = exc.response.json()
        message = body.get("message") or body.get("error") or str(body)
        code = body.get("code", 0)
    except Exception:
        message = exc.response.text[:200]
        code = 0

    if status == 401 or code in (1000, 1001):
        return KlingAuthError(
            f"[{context}] Kling API key invalid or expired (code {code}).\n"
            "  Regenerate at https://platform.klingai.com\n"
            f"  Details: {message}"
        )
    if status == 429 or code == 1002:
        return KlingRateLimitError(
            f"[{context}] Kling rate limit or quota exceeded (code {code}).\n"
            "  Check your plan limits at https://platform.klingai.com\n"
            f"  Details: {message}"
        )
    return KlingError(f"[{context}] HTTP {status} (code {code}): {message}")


def _build_prompt(theme: str) -> str:
    return (
        f"{CHARACTER_STYLE}, {theme}, "
        "smooth cinematic motion, lifestyle video, Instagram Reels style, 9:16 vertical"
    )


# ── Task polling ───────────────────────────────────────────────────────────────

async def _poll_task(
    client: httpx.AsyncClient,
    task_id: str,
    endpoint: str,
    max_wait: int = 300,
) -> str:
    """Poll Kling task until complete. Returns video URL."""
    elapsed = 0
    while elapsed < max_wait:
        await asyncio.sleep(10)
        elapsed += 10
        try:
            r = await client.get(f"{_BASE}/{endpoint}/{task_id}", headers=_headers())
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise _classify(exc, "poll_task") from exc

        data = r.json().get("data", {})
        status = data.get("task_status")
        logger.debug(f"[kling] Task {task_id}: {status} ({elapsed}s)")

        if status == "succeed":
            videos = data.get("task_result", {}).get("videos", [])
            if not videos:
                raise KlingGenerationError(
                    f"Kling task {task_id} succeeded but returned no video URLs.\n"
                    f"  Full response: {data}"
                )
            return videos[0]["url"]

        if status == "failed":
            reason = data.get("task_status_msg", "unknown")
            raise KlingGenerationError(
                f"Kling task {task_id} failed.\n"
                f"  Reason: {reason}\n"
                "  Try again or check https://platform.klingai.com for service status."
            )

    raise KlingGenerationError(
        f"Kling task {task_id} timed out after {max_wait}s.\n"
        "  Kling may be under load. Try again later."
    )


# ── Main ───────────────────────────────────────────────────────────────────────

async def generate_video(image_path: str, theme: str) -> str:
    """
    Generate a short video clip using Kling AI (image-to-video).

    Args:
        image_path: Local path to the reference image from Leonardo.
        theme: Content theme string.

    Returns:
        Local path to the saved video file.

    Raises:
        EnvironmentError: API key not set.
        KlingAuthError: Key invalid/expired.
        KlingRateLimitError: Quota exceeded.
        KlingGenerationError: Task failed or timed out.
    """
    _check_config()
    prompt = _build_prompt(theme)
    logger.debug(f"[kling] Prompt: {prompt[:120]}...")

    # Encode reference image
    try:
        image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    except OSError as exc:
        raise KlingGenerationError(
            f"Cannot read reference image for Kling: {image_path}\n"
            f"  OS error: {exc}"
        ) from exc

    payload = {
        "model": "kling-v1",
        "image": image_b64,
        "prompt": prompt,
        "duration": "5",
        "aspect_ratio": "9:16",
        "cfg_scale": 0.5,
    }
    endpoint_path = "videos/image2video"

    async with httpx.AsyncClient(timeout=30) as client:
        for attempt in range(1, 4):
            try:
                r = await client.post(
                    f"{_BASE}/{endpoint_path}",
                    json=payload,
                    headers=_headers(),
                )
                r.raise_for_status()
                task_id = r.json()["data"]["task_id"]
                logger.debug(f"[kling] Task created: {task_id}")
                break
            except httpx.HTTPStatusError as exc:
                typed = _classify(exc, "create_task")
                if isinstance(typed, (KlingAuthError, KlingRateLimitError)):
                    raise typed from exc
                if attempt == 3:
                    raise typed from exc
                wait = 2 ** attempt * 10
                logger.warning(f"[kling] Attempt {attempt} failed: {typed}. Retrying in {wait}s")
                await asyncio.sleep(wait)
            except httpx.ConnectError as exc:
                if attempt == 3:
                    raise KlingError(
                        "Cannot connect to Kling AI API.\n"
                        "  Check internet or https://platform.klingai.com"
                    ) from exc
                await asyncio.sleep(2 ** attempt * 5)

        video_url = await _poll_task(client, task_id, "videos/image2video")

        # Download video
        out_dir = Path(OUTPUT_DIR) / "videos"
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = out_dir / f"sofia_{timestamp}.mp4"

        try:
            r = await client.get(video_url, timeout=120)
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise KlingGenerationError(
                f"Failed to download video from Kling CDN: HTTP {exc.response.status_code}\n"
                f"  URL: {video_url}"
            ) from exc
        except httpx.ConnectError as exc:
            raise KlingGenerationError(
                f"Network error downloading video from Kling CDN.\n"
                f"  URL: {video_url}\n"
                f"  Error: {exc}"
            ) from exc

        if len(r.content) < 100_000:
            raise KlingGenerationError(
                f"Downloaded video is suspiciously small ({len(r.content)} bytes).\n"
                "  Likely a CDN error or incomplete download."
            )

        filepath.write_bytes(r.content)
        size_mb = len(r.content) / 1024 / 1024
        logger.info(f"[kling] Video saved: {filepath} ({size_mb:.1f} MB)")
        return str(filepath)
