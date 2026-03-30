"""Instagram Reels posting via Meta Graph API v21."""

from __future__ import annotations

import asyncio

import httpx
from loguru import logger

from config import INSTAGRAM_ACCESS_TOKEN, INSTAGRAM_USER_ID

_BASE = "https://graph.facebook.com/v21.0"


# ── Custom exceptions ──────────────────────────────────────────────────────────

class InstagramError(Exception):
    """Base Instagram error."""

class InstagramAuthError(InstagramError):
    """Token invalid, expired, or missing permissions."""

class InstagramRateLimitError(InstagramError):
    """Too many requests."""

class InstagramMediaError(InstagramError):
    """Video/container processing failed."""

class InstagramAccountError(InstagramError):
    """Account not Business/Creator or permissions missing."""


# ── Config check ───────────────────────────────────────────────────────────────

def _check_config() -> None:
    if not INSTAGRAM_ACCESS_TOKEN:
        raise EnvironmentError(
            "INSTAGRAM_ACCESS_TOKEN is not set.\n"
            "  Get it at https://developers.facebook.com/tools/explorer/\n"
            "  Required permissions: instagram_basic, instagram_content_publish"
        )
    if not INSTAGRAM_USER_ID:
        raise EnvironmentError(
            "INSTAGRAM_USER_ID is not set.\n"
            "  Find it via Graph API Explorer: GET /me?fields=id,name"
        )


# ── Error classifier ───────────────────────────────────────────────────────────

def _classify(exc: httpx.HTTPStatusError, context: str) -> InstagramError:
    """Map HTTP errors to typed Instagram exceptions."""
    try:
        body = exc.response.json()
        code = body.get("error", {}).get("code", 0)
        subcode = body.get("error", {}).get("error_subcode", 0)
        message = body.get("error", {}).get("message", str(exc))
    except Exception:
        code, subcode, message = 0, 0, str(exc)

    status = exc.response.status_code

    if status == 401 or code in (190, 102):
        return InstagramAuthError(
            f"[{context}] Token invalid or expired (code {code}).\n"
            "  Refresh your access token at https://developers.facebook.com/tools/explorer/\n"
            f"  Details: {message}"
        )
    if code == 10 or subcode == 458:
        return InstagramAccountError(
            f"[{context}] App not approved or permission missing.\n"
            "  Make sure 'instagram_content_publish' permission is granted in your Meta App.\n"
            f"  Details: {message}"
        )
    if code == 32 or status == 429:
        return InstagramRateLimitError(
            f"[{context}] Rate limit reached (code {code}).\n"
            "  Instagram allows ~25 API calls per hour. Wait before retrying.\n"
            f"  Details: {message}"
        )
    if code == 36 or "video" in message.lower():
        return InstagramMediaError(
            f"[{context}] Video rejected by Instagram.\n"
            "  Requirements: MP4, H.264, 3–90 sec, max 1 GB, 9:16 ratio recommended.\n"
            f"  Details: {message}"
        )
    if code == 100 and "does not exist" in message:
        return InstagramAccountError(
            f"[{context}] Instagram user ID not found: {INSTAGRAM_USER_ID}.\n"
            "  Check INSTAGRAM_USER_ID in .env — it must be the numeric IG Business/Creator ID."
        )

    return InstagramError(
        f"[{context}] HTTP {status} (code {code}): {message}"
    )


# ── Container polling ──────────────────────────────────────────────────────────

async def _wait_for_container(
    client: httpx.AsyncClient,
    container_id: str,
    max_wait: int = 120,
) -> None:
    """Poll until container status is FINISHED. Raises on ERROR or timeout."""
    params = {"fields": "status_code,status", "access_token": INSTAGRAM_ACCESS_TOKEN}
    elapsed = 0
    while elapsed < max_wait:
        await asyncio.sleep(5)
        elapsed += 5
        try:
            r = await client.get(f"{_BASE}/{container_id}", params=params)
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise _classify(exc, "container_poll") from exc

        data = r.json()
        status = data.get("status_code") or data.get("status")
        logger.debug(f"[instagram] Container {container_id} status: {status} ({elapsed}s)")

        if status == "FINISHED":
            return
        if status == "ERROR":
            raise InstagramMediaError(
                f"Instagram container {container_id} failed during processing.\n"
                "  Common causes: video codec not H.264, duration <3 or >90 sec,\n"
                "  unsupported resolution, or corrupt file.\n"
                f"  Full response: {data}"
            )

    raise InstagramMediaError(
        f"Container {container_id} did not finish within {max_wait}s.\n"
        "  Instagram may be slow — try again later."
    )


# ── Publish ────────────────────────────────────────────────────────────────────

async def post_reel_from_url(video_url: str, caption: str) -> str:
    """
    Upload a Reel from a publicly accessible URL.

    Args:
        video_url: Public HTTPS URL to MP4 (H.264, 3–90 sec). Must be S3/CDN.
        caption: Caption with hashtags (max 2200 chars).

    Returns:
        Instagram media ID string.

    Raises:
        EnvironmentError: Token or user ID not configured.
        InstagramAuthError: Token invalid/expired.
        InstagramAccountError: Wrong account type or missing permissions.
        InstagramRateLimitError: Too many requests.
        InstagramMediaError: Video rejected or container failed.
        InstagramError: Other API errors.
    """
    _check_config()

    if len(caption) > 2200:
        logger.warning(f"[instagram] Caption truncated from {len(caption)} to 2200 chars")
        caption = caption[:2197] + "..."

    async with httpx.AsyncClient(timeout=60) as client:
        # ── Step 1: Create media container ────────────────────────────────────
        for attempt in range(1, 4):
            try:
                r = await client.post(
                    f"{_BASE}/{INSTAGRAM_USER_ID}/media",
                    data={
                        "media_type": "REELS",
                        "video_url": video_url,
                        "caption": caption,
                        "access_token": INSTAGRAM_ACCESS_TOKEN,
                    },
                )
                r.raise_for_status()
                container_id = r.json()["id"]
                logger.debug(f"[instagram] Container created: {container_id}")
                break
            except httpx.HTTPStatusError as exc:
                typed = _classify(exc, "create_container")
                if isinstance(typed, InstagramRateLimitError):
                    wait = 2 ** attempt * 30
                    logger.warning(f"[instagram] Rate limit on attempt {attempt}, waiting {wait}s")
                    await asyncio.sleep(wait)
                    continue
                raise typed from exc
            except httpx.ConnectError as exc:
                if attempt == 3:
                    raise InstagramError(
                        "Cannot connect to Facebook Graph API.\n"
                        "  Check your internet connection."
                    ) from exc
                await asyncio.sleep(2 ** attempt * 5)

        # ── Step 2: Wait for container processing ─────────────────────────────
        logger.debug(f"[instagram] Waiting for container {container_id}...")
        await _wait_for_container(client, container_id)

        # ── Step 3: Publish ────────────────────────────────────────────────────
        try:
            r = await client.post(
                f"{_BASE}/{INSTAGRAM_USER_ID}/media_publish",
                data={
                    "creation_id": container_id,
                    "access_token": INSTAGRAM_ACCESS_TOKEN,
                },
            )
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise _classify(exc, "media_publish") from exc

        media_id = r.json()["id"]
        logger.success(f"[instagram] Reel published: {media_id}")
        return media_id


# Keep old name as alias so main.py import doesn't break
post_to_instagram = post_reel_from_url
