"""Instagram Reels posting via Meta Graph API v21."""

import asyncio

import httpx
from loguru import logger

from config import INSTAGRAM_ACCESS_TOKEN, INSTAGRAM_USER_ID

_BASE = "https://graph.facebook.com/v21.0"


def _check_config() -> None:
    if not INSTAGRAM_ACCESS_TOKEN:
        raise EnvironmentError("INSTAGRAM_ACCESS_TOKEN is not set in .env")
    if not INSTAGRAM_USER_ID:
        raise EnvironmentError("INSTAGRAM_USER_ID is not set in .env")


async def _wait_for_container(client: httpx.AsyncClient, container_id: str, max_wait: int = 120) -> None:
    """Poll until container is FINISHED."""
    params = {
        "fields": "status_code",
        "access_token": INSTAGRAM_ACCESS_TOKEN,
    }
    for _ in range(max_wait // 5):
        await asyncio.sleep(5)
        r = await client.get(f"{_BASE}/{container_id}", params=params)
        r.raise_for_status()
        status = r.json().get("status_code")
        if status == "FINISHED":
            return
        if status == "ERROR":
            raise RuntimeError(f"Instagram container {container_id} failed during processing")
    raise TimeoutError(f"Instagram container {container_id} timed out")


async def post_to_instagram(video_path: str, caption: str) -> str:
    """
    Upload a Reel to Instagram via Meta Graph API.

    Flow: upload video URL → get container_id → publish.

    Returns:
        Instagram media ID.
    """
    _check_config()

    # Instagram requires a publicly accessible video URL.
    # In production, upload to S3/CDN first and pass the URL here.
    # For local testing, run a temporary HTTP server (see README).
    raise NotImplementedError(
        "Instagram Graph API requires a public video URL. "
        "Upload the video to S3 or a CDN, then pass the URL. "
        "See README for setup instructions."
    )


async def post_reel_from_url(video_url: str, caption: str) -> str:
    """
    Upload a Reel from a publicly accessible URL.

    Args:
        video_url: Public HTTPS URL to an MP4 file (H.264, 3–90 sec).
        caption: Caption text with hashtags.

    Returns:
        Instagram media ID.
    """
    _check_config()

    async with httpx.AsyncClient(timeout=60) as client:
        # Step 1: Create media container
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
                if r.status_code == 429:
                    wait = 2 ** attempt * 30
                    logger.warning(f"Instagram rate limit, waiting {wait}s")
                    await asyncio.sleep(wait)
                    continue
                r.raise_for_status()
                container_id = r.json()["id"]
                logger.debug(f"Instagram container: {container_id}")
                break
            except (httpx.HTTPStatusError, httpx.ConnectError) as exc:
                if attempt == 3:
                    raise
                await asyncio.sleep(2 ** attempt * 10)

        # Step 2: Wait for processing
        await _wait_for_container(client, container_id)

        # Step 3: Publish
        r = await client.post(
            f"{_BASE}/{INSTAGRAM_USER_ID}/media_publish",
            data={
                "creation_id": container_id,
                "access_token": INSTAGRAM_ACCESS_TOKEN,
            },
        )
        r.raise_for_status()
        media_id = r.json()["id"]
        logger.success(f"Instagram Reel published: {media_id}")
        return media_id
