"""TikTok video posting via Content Posting API v2."""

import asyncio

import httpx
from loguru import logger

from config import TIKTOK_ACCESS_TOKEN

_BASE = "https://open.tiktokapis.com/v2"


def _headers() -> dict:
    if not TIKTOK_ACCESS_TOKEN:
        raise EnvironmentError("TIKTOK_ACCESS_TOKEN is not set in .env")
    return {
        "Authorization": f"Bearer {TIKTOK_ACCESS_TOKEN}",
        "Content-Type": "application/json; charset=UTF-8",
    }


async def _poll_publish_status(client: httpx.AsyncClient, publish_id: str, max_wait: int = 180) -> str:
    """Poll until TikTok post is published, return share URL."""
    for _ in range(max_wait // 10):
        await asyncio.sleep(10)
        r = await client.post(
            f"{_BASE}/post/publish/status/fetch/",
            json={"publish_id": publish_id},
            headers=_headers(),
        )
        r.raise_for_status()
        data = r.json().get("data", {})
        status = data.get("status")
        if status == "PUBLISH_COMPLETE":
            return data.get("share_url", "")
        if status in ("FAILED", "PUBLISH_FAILED"):
            err = data.get("fail_reason", "unknown")
            raise RuntimeError(f"TikTok publish failed: {err}")
    raise TimeoutError(f"TikTok publish {publish_id} timed out after {max_wait}s")


async def post_to_tiktok(video_path: str, caption: str) -> str:
    """
    Post a video to TikTok using PULL_FROM_URL upload method.

    TikTok requires a publicly accessible video URL.
    In production, upload to S3/CDN first.

    Returns:
        TikTok publish ID.
    """
    if not TIKTOK_ACCESS_TOKEN:
        raise EnvironmentError("TIKTOK_ACCESS_TOKEN is not set in .env")

    # Local files need to be hosted publicly. Raise a clear error.
    raise NotImplementedError(
        "TikTok API requires a public video URL (PULL_FROM_URL). "
        "Upload the video to S3 or a CDN, then call post_tiktok_from_url(). "
        "See README for setup instructions."
    )


async def post_tiktok_from_url(video_url: str, title: str) -> str:
    """
    Post to TikTok from a publicly accessible URL.

    Args:
        video_url: Public HTTPS URL to an MP4 file (3–60 sec).
        title: Post title / caption with hashtags (max 150 chars).

    Returns:
        TikTok publish ID.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        for attempt in range(1, 4):
            try:
                r = await client.post(
                    f"{_BASE}/post/publish/video/init/",
                    json={
                        "post_info": {
                            "title": title[:150],
                            "privacy_level": "PUBLIC_TO_EVERYONE",
                            "disable_duet": False,
                            "disable_comment": False,
                            "disable_stitch": False,
                        },
                        "source_info": {
                            "source": "PULL_FROM_URL",
                            "video_url": video_url,
                        },
                    },
                    headers=_headers(),
                )
                if r.status_code == 429:
                    wait = 2 ** attempt * 20
                    logger.warning(f"TikTok rate limit, waiting {wait}s")
                    await asyncio.sleep(wait)
                    continue
                r.raise_for_status()
                publish_id = r.json()["data"]["publish_id"]
                logger.debug(f"TikTok publish_id: {publish_id}")
                break
            except (httpx.HTTPStatusError, httpx.ConnectError) as exc:
                if attempt == 3:
                    raise
                await asyncio.sleep(2 ** attempt * 10)

        share_url = await _poll_publish_status(client, publish_id)
        logger.success(f"TikTok published: {publish_id} | {share_url}")
        return publish_id
