"""Video generation via Kling AI API (Runway as fallback)."""

import asyncio
from datetime import datetime
from pathlib import Path

import httpx
from loguru import logger

from config import KLING_API_KEY, CHARACTER_STYLE, OUTPUT_DIR

_KLING_BASE = "https://api.klingai.com/v1"


def _kling_headers() -> dict:
    if not KLING_API_KEY:
        raise EnvironmentError("KLING_API_KEY is not set in .env")
    return {"Authorization": f"Bearer {KLING_API_KEY}", "Content-Type": "application/json"}


async def _poll_kling_task(client: httpx.AsyncClient, task_id: str, max_wait: int = 300) -> str:
    """Poll Kling until video is ready, return video URL."""
    for _ in range(max_wait // 10):
        await asyncio.sleep(10)
        r = await client.get(f"{_KLING_BASE}/videos/text2video/{task_id}", headers=_kling_headers())
        r.raise_for_status()
        data = r.json().get("data", {})
        status = data.get("task_status")
        if status == "succeed":
            works = data.get("task_result", {}).get("videos", [])
            if works:
                return works[0]["url"]
            raise RuntimeError("Kling returned succeed but no video URL")
        if status == "failed":
            raise RuntimeError(f"Kling task failed: {data.get('task_status_msg')}")
    raise TimeoutError(f"Kling task {task_id} timed out after {max_wait}s")


async def generate_video(image_path: str, theme: str) -> str:
    """
    Generate a short video clip from the reference image using Kling AI.

    Uses image-to-video if image_path is provided, falls back to text-to-video.

    Returns:
        Local path to the saved video file.
    """
    prompt = (
        f"{CHARACTER_STYLE}, {theme}, "
        "smooth cinematic motion, lifestyle video, Instagram Reels style"
    )
    logger.debug(f"Kling prompt: {prompt[:120]}")

    # Try image-to-video with the generated image
    img_bytes = Path(image_path).read_bytes() if image_path else None

    async with httpx.AsyncClient(timeout=30) as client:
        for attempt in range(1, 4):
            try:
                if img_bytes:
                    # Image-to-video endpoint
                    payload = {
                        "model": "kling-v1",
                        "image": _encode_image(image_path),
                        "prompt": prompt,
                        "duration": "5",
                        "aspect_ratio": "9:16",
                        "cfg_scale": 0.5,
                    }
                    r = await client.post(
                        f"{_KLING_BASE}/videos/image2video",
                        json=payload,
                        headers=_kling_headers(),
                    )
                else:
                    payload = {
                        "model": "kling-v1",
                        "prompt": prompt,
                        "duration": "5",
                        "aspect_ratio": "9:16",
                    }
                    r = await client.post(
                        f"{_KLING_BASE}/videos/text2video",
                        json=payload,
                        headers=_kling_headers(),
                    )

                if r.status_code == 429:
                    wait = 2 ** attempt * 15
                    logger.warning(f"Kling rate limit, retrying in {wait}s (attempt {attempt})")
                    await asyncio.sleep(wait)
                    continue
                r.raise_for_status()
                task_id = r.json()["data"]["task_id"]
                logger.debug(f"Kling task ID: {task_id}")
                video_url = await _poll_kling_task(client, task_id)
                break
            except (httpx.HTTPStatusError, httpx.ConnectError) as exc:
                if attempt == 3:
                    raise
                wait = 2 ** attempt * 10
                logger.warning(f"Kling error (attempt {attempt}): {exc}, retrying in {wait}s")
                await asyncio.sleep(wait)

        # Download video
        out_dir = Path(OUTPUT_DIR) / "videos"
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = out_dir / f"sofia_{timestamp}.mp4"

        r = await client.get(video_url, timeout=120)
        r.raise_for_status()
        filepath.write_bytes(r.content)
        logger.info(f"Video downloaded: {filepath} ({len(r.content) / 1024 / 1024:.1f} MB)")
        return str(filepath)


def _encode_image(image_path: str) -> str:
    """Base64-encode image for Kling image2video API."""
    import base64
    data = Path(image_path).read_bytes()
    return base64.b64encode(data).decode()
