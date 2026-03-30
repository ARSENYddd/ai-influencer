"""Image generation via Leonardo AI API."""

import asyncio
from datetime import datetime
from pathlib import Path

import httpx
from loguru import logger

from config import LEONARDO_API_KEY, CHARACTER_STYLE, LEONARDO_MODEL_ID, OUTPUT_DIR

_BASE = "https://cloud.leonardo.ai/api/rest/v1"


def _headers() -> dict:
    if not LEONARDO_API_KEY:
        raise EnvironmentError("LEONARDO_API_KEY is not set in .env")
    return {"Authorization": f"Bearer {LEONARDO_API_KEY}", "Content-Type": "application/json"}


def _build_prompt(theme: str) -> str:
    return (
        f"{CHARACTER_STYLE}, {theme}, "
        "ultra realistic, 8k, Instagram lifestyle photo, cinematic lighting"
    )


async def _poll_generation(client: httpx.AsyncClient, generation_id: str, max_wait: int = 120) -> list[str]:
    """Poll Leonardo until images are ready, return list of URLs."""
    for _ in range(max_wait // 5):
        await asyncio.sleep(5)
        r = await client.get(f"{_BASE}/generations/{generation_id}", headers=_headers())
        r.raise_for_status()
        data = r.json().get("generations_by_pk", {})
        status = data.get("status")
        if status == "COMPLETE":
            return [img["url"] for img in data.get("generated_images", [])]
        if status == "FAILED":
            raise RuntimeError(f"Leonardo generation failed: {data}")
    raise TimeoutError(f"Leonardo generation {generation_id} timed out after {max_wait}s")


async def generate_image(theme: str) -> str:
    """
    Generate a character image via Leonardo Diffusion XL.

    Returns:
        Local path to the saved image.
    """
    prompt = _build_prompt(theme)
    logger.debug(f"Leonardo prompt: {prompt[:120]}")

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
        # Retry logic: 3 attempts with exponential backoff
        for attempt in range(1, 4):
            try:
                r = await client.post(f"{_BASE}/generations", json=payload, headers=_headers())
                if r.status_code == 429:
                    wait = 2 ** attempt * 10
                    logger.warning(f"Leonardo rate limit, retrying in {wait}s (attempt {attempt})")
                    await asyncio.sleep(wait)
                    continue
                r.raise_for_status()
                generation_id = r.json()["sdGenerationJob"]["generationId"]
                logger.debug(f"Generation ID: {generation_id}")
                urls = await _poll_generation(client, generation_id)
                break
            except (httpx.HTTPStatusError, httpx.ConnectError) as exc:
                if attempt == 3:
                    raise
                wait = 2 ** attempt * 5
                logger.warning(f"Leonardo error (attempt {attempt}): {exc}, retrying in {wait}s")
                await asyncio.sleep(wait)

        if not urls:
            raise RuntimeError("Leonardo returned no images")

        # Download first image
        out_dir = Path(OUTPUT_DIR) / "images"
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = out_dir / f"sofia_{timestamp}.jpg"

        r = await client.get(urls[0], timeout=60)
        r.raise_for_status()
        filepath.write_bytes(r.content)
        logger.info(f"Image downloaded: {filepath} ({len(r.content) / 1024:.1f} KB)")
        return str(filepath)
