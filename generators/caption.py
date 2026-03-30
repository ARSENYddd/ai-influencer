"""Caption and hashtag generation via OpenAI GPT-4o."""

import asyncio

import httpx
from loguru import logger

from config import OPENAI_API_KEY

_OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def _headers() -> dict:
    if not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY is not set in .env")
    return {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}


def _build_prompt(theme: str) -> str:
    return (
        f"Write an Instagram/TikTok caption for a lifestyle post about {theme}. "
        "Style: casual, engaging, first-person. Include 5 relevant hashtags. "
        "Max 150 characters for caption + hashtags on new line. "
        "Language: Russian"
    )


async def generate_caption(theme: str) -> str:
    """
    Generate caption + hashtags using GPT-4o.

    Returns:
        Full caption string with hashtags.
    """
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a social media copywriter for lifestyle influencers."},
            {"role": "user", "content": _build_prompt(theme)},
        ],
        "max_tokens": 200,
        "temperature": 0.8,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        for attempt in range(1, 4):
            try:
                r = await client.post(_OPENAI_URL, json=payload, headers=_headers())
                if r.status_code == 429:
                    wait = 2 ** attempt * 5
                    logger.warning(f"OpenAI rate limit, retrying in {wait}s (attempt {attempt})")
                    await asyncio.sleep(wait)
                    continue
                r.raise_for_status()
                caption = r.json()["choices"][0]["message"]["content"].strip()
                logger.debug(f"Caption generated ({len(caption)} chars)")
                return caption
            except (httpx.HTTPStatusError, httpx.ConnectError) as exc:
                if attempt == 3:
                    raise
                wait = 2 ** attempt * 5
                logger.warning(f"OpenAI error (attempt {attempt}): {exc}, retrying in {wait}s")
                await asyncio.sleep(wait)

    raise RuntimeError("Failed to generate caption after 3 attempts")
