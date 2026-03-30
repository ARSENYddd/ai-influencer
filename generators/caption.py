"""Caption and hashtag generation via OpenAI GPT-4o."""

from __future__ import annotations

import asyncio

import httpx
from loguru import logger

from config import OPENAI_API_KEY

_OPENAI_URL = "https://api.openai.com/v1/chat/completions"


# ── Custom exceptions ──────────────────────────────────────────────────────────

class OpenAIError(Exception):
    """Base OpenAI error."""

class OpenAIAuthError(OpenAIError):
    """API key invalid, expired, or no billing."""

class OpenAIRateLimitError(OpenAIError):
    """Too many requests or monthly quota exceeded."""

class OpenAISafetyError(OpenAIError):
    """Content policy violation."""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _check_config() -> None:
    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set.\n"
            "  Get your key at https://platform.openai.com/api-keys\n"
            "  Make sure your account has a positive balance."
        )


def _headers() -> dict:
    return {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}


def _classify(exc: httpx.HTTPStatusError, context: str) -> OpenAIError:
    status = exc.response.status_code
    try:
        body = exc.response.json()
        err = body.get("error", {})
        code = err.get("code", "")
        message = err.get("message", str(body))
    except Exception:
        code, message = "", exc.response.text[:200]

    if status == 401 or code in ("invalid_api_key", "invalid_organization"):
        return OpenAIAuthError(
            f"[{context}] OpenAI API key invalid (code: {code}).\n"
            "  Check OPENAI_API_KEY in .env — make sure there are no extra spaces.\n"
            "  Regenerate at https://platform.openai.com/api-keys\n"
            f"  Details: {message}"
        )
    if status == 429:
        if "insufficient_quota" in code or "quota" in message.lower():
            return OpenAIRateLimitError(
                f"[{context}] OpenAI quota exceeded.\n"
                "  Add credits at https://platform.openai.com/account/billing\n"
                f"  Details: {message}"
            )
        return OpenAIRateLimitError(
            f"[{context}] OpenAI rate limit hit.\n"
            "  Wait a moment and retry.\n"
            f"  Details: {message}"
        )
    if status == 400 and code == "content_policy_violation":
        return OpenAISafetyError(
            f"[{context}] OpenAI content policy violation.\n"
            "  Modify the prompt or theme.\n"
            f"  Details: {message}"
        )
    return OpenAIError(f"[{context}] HTTP {status} (code: {code}): {message}")


def _build_prompt(theme: str) -> str:
    return (
        f"Write an Instagram/TikTok caption for a lifestyle post about {theme}. "
        "Style: casual, engaging, first-person. Include 5 relevant hashtags. "
        "Max 150 characters for caption + hashtags on new line. "
        "Language: Russian"
    )


# ── Main ───────────────────────────────────────────────────────────────────────

async def generate_caption(theme: str) -> str:
    """
    Generate caption + hashtags using GPT-4o.

    Args:
        theme: Content theme string.

    Returns:
        Full caption string with hashtags (Russian).

    Raises:
        EnvironmentError: API key not set.
        OpenAIAuthError: Key invalid/expired or no billing.
        OpenAIRateLimitError: Rate limit or quota exceeded.
        OpenAISafetyError: Content policy violation.
        OpenAIError: Other API errors.
    """
    _check_config()

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": "You are a social media copywriter for lifestyle influencers. Write concise, engaging captions in Russian.",
            },
            {"role": "user", "content": _build_prompt(theme)},
        ],
        "max_tokens": 200,
        "temperature": 0.8,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        for attempt in range(1, 4):
            try:
                r = await client.post(_OPENAI_URL, json=payload, headers=_headers())
                r.raise_for_status()
                caption = r.json()["choices"][0]["message"]["content"].strip()
                logger.debug(f"[openai] Caption generated ({len(caption)} chars)")
                return caption
            except httpx.HTTPStatusError as exc:
                typed = _classify(exc, "generate_caption")
                # Auth and safety errors — don't retry
                if isinstance(typed, (OpenAIAuthError, OpenAISafetyError)):
                    raise typed from exc
                if isinstance(typed, OpenAIRateLimitError):
                    if attempt == 3:
                        raise typed from exc
                    wait = 2 ** attempt * 10
                    logger.warning(f"[openai] Rate limit on attempt {attempt}, waiting {wait}s")
                    await asyncio.sleep(wait)
                    continue
                if attempt == 3:
                    raise typed from exc
                await asyncio.sleep(2 ** attempt * 5)
            except httpx.ConnectError as exc:
                if attempt == 3:
                    raise OpenAIError(
                        "Cannot connect to OpenAI API.\n"
                        "  Check internet or https://status.openai.com"
                    ) from exc
                await asyncio.sleep(2 ** attempt * 5)
            except (KeyError, IndexError) as exc:
                raise OpenAIError(
                    f"Unexpected OpenAI response format: {exc}\n"
                    "  OpenAI may have changed their API schema."
                ) from exc

    raise OpenAIError("Failed to generate caption after 3 attempts.")
