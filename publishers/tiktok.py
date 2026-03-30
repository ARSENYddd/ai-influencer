"""TikTok video posting via Content Posting API v2."""

from __future__ import annotations

import asyncio

import httpx
from loguru import logger

from config import TIKTOK_ACCESS_TOKEN

_BASE = "https://open.tiktokapis.com/v2"


# ── Custom exceptions ──────────────────────────────────────────────────────────

class TikTokError(Exception):
    """Base TikTok error."""

class TikTokAuthError(TikTokError):
    """Token invalid, expired, or app not approved."""

class TikTokRateLimitError(TikTokError):
    """Too many requests."""

class TikTokMediaError(TikTokError):
    """Video rejected or publish failed."""

class TikTokAccountError(TikTokError):
    """Account or app configuration issue."""


# ── Config check ───────────────────────────────────────────────────────────────

def _check_config() -> None:
    if not TIKTOK_ACCESS_TOKEN:
        raise EnvironmentError(
            "TIKTOK_ACCESS_TOKEN is not set.\n"
            "  Register your app at https://developers.tiktok.com/\n"
            "  Enable 'Content Posting API' and complete app review (2–5 days).\n"
            "  Then obtain an access token via OAuth 2.0."
        )


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {TIKTOK_ACCESS_TOKEN}",
        "Content-Type": "application/json; charset=UTF-8",
    }


# ── Error classifier ───────────────────────────────────────────────────────────

def _classify(response_json: dict, context: str) -> TikTokError:
    """Map TikTok error codes to typed exceptions."""
    error = response_json.get("error", {})
    code = error.get("code", "unknown")
    message = error.get("message", str(response_json))
    log_id = error.get("log_id", "")

    if code in ("access_token_invalid", "access_token_expired"):
        return TikTokAuthError(
            f"[{context}] TikTok token invalid or expired (code: {code}).\n"
            "  Refresh your access token via OAuth 2.0.\n"
            f"  TikTok log_id: {log_id}"
        )
    if code == "app_not_approved":
        return TikTokAccountError(
            f"[{context}] TikTok app not approved for Content Posting API.\n"
            "  Submit your app for review at https://developers.tiktok.com/\n"
            "  Review takes 2–5 business days.\n"
            f"  TikTok log_id: {log_id}"
        )
    if code in ("scope_not_authorized", "permission_denied"):
        return TikTokAccountError(
            f"[{context}] Missing TikTok scope (code: {code}).\n"
            "  Required scope: video.publish\n"
            "  Re-authorize your app with this scope.\n"
            f"  TikTok log_id: {log_id}"
        )
    if code in ("rate_limit_exceeded", "spam_risk_too_many_pending_share"):
        return TikTokRateLimitError(
            f"[{context}] TikTok rate limit (code: {code}).\n"
            "  Wait before retrying. TikTok limits: ~10 posts/day per account.\n"
            f"  TikTok log_id: {log_id}"
        )
    if "video" in code or "media" in code or code in ("invalid_param", "video_pull_failed"):
        return TikTokMediaError(
            f"[{context}] TikTok rejected the video (code: {code}).\n"
            "  Requirements: MP4/WebM, 3–60 sec, max 4 GB, public HTTPS URL.\n"
            f"  Details: {message}\n"
            f"  TikTok log_id: {log_id}"
        )

    return TikTokError(f"[{context}] TikTok error [{code}]: {message} (log_id: {log_id})")


# ── Publish status polling ─────────────────────────────────────────────────────

async def _poll_publish_status(
    client: httpx.AsyncClient,
    publish_id: str,
    max_wait: int = 180,
) -> str:
    """Poll until TikTok post is published. Returns share_url (may be empty)."""
    elapsed = 0
    while elapsed < max_wait:
        await asyncio.sleep(10)
        elapsed += 10
        try:
            r = await client.post(
                f"{_BASE}/post/publish/status/fetch/",
                json={"publish_id": publish_id},
                headers=_headers(),
            )
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            try:
                body = exc.response.json()
                raise _classify(body, "status_fetch") from exc
            except (ValueError, KeyError):
                raise TikTokError(
                    f"[status_fetch] HTTP {exc.response.status_code}: {exc.response.text[:200]}"
                ) from exc

        data = r.json().get("data", {})
        status = data.get("status")
        logger.debug(f"[tiktok] Publish {publish_id} status: {status} ({elapsed}s)")

        if status == "PUBLISH_COMPLETE":
            return data.get("share_url", "")
        if status in ("FAILED", "PUBLISH_FAILED"):
            reason = data.get("fail_reason", "unknown")
            raise TikTokMediaError(
                f"TikTok publish failed (publish_id: {publish_id}).\n"
                f"  Reason: {reason}\n"
                "  Common causes: video too short (<3s), unsupported codec,\n"
                "  URL not publicly accessible, or duplicate content."
            )

    raise TikTokMediaError(
        f"TikTok publish {publish_id} timed out after {max_wait}s.\n"
        "  TikTok may be slow — check status manually in TikTok Studio."
    )


# ── Publish ────────────────────────────────────────────────────────────────────

async def post_tiktok_from_url(video_url: str, title: str) -> str:
    """
    Post a video to TikTok from a publicly accessible URL.

    Args:
        video_url: Public HTTPS URL to MP4 (3–60 sec). Must be S3/CDN.
        title: Post title / caption with hashtags (max 150 chars).

    Returns:
        TikTok publish ID.

    Raises:
        EnvironmentError: Token not configured.
        TikTokAuthError: Token invalid/expired.
        TikTokAccountError: App not approved or missing scope.
        TikTokRateLimitError: Too many requests.
        TikTokMediaError: Video rejected or publish failed.
        TikTokError: Other API errors.
    """
    _check_config()

    if len(title) > 150:
        logger.warning(f"[tiktok] Title truncated from {len(title)} to 150 chars")
        title = title[:147] + "..."

    payload = {
        "post_info": {
            "title": title,
            "privacy_level": "PUBLIC_TO_EVERYONE",
            "disable_duet": False,
            "disable_comment": False,
            "disable_stitch": False,
        },
        "source_info": {
            "source": "PULL_FROM_URL",
            "video_url": video_url,
        },
    }

    async with httpx.AsyncClient(timeout=30) as client:
        for attempt in range(1, 4):
            try:
                r = await client.post(
                    f"{_BASE}/post/publish/video/init/",
                    json=payload,
                    headers=_headers(),
                )
                r.raise_for_status()
                body = r.json()

                # TikTok sometimes returns 200 with an error body
                if body.get("error", {}).get("code", "ok") != "ok":
                    raise _classify(body, "video_init")

                publish_id = body["data"]["publish_id"]
                logger.debug(f"[tiktok] publish_id: {publish_id}")
                break

            except httpx.HTTPStatusError as exc:
                try:
                    body = exc.response.json()
                    typed = _classify(body, "video_init")
                except (ValueError, KeyError):
                    typed = TikTokError(
                        f"[video_init] HTTP {exc.response.status_code}: {exc.response.text[:200]}"
                    )
                if isinstance(typed, TikTokRateLimitError):
                    wait = 2 ** attempt * 20
                    logger.warning(f"[tiktok] Rate limit on attempt {attempt}, waiting {wait}s")
                    await asyncio.sleep(wait)
                    continue
                raise typed from exc
            except httpx.ConnectError as exc:
                if attempt == 3:
                    raise TikTokError(
                        "Cannot connect to TikTok API.\n"
                        "  Check your internet connection."
                    ) from exc
                await asyncio.sleep(2 ** attempt * 5)

        share_url = await _poll_publish_status(client, publish_id)
        if share_url:
            logger.success(f"[tiktok] Published: {share_url}")
        else:
            logger.success(f"[tiktok] Published (publish_id: {publish_id})")
        return publish_id


# Keep old name as alias
post_to_tiktok = post_tiktok_from_url
