"""
Scrapes trending Instagram Reels using instagrapi.
Returns list of TrendItem dicts with: url, audio_name, likes, views, hashtags, video_path
"""
import os
import time
import logging
from pathlib import Path
from instagrapi import Client
from instagrapi.exceptions import LoginRequired, RateLimitError

logger = logging.getLogger(__name__)

TRENDING_HASHTAGS = [
    "dancechallenge", "trending", "reelsviral", "newdance",
    "fyp", "choreography", "dancetrend"
]

class ReelsScraper:
    def __init__(self, username: str, password: str, session_file: str = "scraper_session.json"):
        self.client = Client()
        self.session_file = session_file
        self._login(username, password)

    def _login(self, username: str, password: str):
        if Path(self.session_file).exists():
            try:
                self.client.load_settings(self.session_file)
                self.client.login(username, password)
                logger.info("Logged in via cached session")
                return
            except Exception:
                logger.warning("Session expired, logging in fresh")

        self.client.login(username, password)
        self.client.dump_settings(self.session_file)
        logger.info("Fresh login successful")

    def get_trending_reels(self, count_per_tag: int = 10, min_likes: int = 10000) -> list[dict]:
        results = []
        seen_ids = set()

        for hashtag in TRENDING_HASHTAGS:
            try:
                medias = self.client.hashtag_medias_recent(hashtag, amount=count_per_tag)
                for media in medias:
                    if media.pk in seen_ids:
                        continue
                    if media.media_type != 2:
                        continue
                    if (media.like_count or 0) < min_likes:
                        continue

                    seen_ids.add(media.pk)
                    results.append({
                        "id": str(media.pk),
                        "url": f"https://www.instagram.com/reel/{media.code}/",
                        "audio_name": media.clips_metadata.get("audio_type", "unknown") if media.clips_metadata else "unknown",
                        "likes": media.like_count or 0,
                        "views": media.view_count or 0,
                        "hashtags": [tag.name for tag in (media.usertags or [])],
                        "caption": media.caption_text or "",
                        "video_url": str(media.video_url) if media.video_url else None,
                        "thumbnail_url": str(media.thumbnail_url) if media.thumbnail_url else None,
                        "scraped_hashtag": hashtag,
                    })

                time.sleep(2)

            except RateLimitError:
                logger.warning(f"Rate limited on #{hashtag}, sleeping 60s")
                time.sleep(60)
            except Exception as e:
                logger.error(f"Error scraping #{hashtag}: {e}")

        results.sort(key=lambda x: x["likes"] + x["views"] * 0.1, reverse=True)
        logger.info(f"Scraped {len(results)} trending reels")
        return results

    def download_video(self, reel: dict, output_dir: str) -> str | None:
        if not reel.get("video_url"):
            return None
        try:
            import requests
            path = Path(output_dir) / f"reel_{reel['id']}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            r = requests.get(reel["video_url"], timeout=30)
            path.write_bytes(r.content)
            return str(path)
        except Exception as e:
            logger.error(f"Failed to download reel {reel['id']}: {e}")
            return None
