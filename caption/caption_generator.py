"""
Generates Instagram captions using Groq (Llama 3).
Caption is written in the persona's voice with matching hashtags.
Includes retry logic (3 attempts).
"""
import time
import logging
from groq import Groq

logger = logging.getLogger(__name__)
MAX_RETRIES = 3


class CaptionGenerator:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)

    def generate(self, persona: dict, trend: dict) -> str | None:
        prompt = self._build_prompt(persona, trend)

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.8,
                )
                caption = response.choices[0].message.content.strip()
                logger.info(f"Caption generated for {persona['id']}")
                return caption

            except Exception as e:
                logger.warning(f"Caption attempt {attempt+1} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(3 * (attempt + 1))
                else:
                    logger.error("All caption attempts failed")
                    return None

    def _build_prompt(self, persona: dict, trend: dict) -> str:
        style = persona["caption_style"]
        hashtags = " ".join(style.get("hashtags", []))
        tone = style.get("tone", "fun and casual")
        use_emojis = style.get("emojis", True)
        hashtag_count = style.get("hashtag_count", 15)
        personality = persona.get("personality", "")
        dance_type = trend.get("dance_type", "this dance trend")

        return f"""Write an Instagram caption for an AI influencer named {persona['display_name']}.

Personality: {personality}
Tone: {tone}
{"Use emojis" if use_emojis else "No emojis"}
Include {hashtag_count} relevant hashtags at the end.

Context: She just did the "{dance_type}" trend.

Pre-approved hashtags to include some of: {hashtags}

Write ONLY the caption text (1-3 short sentences) followed by hashtags.
Make it feel authentic, not robotic. Stay in character."""
