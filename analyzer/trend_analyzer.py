"""
Uses Groq (Llama 3) to analyze scraped reels and score/select top trends.
"""
import json
import logging
from groq import Groq

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)

    def analyze_trends(self, reels: list[dict], personas: list[dict]) -> list[dict]:
        if not reels:
            return []

        reels_summary = [
            {
                "id": r["id"],
                "likes": r["likes"],
                "views": r["views"],
                "hashtags": r["scraped_hashtag"],
                "caption_snippet": r["caption"][:100],
            }
            for r in reels[:20]
        ]

        persona_names = [p["id"] for p in personas]

        prompt = f"""You are analyzing trending Instagram Reels to select the best ones for AI influencer content.

Here are {len(reels_summary)} trending reels data:
{json.dumps(reels_summary, indent=2)}

Personas available: {persona_names}

Analyze and return a JSON array of the TOP 5 reels with this structure:
{{
  "id": "reel_id",
  "dance_type": "name or description of dance/challenge",
  "energy_level": "low|medium|high",
  "score": 0-100,
  "best_persona": "persona_id most suited",
  "pipeline_recommendation": "A or C",
  "image_prompt_keywords": ["keyword1", "keyword2"],
  "reason": "brief reason why this trend is good"
}}

Pipeline A = face swap on actual video (best for high-energy dance videos with clear face shots)
Pipeline C = static photos + montage (best for aesthetic/posing trends or when video quality is poor)

Return ONLY valid JSON array, no other text."""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.3,
            )
            text = response.choices[0].message.content.strip()
            # Strip markdown code blocks if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            result = json.loads(text)
            logger.info(f"Analyzed {len(result)} top trends")
            return result
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return []
