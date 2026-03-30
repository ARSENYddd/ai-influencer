"""Caption generation module using Anthropic Claude Haiku."""

import logging
import random
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import anthropic

logger = logging.getLogger(__name__)

CAPTION_STYLES = [
    "motivational and inspiring",
    "fun and playful with emojis",
    "mysterious and aesthetic",
    "lifestyle and wellness focused",
    "travel and adventure themed",
    "fashion and beauty focused",
    "self-love and confidence boosting",
]

HASHTAG_SETS = [
    "#aesthetic #lifestyle #vibes #instagood #photooftheday",
    "#fashion #ootd #style #trendy #instafashion",
    "#travel #wanderlust #explore #adventure #travelgram",
    "#fitness #wellness #health #motivation #selfcare",
    "#beauty #makeup #glow #skincare #naturalbeauty",
    "#coffee #cozy #morningvibes #dailylife #homestyle",
    "#sunset #goldenhour #nature #photography #beautiful",
]

SYSTEM_PROMPT = """You are a social media expert writing Instagram captions for a popular AI lifestyle influencer.
Write engaging, authentic-sounding captions that:
- Feel personal and relatable
- Use appropriate emojis naturally (not excessively)
- Include a subtle call-to-action or question to boost engagement
- Sound like a real 20-something woman posting her daily life
- Are 2-4 sentences maximum
- Never mention AI, algorithms, or automation
Output ONLY the caption text, nothing else."""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type((anthropic.APIError, anthropic.APIConnectionError, Exception)),
    before_sleep=lambda retry_state: logger.warning(
        f"Caption attempt {retry_state.attempt_number} failed, retrying..."
    ),
)
def generate_caption(style_hint: str = None) -> tuple[str, str]:
    """
    Generate an Instagram caption using Claude Haiku.

    Args:
        style_hint: Optional style context (e.g., 'beach', 'coffee shop').

    Returns:
        Tuple of (caption, hashtags).
    """
    client = anthropic.Anthropic()

    style = style_hint or random.choice(CAPTION_STYLES)
    hashtags = random.choice(HASHTAG_SETS)

    user_prompt = f"Write an Instagram caption in {style} style for a lifestyle photo. Keep it short and engaging."

    logger.info(f"Generating caption (style: {style})...")

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    caption = message.content[0].text.strip()
    full_caption = f"{caption}\n\n{hashtags}"

    logger.info(f"Caption generated: {caption[:60]}...")
    return caption, full_caption


def generate_caption_dry_run() -> tuple[str, str]:
    """Dry-run: return a pre-written sample caption."""
    samples = [
        ("Living for these golden hour moments ✨ Life is too short not to chase the light. What's your favorite time of day?", "#aesthetic #lifestyle #vibes #instagood #photooftheday"),
        ("Coffee first, everything else second ☕ Starting the morning right — what's your morning ritual?", "#coffee #cozy #morningvibes #dailylife #homestyle"),
        ("New day, new energy 🌿 Grateful for every little moment that makes life beautiful. Drop a ❤️ if you feel this!", "#fitness #wellness #health #motivation #selfcare"),
    ]
    caption_text, hashtags = random.choice(samples)
    full = f"{caption_text}\n\n{hashtags}"
    logger.info(f"Dry-run caption: {caption_text[:60]}...")
    return caption_text, full
