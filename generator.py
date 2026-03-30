"""Image generation module using Replicate Flux Dev model."""

import os
import time
import logging
import requests
from pathlib import Path
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import replicate

logger = logging.getLogger(__name__)

FLUX_MODEL = "black-forest-labs/flux-dev"

PROMPT_TEMPLATES = [
    "beautiful AI girl, {style}, natural lighting, Instagram photo, ultra realistic, 8k",
    "gorgeous young woman, {style}, golden hour light, lifestyle photography, photorealistic",
    "stunning AI influencer, {style}, aesthetic background, professional photography, highly detailed",
    "attractive girl, {style}, soft bokeh background, fashion shoot, hyperrealistic",
    "modern AI girl, {style}, candid street photography style, warm tones, cinematic",
]

STYLES = [
    "casual chic outfit, coffee shop background",
    "beach sunset, summer dress, ocean view",
    "urban fashion, city streets, modern architecture",
    "cozy home aesthetic, warm interior, morning light",
    "fitness lifestyle, activewear, gym or outdoor",
    "travel blogger, exotic location, adventurous",
    "luxury fashion, elegant dress, upscale setting",
    "bohemian style, nature background, golden hour",
]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type((replicate.exceptions.ReplicateError, requests.exceptions.RequestException, Exception)),
    before_sleep=lambda retry_state: logger.warning(
        f"Attempt {retry_state.attempt_number} failed, retrying in {retry_state.next_action.sleep:.1f}s..."
    ),
)
def generate_image(prompt: str = None, style_index: int = None) -> str:
    """
    Generate an AI image using Replicate Flux Dev.
    Returns the local path to the saved image.

    Args:
        prompt: Custom prompt override. If None, auto-selects.
        style_index: Style index override. If None, uses time-based rotation.

    Returns:
        Path to the saved image file.
    """
    # Select style based on time-of-day rotation if not specified
    if style_index is None:
        style_index = int(time.time() / 3600) % len(STYLES)

    style = STYLES[style_index]

    if prompt is None:
        template_index = style_index % len(PROMPT_TEMPLATES)
        prompt = PROMPT_TEMPLATES[template_index].format(style=style)

    logger.info(f"Generating image with prompt: {prompt[:80]}...")

    output = replicate.run(
        FLUX_MODEL,
        input={
            "prompt": prompt,
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 28,
            "guidance": 3.5,
            "output_format": "jpg",
            "output_quality": 90,
        },
    )

    # output is a list of FileOutput objects
    image_url = str(output[0]) if isinstance(output, list) else str(output)
    logger.info(f"Image generated, downloading from Replicate...")

    return _download_image(image_url)


def _download_image(url: str) -> str:
    """Download image from URL and save to output/images/."""
    output_dir = Path("output/images")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ai_girl_{timestamp}.jpg"
    filepath = output_dir / filename

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with open(filepath, "wb") as f:
        f.write(response.content)

    logger.info(f"Image saved: {filepath} ({len(response.content) / 1024:.1f} KB)")
    return str(filepath)


def generate_image_dry_run() -> str:
    """Dry-run: skip API call, return a placeholder path."""
    output_dir = Path("output/images")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"dry_run_{timestamp}.jpg"

    # Create a tiny placeholder image using Pillow
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (512, 512), color=(180, 120, 200))
        draw = ImageDraw.Draw(img)
        draw.text((160, 240), "DRY RUN\nAI Girl Placeholder", fill=(255, 255, 255))
        img.save(str(filepath))
        logger.info(f"Dry-run placeholder image created: {filepath}")
    except Exception as e:
        logger.warning(f"Could not create placeholder image: {e}")
        filepath.touch()

    return str(filepath)
