"""
Generates images using Replicate Flux Dev with persona LoRA.
Includes retry logic (3 attempts) and consistent style enforcement.
"""
import time
import logging
import replicate
from pathlib import Path
import requests

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 5

class ImageGenerator:
    def __init__(self, api_token: str):
        self.client = replicate.Client(api_token=api_token)

    def generate(self, persona: dict, trend: dict, output_dir: str, count: int = 4) -> list[str]:
        prompt = self._build_prompt(persona, trend)
        paths = []

        for i in range(count):
            for attempt in range(MAX_RETRIES):
                try:
                    output = self.client.run(
                        "black-forest-labs/flux-dev-lora",
                        input={
                            "prompt": prompt,
                            "hf_lora": persona["lora_model_id"],
                            "lora_scale": 0.85,
                            "num_outputs": 1,
                            "aspect_ratio": "9:16",
                            "output_format": "jpg",
                            "guidance_scale": 3.5,
                            "num_inference_steps": 28,
                        }
                    )
                    image_url = str(output[0])
                    path = self._download_image(image_url, output_dir, f"{persona['id']}_{i}")
                    if path:
                        paths.append(path)
                    break

                except Exception as e:
                    logger.warning(f"Generation attempt {attempt+1} failed: {e}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY * (attempt + 1))
                    else:
                        logger.error(f"All {MAX_RETRIES} attempts failed for image {i}")

        logger.info(f"Generated {len(paths)}/{count} images for {persona['id']}")
        return paths

    def _build_prompt(self, persona: dict, trend: dict) -> str:
        trigger = persona.get("lora_trigger_word", "")
        keywords = " ".join(trend.get("image_prompt_keywords", []))
        appearance = persona.get("appearance", {})
        style = persona.get("style", "")

        return (
            f"{trigger}, {appearance.get('hair', '')} hair, {appearance.get('eyes', '')} eyes, "
            f"{style}, {keywords}, "
            f"dancing pose, Instagram photo, professional lighting, "
            f"high quality, sharp focus, 8k, photorealistic"
        )

    def _download_image(self, url: str, output_dir: str, name: str) -> str | None:
        try:
            path = Path(output_dir) / f"{name}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            r = requests.get(url, timeout=30)
            path.write_bytes(r.content)
            return str(path)
        except Exception as e:
            logger.error(f"Image download failed: {e}")
            return None
