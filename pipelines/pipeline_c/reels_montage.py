"""
Assembles a Reels video from static images.
Uses moviepy: fade transitions, trend audio if available, 9:16 format.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REELS_SIZE = (1080, 1920)
DEFAULT_DURATION_PER_CLIP = 2.5

class ReelsMontage:
    def build(
        self,
        image_paths: list[str],
        output_path: str,
        audio_path: str | None = None,
        duration_per_clip: float = DEFAULT_DURATION_PER_CLIP
    ) -> str | None:
        if not image_paths:
            logger.error("No images provided for montage")
            return None

        try:
            from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip

            clips = []
            for img_path in image_paths:
                clip = (
                    ImageClip(img_path)
                    .set_duration(duration_per_clip)
                    .resize(REELS_SIZE)
                    .crossfadein(0.3)
                )
                clips.append(clip)

            video = concatenate_videoclips(clips, method="compose", padding=-0.3)

            if audio_path and Path(audio_path).exists():
                audio = AudioFileClip(audio_path).subclip(0, video.duration)
                video = video.set_audio(audio)

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            video.write_videofile(
                output_path,
                fps=30,
                codec="libx264",
                audio_codec="aac",
                logger=None
            )
            logger.info(f"Montage saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Montage failed: {e}")
            return None
