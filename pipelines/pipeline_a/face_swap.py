"""
FaceFusion wrapper for Pipeline A.
Calls FaceFusion via Gradio API or folder-watcher approach.
"""
import logging
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

class FaceSwapper:
    def __init__(self, host: str = "localhost", port: int = 7860):
        self.base_url = f"http://{host}:{port}"

    def swap(self, source_video: str, face_image: str, output_path: str) -> str | None:
        """
        Trigger face swap via FaceFusion Gradio API.
        Returns output path or None on failure.
        """
        try:
            # FaceFusion Gradio API endpoint
            response = requests.post(
                f"{self.base_url}/run/predict",
                json={
                    "data": [source_video, face_image, output_path]
                },
                timeout=300
            )
            if response.status_code == 200:
                logger.info(f"Face swap complete: {output_path}")
                return output_path
            else:
                logger.error(f"FaceFusion API error: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Face swap failed: {e}")
            return None

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/", timeout=5)
            return r.status_code == 200
        except Exception:
            return False
