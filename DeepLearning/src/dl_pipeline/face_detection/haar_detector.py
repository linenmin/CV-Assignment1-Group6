from __future__ import annotations

from pathlib import Path
from urllib import request

import cv2
import numpy as np

from dl_pipeline.common.paths import ensure_dir


CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"


class HaarFaceDetector:
    def __init__(self, cache_dir: str | Path, face_size: int) -> None:
        self.face_size = face_size
        cache_dir = ensure_dir(Path(cache_dir))
        self.cascade_path = cache_dir / "haarcascade_frontalface_default.xml"
        if not self.cascade_path.exists():
            with request.urlopen(CASCADE_URL) as response, self.cascade_path.open("wb") as handle:
                handle.write(response.read())
        self.detector = cv2.CascadeClassifier(str(self.cascade_path))

    def _center_crop(self, image_rgb: np.ndarray) -> np.ndarray:
        height, width = image_rgb.shape[:2]
        side = min(height, width)
        start_y = (height - side) // 2
        start_x = (width - side) // 2
        return image_rgb[start_y:start_y + side, start_x:start_x + side]

    def extract_primary_face(self, image_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) == 0:
            face = self._center_crop(image_rgb)
        else:
            x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
            face = image_rgb[y:y + h, x:x + w]
        return cv2.resize(face, (self.face_size, self.face_size), interpolation=cv2.INTER_AREA)
