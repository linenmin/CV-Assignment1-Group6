from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from dl_pipeline.common.paths import ensure_dir


ARCFACE_REFERENCE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def build_reference_landmarks(face_size: int) -> np.ndarray:
    scale = face_size / 112.0
    return ARCFACE_REFERENCE * scale


def select_primary_detection(boxes: np.ndarray, probs: np.ndarray, landmarks: np.ndarray):
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best_index = int(np.argmax(areas))
    return boxes[best_index], landmarks[best_index]


class MTCNNFaceDetector:
    def __init__(self, cache_dir: str | Path, face_size: int) -> None:
        from facenet_pytorch import MTCNN

        self.face_size = face_size
        ensure_dir(Path(cache_dir))
        self.detector = MTCNN(keep_all=True, post_process=False, device="cpu")
        self.reference_landmarks = build_reference_landmarks(face_size)

    def _center_crop(self, image_rgb: np.ndarray) -> np.ndarray:
        height, width = image_rgb.shape[:2]
        side = min(height, width)
        start_y = (height - side) // 2
        start_x = (width - side) // 2
        return image_rgb[start_y:start_y + side, start_x:start_x + side]

    def _align_face(self, image_rgb: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        transform, _ = cv2.estimateAffinePartial2D(
            landmarks.astype(np.float32),
            self.reference_landmarks,
            method=cv2.LMEDS,
        )
        if transform is None:
            face = self._center_crop(image_rgb)
            return cv2.resize(face, (self.face_size, self.face_size), interpolation=cv2.INTER_AREA)
        return cv2.warpAffine(
            image_rgb,
            transform,
            (self.face_size, self.face_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    def extract_primary_face(self, image_rgb: np.ndarray) -> np.ndarray:
        boxes, probs, landmarks = self.detector.detect(image_rgb, landmarks=True)
        if boxes is None or landmarks is None or len(boxes) == 0:
            face = self._center_crop(image_rgb)
            return cv2.resize(face, (self.face_size, self.face_size), interpolation=cv2.INTER_AREA)

        _, primary_landmarks = select_primary_detection(boxes, probs, landmarks)
        return self._align_face(image_rgb, primary_landmarks)
