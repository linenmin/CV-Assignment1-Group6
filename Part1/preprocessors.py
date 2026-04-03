"""
preprocessors.py
----------------
Face detection & preprocessing pipeline (HAAR-based) and quality-check helpers.
"""

import os
import cv2
import numpy as np


class HAARPreprocessor:
    """Preprocessing pipeline built around HAAR feature based cascade classifiers.

    Improvements over the template:
    - Uses OpenCV's bundled haarcascade XML (no download needed).
    - Fixes the colour-space bug: images are stored as RGB, so grayscale
      conversion must use COLOR_RGB2GRAY, not COLOR_BGR2GRAY.
    - Falls back to a centre-crop when no face is detected, so every sample
      has a valid array (no NaN rows that break downstream PCA/SVM).
    """

    def __init__(self, face_size):
        self.face_size = face_size
        haar_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        self.classifier = cv2.CascadeClassifier(haar_path)
        assert not self.classifier.empty(), f"Could not load HAAR cascade from {haar_path}"

    def detect_faces(self, img):
        """Detect all faces in an RGB image."""
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return self.classifier.detectMultiScale(
            img_gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30),
        )

    def preprocess(self, data_row):
        img = data_row['img']
        faces = self.detect_faces(img)

        if len(faces) == 0:
            # Fallback: square centre-crop so no sample is lost
            h, w = img.shape[:2]
            side = min(h, w)
            y0, x0 = (h - side) // 2, (w - side) // 2
            face = img[y0:y0 + side, x0:x0 + side]
        else:
            x, y, w, h = faces[0]   # take the first (most prominent) face
            face = img[y:y + h, x:x + w]

        return cv2.resize(face, self.face_size, interpolation=cv2.INTER_AREA)

    def __call__(self, data):
        return np.stack([self.preprocess(row) for _, row in data.iterrows()])


# ── Quality-check helpers ─────────────────────────────────────────────────────

_eye_cascade = None  # lazy-loaded

def _get_eye_cascade():
    global _eye_cascade
    if _eye_cascade is None:
        _eye_cascade = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, 'haarcascade_eye.xml')
        )
    return _eye_cascade


def laplacian_variance(img_rgb):
    """Measure of sharpness: higher = sharper."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def has_eyes(img_rgb, face_box=None):
    """Returns True if at least one eye is detected inside the face region."""
    eye_cascade = _get_eye_cascade()
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    if face_box is not None:
        x, y, w, h = face_box
        roi = gray[y:y + h, x:x + w]
    else:
        roi = gray
    eyes = eye_cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=4)
    return len(eyes) > 0


def quality_check(train_df, preprocessor, blur_threshold=80.0):
    """Run quality checks on all training samples.

    Returns a DataFrame with columns: idx, face_detected, blur_score, eye_detected, issues.
    """
    import pandas as pd

    records = []
    for idx, row in train_df.iterrows():
        img = row['img']
        issues = []

        # Check 1: face detection
        faces = preprocessor.detect_faces(img)
        face_detected = len(faces) > 0
        face_box = tuple(faces[0]) if face_detected else None
        if not face_detected:
            issues.append('no_face')

        # Check 2: blurriness
        blur_score = laplacian_variance(img)
        if blur_score < blur_threshold:
            issues.append(f'blur({blur_score:.0f})')

        # Check 3: eye occlusion (only meaningful when a face was found)
        if face_detected:
            eye_found = has_eyes(img, face_box)
            if not eye_found:
                issues.append('no_eyes')
        else:
            eye_found = False

        records.append({
            'idx': idx,
            'face_detected': face_detected,
            'blur_score': blur_score,
            'eye_detected': eye_found,
            'issues': ', '.join(issues) if issues else 'ok',
        })

    return pd.DataFrame(records).set_index('idx')
