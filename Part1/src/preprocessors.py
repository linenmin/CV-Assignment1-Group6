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
    - When multiple faces are detected, selects the most prominent real face
      by scoring each candidate as relative_area × Gaussian(distance_to_centre).
      Skin ratio acts as a hard gate to reject non-face detections.
    - Falls back to lenient cascade parameters when strict detection finds nothing.
    """

    def __init__(self, face_size):
        self.face_size = face_size
        haar_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        self.classifier = cv2.CascadeClassifier(haar_path)
        assert not self.classifier.empty(), f"Could not load HAAR cascade from {haar_path}"

    # ── Noise detection helpers ───────────────────────────────────────────────

    def _is_corrupt(self, img_rgb):
        """Return True for near-uniform (blank/broken) images."""
        return float(img_rgb.std()) < 5.0

    def _skin_ratio(self, img_rgb, face_box):
        """Fraction of pixels inside face_box that fall in the skin-colour range (HSV)."""
        x, y, w, h = face_box
        region = img_rgb[y:y + h, x:x + w]
        hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        # Skin tones span two hue segments in OpenCV HSV (H in [0,180])
        lo1 = np.array([0,  20,  70], dtype=np.uint8)
        hi1 = np.array([25, 255, 255], dtype=np.uint8)
        lo2 = np.array([170, 20,  70], dtype=np.uint8)
        hi2 = np.array([180, 255, 255], dtype=np.uint8)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, lo1, hi1),
            cv2.inRange(hsv, lo2, hi2),
        )
        return float(mask.mean()) / 255.0

    def _is_cartoon(self, img_rgb):
        """Heuristic: cartoons have flat, hyper-saturated colour palettes."""
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        return float(sat.mean()) > 120 and float(sat.std()) < 50

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _center_crop(self, img):
        h, w = img.shape[:2]
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        return img[y0:y0 + side, x0:x0 + side]

    def _sharpness(self, img_rgb, face_box):
        """Laplacian variance of the face region — higher means sharper (more in-focus)."""
        x, y, w, h = face_box
        region = img_rgb[y:y + h, x:x + w]
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _score_face(self, img_rgb, face_box):
        """Score a face candidate using three orthogonal signals.

        score = relative_area × Gaussian(distance_to_centre) × sharpness_weight

        - relative_area:      subject is usually closer → larger bounding box
        - centre distance:    subject is usually centred in the frame
        - sharpness_weight:   subject is in focus; background people are blurred
                              by depth-of-field — this is the key discriminator

        Skin ratio acts as a hard gate to reject cartoon / object false-positives.
        """
        x, y, w, h = face_box
        ih, iw = img_rgb.shape[:2]

        # Hard gate: skip detections with almost no skin pixels
        if self._skin_ratio(img_rgb, face_box) < 0.05:
            return -1.0

        # Relative area in [0, 1]
        area_score = (w * h) / (iw * ih)

        # Gaussian centre weight (σ = 35 % of shorter dimension)
        face_cx = x + w / 2.0
        face_cy = y + h / 2.0
        sigma = min(iw, ih) * 0.35
        dist_sq = (face_cx - iw / 2.0) ** 2 + (face_cy - ih / 2.0) ** 2
        center_score = np.exp(-dist_sq / (2.0 * sigma ** 2))

        # Sharpness weight in [0, 1): f(v) = v / (v + 100)
        # blurry background face (v≈10) → ~0.09; sharp subject (v≈200) → ~0.67
        sharpness_score = self._sharpness(img_rgb, face_box)
        sharpness_weight = sharpness_score / (sharpness_score + 100.0)

        return area_score * center_score * sharpness_weight

    def _select_best_face(self, img_rgb, faces):
        """Pick the face with the highest area × centre-distance score.

        Falls back to pure area if all candidates fail the skin gate.
        """
        best_score, best_box = -1.0, faces[0]
        for box in faces:
            score = self._score_face(img_rgb, box)
            if score > best_score:
                best_score, best_box = score, box

        # If every candidate was gated out (all score == -1), fall back to largest area
        if best_score < 0:
            best_box = max(faces, key=lambda b: b[2] * b[3])

        return best_box

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_faces(self, img, lenient=False):
        """Detect all faces in an RGB image.

        Parameters
        ----------
        lenient : bool
            When True, uses relaxed cascade parameters for a second-pass attempt
            on images where strict detection found nothing.
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if lenient:
            return self.classifier.detectMultiScale(
                img_gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(20, 20),
            )
        return self.classifier.detectMultiScale(
            img_gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30),
        )

    def preprocess(self, data_row):
        img = data_row['img']

        if self._is_corrupt(img):
            face = self._center_crop(img)
        else:
            faces = self.detect_faces(img)
            if len(faces) == 0:
                # Second attempt with more lenient parameters before giving up
                faces = self.detect_faces(img, lenient=True)

            if len(faces) == 0:
                face = self._center_crop(img)
            elif len(faces) == 1:
                x, y, w, h = faces[0]
                face = img[y:y + h, x:x + w]
            else:
                x, y, w, h = self._select_best_face(img, faces)
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

    Returns a DataFrame with columns:
        idx, face_detected, num_faces, blur_score, eye_detected,
        skin_ratio, is_cartoon, is_corrupt, issues.
    """
    import pandas as pd

    records = []
    for idx, row in train_df.iterrows():
        img = row['img']
        issues = []

        # Corruption check
        is_corrupt = preprocessor._is_corrupt(img)
        if is_corrupt:
            issues.append('corrupt')

        # Cartoon check
        is_cartoon = preprocessor._is_cartoon(img)
        if is_cartoon:
            issues.append('cartoon')

        # Face detection
        faces = preprocessor.detect_faces(img)
        num_faces = len(faces)
        face_detected = num_faces > 0
        if not face_detected:
            issues.append('no_face')
        elif num_faces > 1:
            issues.append(f'multi_face({num_faces})')

        # Skin ratio and sharpness on the chosen face
        if face_detected:
            chosen_box = tuple(
                preprocessor._select_best_face(img, faces)
                if num_faces > 1
                else faces[0]
            )
            skin_ratio = preprocessor._skin_ratio(img, chosen_box)
            face_sharpness = preprocessor._sharpness(img, chosen_box)
            face_box = chosen_box
        else:
            chosen_box = None
            skin_ratio = 0.0
            face_sharpness = 0.0
            face_box = None

        # Blurriness
        blur_score = laplacian_variance(img)
        if blur_score < blur_threshold:
            issues.append(f'blur({blur_score:.0f})')

        # Eye detection (only when face found)
        if face_detected:
            eye_found = has_eyes(img, face_box)
            if not eye_found:
                issues.append('no_eyes')
        else:
            eye_found = False

        records.append({
            'idx':             idx,
            'face_detected':   face_detected,
            'num_faces':       num_faces,
            'blur_score':      blur_score,
            'face_sharpness':  face_sharpness,
            'eye_detected':    eye_found,
            'skin_ratio':      skin_ratio,
            'is_cartoon':      is_cartoon,
            'is_corrupt':      is_corrupt,
            'issues':          ', '.join(issues) if issues else 'ok',
        })

    return pd.DataFrame(records).set_index('idx')


def filter_train(train_df, quality_df, skin_threshold=0.10):
    """Return a filtered copy of train_df with clearly-noisy samples removed.

    Drops rows where:
    - is_corrupt is True
    - is_cartoon is True
    - face was detected but skin_ratio < skin_threshold

    Parameters
    ----------
    train_df     : original training DataFrame (must include 'img' and 'class')
    quality_df   : output of quality_check()
    skin_threshold : minimum skin fraction to keep a face-detected sample

    Returns
    -------
    pd.DataFrame — filtered training set (index preserved)
    """
    q = quality_df.reindex(train_df.index)

    keep = ~(
        q['is_corrupt'] |
        q['is_cartoon'] |
        (q['face_detected'] & (q['skin_ratio'] < skin_threshold))
    )

    dropped = (~keep).sum()
    print(f"filter_train: dropped {dropped} / {len(train_df)} samples")
    return train_df[keep]
