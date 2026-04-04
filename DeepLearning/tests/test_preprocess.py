import unittest
from unittest.mock import patch
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_pipeline.preprocess.build_processed_dataset import _build_detector, build_processed_record
from dl_pipeline.preprocess.build_multiface_selected_dataset import (
    _load_raw_bgr,
    _build_rescue_views,
    _detect_faces_with_optional_rescue,
)


class PreprocessRecordTests(unittest.TestCase):
    def test_build_processed_record_keeps_class_label_when_present(self):
        row = {
            "id": 7,
            "source_path": "data/raw/train/train_7.npy",
            "class": 2,
        }

        record = build_processed_record(
            row=row,
            output_path="data/processed/train/7.png",
        )

        self.assertEqual(record["id"], 7)
        self.assertEqual(record["source_path"], "data/raw/train/train_7.npy")
        self.assertEqual(record["image_path"], "data/processed/train/7.png")
        self.assertEqual(record["class"], 2)

    @patch("dl_pipeline.preprocess.build_processed_dataset.HaarFaceDetector")
    def test_build_detector_supports_haar(self, mock_haar_detector):
        detector = _build_detector("haar", "cache", 224)

        self.assertEqual(detector, mock_haar_detector.return_value)
        mock_haar_detector.assert_called_once_with(cache_dir="cache", face_size=224)

    @patch("dl_pipeline.preprocess.build_processed_dataset.MTCNNFaceDetector")
    def test_build_detector_supports_mtcnn(self, mock_mtcnn_detector):
        detector = _build_detector("mtcnn", "cache", 112)

        self.assertEqual(detector, mock_mtcnn_detector.return_value)
        mock_mtcnn_detector.assert_called_once_with(cache_dir="cache", face_size=112)

    def test_build_rescue_views_adds_single_padded_variant(self):
        image = np.zeros((100, 80, 3), dtype=np.uint8)

        views = _build_rescue_views(image, [0.18, 0.18, 0.0])

        self.assertEqual(len(views), 1)
        source, padded = views[0]
        self.assertEqual(source, "detect_align_rescue_pad18")
        self.assertGreater(padded.shape[0], image.shape[0])
        self.assertGreater(padded.shape[1], image.shape[1])

    def test_detect_faces_with_optional_rescue_uses_padded_image_after_original_miss(self):
        image = np.zeros((100, 80, 3), dtype=np.uint8)
        fake_face = object()

        class FakeApp:
            def __init__(self):
                self.calls = []

            def get(self, img):
                self.calls.append(img.shape[:2])
                if len(self.calls) == 1:
                    return []
                return [fake_face]

        app = FakeApp()

        faces, detect_bgr, detect_source = _detect_faces_with_optional_rescue(
            image,
            app,
            [0.18],
        )

        self.assertEqual(faces, [fake_face])
        self.assertEqual(detect_source, "detect_align_rescue_pad18")
        self.assertGreater(detect_bgr.shape[0], image.shape[0])
        self.assertGreater(detect_bgr.shape[1], image.shape[1])
        self.assertEqual(len(app.calls), 2)

    def test_multiface_selected_loader_preserves_bgr_npy_layout(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.npy"
            bgr = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
            np.save(path, bgr)

            loaded = _load_raw_bgr(path)

            np.testing.assert_array_equal(loaded, bgr)


if __name__ == "__main__":
    unittest.main()
