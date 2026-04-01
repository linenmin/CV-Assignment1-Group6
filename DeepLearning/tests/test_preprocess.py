import unittest
from unittest.mock import patch

from dl_pipeline.preprocess.build_processed_dataset import _build_detector, build_processed_record


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


if __name__ == "__main__":
    unittest.main()
