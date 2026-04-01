import unittest

import numpy as np

from dl_pipeline.face_detection.mtcnn_detector import build_reference_landmarks, select_primary_detection


class MTCNNDetectorTests(unittest.TestCase):
    def test_reference_landmarks_scale_with_output_size(self):
        reference = build_reference_landmarks(face_size=112)

        self.assertEqual(reference.shape, (5, 2))
        self.assertAlmostEqual(reference[0, 0], 38.2946, places=3)
        self.assertAlmostEqual(reference[4, 1], 92.2041, places=3)

    def test_select_primary_detection_prefers_largest_face(self):
        boxes = np.array([[0.0, 0.0, 20.0, 20.0], [5.0, 5.0, 40.0, 45.0]], dtype=np.float32)
        probs = np.array([0.99, 0.80], dtype=np.float32)
        landmarks = np.stack([np.zeros((5, 2), dtype=np.float32), np.ones((5, 2), dtype=np.float32)])

        box, landmarks_out = select_primary_detection(boxes, probs, landmarks)

        np.testing.assert_allclose(box, boxes[1])
        np.testing.assert_allclose(landmarks_out, landmarks[1])


if __name__ == "__main__":
    unittest.main()
