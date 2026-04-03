"""不依赖 insightface 包：仅测相似度、多脸聚合与决策逻辑。"""
from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_pipeline.inference.insightface_dual_verifier import (
    aggregate_multi_face_scores,
    describe_multi_face_scores,
    dual_verifier_predict,
    top_k_mean_cosine_similarity,
    youden_threshold,
)


class InsightFaceDualVerifierTests(unittest.TestCase):
    def test_top_k_mean_cosine(self):
        g = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        p = np.array([1.0, 0.0], dtype=np.float32)
        p /= np.linalg.norm(p)
        m = top_k_mean_cosine_similarity(p, g, k=2)
        self.assertAlmostEqual(m, 1.0, places=5)

    def test_youden_simple(self):
        pos = np.array([0.9, 0.8, 0.85])
        neg = np.array([0.1, 0.2, 0.15])
        t, meta = youden_threshold(pos, neg)
        self.assertGreater(t, 0.15)
        self.assertLessEqual(t, 0.9)
        self.assertIn("youden_j", meta)

    def test_dual_verifier_branches(self):
        self.assertEqual(dual_verifier_predict(0.9, 0.1, 0.5, 0.5), 1)
        self.assertEqual(dual_verifier_predict(0.1, 0.9, 0.5, 0.5), 2)
        self.assertEqual(dual_verifier_predict(0.1, 0.1, 0.5, 0.5), 0)
        self.assertEqual(dual_verifier_predict(0.9, 0.9, 0.5, 0.5), 1)
        self.assertEqual(dual_verifier_predict(0.6, 0.9, 0.5, 0.5), 2)

    def test_aggregate_multi_face_scores_uses_best_face_per_identity(self):
        result = aggregate_multi_face_scores(
            score_jesse=np.array([0.92, 0.20], dtype=np.float32),
            score_mila=np.array([0.10, 0.88], dtype=np.float32),
            theta_jesse=0.50,
            theta_mila=0.50,
        )

        self.assertAlmostEqual(result["best_score_jesse"], 0.92, places=5)
        self.assertAlmostEqual(result["best_score_mila"], 0.88, places=5)
        self.assertEqual(result["best_face_index_jesse"], 0)
        self.assertEqual(result["best_face_index_mila"], 1)
        self.assertEqual(result["prediction"], 1)

    def test_aggregate_multi_face_scores_returns_other_when_all_faces_are_below_threshold(self):
        result = aggregate_multi_face_scores(
            score_jesse=np.array([0.40, 0.30], dtype=np.float32),
            score_mila=np.array([0.45, 0.48], dtype=np.float32),
            theta_jesse=0.50,
            theta_mila=0.50,
        )

        self.assertEqual(result["prediction"], 0)
        self.assertLess(result["best_excess_jesse"], 0.0)
        self.assertLess(result["best_excess_mila"], 0.0)

    def test_aggregate_multi_face_scores_prefers_identity_with_stronger_excess(self):
        result = aggregate_multi_face_scores(
            score_jesse=np.array([0.70, 0.51], dtype=np.float32),
            score_mila=np.array([0.52, 0.95], dtype=np.float32),
            theta_jesse=0.50,
            theta_mila=0.60,
        )

        self.assertAlmostEqual(result["best_excess_jesse"], 0.20, places=5)
        self.assertAlmostEqual(result["best_excess_mila"], 0.35, places=5)
        self.assertEqual(result["prediction"], 2)

    def test_describe_multi_face_scores_marks_selected_faces_and_threshold_hits(self):
        rows = describe_multi_face_scores(
            score_jesse=np.array([0.82, 0.20, 0.61], dtype=np.float32),
            score_mila=np.array([0.10, 0.91, 0.40], dtype=np.float32),
            theta_jesse=0.60,
            theta_mila=0.50,
        )

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["face_index"], 0)
        self.assertTrue(rows[0]["is_best_jesse"])
        self.assertTrue(rows[0]["passes_jesse"])
        self.assertFalse(rows[0]["is_best_mila"])
        self.assertEqual(rows[1]["face_index"], 1)
        self.assertTrue(rows[1]["is_best_mila"])
        self.assertTrue(rows[1]["passes_mila"])
        self.assertAlmostEqual(rows[2]["excess_jesse"], 0.01, places=5)

    def test_describe_multi_face_scores_returns_empty_list_for_missing_faces(self):
        self.assertEqual(
            describe_multi_face_scores(
                score_jesse=np.array([], dtype=np.float32),
                score_mila=np.array([], dtype=np.float32),
                theta_jesse=0.60,
                theta_mila=0.50,
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
