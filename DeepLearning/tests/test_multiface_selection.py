from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_pipeline.preprocess.multiface_selection import (
    build_face_selection_rows,
    select_face_index_for_label,
    select_face_indices_for_label,
)


class MultifaceSelectionTests(unittest.TestCase):
    def test_select_face_index_for_class1_uses_highest_class1_score(self):
        rows = build_face_selection_rows(
            np.array([0.2, 0.9, 0.7], dtype=np.float32),
            np.array([0.8, 0.1, 0.2], dtype=np.float32),
            theta_jesse=0.5,
            theta_mila=0.5,
        )
        self.assertEqual(select_face_index_for_label(rows, label=1), 1)

    def test_select_face_index_for_class2_uses_highest_class2_score(self):
        rows = build_face_selection_rows(
            np.array([0.9, 0.2, 0.1], dtype=np.float32),
            np.array([0.1, 0.6, 0.85], dtype=np.float32),
            theta_jesse=0.5,
            theta_mila=0.5,
        )
        self.assertEqual(select_face_index_for_label(rows, label=2), 2)

    def test_select_face_index_for_other_uses_hardest_negative(self):
        rows = build_face_selection_rows(
            np.array([0.55, 0.20, 0.40], dtype=np.float32),
            np.array([0.10, 0.62, 0.59], dtype=np.float32),
            theta_jesse=0.50,
            theta_mila=0.60,
        )
        self.assertEqual(select_face_index_for_label(rows, label=0), 0)

    def test_select_face_index_for_unlabeled_probe_uses_strongest_excess(self):
        rows = build_face_selection_rows(
            np.array([0.45, 0.61, 0.80], dtype=np.float32),
            np.array([0.70, 0.20, 0.10], dtype=np.float32),
            theta_jesse=0.60,
            theta_mila=0.50,
        )
        self.assertEqual(select_face_index_for_label(rows, label=None), 2)

    def test_select_face_index_returns_none_for_empty_rows(self):
        self.assertIsNone(select_face_index_for_label([], label=1))

    def test_select_face_indices_for_other_train_can_expand_top_k_hard_negatives(self):
        rows = build_face_selection_rows(
            np.array([0.55, 0.52, 0.20, 0.40], dtype=np.float32),
            np.array([0.10, 0.64, 0.62, 0.59], dtype=np.float32),
            theta_jesse=0.50,
            theta_mila=0.60,
        )
        self.assertEqual(
            select_face_indices_for_label(rows, label=0, max_faces=2),
            [0, 1],
        )

    def test_select_face_indices_for_class_targets_and_unlabeled_remain_single_face(self):
        rows = build_face_selection_rows(
            np.array([0.45, 0.61, 0.80], dtype=np.float32),
            np.array([0.70, 0.20, 0.10], dtype=np.float32),
            theta_jesse=0.60,
            theta_mila=0.50,
        )
        self.assertEqual(select_face_indices_for_label(rows, label=1, max_faces=3), [2])
        self.assertEqual(select_face_indices_for_label(rows, label=None, max_faces=3), [2])


if __name__ == "__main__":
    unittest.main()
