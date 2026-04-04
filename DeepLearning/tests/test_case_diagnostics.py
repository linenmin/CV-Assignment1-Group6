from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_pipeline.evaluation.case_diagnostics import prepare_test_audit_index, rank_class_neighbors


class CaseDiagnosticsTests(unittest.TestCase):
    def test_rank_class_neighbors_returns_top_k_within_target_label(self):
        query = np.array([1.0, 0.0], dtype=np.float32)
        gallery_embeddings = np.array(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.0, 1.0],
                [0.6, 0.1],
            ],
            dtype=np.float32,
        )
        gallery_labels = np.array([2, 2, 1, 2], dtype=np.int64)
        gallery_ids = np.array([10, 11, 12, 13], dtype=np.int64)

        rows = rank_class_neighbors(query, gallery_embeddings, gallery_labels, gallery_ids, target_label=2, top_k=2)

        self.assertEqual([row["id"] for row in rows], [10, 11])
        self.assertEqual([row["rank"] for row in rows], [1, 2])
        self.assertTrue(rows[0]["similarity"] >= rows[1]["similarity"])

    def test_rank_class_neighbors_returns_empty_for_missing_target_label(self):
        query = np.array([1.0, 0.0], dtype=np.float32)
        gallery_embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
        gallery_labels = np.array([1], dtype=np.int64)
        gallery_ids = np.array([10], dtype=np.int64)

        rows = rank_class_neighbors(query, gallery_embeddings, gallery_labels, gallery_ids, target_label=2, top_k=3)

        self.assertEqual(rows, [])

    def test_prepare_test_audit_index_filters_non_test_rows_before_indexing(self):
        audit_df = pd.DataFrame(
            [
                {"split": "train", "id": 48, "selection_source": "detect_align_selected"},
                {"split": "test", "id": 48, "selection_source": "crop_fallback"},
                {"split": "test", "id": 543, "selection_source": "detect_align_rescue_pad18"},
            ]
        )

        indexed = prepare_test_audit_index(audit_df)

        self.assertEqual(indexed.loc[48, "selection_source"], "crop_fallback")
        self.assertEqual(indexed.loc[543, "selection_source"], "detect_align_rescue_pad18")
        self.assertEqual(indexed.index.tolist(), [48, 543])

    def test_prepare_test_audit_index_raises_when_test_ids_still_duplicate(self):
        audit_df = pd.DataFrame(
            [
                {"split": "test", "id": 48, "selection_source": "crop_fallback"},
                {"split": "test", "id": 48, "selection_source": "detect_align_rescue_pad18"},
            ]
        )

        with self.assertRaisesRegex(ValueError, "重复 id"):
            prepare_test_audit_index(audit_df)


if __name__ == "__main__":
    unittest.main()
