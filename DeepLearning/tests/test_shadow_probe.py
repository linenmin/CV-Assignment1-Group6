import unittest

import pandas as pd
import torch

from dl_pipeline.training.shadow_probe import (
    build_shadow_probe_frame,
    compute_shadow_probe_epoch_records,
)


class ShadowProbeTests(unittest.TestCase):
    def test_build_shadow_probe_frame_keeps_requested_id_order(self):
        test_df = pd.DataFrame(
            [
                {"id": 48, "image_path": "test/48.png"},
                {"id": 289, "image_path": "test/289.png"},
                {"id": 361, "image_path": "test/361.png"},
            ]
        )

        selected = build_shadow_probe_frame(test_df, [289, 48])

        self.assertEqual(selected["id"].tolist(), [289, 48])

    def test_compute_shadow_probe_epoch_records_reports_argmax_and_thresholded_predictions(self):
        train_embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        train_labels = torch.tensor([1, 2], dtype=torch.long)
        val_embeddings = torch.tensor(
            [
                [0.99, 0.01],
                [0.01, 0.99],
                [0.50, 0.50],
            ],
            dtype=torch.float32,
        )
        val_labels = torch.tensor([1, 2, 0], dtype=torch.long)
        test_embeddings = torch.tensor(
            [
                [0.50, 0.50],
                [0.99, 0.01],
                [0.20, 0.90],
            ],
            dtype=torch.float32,
        )
        test_ids = torch.tensor([48, 99, 289], dtype=torch.long)

        records = compute_shadow_probe_epoch_records(
            epoch=3,
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            val_embeddings=val_embeddings,
            val_labels=val_labels,
            test_embeddings=test_embeddings,
            test_ids=test_ids,
            probe_ids=[289, 48],
            inference_mode="neighborhood_aware",
            prototype_labels=[1, 2],
            other_label=0,
            config_threshold=0.55,
            threshold_values=[0.55, 0.75],
            neighborhood_top_k=1,
            neighborhood_base_weight=1.0,
        )

        self.assertEqual([record["id"] for record in records], [289, 48])
        self.assertEqual(records[0]["epoch"], 3)
        self.assertEqual(records[0]["argmax_label"], 2)
        self.assertEqual(records[0]["pred_config_threshold"], 2)
        self.assertEqual(records[0]["pred_val_selected_threshold"], 2)
        self.assertAlmostEqual(records[0]["selected_threshold"], 0.75, places=6)
        self.assertAlmostEqual(records[0]["score_class2"], records[0]["base_score"], places=6)

        self.assertEqual(records[1]["argmax_label"], 1)
        self.assertEqual(records[1]["pred_config_threshold"], 1)
        self.assertEqual(records[1]["pred_val_selected_threshold"], 0)
        self.assertAlmostEqual(records[1]["selected_threshold"], 0.75, places=6)
        self.assertIn("neighbor_score", records[1])
        self.assertIn("final_score", records[1])


if __name__ == "__main__":
    unittest.main()
