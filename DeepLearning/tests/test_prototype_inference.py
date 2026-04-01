import unittest

import torch

from dl_pipeline.inference.prototype import (
    compute_class_prototypes,
    predict_open_set,
    predict_open_set_with_class_thresholds,
    select_best_threshold,
    select_best_class_thresholds,
)


class PrototypeInferenceTests(unittest.TestCase):
    def test_compute_class_prototypes_averages_and_normalizes_embeddings(self):
        embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.0, 1.0],
                [0.2, 0.8],
            ]
        )
        labels = torch.tensor([1, 1, 2, 2])

        prototypes = compute_class_prototypes(embeddings, labels, prototype_labels=[1, 2])

        self.assertEqual(set(prototypes.keys()), {1, 2})
        self.assertAlmostEqual(torch.linalg.norm(prototypes[1]).item(), 1.0, places=5)
        self.assertAlmostEqual(torch.linalg.norm(prototypes[2]).item(), 1.0, places=5)
        self.assertGreater(prototypes[1][0].item(), prototypes[1][1].item())
        self.assertGreater(prototypes[2][1].item(), prototypes[2][0].item())

    def test_predict_open_set_assigns_other_when_best_similarity_is_below_threshold(self):
        prototypes = {
            1: torch.tensor([1.0, 0.0]),
            2: torch.tensor([0.0, 1.0]),
        }
        query_embeddings = torch.tensor(
            [
                [0.9, 0.1],
                [0.1, 0.9],
                [0.6, 0.6],
            ]
        )

        predictions, scores = predict_open_set(
            query_embeddings,
            prototypes=prototypes,
            other_label=0,
            threshold=0.85,
        )

        self.assertEqual(predictions.tolist(), [1, 2, 0])
        self.assertGreater(scores[0].item(), 0.85)
        self.assertLess(scores[2].item(), 0.85)

    def test_select_best_threshold_maximizes_validation_accuracy(self):
        prototypes = {
            1: torch.tensor([1.0, 0.0]),
            2: torch.tensor([0.0, 1.0]),
        }
        val_embeddings = torch.tensor(
            [
                [0.95, 0.05],
                [0.05, 0.95],
                [0.65, 0.65],
            ]
        )
        val_labels = torch.tensor([1, 2, 0])

        threshold, accuracy = select_best_threshold(
            val_embeddings=val_embeddings,
            val_labels=val_labels,
            prototypes=prototypes,
            other_label=0,
            threshold_values=[0.7, 0.8, 0.9],
        )

        self.assertEqual(threshold, 0.8)
        self.assertAlmostEqual(accuracy, 1.0)

    def test_predict_open_set_with_class_thresholds_uses_threshold_for_predicted_class(self):
        prototypes = {
            1: torch.tensor([1.0, 0.0]),
            2: torch.tensor([0.0, 1.0]),
        }
        query_embeddings = torch.tensor(
            [
                [0.7, 0.3],
                [0.3, 0.7],
                [0.98, 0.02],
            ]
        )

        predictions, scores = predict_open_set_with_class_thresholds(
            query_embeddings,
            prototypes=prototypes,
            other_label=0,
            thresholds_by_class={1: 0.95, 2: 0.90},
        )

        self.assertEqual(predictions.tolist(), [0, 2, 1])
        self.assertLess(scores[0].item(), 0.95)
        self.assertGreater(scores[1].item(), 0.90)

    def test_select_best_class_thresholds_maximizes_validation_accuracy(self):
        prototypes = {
            1: torch.tensor([1.0, 0.0]),
            2: torch.tensor([0.0, 1.0]),
        }
        val_embeddings = torch.tensor(
            [
                [0.98, 0.20],
                [0.80, 0.60],
                [0.74, 0.67],
                [0.20, 0.98],
                [0.60, 0.80],
                [0.67, 0.74],
            ]
        )
        val_labels = torch.tensor([1, 1, 0, 2, 2, 0])

        thresholds, accuracy = select_best_class_thresholds(
            val_embeddings=val_embeddings,
            val_labels=val_labels,
            prototypes=prototypes,
            other_label=0,
            threshold_values_by_class={
                1: [0.70, 0.75, 0.85],
                2: [0.70, 0.80, 0.90],
            },
        )

        self.assertEqual(thresholds, {1: 0.75, 2: 0.80})
        self.assertAlmostEqual(accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
