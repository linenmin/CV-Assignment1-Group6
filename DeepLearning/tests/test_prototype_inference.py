import unittest

import torch
from sklearn.model_selection import StratifiedKFold

from dl_pipeline.inference.prototype import (
    compute_class_prototypes,
    predict_open_set,
    predict_open_set_with_class_thresholds,
    select_best_threshold_crossval,
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

    def test_select_best_threshold_crossval_uses_mean_fold_accuracy(self):
        embeddings = torch.tensor(
            [
                [0.98, 0.02],
                [0.80, 0.60],
                [0.02, 0.98],
                [0.60, 0.80],
                [0.70, 0.68],
                [0.68, 0.70],
            ]
        )
        labels = torch.tensor([1, 1, 2, 2, 0, 0])
        threshold_values = [0.70, 0.85, 0.95]

        threshold, accuracy = select_best_threshold_crossval(
            embeddings=embeddings,
            labels=labels,
            prototype_labels=[1, 2],
            other_label=0,
            threshold_values=threshold_values,
            n_splits=2,
            random_state=42,
        )

        splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        expected_scores = {}
        for candidate_threshold in threshold_values:
            fold_accuracies = []
            for train_indices, val_indices in splitter.split(embeddings.numpy(), labels.numpy()):
                train_indices = torch.as_tensor(train_indices, dtype=torch.long)
                val_indices = torch.as_tensor(val_indices, dtype=torch.long)
                prototypes = compute_class_prototypes(
                    embeddings[train_indices],
                    labels[train_indices],
                    prototype_labels=[1, 2],
                )
                predictions, _ = predict_open_set(
                    embeddings[val_indices],
                    prototypes=prototypes,
                    other_label=0,
                    threshold=candidate_threshold,
                )
                fold_accuracies.append(
                    (predictions == labels[val_indices]).float().mean().item()
                )
            expected_scores[candidate_threshold] = sum(fold_accuracies) / len(fold_accuracies)

        expected_threshold = max(expected_scores, key=expected_scores.get)
        expected_accuracy = expected_scores[expected_threshold]
        self.assertEqual(threshold, expected_threshold)
        self.assertAlmostEqual(accuracy, expected_accuracy)


if __name__ == "__main__":
    unittest.main()
