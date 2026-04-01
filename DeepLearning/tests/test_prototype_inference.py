import unittest

import torch
from sklearn.model_selection import StratifiedKFold

from dl_pipeline.inference.prototype import (
    combine_embedding_sets,
    compute_class_prototypes,
    predict_open_set_with_richer_scorer,
    predict_open_set_with_exemplars,
    predict_open_set,
    predict_open_set_with_class_thresholds,
    select_best_richer_scorer_params_leave_one_out,
    select_best_exemplar_params,
    select_best_threshold_crossval,
    select_best_threshold,
    select_best_class_thresholds,
)


class PrototypeInferenceTests(unittest.TestCase):
    def test_combine_embedding_sets_concatenates_embeddings_and_labels(self):
        first_embeddings = torch.tensor([[1.0, 0.0], [0.8, 0.2]])
        first_labels = torch.tensor([1, 1])
        second_embeddings = torch.tensor([[0.0, 1.0]])
        second_labels = torch.tensor([2])

        embeddings, labels = combine_embedding_sets(
            [
                (first_embeddings, first_labels),
                (second_embeddings, second_labels),
            ]
        )

        self.assertTrue(
            torch.allclose(
                embeddings,
                torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]),
            )
        )
        self.assertEqual(labels.tolist(), [1, 1, 2])

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

    def test_predict_open_set_with_exemplars_rejects_when_other_margin_is_too_small(self):
        gallery_embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
                [0.8, 0.6],
                [0.75, 0.65],
            ]
        )
        gallery_labels = torch.tensor([1, 1, 2, 2, 0, 0])
        query_embeddings = torch.tensor(
            [
                [0.95, 0.05],
                [0.78, 0.62],
                [0.05, 0.95],
            ]
        )

        predictions, target_scores, other_scores = predict_open_set_with_exemplars(
            query_embeddings=query_embeddings,
            gallery_embeddings=gallery_embeddings,
            gallery_labels=gallery_labels,
            target_labels=[1, 2],
            other_label=0,
            threshold=0.55,
            top_k=2,
            margin=0.05,
        )

        self.assertEqual(predictions.tolist(), [1, 0, 2])
        self.assertGreater(target_scores[0].item(), 0.55)
        self.assertLess((target_scores[1] - other_scores[1]).item(), 0.05)

    def test_select_best_exemplar_params_finds_best_topk_and_margin(self):
        gallery_embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
                [0.8, 0.6],
                [0.75, 0.65],
            ]
        )
        gallery_labels = torch.tensor([1, 1, 2, 2, 0, 0])
        val_embeddings = torch.tensor(
            [
                [0.95, 0.05],
                [0.78, 0.62],
                [0.05, 0.95],
            ]
        )
        val_labels = torch.tensor([1, 0, 2])

        best_params, best_accuracy = select_best_exemplar_params(
            val_embeddings=val_embeddings,
            val_labels=val_labels,
            gallery_embeddings=gallery_embeddings,
            gallery_labels=gallery_labels,
            target_labels=[1, 2],
            other_label=0,
            threshold=0.55,
            top_k_values=[1, 2],
            margin_values=[0.0, 0.05],
        )

        expected_scores = {}
        for top_k in [1, 2]:
            for margin in [0.0, 0.05]:
                predictions, _, _ = predict_open_set_with_exemplars(
                    query_embeddings=val_embeddings,
                    gallery_embeddings=gallery_embeddings,
                    gallery_labels=gallery_labels,
                    target_labels=[1, 2],
                    other_label=0,
                    threshold=0.55,
                    top_k=top_k,
                    margin=margin,
                )
                expected_scores[(top_k, margin)] = (
                    predictions == val_labels
                ).float().mean().item()

        expected_combo = max(expected_scores, key=expected_scores.get)
        self.assertEqual(best_params, {"top_k": expected_combo[0], "margin": expected_combo[1]})
        self.assertAlmostEqual(best_accuracy, expected_scores[expected_combo])

    def test_predict_open_set_with_richer_scorer_uses_other_and_target_margins(self):
        gallery_embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [0.95, 0.05],
                [0.0, 1.0],
                [0.05, 0.95],
                [0.78, 0.62],
                [0.74, 0.66],
            ]
        )
        gallery_labels = torch.tensor([1, 1, 2, 2, 0, 0])
        query_embeddings = torch.tensor(
            [
                [0.97, 0.03],
                [0.70, 0.70],
                [0.77, 0.63],
            ]
        )

        predictions, best_target_scores, other_scores, second_target_scores = (
            predict_open_set_with_richer_scorer(
                query_embeddings=query_embeddings,
                gallery_embeddings=gallery_embeddings,
                gallery_labels=gallery_labels,
                target_labels=[1, 2],
                other_label=0,
                threshold=0.55,
                target_top_k=2,
                other_top_k=2,
                other_margin=0.03,
                target_margin=0.03,
            )
        )

        self.assertEqual(predictions.tolist(), [1, 0, 0])
        self.assertGreater(best_target_scores[0].item(), 0.55)
        self.assertLess(
            (best_target_scores[1] - second_target_scores[1]).item(),
            0.03,
        )
        self.assertLess(
            (best_target_scores[2] - other_scores[2]).item(),
            0.03,
        )

    def test_select_best_richer_scorer_params_leave_one_out_matches_manual_search(self):
        embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [0.95, 0.05],
                [0.0, 1.0],
                [0.05, 0.95],
                [0.78, 0.62],
                [0.74, 0.66],
            ]
        )
        labels = torch.tensor([1, 1, 2, 2, 0, 0])

        best_params, best_accuracy = select_best_richer_scorer_params_leave_one_out(
            embeddings=embeddings,
            labels=labels,
            target_labels=[1, 2],
            other_label=0,
            threshold_values=[0.55],
            target_top_k_values=[1, 2],
            other_top_k_values=[1, 2],
            other_margin_values=[0.0, 0.03],
            target_margin_values=[0.0, 0.03],
        )

        expected_scores = {}
        for target_top_k in [1, 2]:
            for other_top_k in [1, 2]:
                for other_margin in [0.0, 0.03]:
                    for target_margin in [0.0, 0.03]:
                        predictions = []
                        for query_index in range(len(labels)):
                            query = embeddings[query_index : query_index + 1]
                            keep_mask = torch.ones(len(labels), dtype=torch.bool)
                            keep_mask[query_index] = False
                            gallery_embeddings = embeddings[keep_mask]
                            gallery_labels = labels[keep_mask]
                            fold_predictions, _, _, _ = predict_open_set_with_richer_scorer(
                                query_embeddings=query,
                                gallery_embeddings=gallery_embeddings,
                                gallery_labels=gallery_labels,
                                target_labels=[1, 2],
                                other_label=0,
                                threshold=0.55,
                                target_top_k=target_top_k,
                                other_top_k=other_top_k,
                                other_margin=other_margin,
                                target_margin=target_margin,
                            )
                            predictions.append(int(fold_predictions.item()))
                        predictions_tensor = torch.tensor(predictions)
                        expected_scores[
                            (
                                0.55,
                                target_top_k,
                                other_top_k,
                                other_margin,
                                target_margin,
                            )
                        ] = (predictions_tensor == labels).float().mean().item()

        expected_combo = max(expected_scores, key=expected_scores.get)
        self.assertEqual(
            best_params,
            {
                "threshold": expected_combo[0],
                "target_top_k": expected_combo[1],
                "other_top_k": expected_combo[2],
                "other_margin": expected_combo[3],
                "target_margin": expected_combo[4],
            },
        )
        self.assertAlmostEqual(best_accuracy, expected_scores[expected_combo])


if __name__ == "__main__":
    unittest.main()
