import unittest

import torch
from sklearn.model_selection import StratifiedKFold

from dl_pipeline.inference.prototype import (
    average_normalized_embedding_sets,
    combine_embedding_sets,
    compute_similarity_matrix,
    compute_class_prototypes,
    fuse_similarity_matrices,
    predict_open_set_with_lookalikes,
    predict_open_set_with_quality_aware_scorer,
    predict_open_set_with_richer_scorer,
    predict_open_set_with_verifiers,
    predict_open_set_with_exemplars,
    predict_open_set,
    predict_open_set_from_similarity_matrix,
    predict_open_set_with_class_thresholds,
    select_best_quality_aware_params_leave_one_out,
    select_best_richer_scorer_params_leave_one_out,
    select_best_verifier_thresholds_leave_one_out,
    select_best_exemplar_params,
    select_best_threshold_crossval,
    select_best_threshold,
    select_best_class_thresholds,
)


class PrototypeInferenceTests(unittest.TestCase):
    def test_compute_similarity_matrix_returns_scores_in_requested_label_order(self):
        prototypes = {
            2: torch.tensor([0.0, 1.0]),
            1: torch.tensor([1.0, 0.0]),
        }
        query_embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )

        similarity_matrix, prototype_labels = compute_similarity_matrix(
            query_embeddings=query_embeddings,
            prototypes=prototypes,
            prototype_labels=[1, 2],
        )

        self.assertEqual(prototype_labels, [1, 2])
        expected = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        self.assertTrue(torch.allclose(similarity_matrix, expected, atol=1e-5))

    def test_fuse_similarity_matrices_supports_mean_and_min(self):
        first = torch.tensor([[0.80, 0.20], [0.30, 0.70]])
        second = torch.tensor([[0.60, 0.40], [0.40, 0.60]])

        mean_fused = fuse_similarity_matrices([first, second], method="mean")
        min_fused = fuse_similarity_matrices([first, second], method="min")

        self.assertTrue(
            torch.allclose(
                mean_fused,
                torch.tensor([[0.70, 0.30], [0.35, 0.65]]),
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                min_fused,
                torch.tensor([[0.60, 0.20], [0.30, 0.60]]),
                atol=1e-5,
            )
        )

    def test_predict_open_set_from_similarity_matrix_applies_threshold_after_fusion(self):
        similarity_matrix = torch.tensor(
            [
                [0.82, 0.20],
                [0.55, 0.60],
                [0.51, 0.49],
            ]
        )

        predictions, scores = predict_open_set_from_similarity_matrix(
            similarity_matrix=similarity_matrix,
            prototype_labels=[1, 2],
            other_label=0,
            threshold=0.60,
        )

        self.assertEqual(predictions.tolist(), [1, 2, 0])
        self.assertTrue(
            torch.allclose(
                scores,
                torch.tensor([0.82, 0.60, 0.51]),
                atol=1e-6,
            )
        )

    def test_predict_open_set_with_lookalikes_rejects_when_nearest_is_lookalike(self):
        prototypes = {
            1: torch.tensor([1.0, 0.0]),
            2: torch.tensor([0.0, 1.0]),
            3: torch.tensor([0.6, 0.4]),
            4: torch.tensor([0.4, 0.6]),
        }
        query_embeddings = torch.tensor(
            [
                [0.98, 0.02],  # clear Jesse
                [0.60, 0.40],  # nearest Michael-like
            ]
        )

        predictions, nearest_labels, scores = predict_open_set_with_lookalikes(
            query_embeddings=query_embeddings,
            prototypes=prototypes,
            target_labels=[1, 2],
            lookalike_labels=[3, 4],
            other_label=0,
            threshold=0.85,
        )

        self.assertEqual(predictions.tolist(), [1, 0])
        self.assertEqual(nearest_labels.tolist(), [1, 3])
        self.assertGreater(scores[0].item(), 0.85)

    def test_average_normalized_embedding_sets_normalizes_before_and_after_mean(self):
        first = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        second = torch.tensor([[1.0, 1.0], [1.0, 1.0]])

        averaged = average_normalized_embedding_sets([first, second])

        expected = torch.tensor(
            [
                [0.9238795, 0.3826834],
                [0.3826834, 0.9238795],
            ]
        )
        self.assertTrue(torch.allclose(averaged, expected, atol=1e-5))
        norms = torch.linalg.norm(averaged, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

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

    def test_predict_open_set_with_quality_aware_scorer_uses_gallery_and_probe_quality(self):
        gallery_embeddings = torch.tensor(
            [
                [2.0, 0.0],   # higher-quality class 1 exemplar
                [0.8, 0.6],   # lower-quality class 1 exemplar
                [0.0, 2.0],   # higher-quality class 2 exemplar
                [0.0, 0.9],   # lower-quality class 2 exemplar
                [1.4, 1.2],   # higher-quality other exemplar
                [0.7, 0.7],   # lower-quality other exemplar
            ]
        )
        gallery_labels = torch.tensor([1, 1, 2, 2, 0, 0])
        query_embeddings = torch.tensor(
            [
                [1.7, 0.2],   # should be accepted as class 1
                [0.35, 0.32], # low-quality ambiguous sample should be rejected
            ]
        )

        predictions, best_target_scores, other_scores, second_target_scores, effective_thresholds = (
            predict_open_set_with_quality_aware_scorer(
                query_embeddings=query_embeddings,
                gallery_embeddings=gallery_embeddings,
                gallery_labels=gallery_labels,
                target_labels=[1, 2],
                other_label=0,
                threshold=0.55,
                target_top_k=2,
                other_top_k=2,
                other_margin=0.02,
                target_margin=0.02,
                quality_alpha=2.0,
                low_quality_threshold_boost=0.08,
            )
        )

        self.assertEqual(predictions.tolist(), [1, 0])
        self.assertGreater(best_target_scores[0].item(), effective_thresholds[0].item())
        self.assertGreater(effective_thresholds[1].item(), 0.55)
        self.assertLess((best_target_scores[1] - other_scores[1]).item(), 0.02)
        self.assertGreater(second_target_scores[1].item(), 0.0)

    def test_select_best_quality_aware_params_leave_one_out_matches_manual_search(self):
        embeddings = torch.tensor(
            [
                [2.0, 0.0],
                [0.8, 0.6],
                [0.0, 2.0],
                [0.0, 0.9],
                [1.4, 1.2],
                [0.7, 0.7],
            ]
        )
        labels = torch.tensor([1, 1, 2, 2, 0, 0])

        best_params, best_accuracy = select_best_quality_aware_params_leave_one_out(
            embeddings=embeddings,
            labels=labels,
            target_labels=[1, 2],
            other_label=0,
            threshold_values=[0.55],
            target_top_k_values=[1, 2],
            other_top_k_values=[1, 2],
            other_margin_values=[0.0, 0.02],
            target_margin_values=[0.0, 0.02],
            quality_alpha_values=[0.0, 2.0],
            low_quality_threshold_boost_values=[0.0, 0.08],
        )

        expected_scores = {}
        for target_top_k in [1, 2]:
            for other_top_k in [1, 2]:
                for other_margin in [0.0, 0.02]:
                    for target_margin in [0.0, 0.02]:
                        for quality_alpha in [0.0, 2.0]:
                            for low_quality_threshold_boost in [0.0, 0.08]:
                                predictions = []
                                for query_index in range(len(labels)):
                                    query = embeddings[query_index : query_index + 1]
                                    keep_mask = torch.ones(len(labels), dtype=torch.bool)
                                    keep_mask[query_index] = False
                                    gallery_embeddings = embeddings[keep_mask]
                                    gallery_labels = labels[keep_mask]
                                    fold_predictions, _, _, _, _ = (
                                        predict_open_set_with_quality_aware_scorer(
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
                                            quality_alpha=quality_alpha,
                                            low_quality_threshold_boost=low_quality_threshold_boost,
                                        )
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
                                        quality_alpha,
                                        low_quality_threshold_boost,
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
                "quality_alpha": expected_combo[5],
                "low_quality_threshold_boost": expected_combo[6],
            },
        )
        self.assertAlmostEqual(best_accuracy, expected_scores[expected_combo])

    def test_predict_open_set_with_verifiers_uses_one_vs_rest_scores(self):
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
                [0.95, 0.05],  # clear class 1
                [0.05, 0.95],  # clear class 2
                [0.78, 0.62],  # looks like other
            ]
        )

        predictions, verifier_score_matrix = predict_open_set_with_verifiers(
            query_embeddings=query_embeddings,
            gallery_embeddings=gallery_embeddings,
            gallery_labels=gallery_labels,
            target_labels=[1, 2],
            other_label=0,
            target_top_k=2,
            negative_top_k=2,
            thresholds_by_class={1: 0.10, 2: 0.10},
        )

        self.assertEqual(predictions.tolist(), [1, 2, 0])
        self.assertGreater(verifier_score_matrix[0, 0].item(), 0.10)
        self.assertGreater(verifier_score_matrix[1, 1].item(), 0.10)
        self.assertLess(verifier_score_matrix[2, 0].item(), 0.10)
        self.assertLess(verifier_score_matrix[2, 1].item(), 0.10)

    def test_select_best_verifier_thresholds_leave_one_out_matches_manual_search(self):
        embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
                [0.8, 0.6],
                [0.75, 0.65],
            ]
        )
        labels = torch.tensor([1, 1, 2, 2, 0, 0])

        best_params, best_accuracy = select_best_verifier_thresholds_leave_one_out(
            embeddings=embeddings,
            labels=labels,
            target_labels=[1, 2],
            other_label=0,
            target_top_k_values=[1, 2],
            negative_top_k_values=[1, 2],
            threshold_values_by_class={
                1: [0.10, 0.20, 0.30],
                2: [0.10, 0.20, 0.30],
            },
        )

        expected_scores = {}
        for target_top_k in [1, 2]:
            for negative_top_k in [1, 2]:
                for threshold_1 in [0.10, 0.20, 0.30]:
                    for threshold_2 in [0.10, 0.20, 0.30]:
                        predictions = []
                        for query_index in range(len(labels)):
                            query = embeddings[query_index : query_index + 1]
                            keep_mask = torch.ones(len(labels), dtype=torch.bool)
                            keep_mask[query_index] = False
                            fold_predictions, _ = predict_open_set_with_verifiers(
                                query_embeddings=query,
                                gallery_embeddings=embeddings[keep_mask],
                                gallery_labels=labels[keep_mask],
                                target_labels=[1, 2],
                                other_label=0,
                                target_top_k=target_top_k,
                                negative_top_k=negative_top_k,
                                thresholds_by_class={1: threshold_1, 2: threshold_2},
                            )
                            predictions.append(int(fold_predictions.item()))
                        predictions_tensor = torch.tensor(predictions)
                        expected_scores[
                            (
                                target_top_k,
                                negative_top_k,
                                threshold_1,
                                threshold_2,
                            )
                        ] = (predictions_tensor == labels).float().mean().item()

        expected_combo = max(expected_scores, key=expected_scores.get)
        self.assertEqual(
            best_params,
            {
                "target_top_k": expected_combo[0],
                "negative_top_k": expected_combo[1],
                "thresholds_by_class": {1: expected_combo[2], 2: expected_combo[3]},
            },
        )
        self.assertAlmostEqual(best_accuracy, expected_scores[expected_combo])


if __name__ == "__main__":
    unittest.main()
