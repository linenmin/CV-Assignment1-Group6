from __future__ import annotations

from itertools import product

import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold


def _normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    return F.normalize(embeddings, p=2, dim=1)


def combine_embedding_sets(
    embedding_sets: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not embedding_sets:
        raise ValueError("embedding_sets 不能为空。")

    embeddings = [item[0] for item in embedding_sets]
    labels = [item[1] for item in embedding_sets]
    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)


def _compute_best_scores(
    query_embeddings: torch.Tensor,
    prototypes: dict[int, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not prototypes:
        raise ValueError("prototypes 不能为空。")

    normalized_queries = _normalize_embeddings(query_embeddings)
    prototype_labels = list(prototypes.keys())
    prototype_matrix = torch.stack([prototypes[label] for label in prototype_labels], dim=0)
    similarities = normalized_queries @ prototype_matrix.T
    best_scores, best_indices = similarities.max(dim=1)
    predicted_labels = torch.tensor(
        [prototype_labels[index] for index in best_indices.tolist()],
        device=query_embeddings.device,
        dtype=torch.long,
    )
    return predicted_labels, best_scores


def _mean_topk_similarity(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    k: int,
) -> torch.Tensor:
    if gallery_embeddings.numel() == 0:
        raise ValueError("gallery_embeddings 不能为空。")
    normalized_queries = _normalize_embeddings(query_embeddings)
    normalized_gallery = _normalize_embeddings(gallery_embeddings)
    similarities = normalized_queries @ normalized_gallery.T
    top_k = min(k, normalized_gallery.shape[0])
    topk_values, _ = similarities.topk(top_k, dim=1)
    return topk_values.mean(dim=1)


def compute_class_prototypes(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    prototype_labels: list[int],
) -> dict[int, torch.Tensor]:
    normalized_embeddings = _normalize_embeddings(embeddings)
    prototypes: dict[int, torch.Tensor] = {}
    for label in prototype_labels:
        class_embeddings = normalized_embeddings[labels == label]
        if class_embeddings.numel() == 0:
            raise ValueError(f"标签 {label} 没有可用于构建 prototype 的样本。")
        prototype = class_embeddings.mean(dim=0, keepdim=True)
        prototypes[label] = F.normalize(prototype, p=2, dim=1).squeeze(0)
    return prototypes


def predict_open_set(
    query_embeddings: torch.Tensor,
    prototypes: dict[int, torch.Tensor],
    other_label: int,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    predicted_labels, best_scores = _compute_best_scores(query_embeddings, prototypes)
    predictions = predicted_labels.clone()
    predictions[best_scores < threshold] = other_label
    return predictions, best_scores


def predict_open_set_with_class_thresholds(
    query_embeddings: torch.Tensor,
    prototypes: dict[int, torch.Tensor],
    other_label: int,
    thresholds_by_class: dict[int, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not thresholds_by_class:
        raise ValueError("thresholds_by_class 不能为空。")

    predicted_labels, best_scores = _compute_best_scores(query_embeddings, prototypes)
    predictions = predicted_labels.clone()
    for index, predicted_label in enumerate(predicted_labels.tolist()):
        if predicted_label not in thresholds_by_class:
            raise ValueError(f"类别 {predicted_label} 缺少对应阈值。")
        if best_scores[index].item() < thresholds_by_class[predicted_label]:
            predictions[index] = other_label
    return predictions, best_scores


def predict_open_set_with_exemplars(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
    target_labels: list[int],
    other_label: int,
    threshold: float,
    top_k: int,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if top_k < 1:
        raise ValueError("top_k 至少为 1。")
    if not target_labels:
        raise ValueError("target_labels 不能为空。")

    target_scores_by_label: dict[int, torch.Tensor] = {}
    for label in target_labels:
        class_gallery = gallery_embeddings[gallery_labels == label]
        if class_gallery.numel() == 0:
            raise ValueError(f"类别 {label} 没有可用 exemplar。")
        target_scores_by_label[label] = _mean_topk_similarity(query_embeddings, class_gallery, top_k)

    other_gallery = gallery_embeddings[gallery_labels == other_label]
    if other_gallery.numel() == 0:
        raise ValueError("other 类没有可用 exemplar。")
    other_scores = _mean_topk_similarity(query_embeddings, other_gallery, top_k)

    ordered_labels = list(target_scores_by_label.keys())
    target_score_matrix = torch.stack([target_scores_by_label[label] for label in ordered_labels], dim=1)
    best_target_scores, best_target_indices = target_score_matrix.max(dim=1)
    predicted_targets = torch.tensor(
        [ordered_labels[index] for index in best_target_indices.tolist()],
        device=query_embeddings.device,
        dtype=torch.long,
    )

    predictions = predicted_targets.clone()
    reject_mask = (best_target_scores < threshold) | ((best_target_scores - other_scores) < margin)
    predictions[reject_mask] = other_label
    return predictions, best_target_scores, other_scores


def predict_open_set_with_richer_scorer(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
    target_labels: list[int],
    other_label: int,
    threshold: float,
    target_top_k: int,
    other_top_k: int,
    other_margin: float,
    target_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if target_top_k < 1:
        raise ValueError("target_top_k 至少为 1。")
    if other_top_k < 1:
        raise ValueError("other_top_k 至少为 1。")
    if not target_labels:
        raise ValueError("target_labels 不能为空。")

    target_scores_by_label: dict[int, torch.Tensor] = {}
    for label in target_labels:
        class_gallery = gallery_embeddings[gallery_labels == label]
        if class_gallery.numel() == 0:
            raise ValueError(f"类别 {label} 没有可用 gallery。")
        target_scores_by_label[label] = _mean_topk_similarity(
            query_embeddings,
            class_gallery,
            target_top_k,
        )

    other_gallery = gallery_embeddings[gallery_labels == other_label]
    if other_gallery.numel() == 0:
        raise ValueError("other 类没有可用 gallery。")
    other_scores = _mean_topk_similarity(query_embeddings, other_gallery, other_top_k)

    ordered_labels = list(target_scores_by_label.keys())
    target_score_matrix = torch.stack([target_scores_by_label[label] for label in ordered_labels], dim=1)
    best_target_scores, best_target_indices = target_score_matrix.max(dim=1)
    predicted_targets = torch.tensor(
        [ordered_labels[index] for index in best_target_indices.tolist()],
        device=query_embeddings.device,
        dtype=torch.long,
    )

    if target_score_matrix.shape[1] > 1:
        top2_values, _ = target_score_matrix.topk(2, dim=1)
        second_target_scores = top2_values[:, 1]
    else:
        second_target_scores = torch.full_like(best_target_scores, float("-inf"))

    predictions = predicted_targets.clone()
    reject_mask = (
        (best_target_scores < threshold)
        | ((best_target_scores - other_scores) < other_margin)
        | ((best_target_scores - second_target_scores) < target_margin)
    )
    predictions[reject_mask] = other_label
    return predictions, best_target_scores, other_scores, second_target_scores


def select_best_threshold(
    val_embeddings: torch.Tensor,
    val_labels: torch.Tensor,
    prototypes: dict[int, torch.Tensor],
    other_label: int,
    threshold_values: list[float],
) -> tuple[float, float]:
    if not threshold_values:
        raise ValueError("threshold_values 不能为空。")

    best_threshold = threshold_values[0]
    best_accuracy = -1.0
    for threshold in threshold_values:
        predictions, _ = predict_open_set(
            val_embeddings,
            prototypes=prototypes,
            other_label=other_label,
            threshold=threshold,
        )
        accuracy = (predictions == val_labels).float().mean().item()
        if accuracy > best_accuracy:
            best_threshold = threshold
            best_accuracy = accuracy
    return best_threshold, best_accuracy


def select_best_threshold_crossval(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    prototype_labels: list[int],
    other_label: int,
    threshold_values: list[float],
    n_splits: int = 4,
    random_state: int = 42,
) -> tuple[float, float]:
    if not threshold_values:
        raise ValueError("threshold_values 不能为空。")
    if n_splits < 2:
        raise ValueError("n_splits 至少为 2。")

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_threshold = threshold_values[0]
    best_accuracy = -1.0

    for threshold in threshold_values:
        fold_accuracies: list[float] = []
        for train_indices, val_indices in splitter.split(
            embeddings.cpu().numpy(),
            labels.cpu().numpy(),
        ):
            train_indices_tensor = torch.as_tensor(train_indices, dtype=torch.long)
            val_indices_tensor = torch.as_tensor(val_indices, dtype=torch.long)
            prototypes = compute_class_prototypes(
                embeddings[train_indices_tensor],
                labels[train_indices_tensor],
                prototype_labels=prototype_labels,
            )
            predictions, _ = predict_open_set(
                embeddings[val_indices_tensor],
                prototypes=prototypes,
                other_label=other_label,
                threshold=threshold,
            )
            fold_accuracies.append(
                (predictions == labels[val_indices_tensor]).float().mean().item()
            )

        mean_accuracy = sum(fold_accuracies) / len(fold_accuracies)
        if mean_accuracy > best_accuracy:
            best_threshold = threshold
            best_accuracy = mean_accuracy

    return best_threshold, best_accuracy


def select_best_exemplar_params(
    val_embeddings: torch.Tensor,
    val_labels: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
    target_labels: list[int],
    other_label: int,
    threshold: float,
    top_k_values: list[int],
    margin_values: list[float],
) -> tuple[dict[str, float | int], float]:
    if not top_k_values:
        raise ValueError("top_k_values 不能为空。")
    if not margin_values:
        raise ValueError("margin_values 不能为空。")

    best_params = {"top_k": top_k_values[0], "margin": margin_values[0]}
    best_accuracy = -1.0
    for top_k, margin in product(top_k_values, margin_values):
        predictions, _, _ = predict_open_set_with_exemplars(
            query_embeddings=val_embeddings,
            gallery_embeddings=gallery_embeddings,
            gallery_labels=gallery_labels,
            target_labels=target_labels,
            other_label=other_label,
            threshold=threshold,
            top_k=top_k,
            margin=margin,
        )
        accuracy = (predictions == val_labels).float().mean().item()
        if accuracy > best_accuracy:
            best_params = {"top_k": top_k, "margin": margin}
            best_accuracy = accuracy

    return best_params, best_accuracy


def select_best_richer_scorer_params_leave_one_out(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    target_labels: list[int],
    other_label: int,
    threshold_values: list[float],
    target_top_k_values: list[int],
    other_top_k_values: list[int],
    other_margin_values: list[float],
    target_margin_values: list[float],
) -> tuple[dict[str, float | int], float]:
    if not threshold_values:
        raise ValueError("threshold_values 不能为空。")
    if not target_top_k_values:
        raise ValueError("target_top_k_values 不能为空。")
    if not other_top_k_values:
        raise ValueError("other_top_k_values 不能为空。")
    if not other_margin_values:
        raise ValueError("other_margin_values 不能为空。")
    if not target_margin_values:
        raise ValueError("target_margin_values 不能为空。")

    best_params = {
        "threshold": threshold_values[0],
        "target_top_k": target_top_k_values[0],
        "other_top_k": other_top_k_values[0],
        "other_margin": other_margin_values[0],
        "target_margin": target_margin_values[0],
    }
    best_accuracy = -1.0

    for threshold, target_top_k, other_top_k, other_margin, target_margin in product(
        threshold_values,
        target_top_k_values,
        other_top_k_values,
        other_margin_values,
        target_margin_values,
    ):
        predictions: list[int] = []
        for query_index in range(labels.shape[0]):
            keep_mask = torch.ones(labels.shape[0], dtype=torch.bool, device=labels.device)
            keep_mask[query_index] = False
            fold_predictions, _, _, _ = predict_open_set_with_richer_scorer(
                query_embeddings=embeddings[query_index : query_index + 1],
                gallery_embeddings=embeddings[keep_mask],
                gallery_labels=labels[keep_mask],
                target_labels=target_labels,
                other_label=other_label,
                threshold=threshold,
                target_top_k=target_top_k,
                other_top_k=other_top_k,
                other_margin=other_margin,
                target_margin=target_margin,
            )
            predictions.append(int(fold_predictions.item()))

        predictions_tensor = torch.tensor(predictions, device=labels.device, dtype=torch.long)
        accuracy = (predictions_tensor == labels).float().mean().item()
        if accuracy > best_accuracy:
            best_params = {
                "threshold": threshold,
                "target_top_k": target_top_k,
                "other_top_k": other_top_k,
                "other_margin": other_margin,
                "target_margin": target_margin,
            }
            best_accuracy = accuracy

    return best_params, best_accuracy


def select_best_class_thresholds(
    val_embeddings: torch.Tensor,
    val_labels: torch.Tensor,
    prototypes: dict[int, torch.Tensor],
    other_label: int,
    threshold_values_by_class: dict[int, list[float]],
) -> tuple[dict[int, float], float]:
    if not threshold_values_by_class:
        raise ValueError("threshold_values_by_class 不能为空。")

    class_labels = list(threshold_values_by_class.keys())
    if any(not threshold_values_by_class[label] for label in class_labels):
        raise ValueError("每个类别都必须提供至少一个候选阈值。")

    best_thresholds = {label: threshold_values_by_class[label][0] for label in class_labels}
    best_accuracy = -1.0
    threshold_grids = [threshold_values_by_class[label] for label in class_labels]
    for candidate_values in product(*threshold_grids):
        thresholds = {label: value for label, value in zip(class_labels, candidate_values)}
        predictions, _ = predict_open_set_with_class_thresholds(
            val_embeddings,
            prototypes=prototypes,
            other_label=other_label,
            thresholds_by_class=thresholds,
        )
        accuracy = (predictions == val_labels).float().mean().item()
        if accuracy > best_accuracy:
            best_thresholds = thresholds
            best_accuracy = accuracy
    return best_thresholds, best_accuracy
