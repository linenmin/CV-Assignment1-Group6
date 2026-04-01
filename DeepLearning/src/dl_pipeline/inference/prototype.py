from __future__ import annotations

import torch
import torch.nn.functional as F
from itertools import product


def _normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    return F.normalize(embeddings, p=2, dim=1)


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
