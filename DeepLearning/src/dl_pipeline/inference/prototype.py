from __future__ import annotations

from itertools import product

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold


def _normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    return F.normalize(embeddings, p=2, dim=1)


def average_normalized_embedding_sets(
    embedding_sets: list[torch.Tensor],
) -> torch.Tensor:
    if not embedding_sets:
        raise ValueError("embedding_sets 不能为空。")

    normalized_sets = [_normalize_embeddings(embeddings) for embeddings in embedding_sets]
    mean_embeddings = torch.stack(normalized_sets, dim=0).mean(dim=0)
    return _normalize_embeddings(mean_embeddings)


def combine_embedding_sets(
    embedding_sets: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not embedding_sets:
        raise ValueError("embedding_sets 不能为空。")

    embeddings = [item[0] for item in embedding_sets]
    labels = [item[1] for item in embedding_sets]
    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)


def pca_whiten_embedding_sets(
    gallery_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    n_components: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if n_components < 1:
        raise ValueError("n_components 至少为 1。")

    normalized_gallery = _normalize_embeddings(gallery_embeddings)
    normalized_query = _normalize_embeddings(query_embeddings)
    max_components = min(
        n_components,
        normalized_gallery.shape[0],
        normalized_gallery.shape[1],
    )
    pca = PCA(n_components=max_components, whiten=True, random_state=42)
    transformed_gallery = pca.fit_transform(normalized_gallery.detach().cpu().numpy())
    transformed_query = pca.transform(normalized_query.detach().cpu().numpy())

    gallery_tensor = torch.as_tensor(
        transformed_gallery,
        dtype=gallery_embeddings.dtype,
        device=gallery_embeddings.device,
    )
    query_tensor = torch.as_tensor(
        transformed_query,
        dtype=query_embeddings.dtype,
        device=query_embeddings.device,
    )
    return _normalize_embeddings(gallery_tensor), _normalize_embeddings(query_tensor)


def spectral_cluster_with_gallery_label_matching(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
    n_clusters: int,
    top_k_gallery: int,
    random_state: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if n_clusters < 2:
        raise ValueError("n_clusters 至少为 2。")
    if top_k_gallery < 1:
        raise ValueError("top_k_gallery 至少为 1。")

    normalized_gallery = _normalize_embeddings(gallery_embeddings)
    normalized_query = _normalize_embeddings(query_embeddings)
    all_embeddings = torch.cat([normalized_gallery, normalized_query], dim=0)
    affinity = (all_embeddings @ all_embeddings.T).clamp_min(0.0)
    affinity.fill_diagonal_(1.0)

    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=random_state,
        assign_labels="kmeans",
    )
    cluster_ids = clustering.fit_predict(affinity.detach().cpu().numpy())
    cluster_ids_tensor = torch.as_tensor(
        cluster_ids,
        dtype=torch.long,
        device=query_embeddings.device,
    )

    gallery_cluster_ids = cluster_ids_tensor[: normalized_gallery.shape[0]]
    query_cluster_ids = cluster_ids_tensor[normalized_gallery.shape[0] :]
    unique_labels = sorted(set(gallery_labels.tolist()))

    cluster_to_label: dict[int, int] = {}
    for cluster_id in sorted(set(query_cluster_ids.tolist())):
        cluster_mask = query_cluster_ids == cluster_id
        cluster_queries = normalized_query[cluster_mask]
        similarity = cluster_queries @ normalized_gallery.T
        top_k = min(top_k_gallery, normalized_gallery.shape[0])
        _, top_indices = similarity.topk(top_k, dim=1)
        neighbor_labels = gallery_labels[top_indices.reshape(-1)]
        label_scores = {
            label: int((neighbor_labels == label).sum().item()) for label in unique_labels
        }
        selected_label = max(label_scores, key=lambda label: (label_scores[label], -label))
        cluster_to_label[int(cluster_id)] = int(selected_label)

    predictions = torch.tensor(
        [cluster_to_label[int(cluster_id)] for cluster_id in query_cluster_ids.tolist()],
        dtype=torch.long,
        device=query_embeddings.device,
    )
    return predictions, query_cluster_ids


def _compute_best_scores(
    query_embeddings: torch.Tensor,
    prototypes: dict[int, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    similarity_matrix, prototype_labels = compute_similarity_matrix(query_embeddings, prototypes)
    return _best_scores_from_similarity_matrix(similarity_matrix, prototype_labels)


def compute_similarity_matrix(
    query_embeddings: torch.Tensor,
    prototypes: dict[int, torch.Tensor],
    prototype_labels: list[int] | None = None,
) -> tuple[torch.Tensor, list[int]]:
    if not prototypes:
        raise ValueError("prototypes 不能为空。")

    ordered_labels = list(prototype_labels) if prototype_labels is not None else list(prototypes.keys())
    missing_labels = [label for label in ordered_labels if label not in prototypes]
    if missing_labels:
        raise ValueError(f"prototypes 缺少标签: {missing_labels}")

    normalized_queries = _normalize_embeddings(query_embeddings)
    prototype_matrix = _normalize_embeddings(
        torch.stack([prototypes[label] for label in ordered_labels], dim=0)
    )
    similarities = normalized_queries @ prototype_matrix.T
    return similarities, ordered_labels


def _best_scores_from_similarity_matrix(
    similarity_matrix: torch.Tensor,
    prototype_labels: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if similarity_matrix.ndim != 2:
        raise ValueError("similarity_matrix 必须是二维张量。")
    if not prototype_labels:
        raise ValueError("prototype_labels 不能为空。")
    if similarity_matrix.shape[1] != len(prototype_labels):
        raise ValueError("similarity_matrix 列数必须与 prototype_labels 数量一致。")

    best_scores, best_indices = similarity_matrix.max(dim=1)
    predicted_labels = torch.tensor(
        [prototype_labels[index] for index in best_indices.tolist()],
        device=similarity_matrix.device,
        dtype=torch.long,
    )
    return predicted_labels, best_scores


def predict_open_set_from_similarity_matrix(
    similarity_matrix: torch.Tensor,
    prototype_labels: list[int],
    other_label: int,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    predicted_labels, best_scores = _best_scores_from_similarity_matrix(
        similarity_matrix,
        prototype_labels,
    )
    predictions = predicted_labels.clone()
    predictions[best_scores < threshold] = other_label
    return predictions, best_scores


def fuse_similarity_matrices(
    similarity_matrices: list[torch.Tensor],
    method: str,
    weights: list[float] | None = None,
) -> torch.Tensor:
    if not similarity_matrices:
        raise ValueError("similarity_matrices 不能为空。")

    reference_shape = similarity_matrices[0].shape
    if any(matrix.shape != reference_shape for matrix in similarity_matrices):
        raise ValueError("所有 similarity_matrix 的形状必须一致。")

    stacked = torch.stack(similarity_matrices, dim=0)
    if method == "mean":
        if weights is None:
            return stacked.mean(dim=0)
        if len(weights) != len(similarity_matrices):
            raise ValueError("weights 数量必须与 similarity_matrices 数量一致。")
        weight_tensor = torch.tensor(
            weights,
            device=stacked.device,
            dtype=stacked.dtype,
        )
        normalized_weights = weight_tensor / weight_tensor.sum().clamp_min(1e-12)
        return (stacked * normalized_weights.view(-1, 1, 1)).sum(dim=0)

    if method == "min":
        return stacked.min(dim=0).values

    raise ValueError(f"未知的 fusion method: {method}")


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


def _normalize_quality_from_norms(norms: torch.Tensor) -> torch.Tensor:
    min_value = norms.min()
    max_value = norms.max()
    if torch.isclose(max_value, min_value):
        return torch.ones_like(norms)
    return ((norms - min_value) / (max_value - min_value)).clamp(0.0, 1.0)


def _weighted_topk_similarity(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_weights: torch.Tensor,
    k: int,
) -> torch.Tensor:
    if gallery_embeddings.numel() == 0:
        raise ValueError("gallery_embeddings 不能为空。")
    if gallery_weights.numel() == 0:
        raise ValueError("gallery_weights 不能为空。")

    normalized_queries = _normalize_embeddings(query_embeddings)
    normalized_gallery = _normalize_embeddings(gallery_embeddings)
    similarities = normalized_queries @ normalized_gallery.T
    top_k = min(k, normalized_gallery.shape[0])
    topk_values, topk_indices = similarities.topk(top_k, dim=1)
    selected_weights = gallery_weights[topk_indices]
    normalized_weights = selected_weights / selected_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return (topk_values * normalized_weights).sum(dim=1)


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
    similarity_matrix, prototype_labels = compute_similarity_matrix(query_embeddings, prototypes)
    return predict_open_set_from_similarity_matrix(
        similarity_matrix=similarity_matrix,
        prototype_labels=prototype_labels,
        other_label=other_label,
        threshold=threshold,
    )


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


def predict_open_set_with_lookalikes(
    query_embeddings: torch.Tensor,
    prototypes: dict[int, torch.Tensor],
    target_labels: list[int],
    lookalike_labels: list[int],
    other_label: int,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not target_labels:
        raise ValueError("target_labels 不能为空。")
    if not lookalike_labels:
        raise ValueError("lookalike_labels 不能为空。")

    required_labels = set(target_labels) | set(lookalike_labels)
    missing_labels = [label for label in required_labels if label not in prototypes]
    if missing_labels:
        raise ValueError(f"prototypes 缺少标签: {missing_labels}")

    nearest_labels, best_scores = _compute_best_scores(
        query_embeddings,
        {label: prototypes[label] for label in target_labels + lookalike_labels},
    )
    predictions = torch.full_like(nearest_labels, other_label)
    target_mask = torch.isin(nearest_labels, torch.tensor(target_labels, device=nearest_labels.device))
    accept_mask = target_mask & (best_scores >= threshold)
    predictions[accept_mask] = nearest_labels[accept_mask]
    return predictions, nearest_labels, best_scores


def predict_open_set_with_lookalike_margin_rejection(
    query_embeddings: torch.Tensor,
    prototypes: dict[int, torch.Tensor],
    target_labels: list[int],
    lookalike_by_target: dict[int, int],
    other_label: int,
    threshold: float,
    margins_by_target: dict[int, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not target_labels:
        raise ValueError("target_labels 不能为空。")
    if not lookalike_by_target:
        raise ValueError("lookalike_by_target 不能为空。")
    if not margins_by_target:
        raise ValueError("margins_by_target 不能为空。")

    target_predictions, target_scores = _compute_best_scores(
        query_embeddings,
        {label: prototypes[label] for label in target_labels},
    )

    lookalike_scores = []
    for target_label in target_predictions.tolist():
        if target_label not in lookalike_by_target:
            raise ValueError(f"类别 {target_label} 缺少 look-alike label。")
        if target_label not in margins_by_target:
            raise ValueError(f"类别 {target_label} 缺少对应 margin。")
        lookalike_label = lookalike_by_target[target_label]
        if lookalike_label not in prototypes:
            raise ValueError(f"look-alike 类别 {lookalike_label} 缺少 prototype。")
        normalized_queries = _normalize_embeddings(query_embeddings)
        normalized_lookalike = _normalize_embeddings(
            torch.stack([prototypes[lookalike_by_target[label]] for label in target_predictions.tolist()], dim=0)
        )
        lookalike_scores = (normalized_queries * normalized_lookalike).sum(dim=1)
        break

    margins = target_scores - lookalike_scores
    predictions = target_predictions.clone()
    predictions[target_scores < threshold] = other_label
    for index, target_label in enumerate(target_predictions.tolist()):
        if predictions[index].item() == other_label:
            continue
        if margins[index].item() < margins_by_target[target_label]:
            predictions[index] = other_label
    return predictions, target_scores, lookalike_scores, margins


def global_label_spread_predictions(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
    class_labels: list[int],
    alpha: float,
    top_k: int,
    max_iter: int,
    tol: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not class_labels:
        raise ValueError("class_labels 不能为空。")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha 必须位于 (0, 1) 区间。")
    if top_k < 1:
        raise ValueError("top_k 至少为 1。")
    if max_iter < 1:
        raise ValueError("max_iter 至少为 1。")

    all_embeddings = _normalize_embeddings(torch.cat([gallery_embeddings, query_embeddings], dim=0))
    similarity = all_embeddings @ all_embeddings.T
    similarity.fill_diagonal_(0.0)
    similarity = similarity.clamp_min(0.0)

    neighbor_count = min(top_k, similarity.shape[1] - 1)
    if neighbor_count < 1:
        raise ValueError("样本数不足以构建图。")
    topk_values, topk_indices = similarity.topk(neighbor_count, dim=1)
    affinity = torch.zeros_like(similarity)
    affinity.scatter_(1, topk_indices, topk_values)
    affinity = torch.maximum(affinity, affinity.T)
    transition = affinity / affinity.sum(dim=1, keepdim=True).clamp_min(1e-6)

    class_to_index = {label: index for index, label in enumerate(class_labels)}
    y = torch.zeros(
        (all_embeddings.shape[0], len(class_labels)),
        dtype=all_embeddings.dtype,
        device=all_embeddings.device,
    )
    for row_index, label in enumerate(gallery_labels.tolist()):
        if label not in class_to_index:
            raise ValueError(f"gallery label {label} 不在 class_labels 中。")
        y[row_index, class_to_index[label]] = 1.0

    f = y.clone()
    labeled_mask = torch.zeros(all_embeddings.shape[0], dtype=torch.bool, device=all_embeddings.device)
    labeled_mask[: gallery_embeddings.shape[0]] = True

    for _ in range(max_iter):
        propagated = alpha * (transition @ f) + (1.0 - alpha) * y
        propagated[labeled_mask] = y[labeled_mask]
        propagated = propagated / propagated.sum(dim=1, keepdim=True).clamp_min(1e-6)
        if torch.max(torch.abs(propagated - f)).item() < tol:
            f = propagated
            break
        f = propagated

    query_probabilities = f[gallery_embeddings.shape[0] :]
    predicted_indices = query_probabilities.argmax(dim=1)
    predictions = torch.tensor(
        [class_labels[index] for index in predicted_indices.tolist()],
        dtype=torch.long,
        device=query_embeddings.device,
    )
    return predictions, query_probabilities


def neighborhood_aware_predictions(
    query_embeddings: torch.Tensor,
    prototypes: dict[int, torch.Tensor],
    other_label: int,
    threshold: float,
    top_k: int,
    base_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if top_k < 1:
        raise ValueError("top_k 至少为 1。")
    if not (0.0 <= base_weight <= 1.0):
        raise ValueError("base_weight 必须位于 [0, 1] 区间。")

    predicted_labels, base_scores = _compute_best_scores(query_embeddings, prototypes)

    normalized_queries = _normalize_embeddings(query_embeddings)
    similarity = normalized_queries @ normalized_queries.T
    similarity.fill_diagonal_(0.0)
    current_top_k = min(top_k, max(1, similarity.shape[1] - 1))
    _, top_indices = similarity.topk(current_top_k, dim=1)
    neighbor_mean_scores = base_scores[top_indices].mean(dim=1)

    final_scores = (base_weight * base_scores) + ((1.0 - base_weight) * neighbor_mean_scores)
    predictions = predicted_labels.clone()
    predictions[final_scores < threshold] = other_label
    return predictions, base_scores, neighbor_mean_scores, final_scores


def neighborhood_aware_predictions_adaptive_threshold(
    query_embeddings: torch.Tensor,
    prototypes: dict[int, torch.Tensor],
    other_label: int,
    top_k: int,
    base_weight: float,
    th_min: float,
    th_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """与 ``neighborhood_aware_predictions`` 相同的打分，但拒识阈值按邻居标签一致性在 ``[th_min, th_max]`` 内逐样本变化。

    对第 ``i`` 个 query，取其 top-k 近邻（在 query 集合内、按 embedding 余弦相似度），
    统计邻居的 prototype argmax 标签与当前 query 的 argmax 标签一致的比例 ``agree_frac``，
    并设 ``effective_th[i] = th_max - (th_max - th_min) * agree_frac``（邻居越一致，阈值越低、越易接受）。

    返回的第五个张量为 ``effective_threshold``，形状与 query 数相同。
    """
    if top_k < 1:
        raise ValueError("top_k 至少为 1。")
    if not (0.0 <= base_weight <= 1.0):
        raise ValueError("base_weight 必须位于 [0, 1] 区间。")
    if not (0.0 <= th_min <= 1.0 and 0.0 <= th_max <= 1.0):
        raise ValueError("th_min、th_max 必须位于 [0, 1] 区间。")
    if th_min > th_max:
        raise ValueError("th_min 不能大于 th_max。")

    predicted_labels, base_scores = _compute_best_scores(query_embeddings, prototypes)

    normalized_queries = _normalize_embeddings(query_embeddings)
    similarity = normalized_queries @ normalized_queries.T
    similarity.fill_diagonal_(0.0)
    current_top_k = min(top_k, max(1, similarity.shape[1] - 1))
    _, top_indices = similarity.topk(current_top_k, dim=1)
    neighbor_mean_scores = base_scores[top_indices].mean(dim=1)

    final_scores = (base_weight * base_scores) + ((1.0 - base_weight) * neighbor_mean_scores)

    neighbor_argmax_labels = predicted_labels[top_indices]
    agree_frac = (neighbor_argmax_labels == predicted_labels.unsqueeze(1)).float().mean(dim=1)
    effective_threshold = th_max - (th_max - th_min) * agree_frac

    predictions = predicted_labels.clone()
    predictions[final_scores < effective_threshold] = other_label
    return predictions, base_scores, neighbor_mean_scores, final_scores, effective_threshold


def compute_subcenter_prototypes(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    prototype_labels: list[int],
    k_subcenters: int,
    random_state: int = 42,
) -> dict[int, torch.Tensor]:
    """Per-class KMeans on L2-normalized embeddings; each class has K_eff <= k_subcenters centers (L2-normalized rows)."""
    if k_subcenters < 1:
        raise ValueError("k_subcenters 至少为 1。")
    if not prototype_labels:
        raise ValueError("prototype_labels 不能为空。")

    normalized_embeddings = _normalize_embeddings(embeddings)
    subcenters: dict[int, torch.Tensor] = {}
    for label in prototype_labels:
        class_mask = labels == label
        class_emb = normalized_embeddings[class_mask]
        if class_emb.numel() == 0:
            raise ValueError(f"标签 {label} 没有可用于构建 subcenter 的样本。")
        n = class_emb.shape[0]
        k_eff = min(k_subcenters, n)
        if k_eff == 1:
            center = class_emb.mean(dim=0, keepdim=True)
            subcenters[label] = F.normalize(center, p=2, dim=1)
        else:
            kmeans = KMeans(
                n_clusters=k_eff,
                random_state=random_state,
                n_init=10,
            )
            kmeans.fit(class_emb.detach().cpu().numpy())
            centers = torch.as_tensor(
                kmeans.cluster_centers_,
                dtype=class_emb.dtype,
                device=class_emb.device,
            )
            subcenters[label] = _normalize_embeddings(centers)
    return subcenters


def _compute_best_scores_subcenters(
    query_embeddings: torch.Tensor,
    subcenter_prototypes: dict[int, torch.Tensor],
    prototype_labels: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not prototype_labels:
        raise ValueError("prototype_labels 不能为空。")

    normalized_queries = _normalize_embeddings(query_embeddings)
    scores_per_class = []
    for label in prototype_labels:
        centers = subcenter_prototypes[label]
        sim = normalized_queries @ centers.T
        max_sim, _ = sim.max(dim=1)
        scores_per_class.append(max_sim)
    score_matrix = torch.stack(scores_per_class, dim=1)
    best_scores, best_indices = score_matrix.max(dim=1)
    predicted_labels = torch.tensor(
        [prototype_labels[index] for index in best_indices.tolist()],
        device=query_embeddings.device,
        dtype=torch.long,
    )
    return predicted_labels, best_scores


def neighborhood_aware_subcenter_predictions(
    query_embeddings: torch.Tensor,
    subcenter_prototypes: dict[int, torch.Tensor],
    prototype_labels: list[int],
    other_label: int,
    threshold: float,
    top_k: int,
    base_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """与 neighborhood_aware_predictions 相同，但 prototype 为每类多子中心（取 max similarity）。"""
    if top_k < 1:
        raise ValueError("top_k 至少为 1。")
    if not (0.0 <= base_weight <= 1.0):
        raise ValueError("base_weight 必须位于 [0, 1] 区间。")

    predicted_labels, base_scores = _compute_best_scores_subcenters(
        query_embeddings,
        subcenter_prototypes,
        prototype_labels,
    )

    normalized_queries = _normalize_embeddings(query_embeddings)
    similarity = normalized_queries @ normalized_queries.T
    similarity.fill_diagonal_(0.0)
    current_top_k = min(top_k, max(1, similarity.shape[1] - 1))
    _, top_indices = similarity.topk(current_top_k, dim=1)
    neighbor_mean_scores = base_scores[top_indices].mean(dim=1)

    final_scores = (base_weight * base_scores) + ((1.0 - base_weight) * neighbor_mean_scores)
    predictions = predicted_labels.clone()
    predictions[final_scores < threshold] = other_label
    return predictions, base_scores, neighbor_mean_scores, final_scores


def cross_model_disagreement_resolve(
    pred_primary: torch.Tensor,
    score_primary: torch.Tensor,
    pred_secondary: torch.Tensor,
    score_secondary: torch.Tensor,
    other_label: int,
    secondary_confirm_margin: float = 0.05,
) -> torch.Tensor:
    """主模型（通常为 ViT）优先；仅当主模型判 other 而次模型判目标类且次模型分数明显更高时，采纳次模型。"""
    out = pred_primary.clone()
    mask = (pred_primary == other_label) & (pred_secondary != other_label)
    adopt = mask & (score_secondary > score_primary + secondary_confirm_margin)
    out[adopt] = pred_secondary[adopt]
    return out


def neighborhood_score_fusion_predictions(
    pred_primary_argmax: torch.Tensor,
    final_score_primary: torch.Tensor,
    final_score_secondary: torch.Tensor,
    other_label: int,
    alpha: float,
    threshold: float,
) -> torch.Tensor:
    """对两路 neighborhood_aware 的标量 ``final_score`` 做线性融合，再按阈值做 open-set。

    ``s_fused = alpha * s_primary + (1 - alpha) * s_secondary``；若 ``s_fused >= threshold`` 则保留
    **主模型（通常为 ViT）的 argmax 类别**，否则判为 ``other_label``。

    注意：``pred_primary_argmax`` 应来自 ``neighborhood_aware_predictions`` 在极低阈值下得到的类别
    （即 prototype 上的 argmax，不经 open-set 拒识），与 ``final_score_*`` 同一次前向一致。
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha 必须位于 [0, 1] 区间。")
    s_fused = alpha * final_score_primary + (1.0 - alpha) * final_score_secondary
    pred = pred_primary_argmax.clone()
    pred[s_fused < threshold] = other_label
    return pred


def transductive_cluster_refinement(
    test_embeddings: torch.Tensor,
    base_predictions: torch.Tensor,
    final_scores: torch.Tensor,
    top_k_connect: int,
    sim_connect_threshold: float,
    min_cluster_size: int,
    weighted_vote_fraction: float,
) -> torch.Tensor:
    """在 test 集上按 cosine 相似度建 kNN 边，求连通分量；足够大且加权票达阈值的簇统一为多数类。"""
    n = test_embeddings.shape[0]
    if n < 2:
        return base_predictions.clone()

    normalized = _normalize_embeddings(test_embeddings)
    sim = normalized @ normalized.T
    sim.fill_diagonal_(0.0)
    k_eff = min(top_k_connect, max(1, n - 1))
    topk_values, topk_indices = sim.topk(k_eff, dim=1)

    parent = list(range(n))

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(k_eff):
            if topk_values[i, j] < sim_connect_threshold:
                continue
            jj = int(topk_indices[i, j].item())
            if sim[i, jj] >= sim_connect_threshold:
                union(i, jj)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        r = find(i)
        clusters.setdefault(r, []).append(i)

    out = base_predictions.clone()
    for members in clusters.values():
        if len(members) < min_cluster_size:
            continue
        label_weights: dict[int, float] = {}
        for idx in members:
            lab = int(base_predictions[idx].item())
            label_weights[lab] = label_weights.get(lab, 0.0) + float(final_scores[idx].item())
        total_w = sum(label_weights.values())
        if total_w <= 0:
            continue
        best_label = max(label_weights, key=lambda k: label_weights[k])
        if label_weights[best_label] / total_w >= weighted_vote_fraction:
            for idx in members:
                out[idx] = best_label
    return out


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


def predict_open_set_with_quality_aware_scorer(
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
    quality_alpha: float,
    low_quality_threshold_boost: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if target_top_k < 1:
        raise ValueError("target_top_k 至少为 1。")
    if other_top_k < 1:
        raise ValueError("other_top_k 至少为 1。")
    if not target_labels:
        raise ValueError("target_labels 不能为空。")

    gallery_norms = torch.linalg.norm(gallery_embeddings, dim=1)
    gallery_quality = _normalize_quality_from_norms(gallery_norms)
    gallery_weights = 1.0 + quality_alpha * gallery_quality

    query_norms = torch.linalg.norm(query_embeddings, dim=1)
    query_quality = _normalize_quality_from_norms(
        torch.cat([gallery_norms, query_norms], dim=0)
    )[gallery_norms.shape[0] :]
    effective_thresholds = threshold + low_quality_threshold_boost * (1.0 - query_quality)

    target_scores_by_label: dict[int, torch.Tensor] = {}
    for label in target_labels:
        class_mask = gallery_labels == label
        class_gallery = gallery_embeddings[class_mask]
        class_weights = gallery_weights[class_mask]
        if class_gallery.numel() == 0:
            raise ValueError(f"类别 {label} 没有可用 gallery。")
        target_scores_by_label[label] = _weighted_topk_similarity(
            query_embeddings,
            class_gallery,
            class_weights,
            target_top_k,
        )

    other_mask = gallery_labels == other_label
    other_gallery = gallery_embeddings[other_mask]
    other_weights = gallery_weights[other_mask]
    if other_gallery.numel() == 0:
        raise ValueError("other 类没有可用 gallery。")
    other_scores = _weighted_topk_similarity(
        query_embeddings,
        other_gallery,
        other_weights,
        other_top_k,
    )

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
        (best_target_scores < effective_thresholds)
        | ((best_target_scores - other_scores) < other_margin)
        | ((best_target_scores - second_target_scores) < target_margin)
    )
    predictions[reject_mask] = other_label
    return predictions, best_target_scores, other_scores, second_target_scores, effective_thresholds


def predict_open_set_with_verifiers(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
    target_labels: list[int],
    other_label: int,
    target_top_k: int,
    negative_top_k: int,
    thresholds_by_class: dict[int, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    if target_top_k < 1:
        raise ValueError("target_top_k 至少为 1。")
    if negative_top_k < 1:
        raise ValueError("negative_top_k 至少为 1。")
    if not target_labels:
        raise ValueError("target_labels 不能为空。")

    verifier_scores: list[torch.Tensor] = []
    ordered_labels = list(target_labels)
    for label in ordered_labels:
        if label not in thresholds_by_class:
            raise ValueError(f"类别 {label} 缺少对应阈值。")
        target_gallery = gallery_embeddings[gallery_labels == label]
        negative_gallery = gallery_embeddings[gallery_labels != label]
        if target_gallery.numel() == 0:
            raise ValueError(f"类别 {label} 没有可用 gallery。")
        if negative_gallery.numel() == 0:
            raise ValueError(f"类别 {label} 没有可用 negative gallery。")

        target_scores = _mean_topk_similarity(query_embeddings, target_gallery, target_top_k)
        negative_scores = _mean_topk_similarity(query_embeddings, negative_gallery, negative_top_k)
        verifier_scores.append(target_scores - negative_scores)

    verifier_score_matrix = torch.stack(verifier_scores, dim=1)
    best_scores, best_indices = verifier_score_matrix.max(dim=1)
    predicted_targets = torch.tensor(
        [ordered_labels[index] for index in best_indices.tolist()],
        device=query_embeddings.device,
        dtype=torch.long,
    )
    predictions = torch.full_like(predicted_targets, other_label)
    for row_index, predicted_label in enumerate(predicted_targets.tolist()):
        if best_scores[row_index].item() >= thresholds_by_class[predicted_label]:
            predictions[row_index] = predicted_label
    return predictions, verifier_score_matrix


def conservative_graph_refine_predictions(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
    base_predictions: torch.Tensor,
    base_scores: torch.Tensor,
    target_labels: list[int],
    other_label: int,
    threshold: float,
    low_margin_delta: float,
    top_k: int,
    accept_consensus: float,
    reject_consensus: float,
    pseudo_accept_margin: float,
    pseudo_reject_margin: float,
    min_anchor_votes: int,
) -> torch.Tensor:
    if top_k < 1:
        raise ValueError("top_k 至少为 1。")
    if low_margin_delta < 0:
        raise ValueError("low_margin_delta 不能为负数。")
    if min_anchor_votes < 1:
        raise ValueError("min_anchor_votes 至少为 1。")
    if not target_labels:
        raise ValueError("target_labels 不能为空。")

    normalized_gallery = _normalize_embeddings(gallery_embeddings)
    normalized_queries = _normalize_embeddings(query_embeddings)
    refined_predictions = base_predictions.clone()
    target_label_set = set(target_labels)

    confident_accept_mask = (base_predictions != other_label) & (
        base_scores >= (threshold + pseudo_accept_margin)
    )
    confident_reject_mask = (base_predictions == other_label) & (
        base_scores <= (threshold - pseudo_reject_margin)
    )
    confident_mask = confident_accept_mask | confident_reject_mask

    pseudo_embeddings = normalized_queries[confident_mask]
    pseudo_labels = base_predictions[confident_mask]
    pseudo_query_indices = torch.arange(
        query_embeddings.shape[0],
        device=query_embeddings.device,
        dtype=torch.long,
    )[confident_mask]

    for query_index in range(query_embeddings.shape[0]):
        if abs(base_scores[query_index].item() - threshold) > low_margin_delta:
            continue

        anchor_embeddings = [normalized_gallery]
        anchor_labels = [gallery_labels]
        anchor_sources = [torch.zeros(gallery_labels.shape[0], dtype=torch.bool, device=gallery_labels.device)]

        if pseudo_embeddings.numel() > 0:
            include_mask = pseudo_query_indices != query_index
            if include_mask.any():
                anchor_embeddings.append(pseudo_embeddings[include_mask])
                anchor_labels.append(pseudo_labels[include_mask])
                anchor_sources.append(
                    torch.ones(int(include_mask.sum().item()), dtype=torch.bool, device=gallery_labels.device)
                )

        current_anchor_embeddings = torch.cat(anchor_embeddings, dim=0)
        current_anchor_labels = torch.cat(anchor_labels, dim=0)
        current_anchor_is_pseudo = torch.cat(anchor_sources, dim=0)

        similarities = normalized_queries[query_index : query_index + 1] @ current_anchor_embeddings.T
        positive_similarities = similarities.squeeze(0).clamp_min(0.0)
        if torch.allclose(positive_similarities, torch.zeros_like(positive_similarities)):
            continue

        current_top_k = min(top_k, positive_similarities.shape[0])
        topk_values, topk_indices = positive_similarities.topk(current_top_k)
        positive_neighbor_mask = topk_values > 0
        if not positive_neighbor_mask.any():
            continue

        topk_indices = topk_indices[positive_neighbor_mask]
        topk_values = topk_values[positive_neighbor_mask]
        topk_labels = current_anchor_labels[topk_indices]
        topk_is_pseudo = current_anchor_is_pseudo[topk_indices]
        total_vote = topk_values.sum().item()
        if total_vote <= 0:
            continue

        vote_by_label: dict[int, float] = {}
        gallery_count_by_label: dict[int, int] = {}
        for neighbor_label, neighbor_value, is_pseudo in zip(
            topk_labels.tolist(),
            topk_values.tolist(),
            topk_is_pseudo.tolist(),
        ):
            vote_by_label[neighbor_label] = vote_by_label.get(neighbor_label, 0.0) + float(neighbor_value)
            if not is_pseudo:
                gallery_count_by_label[neighbor_label] = gallery_count_by_label.get(neighbor_label, 0) + 1

        winning_label, winning_vote = max(vote_by_label.items(), key=lambda item: item[1])
        winning_ratio = winning_vote / total_vote
        current_prediction = int(base_predictions[query_index].item())

        if current_prediction == other_label:
            if (
                winning_label in target_label_set
                and winning_ratio >= accept_consensus
                and gallery_count_by_label.get(winning_label, 0) >= min_anchor_votes
            ):
                refined_predictions[query_index] = winning_label
            continue

        current_vote_ratio = vote_by_label.get(current_prediction, 0.0) / total_vote
        if current_vote_ratio < reject_consensus:
            refined_predictions[query_index] = other_label

    return refined_predictions


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


def select_best_neighborhood_threshold(
    val_embeddings: torch.Tensor,
    val_labels: torch.Tensor,
    prototypes: dict[int, torch.Tensor],
    other_label: int,
    threshold_values: list[float],
    top_k: int,
    base_weight: float,
) -> tuple[float, float, list[dict[str, float]]]:
    """在验证集上遍历 ``threshold_values``，按 neighborhood_aware 最终分数选最优阈值。"""
    if not threshold_values:
        raise ValueError("threshold_values 不能为空。")

    best_threshold = float(threshold_values[0])
    best_accuracy = -1.0
    details: list[dict[str, float]] = []
    for threshold in threshold_values:
        predictions, _, _, _ = neighborhood_aware_predictions(
            query_embeddings=val_embeddings,
            prototypes=prototypes,
            other_label=other_label,
            threshold=float(threshold),
            top_k=top_k,
            base_weight=base_weight,
        )
        accuracy = (predictions == val_labels).float().mean().item()
        details.append({"threshold": float(threshold), "val_accuracy": float(accuracy)})
        if accuracy > best_accuracy:
            best_threshold = float(threshold)
            best_accuracy = float(accuracy)
    return best_threshold, best_accuracy, details


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


def select_best_quality_aware_params_leave_one_out(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    target_labels: list[int],
    other_label: int,
    threshold_values: list[float],
    target_top_k_values: list[int],
    other_top_k_values: list[int],
    other_margin_values: list[float],
    target_margin_values: list[float],
    quality_alpha_values: list[float],
    low_quality_threshold_boost_values: list[float],
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
    if not quality_alpha_values:
        raise ValueError("quality_alpha_values 不能为空。")
    if not low_quality_threshold_boost_values:
        raise ValueError("low_quality_threshold_boost_values 不能为空。")

    best_params = {
        "threshold": threshold_values[0],
        "target_top_k": target_top_k_values[0],
        "other_top_k": other_top_k_values[0],
        "other_margin": other_margin_values[0],
        "target_margin": target_margin_values[0],
        "quality_alpha": quality_alpha_values[0],
        "low_quality_threshold_boost": low_quality_threshold_boost_values[0],
    }
    best_accuracy = -1.0

    for (
        threshold,
        target_top_k,
        other_top_k,
        other_margin,
        target_margin,
        quality_alpha,
        low_quality_threshold_boost,
    ) in product(
        threshold_values,
        target_top_k_values,
        other_top_k_values,
        other_margin_values,
        target_margin_values,
        quality_alpha_values,
        low_quality_threshold_boost_values,
    ):
        predictions: list[int] = []
        for query_index in range(labels.shape[0]):
            keep_mask = torch.ones(labels.shape[0], dtype=torch.bool, device=labels.device)
            keep_mask[query_index] = False
            fold_predictions, _, _, _, _ = predict_open_set_with_quality_aware_scorer(
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
                quality_alpha=quality_alpha,
                low_quality_threshold_boost=low_quality_threshold_boost,
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
                "quality_alpha": quality_alpha,
                "low_quality_threshold_boost": low_quality_threshold_boost,
            }
            best_accuracy = accuracy

    return best_params, best_accuracy


def select_best_verifier_thresholds_leave_one_out(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    target_labels: list[int],
    other_label: int,
    target_top_k_values: list[int],
    negative_top_k_values: list[int],
    threshold_values_by_class: dict[int, list[float]],
) -> tuple[dict[str, object], float]:
    if not target_top_k_values:
        raise ValueError("target_top_k_values 不能为空。")
    if not negative_top_k_values:
        raise ValueError("negative_top_k_values 不能为空。")
    if not threshold_values_by_class:
        raise ValueError("threshold_values_by_class 不能为空。")

    ordered_labels = list(target_labels)
    threshold_grids = [threshold_values_by_class[label] for label in ordered_labels]
    if any(not grid for grid in threshold_grids):
        raise ValueError("每个类别都必须提供至少一个候选阈值。")

    best_params: dict[str, object] = {
        "target_top_k": target_top_k_values[0],
        "negative_top_k": negative_top_k_values[0],
        "thresholds_by_class": {
            label: threshold_values_by_class[label][0] for label in ordered_labels
        },
    }
    best_accuracy = -1.0

    for target_top_k, negative_top_k in product(target_top_k_values, negative_top_k_values):
        for threshold_values in product(*threshold_grids):
            thresholds_by_class = {
                label: value for label, value in zip(ordered_labels, threshold_values)
            }
            predictions: list[int] = []
            for query_index in range(labels.shape[0]):
                keep_mask = torch.ones(labels.shape[0], dtype=torch.bool, device=labels.device)
                keep_mask[query_index] = False
                fold_predictions, _ = predict_open_set_with_verifiers(
                    query_embeddings=embeddings[query_index : query_index + 1],
                    gallery_embeddings=embeddings[keep_mask],
                    gallery_labels=labels[keep_mask],
                    target_labels=ordered_labels,
                    other_label=other_label,
                    target_top_k=target_top_k,
                    negative_top_k=negative_top_k,
                    thresholds_by_class=thresholds_by_class,
                )
                predictions.append(int(fold_predictions.item()))

            predictions_tensor = torch.tensor(predictions, device=labels.device, dtype=torch.long)
            accuracy = (predictions_tensor == labels).float().mean().item()
            if accuracy > best_accuracy:
                best_params = {
                    "target_top_k": target_top_k,
                    "negative_top_k": negative_top_k,
                    "thresholds_by_class": thresholds_by_class,
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
