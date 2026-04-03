from __future__ import annotations

from itertools import product

import numpy as np


def summarize_dual_verifier_scores(
    score_jesse: np.ndarray,
    score_mila: np.ndarray,
    theta_jesse: float,
    theta_mila: float,
) -> dict[str, np.ndarray]:
    score_jesse = np.asarray(score_jesse, dtype=np.float32)
    score_mila = np.asarray(score_mila, dtype=np.float32)
    if score_jesse.shape != score_mila.shape:
        raise ValueError("score_jesse 与 score_mila 形状必须一致。")

    excess_jesse = score_jesse - float(theta_jesse)
    excess_mila = score_mila - float(theta_mila)

    is_jesse = excess_jesse > 0.0
    is_mila = excess_mila > 0.0
    predictions = np.zeros_like(score_jesse, dtype=np.int64)

    only_jesse = is_jesse & ~is_mila
    only_mila = is_mila & ~is_jesse
    both_positive = is_jesse & is_mila

    predictions[only_jesse] = 1
    predictions[only_mila] = 2
    predictions[both_positive] = np.where(score_jesse[both_positive] >= score_mila[both_positive], 1, 2)

    selected_excess = np.maximum(excess_jesse, excess_mila)
    selected_excess = selected_excess.astype(np.float32, copy=False)
    best_excess = np.maximum(excess_jesse, excess_mila).astype(np.float32, copy=False)

    choose_jesse = predictions == 1
    choose_mila = predictions == 2
    selected_excess[choose_jesse] = excess_jesse[choose_jesse]
    selected_excess[choose_mila] = excess_mila[choose_mila]

    return {
        "predictions": predictions,
        "selected_excess": selected_excess,
        "best_excess": best_excess,
        "excess_jesse": excess_jesse.astype(np.float32, copy=False),
        "excess_mila": excess_mila.astype(np.float32, copy=False),
    }


def apply_gated_anchor_fusion(
    anchor_predictions: np.ndarray,
    anchor_scores: np.ndarray,
    secondary_predictions: np.ndarray,
    secondary_selected_excess: np.ndarray,
    secondary_best_excess: np.ndarray,
    other_label: int,
    low_conf_threshold: float,
    rescue_margin: float,
    swap_margin: float,
    veto_excess_threshold: float,
) -> np.ndarray:
    anchor_predictions = np.asarray(anchor_predictions, dtype=np.int64)
    anchor_scores = np.asarray(anchor_scores, dtype=np.float32)
    secondary_predictions = np.asarray(secondary_predictions, dtype=np.int64)
    secondary_selected_excess = np.asarray(secondary_selected_excess, dtype=np.float32)
    secondary_best_excess = np.asarray(secondary_best_excess, dtype=np.float32)

    n = anchor_predictions.shape[0]
    for array in (anchor_scores, secondary_predictions, secondary_selected_excess, secondary_best_excess):
        if array.shape[0] != n:
            raise ValueError("所有输入数组长度必须一致。")

    fused = anchor_predictions.copy()
    anchor_low_conf = anchor_scores < float(low_conf_threshold)
    secondary_target = secondary_predictions != int(other_label)
    secondary_other = secondary_predictions == int(other_label)

    rescue_mask = (
        (anchor_predictions == int(other_label))
        & secondary_target
        & (secondary_selected_excess >= float(rescue_margin))
    )
    fused[rescue_mask] = secondary_predictions[rescue_mask]

    swap_mask = (
        anchor_low_conf
        & (anchor_predictions != int(other_label))
        & secondary_target
        & (secondary_predictions != anchor_predictions)
        & (secondary_selected_excess >= float(swap_margin))
    )
    fused[swap_mask] = secondary_predictions[swap_mask]

    veto_mask = (
        anchor_low_conf
        & (anchor_predictions != int(other_label))
        & secondary_other
        & (secondary_best_excess <= float(veto_excess_threshold))
    )
    fused[veto_mask] = int(other_label)

    return fused


def select_best_gated_fusion_params(
    labels: np.ndarray,
    anchor_predictions: np.ndarray,
    anchor_scores: np.ndarray,
    secondary_predictions: np.ndarray,
    secondary_selected_excess: np.ndarray,
    secondary_best_excess: np.ndarray,
    other_label: int,
    low_conf_threshold_values: list[float],
    rescue_margin_values: list[float],
    swap_margin_values: list[float],
    veto_excess_threshold_values: list[float],
) -> tuple[dict[str, float], list[dict[str, float]]]:
    if not low_conf_threshold_values:
        raise ValueError("low_conf_threshold_values 不能为空。")
    if not rescue_margin_values:
        raise ValueError("rescue_margin_values 不能为空。")
    if not swap_margin_values:
        raise ValueError("swap_margin_values 不能为空。")
    if not veto_excess_threshold_values:
        raise ValueError("veto_excess_threshold_values 不能为空。")

    labels = np.asarray(labels, dtype=np.int64)
    anchor_predictions = np.asarray(anchor_predictions, dtype=np.int64)
    records: list[dict[str, float]] = []
    best_record: dict[str, float] | None = None

    for low_conf_threshold, rescue_margin, swap_margin, veto_excess_threshold in product(
        low_conf_threshold_values,
        rescue_margin_values,
        swap_margin_values,
        veto_excess_threshold_values,
    ):
        fused = apply_gated_anchor_fusion(
            anchor_predictions=anchor_predictions,
            anchor_scores=anchor_scores,
            secondary_predictions=secondary_predictions,
            secondary_selected_excess=secondary_selected_excess,
            secondary_best_excess=secondary_best_excess,
            other_label=other_label,
            low_conf_threshold=low_conf_threshold,
            rescue_margin=rescue_margin,
            swap_margin=swap_margin,
            veto_excess_threshold=veto_excess_threshold,
        )
        accuracy = float((fused == labels).mean())
        num_changes = int((fused != anchor_predictions).sum())
        record = {
            "low_conf_threshold": float(low_conf_threshold),
            "rescue_margin": float(rescue_margin),
            "swap_margin": float(swap_margin),
            "veto_excess_threshold": float(veto_excess_threshold),
            "accuracy": accuracy,
            "num_changes": num_changes,
        }
        records.append(record)
        if best_record is None:
            best_record = record
            continue
        if accuracy > best_record["accuracy"]:
            best_record = record
            continue
        if accuracy == best_record["accuracy"] and num_changes < int(best_record["num_changes"]):
            best_record = record

    return best_record or {}, records


def apply_rescue_only_fusion(
    anchor_predictions: np.ndarray,
    anchor_argmax: np.ndarray,
    anchor_scores: np.ndarray,
    secondary_predictions: np.ndarray,
    secondary_excess_jesse: np.ndarray,
    secondary_excess_mila: np.ndarray,
    other_label: int,
    low_conf_threshold: float,
    rescue_margin_jesse: float,
    rescue_margin_mila: float,
    require_anchor_argmax_match: bool,
) -> np.ndarray:
    anchor_predictions = np.asarray(anchor_predictions, dtype=np.int64)
    anchor_argmax = np.asarray(anchor_argmax, dtype=np.int64)
    anchor_scores = np.asarray(anchor_scores, dtype=np.float32)
    secondary_predictions = np.asarray(secondary_predictions, dtype=np.int64)
    secondary_excess_jesse = np.asarray(secondary_excess_jesse, dtype=np.float32)
    secondary_excess_mila = np.asarray(secondary_excess_mila, dtype=np.float32)

    n = anchor_predictions.shape[0]
    for array in (
        anchor_argmax,
        anchor_scores,
        secondary_predictions,
        secondary_excess_jesse,
        secondary_excess_mila,
    ):
        if array.shape[0] != n:
            raise ValueError("所有输入数组长度必须一致。")

    fused = anchor_predictions.copy()
    rescue_base = (anchor_predictions == int(other_label)) & (anchor_scores < float(low_conf_threshold))

    rescue_jesse = rescue_base & (secondary_predictions == 1) & (secondary_excess_jesse >= float(rescue_margin_jesse))
    rescue_mila = rescue_base & (secondary_predictions == 2) & (secondary_excess_mila >= float(rescue_margin_mila))

    if require_anchor_argmax_match:
        rescue_jesse &= anchor_argmax == 1
        rescue_mila &= anchor_argmax == 2

    fused[rescue_jesse] = 1
    fused[rescue_mila] = 2
    return fused


def select_best_rescue_only_params(
    labels: np.ndarray,
    anchor_predictions: np.ndarray,
    anchor_argmax: np.ndarray,
    anchor_scores: np.ndarray,
    secondary_predictions: np.ndarray,
    secondary_excess_jesse: np.ndarray,
    secondary_excess_mila: np.ndarray,
    other_label: int,
    low_conf_threshold_values: list[float],
    rescue_margin_jesse_values: list[float],
    rescue_margin_mila_values: list[float],
    require_anchor_argmax_match_options: list[bool],
) -> tuple[dict[str, float | bool], list[dict[str, float | bool]]]:
    if not low_conf_threshold_values:
        raise ValueError("low_conf_threshold_values 不能为空。")
    if not rescue_margin_jesse_values:
        raise ValueError("rescue_margin_jesse_values 不能为空。")
    if not rescue_margin_mila_values:
        raise ValueError("rescue_margin_mila_values 不能为空。")
    if not require_anchor_argmax_match_options:
        raise ValueError("require_anchor_argmax_match_options 不能为空。")

    labels = np.asarray(labels, dtype=np.int64)
    anchor_predictions = np.asarray(anchor_predictions, dtype=np.int64)
    records: list[dict[str, float | bool]] = []
    best_record: dict[str, float | bool] | None = None

    for low_conf_threshold, rescue_margin_jesse, rescue_margin_mila, require_anchor_argmax_match in product(
        low_conf_threshold_values,
        rescue_margin_jesse_values,
        rescue_margin_mila_values,
        require_anchor_argmax_match_options,
    ):
        fused = apply_rescue_only_fusion(
            anchor_predictions=anchor_predictions,
            anchor_argmax=anchor_argmax,
            anchor_scores=anchor_scores,
            secondary_predictions=secondary_predictions,
            secondary_excess_jesse=secondary_excess_jesse,
            secondary_excess_mila=secondary_excess_mila,
            other_label=other_label,
            low_conf_threshold=low_conf_threshold,
            rescue_margin_jesse=rescue_margin_jesse,
            rescue_margin_mila=rescue_margin_mila,
            require_anchor_argmax_match=require_anchor_argmax_match,
        )
        accuracy = float((fused == labels).mean())
        num_changes = int((fused != anchor_predictions).sum())
        record: dict[str, float | bool] = {
            "low_conf_threshold": float(low_conf_threshold),
            "rescue_margin_jesse": float(rescue_margin_jesse),
            "rescue_margin_mila": float(rescue_margin_mila),
            "require_anchor_argmax_match": bool(require_anchor_argmax_match),
            "accuracy": accuracy,
            "num_changes": num_changes,
        }
        records.append(record)
        if best_record is None:
            best_record = record
            continue
        if accuracy > float(best_record["accuracy"]):
            best_record = record
            continue
        if accuracy == float(best_record["accuracy"]) and num_changes < int(best_record["num_changes"]):
            best_record = record

    return best_record or {}, records
