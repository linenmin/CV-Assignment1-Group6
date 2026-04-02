from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import lightning as L
import pandas as pd
import torch
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_pipeline.common.config import load_experiment_config
from dl_pipeline.common.paths import project_path
from dl_pipeline.common.registry import append_registry_row
from dl_pipeline.data.datamodule import FaceDataModule
from dl_pipeline.inference.prototype import (
    average_normalized_embedding_sets,
    combine_embedding_sets,
    conservative_graph_refine_predictions,
    compute_class_prototypes,
    predict_open_set_with_lookalikes,
    predict_open_set_with_lookalike_margin_rejection,
    predict_open_set_with_quality_aware_scorer,
    predict_open_set_with_richer_scorer,
    predict_open_set_with_verifiers,
    predict_open_set_with_exemplars,
    predict_open_set,
    predict_open_set_with_class_thresholds,
    global_label_spread_predictions,
    neighborhood_aware_predictions,
    pca_whiten_embedding_sets,
    select_best_quality_aware_params_leave_one_out,
    select_best_richer_scorer_params_leave_one_out,
    select_best_verifier_thresholds_leave_one_out,
    select_best_exemplar_params,
    select_best_class_thresholds,
    select_best_threshold_crossval,
    select_best_threshold,
    spectral_cluster_with_gallery_label_matching,
)
from dl_pipeline.inference.submission import build_submission_dataframe, save_submission_dataframe
from dl_pipeline.inference.tta import extract_tta_features
from dl_pipeline.training.progress import AsciiTQDMProgressBar
from dl_pipeline.training.lightning_module import FaceClassifierModule


def _collect_embeddings(model, dataloader, device, use_horizontal_flip_tta: bool = False):
    embeddings = []
    values = []
    model.eval()
    with torch.no_grad():
        for images, batch_values in dataloader:
            features = extract_tta_features(
                model=model,
                images=images,
                device=device,
                use_horizontal_flip=use_horizontal_flip_tta,
            )
            embeddings.append(features)
            values.append(batch_values.detach().cpu())
    return torch.cat(embeddings, dim=0), torch.cat(values, dim=0)


def _load_model_from_checkpoint(config, checkpoint_path: str):
    train_df = pd.read_csv(project_path(config["data"]["splits_dir"]) / "train.csv")
    loss_config = config.get("loss", {})
    return FaceClassifierModule.load_from_checkpoint(
        checkpoint_path,
        model_family=config["model"]["family"],
        backbone_name=config["model"]["backbone_name"],
        num_classes=int(train_df["class"].nunique()),
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
        learning_rate=config["train"]["learning_rate"],
        backbone_learning_rate=config["train"].get("backbone_learning_rate"),
        weight_decay=config["train"]["weight_decay"],
        scheduler_name=config["train"]["scheduler"],
        max_epochs=config["train"]["max_epochs"],
        pretrained_repo_id=config["model"].get("pretrained_repo_id"),
        freeze_backbone=config["model"].get("freeze_backbone", False),
        unfreeze_last_stage=config["model"].get("unfreeze_last_stage", False),
        unfreeze_stage_count=config["model"].get("unfreeze_stage_count", 0),
        unfreeze_cvlface_norm=config["model"].get("unfreeze_cvlface_norm", True),
        unfreeze_cvlface_feature=config["model"].get("unfreeze_cvlface_feature", True),
        loss_name=loss_config.get("name", "cross_entropy"),
        loss_target_labels=loss_config.get("target_labels"),
        arcface_scale=loss_config.get("arcface_scale", 30.0),
        arcface_margin=loss_config.get("arcface_margin", 0.5),
    )


def _run_prototype_inference(config, datamodule, model, test_df, output_root: Path):
    inference_config = config.get("inference", {})
    prototype_labels = inference_config.get("prototype_labels", [1, 2])
    other_label = inference_config.get("other_label", 0)
    threshold_values = inference_config.get("threshold_values", [0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    threshold_values_by_class = inference_config.get("threshold_values_by_class")
    threshold_selection = inference_config.get("threshold_selection", "val")
    crossval_splits = inference_config.get("crossval_splits", 4)
    crossval_random_state = inference_config.get("crossval_random_state", 42)
    gallery_source = inference_config.get("gallery_source", "train")
    use_horizontal_flip_tta = inference_config.get("tta_horizontal_flip", False)

    device = torch.device("cuda" if torch.cuda.is_available() and config["train"]["accelerator"] != "cpu" else "cpu")
    model = model.to(device)

    train_embeddings, train_labels = _collect_embeddings(
        model,
        datamodule.train_dataloader(),
        device,
        use_horizontal_flip_tta=use_horizontal_flip_tta,
    )
    val_embeddings, val_labels = _collect_embeddings(
        model,
        datamodule.val_dataloader(),
        device,
        use_horizontal_flip_tta=use_horizontal_flip_tta,
    )
    test_embeddings, test_ids = _collect_embeddings(
        model,
        datamodule.predict_dataloader(),
        device,
        use_horizontal_flip_tta=use_horizontal_flip_tta,
    )
    gallery_embeddings, gallery_labels = train_embeddings, train_labels
    if gallery_source == "all_labeled":
        gallery_embeddings, gallery_labels = combine_embedding_sets(
            [
                (train_embeddings, train_labels),
                (val_embeddings, val_labels),
            ]
        )

    prototypes = compute_class_prototypes(gallery_embeddings, gallery_labels, prototype_labels=prototype_labels)
    if threshold_selection == "crossval_global":
        threshold, cv_mean_accuracy = select_best_threshold_crossval(
            embeddings=gallery_embeddings,
            labels=gallery_labels,
            prototype_labels=prototype_labels,
            other_label=other_label,
            threshold_values=threshold_values,
            n_splits=crossval_splits,
            random_state=crossval_random_state,
        )
        val_predictions, _ = predict_open_set(
            val_embeddings,
            prototypes=prototypes,
            other_label=other_label,
            threshold=threshold,
        )
        val_accuracy = (val_predictions == val_labels).float().mean().item()
        test_predictions, _ = predict_open_set(
            test_embeddings,
            prototypes=prototypes,
            other_label=other_label,
            threshold=threshold,
        )
        prototype_metrics = {
            "mode": "prototype",
            "threshold_mode": "global_crossval",
            "prototype_labels": prototype_labels,
            "other_label": other_label,
            "gallery_source": gallery_source,
            "tta_horizontal_flip": use_horizontal_flip_tta,
            "selected_threshold": threshold,
            "crossval_mean_accuracy": cv_mean_accuracy,
            "val_accuracy": val_accuracy,
            "crossval_splits": crossval_splits,
            "crossval_random_state": crossval_random_state,
        }
    elif threshold_values_by_class:
        normalized_threshold_values_by_class = {
            int(label): values for label, values in threshold_values_by_class.items()
        }
        thresholds_by_class, val_accuracy = select_best_class_thresholds(
            val_embeddings=val_embeddings,
            val_labels=val_labels,
            prototypes=prototypes,
            other_label=other_label,
            threshold_values_by_class=normalized_threshold_values_by_class,
        )
        test_predictions, _ = predict_open_set_with_class_thresholds(
            test_embeddings,
            prototypes=prototypes,
            other_label=other_label,
            thresholds_by_class=thresholds_by_class,
        )
        prototype_metrics = {
            "mode": "prototype",
            "threshold_mode": "class_specific",
            "prototype_labels": prototype_labels,
            "other_label": other_label,
            "gallery_source": gallery_source,
            "tta_horizontal_flip": use_horizontal_flip_tta,
            "selected_thresholds_by_class": thresholds_by_class,
            "val_accuracy": val_accuracy,
        }
    else:
        threshold, val_accuracy = select_best_threshold(
            val_embeddings=val_embeddings,
            val_labels=val_labels,
            prototypes=prototypes,
            other_label=other_label,
            threshold_values=threshold_values,
        )
        test_predictions, _ = predict_open_set(
            test_embeddings,
            prototypes=prototypes,
            other_label=other_label,
            threshold=threshold,
        )
        prototype_metrics = {
            "mode": "prototype",
            "threshold_mode": "global",
            "prototype_labels": prototype_labels,
            "other_label": other_label,
            "gallery_source": gallery_source,
            "tta_horizontal_flip": use_horizontal_flip_tta,
            "selected_threshold": threshold,
            "val_accuracy": val_accuracy,
        }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "prototype_metrics.json").write_text(
        json.dumps(prototype_metrics, indent=2),
        encoding="utf-8",
    )

    id_to_prediction = {int(sample_id): int(pred) for sample_id, pred in zip(test_ids.tolist(), test_predictions.tolist())}
    return [id_to_prediction[int(sample_id)] for sample_id in test_df["id"].tolist()]


def _run_ensemble_prototype_inference(config, datamodule, test_df, output_root: Path):
    inference_config = config.get("inference", {})
    prototype_labels = inference_config.get("prototype_labels", [1, 2])
    other_label = inference_config.get("other_label", 0)
    threshold = inference_config.get("threshold", 0.55)
    checkpoint_paths = inference_config.get("checkpoint_paths", [])
    if not checkpoint_paths:
        raise ValueError("ensemble_prototype 模式必须提供 checkpoint_paths。")

    device = torch.device("cuda" if torch.cuda.is_available() and config["train"]["accelerator"] != "cpu" else "cpu")

    train_embedding_sets: list[torch.Tensor] = []
    val_embedding_sets: list[torch.Tensor] = []
    test_embedding_sets: list[torch.Tensor] = []
    train_labels = val_labels = test_ids = None

    for checkpoint_path in checkpoint_paths:
        model = _load_model_from_checkpoint(config, checkpoint_path).to(device)
        current_train_embeddings, current_train_labels = _collect_embeddings(model, datamodule.train_dataloader(), device)
        current_val_embeddings, current_val_labels = _collect_embeddings(model, datamodule.val_dataloader(), device)
        current_test_embeddings, current_test_ids = _collect_embeddings(model, datamodule.predict_dataloader(), device)
        train_embedding_sets.append(current_train_embeddings)
        val_embedding_sets.append(current_val_embeddings)
        test_embedding_sets.append(current_test_embeddings)
        if train_labels is None:
            train_labels = current_train_labels
            val_labels = current_val_labels
            test_ids = current_test_ids

    train_embeddings = average_normalized_embedding_sets(train_embedding_sets)
    val_embeddings = average_normalized_embedding_sets(val_embedding_sets)
    test_embeddings = average_normalized_embedding_sets(test_embedding_sets)

    prototypes = compute_class_prototypes(train_embeddings, train_labels, prototype_labels=prototype_labels)
    val_predictions, _ = predict_open_set(
        val_embeddings,
        prototypes=prototypes,
        other_label=other_label,
        threshold=threshold,
    )
    val_accuracy = (val_predictions == val_labels).float().mean().item()
    test_predictions, _ = predict_open_set(
        test_embeddings,
        prototypes=prototypes,
        other_label=other_label,
        threshold=threshold,
    )

    ensemble_metrics = {
        "mode": "ensemble_prototype",
        "prototype_labels": prototype_labels,
        "other_label": other_label,
        "selected_threshold": threshold,
        "checkpoint_paths": checkpoint_paths,
        "num_models": len(checkpoint_paths),
        "val_accuracy": val_accuracy,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "prototype_metrics.json").write_text(
        json.dumps(ensemble_metrics, indent=2),
        encoding="utf-8",
    )

    id_to_prediction = {
        int(sample_id): int(pred)
        for sample_id, pred in zip(test_ids.tolist(), test_predictions.tolist())
    }
    return [id_to_prediction[int(sample_id)] for sample_id in test_df["id"].tolist()]


def _run_conservative_graph_refine_inference(config, datamodule, model, test_df, output_root: Path):
    inference_config = config.get("inference", {})
    prototype_labels = inference_config.get("prototype_labels", [1, 2])
    other_label = inference_config.get("other_label", 0)
    threshold_values = inference_config.get("threshold_values", [0.55])
    use_horizontal_flip_tta = inference_config.get("tta_horizontal_flip", False)
    graph_config = inference_config.get("graph_refine", {})

    device = torch.device("cuda" if torch.cuda.is_available() and config["train"]["accelerator"] != "cpu" else "cpu")
    model = model.to(device)

    train_embeddings, train_labels = _collect_embeddings(
        model,
        datamodule.train_dataloader(),
        device,
        use_horizontal_flip_tta=use_horizontal_flip_tta,
    )
    val_embeddings, val_labels = _collect_embeddings(
        model,
        datamodule.val_dataloader(),
        device,
        use_horizontal_flip_tta=use_horizontal_flip_tta,
    )
    test_embeddings, test_ids = _collect_embeddings(
        model,
        datamodule.predict_dataloader(),
        device,
        use_horizontal_flip_tta=use_horizontal_flip_tta,
    )

    prototypes = compute_class_prototypes(train_embeddings, train_labels, prototype_labels=prototype_labels)
    threshold, base_val_accuracy = select_best_threshold(
        val_embeddings=val_embeddings,
        val_labels=val_labels,
        prototypes=prototypes,
        other_label=other_label,
        threshold_values=threshold_values,
    )
    base_val_predictions, base_val_scores = predict_open_set(
        val_embeddings,
        prototypes=prototypes,
        other_label=other_label,
        threshold=threshold,
    )
    base_test_predictions, base_test_scores = predict_open_set(
        test_embeddings,
        prototypes=prototypes,
        other_label=other_label,
        threshold=threshold,
    )

    refined_val_predictions = conservative_graph_refine_predictions(
        query_embeddings=val_embeddings,
        gallery_embeddings=train_embeddings,
        gallery_labels=train_labels,
        base_predictions=base_val_predictions,
        base_scores=base_val_scores,
        target_labels=prototype_labels,
        other_label=other_label,
        threshold=threshold,
        low_margin_delta=graph_config.get("low_margin_delta", 0.05),
        top_k=graph_config.get("top_k", 7),
        accept_consensus=graph_config.get("accept_consensus", 0.85),
        reject_consensus=graph_config.get("reject_consensus", 0.60),
        pseudo_accept_margin=graph_config.get("pseudo_accept_margin", 0.05),
        pseudo_reject_margin=graph_config.get("pseudo_reject_margin", 0.05),
        min_anchor_votes=graph_config.get("min_anchor_votes", 2),
    )
    refined_test_predictions = conservative_graph_refine_predictions(
        query_embeddings=test_embeddings,
        gallery_embeddings=train_embeddings,
        gallery_labels=train_labels,
        base_predictions=base_test_predictions,
        base_scores=base_test_scores,
        target_labels=prototype_labels,
        other_label=other_label,
        threshold=threshold,
        low_margin_delta=graph_config.get("low_margin_delta", 0.05),
        top_k=graph_config.get("top_k", 7),
        accept_consensus=graph_config.get("accept_consensus", 0.85),
        reject_consensus=graph_config.get("reject_consensus", 0.60),
        pseudo_accept_margin=graph_config.get("pseudo_accept_margin", 0.05),
        pseudo_reject_margin=graph_config.get("pseudo_reject_margin", 0.05),
        min_anchor_votes=graph_config.get("min_anchor_votes", 2),
    )

    refined_val_accuracy = (refined_val_predictions == val_labels).float().mean().item()
    prediction_changes = int((refined_test_predictions != base_test_predictions).sum().item())

    prototype_metrics = {
        "mode": "conservative_graph_refine",
        "prototype_labels": prototype_labels,
        "other_label": other_label,
        "gallery_source": "train",
        "tta_horizontal_flip": use_horizontal_flip_tta,
        "selected_threshold": threshold,
        "base_val_accuracy": base_val_accuracy,
        "refined_val_accuracy": refined_val_accuracy,
        "graph_refine": {
            "low_margin_delta": graph_config.get("low_margin_delta", 0.05),
            "top_k": graph_config.get("top_k", 7),
            "accept_consensus": graph_config.get("accept_consensus", 0.85),
            "reject_consensus": graph_config.get("reject_consensus", 0.60),
            "pseudo_accept_margin": graph_config.get("pseudo_accept_margin", 0.05),
            "pseudo_reject_margin": graph_config.get("pseudo_reject_margin", 0.05),
            "min_anchor_votes": graph_config.get("min_anchor_votes", 2),
        },
        "num_changed_test_predictions": prediction_changes,
    }
    (output_root / "prototype_metrics.json").write_text(
        json.dumps(prototype_metrics, indent=2),
        encoding="utf-8",
    )

    id_to_prediction = {
        int(sample_id): int(pred)
        for sample_id, pred in zip(test_ids.tolist(), refined_test_predictions.tolist())
    }
    return [id_to_prediction[int(sample_id)] for sample_id in test_df["id"].tolist()]


def _run_exemplar_inference(config, datamodule, model, test_df, output_root: Path):
    inference_config = config.get("inference", {})
    target_labels = inference_config.get("target_labels", [1, 2])
    other_label = inference_config.get("other_label", 0)
    threshold = inference_config.get("threshold", 0.55)
    top_k_values = inference_config.get("top_k_values", [1, 3, 5])
    margin_values = inference_config.get("margin_values", [0.0, 0.02, 0.04, 0.06])
    gallery_source = inference_config.get("gallery_source", "train")

    device = torch.device("cuda" if torch.cuda.is_available() and config["train"]["accelerator"] != "cpu" else "cpu")
    model = model.to(device)

    train_embeddings, train_labels = _collect_embeddings(model, datamodule.train_dataloader(), device)
    val_embeddings, val_labels = _collect_embeddings(model, datamodule.val_dataloader(), device)
    test_embeddings, test_ids = _collect_embeddings(model, datamodule.predict_dataloader(), device)
    gallery_embeddings, gallery_labels = train_embeddings, train_labels
    if gallery_source == "all_labeled":
        gallery_embeddings, gallery_labels = combine_embedding_sets(
            [
                (train_embeddings, train_labels),
                (val_embeddings, val_labels),
            ]
        )

    best_params, val_accuracy = select_best_exemplar_params(
        val_embeddings=val_embeddings,
        val_labels=val_labels,
        gallery_embeddings=gallery_embeddings,
        gallery_labels=gallery_labels,
        target_labels=target_labels,
        other_label=other_label,
        threshold=threshold,
        top_k_values=top_k_values,
        margin_values=margin_values,
    )
    test_predictions, test_target_scores, test_other_scores = predict_open_set_with_exemplars(
        query_embeddings=test_embeddings,
        gallery_embeddings=gallery_embeddings,
        gallery_labels=gallery_labels,
        target_labels=target_labels,
        other_label=other_label,
        threshold=threshold,
        top_k=int(best_params["top_k"]),
        margin=float(best_params["margin"]),
    )

    exemplar_metrics = {
        "mode": "exemplar_knn",
        "target_labels": target_labels,
        "other_label": other_label,
        "gallery_source": gallery_source,
        "selected_threshold": threshold,
        "selected_top_k": int(best_params["top_k"]),
        "selected_margin": float(best_params["margin"]),
        "val_accuracy": val_accuracy,
        "mean_test_target_score": float(test_target_scores.mean().item()),
        "mean_test_other_score": float(test_other_scores.mean().item()),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "prototype_metrics.json").write_text(
        json.dumps(exemplar_metrics, indent=2),
        encoding="utf-8",
    )

    id_to_prediction = {
        int(sample_id): int(pred)
        for sample_id, pred in zip(test_ids.tolist(), test_predictions.tolist())
    }
    return [id_to_prediction[int(sample_id)] for sample_id in test_df["id"].tolist()]


def _run_richer_open_set_inference(config, datamodule, model, test_df, output_root: Path):
    inference_config = config.get("inference", {})
    target_labels = inference_config.get("target_labels", [1, 2])
    other_label = inference_config.get("other_label", 0)
    threshold_values = inference_config.get("threshold_values", [0.55])
    target_top_k_values = inference_config.get("target_top_k_values", [3, 5, 7])
    other_top_k_values = inference_config.get("other_top_k_values", [3, 5, 7])
    other_margin_values = inference_config.get("other_margin_values", [0.0, 0.02, 0.04])
    target_margin_values = inference_config.get("target_margin_values", [0.0, 0.02, 0.04])

    device = torch.device("cuda" if torch.cuda.is_available() and config["train"]["accelerator"] != "cpu" else "cpu")
    model = model.to(device)

    train_embeddings, train_labels = _collect_embeddings(model, datamodule.train_dataloader(), device)
    val_embeddings, val_labels = _collect_embeddings(model, datamodule.val_dataloader(), device)
    test_embeddings, test_ids = _collect_embeddings(model, datamodule.predict_dataloader(), device)
    labeled_embeddings, labeled_labels = combine_embedding_sets(
        [
            (train_embeddings, train_labels),
            (val_embeddings, val_labels),
        ]
    )

    best_params, loo_accuracy = select_best_richer_scorer_params_leave_one_out(
        embeddings=labeled_embeddings,
        labels=labeled_labels,
        target_labels=target_labels,
        other_label=other_label,
        threshold_values=threshold_values,
        target_top_k_values=target_top_k_values,
        other_top_k_values=other_top_k_values,
        other_margin_values=other_margin_values,
        target_margin_values=target_margin_values,
    )
    test_predictions, best_target_scores, other_scores, second_target_scores = (
        predict_open_set_with_richer_scorer(
            query_embeddings=test_embeddings,
            gallery_embeddings=labeled_embeddings,
            gallery_labels=labeled_labels,
            target_labels=target_labels,
            other_label=other_label,
            threshold=float(best_params["threshold"]),
            target_top_k=int(best_params["target_top_k"]),
            other_top_k=int(best_params["other_top_k"]),
            other_margin=float(best_params["other_margin"]),
            target_margin=float(best_params["target_margin"]),
        )
    )

    richer_metrics = {
        "mode": "richer_open_set",
        "target_labels": target_labels,
        "other_label": other_label,
        "gallery_source": "all_labeled",
        "selected_threshold": float(best_params["threshold"]),
        "selected_target_top_k": int(best_params["target_top_k"]),
        "selected_other_top_k": int(best_params["other_top_k"]),
        "selected_other_margin": float(best_params["other_margin"]),
        "selected_target_margin": float(best_params["target_margin"]),
        "leave_one_out_accuracy": loo_accuracy,
        "mean_test_best_target_score": float(best_target_scores.mean().item()),
        "mean_test_other_score": float(other_scores.mean().item()),
        "mean_test_second_target_score": float(second_target_scores.mean().item()),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "prototype_metrics.json").write_text(
        json.dumps(richer_metrics, indent=2),
        encoding="utf-8",
    )

    id_to_prediction = {
        int(sample_id): int(pred)
        for sample_id, pred in zip(test_ids.tolist(), test_predictions.tolist())
    }
    return [id_to_prediction[int(sample_id)] for sample_id in test_df["id"].tolist()]


def _run_quality_aware_open_set_inference(config, datamodule, model, test_df, output_root: Path):
    inference_config = config.get("inference", {})
    target_labels = inference_config.get("target_labels", [1, 2])
    other_label = inference_config.get("other_label", 0)
    threshold_values = inference_config.get("threshold_values", [0.55])
    target_top_k_values = inference_config.get("target_top_k_values", [3, 5])
    other_top_k_values = inference_config.get("other_top_k_values", [3, 5])
    other_margin_values = inference_config.get("other_margin_values", [0.02, 0.04, 0.06])
    target_margin_values = inference_config.get("target_margin_values", [0.0, 0.01, 0.02])
    quality_alpha_values = inference_config.get("quality_alpha_values", [0.0, 1.0, 2.0])
    low_quality_threshold_boost_values = inference_config.get(
        "low_quality_threshold_boost_values",
        [0.0, 0.02, 0.04],
    )

    device = torch.device("cuda" if torch.cuda.is_available() and config["train"]["accelerator"] != "cpu" else "cpu")
    model = model.to(device)

    train_embeddings, train_labels = _collect_embeddings(model, datamodule.train_dataloader(), device)
    val_embeddings, val_labels = _collect_embeddings(model, datamodule.val_dataloader(), device)
    test_embeddings, test_ids = _collect_embeddings(model, datamodule.predict_dataloader(), device)
    labeled_embeddings, labeled_labels = combine_embedding_sets(
        [
            (train_embeddings, train_labels),
            (val_embeddings, val_labels),
        ]
    )

    best_params, loo_accuracy = select_best_quality_aware_params_leave_one_out(
        embeddings=labeled_embeddings,
        labels=labeled_labels,
        target_labels=target_labels,
        other_label=other_label,
        threshold_values=threshold_values,
        target_top_k_values=target_top_k_values,
        other_top_k_values=other_top_k_values,
        other_margin_values=other_margin_values,
        target_margin_values=target_margin_values,
        quality_alpha_values=quality_alpha_values,
        low_quality_threshold_boost_values=low_quality_threshold_boost_values,
    )
    (
        test_predictions,
        best_target_scores,
        other_scores,
        second_target_scores,
        effective_thresholds,
    ) = predict_open_set_with_quality_aware_scorer(
        query_embeddings=test_embeddings,
        gallery_embeddings=labeled_embeddings,
        gallery_labels=labeled_labels,
        target_labels=target_labels,
        other_label=other_label,
        threshold=float(best_params["threshold"]),
        target_top_k=int(best_params["target_top_k"]),
        other_top_k=int(best_params["other_top_k"]),
        other_margin=float(best_params["other_margin"]),
        target_margin=float(best_params["target_margin"]),
        quality_alpha=float(best_params["quality_alpha"]),
        low_quality_threshold_boost=float(best_params["low_quality_threshold_boost"]),
    )

    quality_metrics = {
        "mode": "quality_aware_open_set",
        "target_labels": target_labels,
        "other_label": other_label,
        "gallery_source": "all_labeled",
        "selected_threshold": float(best_params["threshold"]),
        "selected_target_top_k": int(best_params["target_top_k"]),
        "selected_other_top_k": int(best_params["other_top_k"]),
        "selected_other_margin": float(best_params["other_margin"]),
        "selected_target_margin": float(best_params["target_margin"]),
        "selected_quality_alpha": float(best_params["quality_alpha"]),
        "selected_low_quality_threshold_boost": float(best_params["low_quality_threshold_boost"]),
        "leave_one_out_accuracy": loo_accuracy,
        "mean_test_best_target_score": float(best_target_scores.mean().item()),
        "mean_test_other_score": float(other_scores.mean().item()),
        "mean_test_second_target_score": float(second_target_scores.mean().item()),
        "mean_test_effective_threshold": float(effective_thresholds.mean().item()),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "prototype_metrics.json").write_text(
        json.dumps(quality_metrics, indent=2),
        encoding="utf-8",
    )

    id_to_prediction = {
        int(sample_id): int(pred)
        for sample_id, pred in zip(test_ids.tolist(), test_predictions.tolist())
    }
    return [id_to_prediction[int(sample_id)] for sample_id in test_df["id"].tolist()]


def _run_verifier_open_set_inference(config, datamodule, model, test_df, output_root: Path):
    inference_config = config.get("inference", {})
    target_labels = inference_config.get("target_labels", [1, 2])
    other_label = inference_config.get("other_label", 0)
    target_top_k_values = inference_config.get("target_top_k_values", [3, 5, 7])
    negative_top_k_values = inference_config.get("negative_top_k_values", [3, 5, 7])
    threshold_values_by_class = {
        int(label): values
        for label, values in inference_config.get("threshold_values_by_class", {}).items()
    }

    device = torch.device("cuda" if torch.cuda.is_available() and config["train"]["accelerator"] != "cpu" else "cpu")
    model = model.to(device)

    train_embeddings, train_labels = _collect_embeddings(model, datamodule.train_dataloader(), device)
    val_embeddings, val_labels = _collect_embeddings(model, datamodule.val_dataloader(), device)
    test_embeddings, test_ids = _collect_embeddings(model, datamodule.predict_dataloader(), device)
    labeled_embeddings, labeled_labels = combine_embedding_sets(
        [
            (train_embeddings, train_labels),
            (val_embeddings, val_labels),
        ]
    )

    best_params, loo_accuracy = select_best_verifier_thresholds_leave_one_out(
        embeddings=labeled_embeddings,
        labels=labeled_labels,
        target_labels=target_labels,
        other_label=other_label,
        target_top_k_values=target_top_k_values,
        negative_top_k_values=negative_top_k_values,
        threshold_values_by_class=threshold_values_by_class,
    )
    test_predictions, verifier_score_matrix = predict_open_set_with_verifiers(
        query_embeddings=test_embeddings,
        gallery_embeddings=labeled_embeddings,
        gallery_labels=labeled_labels,
        target_labels=target_labels,
        other_label=other_label,
        target_top_k=int(best_params["target_top_k"]),
        negative_top_k=int(best_params["negative_top_k"]),
        thresholds_by_class={int(label): float(value) for label, value in best_params["thresholds_by_class"].items()},
    )

    verifier_metrics = {
        "mode": "verifier_open_set",
        "target_labels": target_labels,
        "other_label": other_label,
        "gallery_source": "all_labeled",
        "selected_target_top_k": int(best_params["target_top_k"]),
        "selected_negative_top_k": int(best_params["negative_top_k"]),
        "selected_thresholds_by_class": {
            int(label): float(value) for label, value in best_params["thresholds_by_class"].items()
        },
        "leave_one_out_accuracy": loo_accuracy,
        "mean_test_verifier_score_by_class": {
            str(label): float(verifier_score_matrix[:, index].mean().item())
            for index, label in enumerate(target_labels)
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "prototype_metrics.json").write_text(
        json.dumps(verifier_metrics, indent=2),
        encoding="utf-8",
    )

    id_to_prediction = {
        int(sample_id): int(pred)
        for sample_id, pred in zip(test_ids.tolist(), test_predictions.tolist())
    }
    return [id_to_prediction[int(sample_id)] for sample_id in test_df["id"].tolist()]


def _run_lookalike_prototype_inference(config, datamodule, model, test_df, output_root: Path):
    inference_config = config.get("inference", {})
    target_labels = inference_config.get("target_labels", [1, 2])
    other_label = inference_config.get("other_label", 0)
    threshold = inference_config.get("threshold", 0.55)
    gallery_source = inference_config.get("gallery_source", "all_labeled")
    cluster_random_state = inference_config.get("cluster_random_state", 42)
    michael_like_label = inference_config.get("michael_like_label", 3)
    sarah_like_label = inference_config.get("sarah_like_label", 4)

    device = torch.device("cuda" if torch.cuda.is_available() and config["train"]["accelerator"] != "cpu" else "cpu")
    model = model.to(device)

    train_embeddings, train_labels = _collect_embeddings(model, datamodule.train_dataloader(), device)
    val_embeddings, val_labels = _collect_embeddings(model, datamodule.val_dataloader(), device)
    test_embeddings, test_ids = _collect_embeddings(model, datamodule.predict_dataloader(), device)
    gallery_embeddings, gallery_labels = train_embeddings, train_labels
    if gallery_source == "all_labeled":
        gallery_embeddings, gallery_labels = combine_embedding_sets(
            [
                (train_embeddings, train_labels),
                (val_embeddings, val_labels),
            ]
        )

    target_prototypes = compute_class_prototypes(
        gallery_embeddings,
        gallery_labels,
        prototype_labels=target_labels,
    )

    other_embeddings = gallery_embeddings[gallery_labels == other_label]
    if other_embeddings.shape[0] < 2:
        raise ValueError("lookalike_prototype 模式要求 other 类至少有 2 个样本。")
    normalized_other_embeddings = torch.nn.functional.normalize(other_embeddings, p=2, dim=1)
    cluster_ids = KMeans(
        n_clusters=2,
        random_state=cluster_random_state,
        n_init=50,
    ).fit_predict(normalized_other_embeddings.cpu().numpy())
    cluster_ids_tensor = torch.as_tensor(cluster_ids, dtype=torch.long)

    cluster_records = []
    for cluster_id in sorted(set(cluster_ids)):
        cluster_embeddings = normalized_other_embeddings[cluster_ids_tensor == cluster_id]
        cluster_prototype = torch.nn.functional.normalize(
            cluster_embeddings.mean(dim=0, keepdim=True),
            p=2,
            dim=1,
        ).squeeze(0)
        mean_sim_to_target = {
            int(label): float(torch.dot(cluster_prototype, target_prototypes[label]).item())
            for label in target_labels
        }
        cluster_records.append(
            {
                "cluster_id": int(cluster_id),
                "prototype": cluster_prototype,
                "mean_sim_to_target": mean_sim_to_target,
                "margin_target_1_minus_2": float(
                    mean_sim_to_target[target_labels[0]] - mean_sim_to_target[target_labels[1]]
                ),
                "count": int((cluster_ids_tensor == cluster_id).sum().item()),
            }
        )

    jesse_like_cluster = max(cluster_records, key=lambda item: item["margin_target_1_minus_2"])
    mila_like_cluster = min(cluster_records, key=lambda item: item["margin_target_1_minus_2"])
    prototypes = {
        int(target_labels[0]): target_prototypes[target_labels[0]],
        int(target_labels[1]): target_prototypes[target_labels[1]],
        int(michael_like_label): jesse_like_cluster["prototype"],
        int(sarah_like_label): mila_like_cluster["prototype"],
    }

    lookalike_labels = [int(michael_like_label), int(sarah_like_label)]
    val_predictions, val_nearest_labels, val_scores = predict_open_set_with_lookalikes(
        query_embeddings=val_embeddings,
        prototypes=prototypes,
        target_labels=target_labels,
        lookalike_labels=lookalike_labels,
        other_label=other_label,
        threshold=threshold,
    )
    val_accuracy = (val_predictions == val_labels).float().mean().item()
    test_predictions, test_nearest_labels, test_scores = predict_open_set_with_lookalikes(
        query_embeddings=test_embeddings,
        prototypes=prototypes,
        target_labels=target_labels,
        lookalike_labels=lookalike_labels,
        other_label=other_label,
        threshold=threshold,
    )

    metrics = {
        "mode": "lookalike_prototype",
        "target_labels": target_labels,
        "other_label": other_label,
        "gallery_source": gallery_source,
        "selected_threshold": float(threshold),
        "lookalike_labels": {
            "michael_like": int(michael_like_label),
            "sarah_like": int(sarah_like_label),
        },
        "cluster_random_state": int(cluster_random_state),
        "cluster_summary": [
            {
                "cluster_id": int(item["cluster_id"]),
                "count": int(item["count"]),
                "mean_sim_to_target": item["mean_sim_to_target"],
                "margin_target_1_minus_2": float(item["margin_target_1_minus_2"]),
            }
            for item in cluster_records
        ],
        "assigned_clusters": {
            "michael_like_cluster_id": int(jesse_like_cluster["cluster_id"]),
            "sarah_like_cluster_id": int(mila_like_cluster["cluster_id"]),
        },
        "val_accuracy": float(val_accuracy),
        "val_nearest_label_counts": {
            str(label): int((val_nearest_labels == label).sum().item())
            for label in sorted(set(val_nearest_labels.tolist()))
        },
        "test_nearest_label_counts": {
            str(label): int((test_nearest_labels == label).sum().item())
            for label in sorted(set(test_nearest_labels.tolist()))
        },
        "mean_test_best_score": float(test_scores.mean().item()),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "prototype_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    id_to_prediction = {
        int(sample_id): int(pred)
        for sample_id, pred in zip(test_ids.tolist(), test_predictions.tolist())
    }
    return [id_to_prediction[int(sample_id)] for sample_id in test_df["id"].tolist()]


def _build_lookalike_clustered_prototypes(
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
    target_labels: list[int],
    other_label: int,
    cluster_random_state: int,
    michael_like_label: int,
    sarah_like_label: int,
) -> tuple[dict[int, torch.Tensor], list[dict[str, object]], dict[str, int]]:
    target_prototypes = compute_class_prototypes(
        gallery_embeddings,
        gallery_labels,
        prototype_labels=target_labels,
    )

    other_embeddings = gallery_embeddings[gallery_labels == other_label]
    if other_embeddings.shape[0] < 2:
        raise ValueError("look-alike 模式要求 other 类至少有 2 个样本。")

    normalized_other_embeddings = torch.nn.functional.normalize(other_embeddings, p=2, dim=1)
    cluster_ids = KMeans(
        n_clusters=2,
        random_state=cluster_random_state,
        n_init=50,
    ).fit_predict(normalized_other_embeddings.cpu().numpy())
    cluster_ids_tensor = torch.as_tensor(cluster_ids, dtype=torch.long)

    cluster_records: list[dict[str, object]] = []
    for cluster_id in sorted(set(cluster_ids)):
        cluster_embeddings = normalized_other_embeddings[cluster_ids_tensor == cluster_id]
        cluster_prototype = torch.nn.functional.normalize(
            cluster_embeddings.mean(dim=0, keepdim=True),
            p=2,
            dim=1,
        ).squeeze(0)
        mean_sim_to_target = {
            int(label): float(torch.dot(cluster_prototype, target_prototypes[label]).item())
            for label in target_labels
        }
        cluster_records.append(
            {
                "cluster_id": int(cluster_id),
                "prototype": cluster_prototype,
                "mean_sim_to_target": mean_sim_to_target,
                "margin_target_1_minus_2": float(
                    mean_sim_to_target[target_labels[0]] - mean_sim_to_target[target_labels[1]]
                ),
                "count": int((cluster_ids_tensor == cluster_id).sum().item()),
            }
        )

    jesse_like_cluster = max(cluster_records, key=lambda item: item["margin_target_1_minus_2"])
    mila_like_cluster = min(cluster_records, key=lambda item: item["margin_target_1_minus_2"])
    prototypes = {
        int(target_labels[0]): target_prototypes[target_labels[0]],
        int(target_labels[1]): target_prototypes[target_labels[1]],
        int(michael_like_label): jesse_like_cluster["prototype"],
        int(sarah_like_label): mila_like_cluster["prototype"],
    }
    assigned_clusters = {
        "michael_like_cluster_id": int(jesse_like_cluster["cluster_id"]),
        "sarah_like_cluster_id": int(mila_like_cluster["cluster_id"]),
    }
    return prototypes, cluster_records, assigned_clusters


def _run_global_label_spread_inference(config, datamodule, model, test_df, output_root: Path):
    inference_config = config.get("inference", {})
    class_labels = inference_config.get("class_labels", [0, 1, 2])
    use_horizontal_flip_tta = inference_config.get("tta_horizontal_flip", False)
    graph_config = inference_config.get("graph_label_spread", {})
    alpha = graph_config.get("alpha", 0.2)
    top_k = graph_config.get("top_k", 20)
    max_iter = graph_config.get("max_iter", 50)
    tol = graph_config.get("tol", 1e-6)

    device = torch.device("cuda" if torch.cuda.is_available() and config["train"]["accelerator"] != "cpu" else "cpu")
    model = model.to(device)

    train_embeddings, train_labels = _collect_embeddings(
        model,
        datamodule.train_dataloader(),
        device,
        use_horizontal_flip_tta=use_horizontal_flip_tta,
    )
    val_embeddings, val_labels = _collect_embeddings(
        model,
        datamodule.val_dataloader(),
        device,
        use_horizontal_flip_tta=use_horizontal_flip_tta,
    )
    test_embeddings, test_ids = _collect_embeddings(
        model,
        datamodule.predict_dataloader(),
        device,
        use_horizontal_flip_tta=use_horizontal_flip_tta,
    )

    base_val_predictions, _ = global_label_spread_predictions(
        query_embeddings=val_embeddings,
        gallery_embeddings=train_embeddings,
        gallery_labels=train_labels,
        class_labels=class_labels,
        alpha=alpha,
        top_k=top_k,
        max_iter=max_iter,
        tol=tol,
    )
    base_val_accuracy = (base_val_predictions == val_labels).float().mean().item()

    all_gallery_embeddings, all_gallery_labels = combine_embedding_sets(
        [
            (train_embeddings, train_labels),
            (val_embeddings, val_labels),
        ]
    )
    test_predictions, test_probabilities = global_label_spread_predictions(
        query_embeddings=test_embeddings,
        gallery_embeddings=all_gallery_embeddings,
        gallery_labels=all_gallery_labels,
        class_labels=class_labels,
        alpha=alpha,
        top_k=top_k,
        max_iter=max_iter,
        tol=tol,
    )

    metrics = {
        "mode": "global_label_spread",
        "class_labels": class_labels,
        "tta_horizontal_flip": use_horizontal_flip_tta,
        "val_accuracy": float(base_val_accuracy),
        "graph_label_spread": {
            "alpha": float(alpha),
            "top_k": int(top_k),
            "max_iter": int(max_iter),
            "tol": float(tol),
        },
        "test_prediction_counts": {
            str(label): int((test_predictions == label).sum().item()) for label in class_labels
        },
        "mean_test_confidence": float(test_probabilities.max(dim=1).values.mean().item()),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "prototype_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    id_to_prediction = {
        int(sample_id): int(pred) for sample_id, pred in zip(test_ids.tolist(), test_predictions.tolist())
    }
    return [id_to_prediction[int(sample_id)] for sample_id in test_df["id"].tolist()]


def _run_lookalike_margin_inference(config, datamodule, model, test_df, output_root: Path):
    inference_config = config.get("inference", {})
    target_labels = inference_config.get("target_labels", [1, 2])
    other_label = inference_config.get("other_label", 0)
    threshold = inference_config.get("threshold", 0.55)
    cluster_random_state = inference_config.get("cluster_random_state", 42)
    michael_like_label = inference_config.get("michael_like_label", 3)
    sarah_like_label = inference_config.get("sarah_like_label", 4)
    use_horizontal_flip_tta = inference_config.get("tta_horizontal_flip", False)
    margin_values = inference_config.get("lookalike_margin_values", [0.02, 0.03, 0.04, 0.05])

    device = torch.device("cuda" if torch.cuda.is_available() and config["train"]["accelerator"] != "cpu" else "cpu")
    model = model.to(device)

    train_embeddings, train_labels = _collect_embeddings(
        model,
        datamodule.train_dataloader(),
        device,
        use_horizontal_flip_tta=use_horizontal_flip_tta,
    )
    val_embeddings, val_labels = _collect_embeddings(
        model,
        datamodule.val_dataloader(),
        device,
        use_horizontal_flip_tta=use_horizontal_flip_tta,
    )
    test_embeddings, test_ids = _collect_embeddings(
        model,
        datamodule.predict_dataloader(),
        device,
        use_horizontal_flip_tta=use_horizontal_flip_tta,
    )

    train_prototypes, train_cluster_records, train_assigned_clusters = _build_lookalike_clustered_prototypes(
        gallery_embeddings=train_embeddings,
        gallery_labels=train_labels,
        target_labels=target_labels,
        other_label=other_label,
        cluster_random_state=cluster_random_state,
        michael_like_label=michael_like_label,
        sarah_like_label=sarah_like_label,
    )

    best_margin = margin_values[0]
    best_val_accuracy = -1.0
    for margin in margin_values:
        candidate_predictions, _, _, _ = predict_open_set_with_lookalike_margin_rejection(
            query_embeddings=val_embeddings,
            prototypes=train_prototypes,
            target_labels=target_labels,
            lookalike_by_target={
                int(target_labels[0]): int(michael_like_label),
                int(target_labels[1]): int(sarah_like_label),
            },
            other_label=other_label,
            threshold=threshold,
            margins_by_target={
                int(target_labels[0]): float(margin),
                int(target_labels[1]): float(margin),
            },
        )
        candidate_accuracy = (candidate_predictions == val_labels).float().mean().item()
        if candidate_accuracy > best_val_accuracy:
            best_margin = margin
            best_val_accuracy = candidate_accuracy

    all_gallery_embeddings, all_gallery_labels = combine_embedding_sets(
        [
            (train_embeddings, train_labels),
            (val_embeddings, val_labels),
        ]
    )
    final_prototypes, cluster_records, assigned_clusters = _build_lookalike_clustered_prototypes(
        gallery_embeddings=all_gallery_embeddings,
        gallery_labels=all_gallery_labels,
        target_labels=target_labels,
        other_label=other_label,
        cluster_random_state=cluster_random_state,
        michael_like_label=michael_like_label,
        sarah_like_label=sarah_like_label,
    )
    test_predictions, target_scores, lookalike_scores, margins = predict_open_set_with_lookalike_margin_rejection(
        query_embeddings=test_embeddings,
        prototypes=final_prototypes,
        target_labels=target_labels,
        lookalike_by_target={
            int(target_labels[0]): int(michael_like_label),
            int(target_labels[1]): int(sarah_like_label),
        },
        other_label=other_label,
        threshold=threshold,
        margins_by_target={
            int(target_labels[0]): float(best_margin),
            int(target_labels[1]): float(best_margin),
        },
    )

    metrics = {
        "mode": "lookalike_margin_rejection",
        "target_labels": target_labels,
        "other_label": other_label,
        "tta_horizontal_flip": use_horizontal_flip_tta,
        "selected_threshold": float(threshold),
        "selected_margin": float(best_margin),
        "val_accuracy": float(best_val_accuracy),
        "cluster_random_state": int(cluster_random_state),
        "train_cluster_summary": [
            {
                "cluster_id": int(item["cluster_id"]),
                "count": int(item["count"]),
                "mean_sim_to_target": item["mean_sim_to_target"],
                "margin_target_1_minus_2": float(item["margin_target_1_minus_2"]),
            }
            for item in train_cluster_records
        ],
        "train_assigned_clusters": train_assigned_clusters,
        "final_cluster_summary": [
            {
                "cluster_id": int(item["cluster_id"]),
                "count": int(item["count"]),
                "mean_sim_to_target": item["mean_sim_to_target"],
                "margin_target_1_minus_2": float(item["margin_target_1_minus_2"]),
            }
            for item in cluster_records
        ],
        "final_assigned_clusters": assigned_clusters,
        "mean_test_target_score": float(target_scores.mean().item()),
        "mean_test_lookalike_score": float(lookalike_scores.mean().item()),
        "mean_test_margin": float(margins.mean().item()),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "prototype_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    id_to_prediction = {
        int(sample_id): int(pred) for sample_id, pred in zip(test_ids.tolist(), test_predictions.tolist())
    }
    return [id_to_prediction[int(sample_id)] for sample_id in test_df["id"].tolist()]


def _run_spectral_cluster_inference(config, datamodule, model, test_df, output_root: Path):
    inference_config = config.get("inference", {})
    use_horizontal_flip_tta = inference_config.get("tta_horizontal_flip", False)
    spectral_config = inference_config.get("spectral_cluster", {})
    n_clusters = spectral_config.get("n_clusters", 4)
    top_k_gallery = spectral_config.get("top_k_gallery", 3)
    random_state = spectral_config.get("random_state", 42)

    device = torch.device("cuda" if torch.cuda.is_available() and config["train"]["accelerator"] != "cpu" else "cpu")
    model = model.to(device)

    train_embeddings, train_labels = _collect_embeddings(
        model,
        datamodule.train_dataloader(),
        device,
        use_horizontal_flip_tta=use_horizontal_flip_tta,
    )
    val_embeddings, val_labels = _collect_embeddings(
        model,
        datamodule.val_dataloader(),
        device,
        use_horizontal_flip_tta=use_horizontal_flip_tta,
    )
    test_embeddings, test_ids = _collect_embeddings(
        model,
        datamodule.predict_dataloader(),
        device,
        use_horizontal_flip_tta=use_horizontal_flip_tta,
    )

    val_predictions, val_cluster_ids = spectral_cluster_with_gallery_label_matching(
        query_embeddings=val_embeddings,
        gallery_embeddings=train_embeddings,
        gallery_labels=train_labels,
        n_clusters=n_clusters,
        top_k_gallery=top_k_gallery,
        random_state=random_state,
    )
    val_accuracy = (val_predictions == val_labels).float().mean().item()

    all_gallery_embeddings, all_gallery_labels = combine_embedding_sets(
        [
            (train_embeddings, train_labels),
            (val_embeddings, val_labels),
        ]
    )
    test_predictions, test_cluster_ids = spectral_cluster_with_gallery_label_matching(
        query_embeddings=test_embeddings,
        gallery_embeddings=all_gallery_embeddings,
        gallery_labels=all_gallery_labels,
        n_clusters=n_clusters,
        top_k_gallery=top_k_gallery,
        random_state=random_state,
    )

    metrics = {
        "mode": "spectral_cluster",
        "tta_horizontal_flip": use_horizontal_flip_tta,
        "val_accuracy": float(val_accuracy),
        "spectral_cluster": {
            "n_clusters": int(n_clusters),
            "top_k_gallery": int(top_k_gallery),
            "random_state": int(random_state),
        },
        "val_cluster_counts": {
            str(cluster_id): int((val_cluster_ids == cluster_id).sum().item())
            for cluster_id in sorted(set(val_cluster_ids.tolist()))
        },
        "test_cluster_counts": {
            str(cluster_id): int((test_cluster_ids == cluster_id).sum().item())
            for cluster_id in sorted(set(test_cluster_ids.tolist()))
        },
        "test_prediction_counts": {
            str(label): int((test_predictions == label).sum().item())
            for label in sorted(set(test_predictions.tolist()))
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "prototype_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    id_to_prediction = {
        int(sample_id): int(pred) for sample_id, pred in zip(test_ids.tolist(), test_predictions.tolist())
    }
    return [id_to_prediction[int(sample_id)] for sample_id in test_df["id"].tolist()]


def _run_pca_whitened_prototype_inference(config, datamodule, model, test_df, output_root: Path):
    inference_config = config.get("inference", {})
    prototype_labels = inference_config.get("prototype_labels", [1, 2])
    other_label = inference_config.get("other_label", 0)
    threshold_values = inference_config.get("threshold_values", [0.55])
    use_horizontal_flip_tta = inference_config.get("tta_horizontal_flip", False)
    pca_config = inference_config.get("pca_whitening", {})
    n_components = pca_config.get("n_components", 128)

    device = torch.device("cuda" if torch.cuda.is_available() and config["train"]["accelerator"] != "cpu" else "cpu")
    model = model.to(device)

    train_embeddings, train_labels = _collect_embeddings(
        model,
        datamodule.train_dataloader(),
        device,
        use_horizontal_flip_tta=use_horizontal_flip_tta,
    )
    val_embeddings, val_labels = _collect_embeddings(
        model,
        datamodule.val_dataloader(),
        device,
        use_horizontal_flip_tta=use_horizontal_flip_tta,
    )
    test_embeddings, test_ids = _collect_embeddings(
        model,
        datamodule.predict_dataloader(),
        device,
        use_horizontal_flip_tta=use_horizontal_flip_tta,
    )

    whitened_train_embeddings, whitened_val_embeddings = pca_whiten_embedding_sets(
        gallery_embeddings=train_embeddings,
        query_embeddings=val_embeddings,
        n_components=n_components,
    )
    prototypes = compute_class_prototypes(
        whitened_train_embeddings,
        train_labels,
        prototype_labels=prototype_labels,
    )
    threshold, val_accuracy = select_best_threshold(
        val_embeddings=whitened_val_embeddings,
        val_labels=val_labels,
        prototypes=prototypes,
        other_label=other_label,
        threshold_values=threshold_values,
    )

    combined_gallery_embeddings, combined_gallery_labels = combine_embedding_sets(
        [
            (train_embeddings, train_labels),
            (val_embeddings, val_labels),
        ]
    )
    whitened_gallery_embeddings, whitened_test_embeddings = pca_whiten_embedding_sets(
        gallery_embeddings=combined_gallery_embeddings,
        query_embeddings=test_embeddings,
        n_components=n_components,
    )
    final_prototypes = compute_class_prototypes(
        whitened_gallery_embeddings,
        combined_gallery_labels,
        prototype_labels=prototype_labels,
    )
    test_predictions, _ = predict_open_set(
        whitened_test_embeddings,
        prototypes=final_prototypes,
        other_label=other_label,
        threshold=threshold,
    )

    metrics = {
        "mode": "pca_whitened_prototype",
        "prototype_labels": prototype_labels,
        "other_label": other_label,
        "tta_horizontal_flip": use_horizontal_flip_tta,
        "selected_threshold": float(threshold),
        "val_accuracy": float(val_accuracy),
        "pca_whitening": {
            "n_components": int(min(n_components, train_embeddings.shape[0], train_embeddings.shape[1])),
        },
        "test_prediction_counts": {
            str(label): int((test_predictions == label).sum().item())
            for label in sorted(set(test_predictions.tolist()))
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "prototype_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    id_to_prediction = {
        int(sample_id): int(pred) for sample_id, pred in zip(test_ids.tolist(), test_predictions.tolist())
    }
    return [id_to_prediction[int(sample_id)] for sample_id in test_df["id"].tolist()]


def _run_neighborhood_aware_inference(config, datamodule, model, test_df, output_root: Path):
    inference_config = config.get("inference", {})
    prototype_labels = inference_config.get("prototype_labels", [1, 2])
    other_label = inference_config.get("other_label", 0)
    threshold = inference_config.get("threshold", 0.55)
    use_horizontal_flip_tta = inference_config.get("tta_horizontal_flip", False)
    neighborhood_config = inference_config.get("neighborhood_aware", {})
    top_k = neighborhood_config.get("top_k", 15)
    base_weight = neighborhood_config.get("base_weight", 0.5)

    device = torch.device("cuda" if torch.cuda.is_available() and config["train"]["accelerator"] != "cpu" else "cpu")
    model = model.to(device)

    train_embeddings, train_labels = _collect_embeddings(
        model,
        datamodule.train_dataloader(),
        device,
        use_horizontal_flip_tta=use_horizontal_flip_tta,
    )
    val_embeddings, val_labels = _collect_embeddings(
        model,
        datamodule.val_dataloader(),
        device,
        use_horizontal_flip_tta=use_horizontal_flip_tta,
    )
    test_embeddings, test_ids = _collect_embeddings(
        model,
        datamodule.predict_dataloader(),
        device,
        use_horizontal_flip_tta=use_horizontal_flip_tta,
    )

    prototypes = compute_class_prototypes(
        train_embeddings,
        train_labels,
        prototype_labels=prototype_labels,
    )
    val_predictions, val_base_scores, val_neighbor_scores, val_final_scores = neighborhood_aware_predictions(
        query_embeddings=val_embeddings,
        prototypes=prototypes,
        other_label=other_label,
        threshold=threshold,
        top_k=top_k,
        base_weight=base_weight,
    )
    val_accuracy = (val_predictions == val_labels).float().mean().item()

    all_gallery_embeddings, all_gallery_labels = combine_embedding_sets(
        [
            (train_embeddings, train_labels),
            (val_embeddings, val_labels),
        ]
    )
    final_prototypes = compute_class_prototypes(
        all_gallery_embeddings,
        all_gallery_labels,
        prototype_labels=prototype_labels,
    )
    test_predictions, test_base_scores, test_neighbor_scores, test_final_scores = neighborhood_aware_predictions(
        query_embeddings=test_embeddings,
        prototypes=final_prototypes,
        other_label=other_label,
        threshold=threshold,
        top_k=top_k,
        base_weight=base_weight,
    )

    metrics = {
        "mode": "neighborhood_aware",
        "prototype_labels": prototype_labels,
        "other_label": other_label,
        "tta_horizontal_flip": use_horizontal_flip_tta,
        "selected_threshold": float(threshold),
        "val_accuracy": float(val_accuracy),
        "neighborhood_aware": {
            "top_k": int(top_k),
            "base_weight": float(base_weight),
            "neighbor_weight": float(1.0 - base_weight),
        },
        "mean_val_base_score": float(val_base_scores.mean().item()),
        "mean_val_neighbor_score": float(val_neighbor_scores.mean().item()),
        "mean_val_final_score": float(val_final_scores.mean().item()),
        "mean_test_base_score": float(test_base_scores.mean().item()),
        "mean_test_neighbor_score": float(test_neighbor_scores.mean().item()),
        "mean_test_final_score": float(test_final_scores.mean().item()),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "prototype_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    id_to_prediction = {
        int(sample_id): int(pred) for sample_id, pred in zip(test_ids.tolist(), test_predictions.tolist())
    }
    return [id_to_prediction[int(sample_id)] for sample_id in test_df["id"].tolist()]


def main() -> None:
    parser = argparse.ArgumentParser(description="加载最佳模型并生成 submission.csv。")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    splits_dir = project_path(config["data"]["splits_dir"])
    output_root = project_path("outputs", config["experiment_name"])
    loss_config = config.get("loss", {})

    inference_mode = config.get("inference", {}).get("mode", "softmax")
    checkpoint_path = args.checkpoint
    if checkpoint_path is None and inference_mode != "ensemble_prototype":
        metrics_path = output_root / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError("未找到 metrics.json，无法自动定位最佳模型。")
        checkpoint_path = json.loads(metrics_path.read_text(encoding="utf-8"))["best_model_path"]

    train_df = pd.read_csv(splits_dir / "train.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")

    datamodule = FaceDataModule(
        train_csv=splits_dir / "train.csv",
        val_csv=splits_dir / "val.csv",
        test_csv=splits_dir / "test.csv",
        image_size=config["data"]["face_size"],
        batch_size=config["train"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        normalization=config["data"]["normalization"],
    )
    datamodule.setup()

    model = None
    if inference_mode != "ensemble_prototype":
        model = FaceClassifierModule.load_from_checkpoint(
            checkpoint_path,
            model_family=config["model"]["family"],
            backbone_name=config["model"]["backbone_name"],
            num_classes=int(train_df["class"].nunique()),
            pretrained=config["model"]["pretrained"],
            dropout=config["model"]["dropout"],
            learning_rate=config["train"]["learning_rate"],
            backbone_learning_rate=config["train"].get("backbone_learning_rate"),
            weight_decay=config["train"]["weight_decay"],
            scheduler_name=config["train"]["scheduler"],
            max_epochs=config["train"]["max_epochs"],
            pretrained_repo_id=config["model"].get("pretrained_repo_id"),
            freeze_backbone=config["model"].get("freeze_backbone", False),
            unfreeze_last_stage=config["model"].get("unfreeze_last_stage", False),
            unfreeze_stage_count=config["model"].get("unfreeze_stage_count", 0),
            loss_name=loss_config.get("name", "cross_entropy"),
            loss_target_labels=loss_config.get("target_labels"),
            arcface_scale=loss_config.get("arcface_scale", 30.0),
            arcface_margin=loss_config.get("arcface_margin", 0.5),
        )

    trainer = L.Trainer(
        accelerator=config["train"]["accelerator"],
        devices=config["train"]["devices"],
        precision=config["train"]["precision"],
        logger=False,
        callbacks=[AsciiTQDMProgressBar(refresh_rate=1)],
        enable_progress_bar=True,
        enable_model_summary=False,
    )
    if inference_mode == "prototype":
        ordered_predictions = _run_prototype_inference(config, datamodule, model, test_df, output_root)
    elif inference_mode == "conservative_graph_refine":
        ordered_predictions = _run_conservative_graph_refine_inference(config, datamodule, model, test_df, output_root)
    elif inference_mode == "global_label_spread":
        ordered_predictions = _run_global_label_spread_inference(config, datamodule, model, test_df, output_root)
    elif inference_mode == "spectral_cluster":
        ordered_predictions = _run_spectral_cluster_inference(config, datamodule, model, test_df, output_root)
    elif inference_mode == "pca_whitened_prototype":
        ordered_predictions = _run_pca_whitened_prototype_inference(config, datamodule, model, test_df, output_root)
    elif inference_mode == "neighborhood_aware":
        ordered_predictions = _run_neighborhood_aware_inference(config, datamodule, model, test_df, output_root)
    elif inference_mode == "ensemble_prototype":
        ordered_predictions = _run_ensemble_prototype_inference(config, datamodule, test_df, output_root)
    elif inference_mode == "exemplar_knn":
        ordered_predictions = _run_exemplar_inference(config, datamodule, model, test_df, output_root)
    elif inference_mode == "richer_open_set":
        ordered_predictions = _run_richer_open_set_inference(config, datamodule, model, test_df, output_root)
    elif inference_mode == "quality_aware_open_set":
        ordered_predictions = _run_quality_aware_open_set_inference(config, datamodule, model, test_df, output_root)
    elif inference_mode == "verifier_open_set":
        ordered_predictions = _run_verifier_open_set_inference(config, datamodule, model, test_df, output_root)
    elif inference_mode == "lookalike_prototype":
        ordered_predictions = _run_lookalike_prototype_inference(config, datamodule, model, test_df, output_root)
    elif inference_mode == "lookalike_margin_rejection":
        ordered_predictions = _run_lookalike_margin_inference(config, datamodule, model, test_df, output_root)
    else:
        outputs = trainer.predict(model, datamodule=datamodule)

        id_to_prediction: dict[int, int] = {}
        for batch_output in outputs:
            ids = batch_output["ids"].detach().cpu().tolist()
            preds = batch_output["preds"].detach().cpu().tolist()
            id_to_prediction.update({int(sample_id): int(pred) for sample_id, pred in zip(ids, preds)})

        ordered_predictions = [id_to_prediction[int(sample_id)] for sample_id in test_df["id"].tolist()]
    submission = build_submission_dataframe(test_df, ordered_predictions)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = project_path(
        config["data"]["submissions_dir"],
        f"{timestamp}_{config['experiment_name']}_submission.csv",
    )
    save_submission_dataframe(submission, submission_path)

    append_registry_row(
        project_path("reports", "experiments", "registry.csv"),
        {
            "experiment_name": config["experiment_name"],
            "stage": "predict",
            "submission_path": str(submission_path),
            "checkpoint_path": str(checkpoint_path or json.dumps(config.get("inference", {}).get("checkpoint_paths", []))),
            "tta_horizontal_flip": config.get("inference", {}).get("tta_horizontal_flip", False),
        },
    )
    print(f"submission 已生成: {submission_path}")


if __name__ == "__main__":
    main()
