from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import lightning as L
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_pipeline.common.config import load_experiment_config
from dl_pipeline.common.paths import project_path
from dl_pipeline.common.registry import append_registry_row
from dl_pipeline.data.datamodule import FaceDataModule
from dl_pipeline.inference.prototype import (
    combine_embedding_sets,
    compute_class_prototypes,
    predict_open_set_with_quality_aware_scorer,
    predict_open_set_with_richer_scorer,
    predict_open_set_with_exemplars,
    predict_open_set,
    predict_open_set_with_class_thresholds,
    select_best_quality_aware_params_leave_one_out,
    select_best_richer_scorer_params_leave_one_out,
    select_best_exemplar_params,
    select_best_class_thresholds,
    select_best_threshold_crossval,
    select_best_threshold,
)
from dl_pipeline.inference.submission import build_submission_dataframe, save_submission_dataframe
from dl_pipeline.training.progress import AsciiTQDMProgressBar
from dl_pipeline.training.lightning_module import FaceClassifierModule


def _collect_embeddings(model, dataloader, device):
    embeddings = []
    values = []
    model.eval()
    with torch.no_grad():
        for images, batch_values in dataloader:
            features = model.extract_features(images.to(device))
            embeddings.append(features.detach().cpu())
            values.append(batch_values.detach().cpu())
    return torch.cat(embeddings, dim=0), torch.cat(values, dim=0)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="加载最佳模型并生成 submission.csv。")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    splits_dir = project_path(config["data"]["splits_dir"])
    output_root = project_path("outputs", config["experiment_name"])

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
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
    inference_mode = config.get("inference", {}).get("mode", "softmax")
    if inference_mode == "prototype":
        ordered_predictions = _run_prototype_inference(config, datamodule, model, test_df, output_root)
    elif inference_mode == "exemplar_knn":
        ordered_predictions = _run_exemplar_inference(config, datamodule, model, test_df, output_root)
    elif inference_mode == "richer_open_set":
        ordered_predictions = _run_richer_open_set_inference(config, datamodule, model, test_df, output_root)
    elif inference_mode == "quality_aware_open_set":
        ordered_predictions = _run_quality_aware_open_set_inference(config, datamodule, model, test_df, output_root)
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
            "checkpoint_path": str(checkpoint_path),
        },
    )
    print(f"submission 已生成: {submission_path}")


if __name__ == "__main__":
    main()
