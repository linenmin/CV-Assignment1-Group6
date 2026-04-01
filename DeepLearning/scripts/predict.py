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
    compute_class_prototypes,
    predict_open_set,
    predict_open_set_with_class_thresholds,
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

    device = torch.device("cuda" if torch.cuda.is_available() and config["train"]["accelerator"] != "cpu" else "cpu")
    model = model.to(device)

    train_embeddings, train_labels = _collect_embeddings(model, datamodule.train_dataloader(), device)
    val_embeddings, val_labels = _collect_embeddings(model, datamodule.val_dataloader(), device)
    test_embeddings, test_ids = _collect_embeddings(model, datamodule.predict_dataloader(), device)

    prototypes = compute_class_prototypes(train_embeddings, train_labels, prototype_labels=prototype_labels)
    if threshold_selection == "crossval_global":
        threshold, cv_mean_accuracy = select_best_threshold_crossval(
            embeddings=train_embeddings,
            labels=train_labels,
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
