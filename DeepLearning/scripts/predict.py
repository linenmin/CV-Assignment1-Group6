from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import lightning as L
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_pipeline.common.config import load_experiment_config
from dl_pipeline.common.paths import project_path
from dl_pipeline.common.registry import append_registry_row
from dl_pipeline.data.datamodule import FaceDataModule
from dl_pipeline.inference.submission import build_submission_dataframe, save_submission_dataframe
from dl_pipeline.training.progress import AsciiTQDMProgressBar
from dl_pipeline.training.lightning_module import FaceClassifierModule


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
        weight_decay=config["train"]["weight_decay"],
        scheduler_name=config["train"]["scheduler"],
        max_epochs=config["train"]["max_epochs"],
        pretrained_repo_id=config["model"].get("pretrained_repo_id"),
        freeze_backbone=config["model"].get("freeze_backbone", False),
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
