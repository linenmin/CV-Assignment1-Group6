from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import lightning as L
import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_pipeline.common.config import load_experiment_config
from dl_pipeline.common.paths import ensure_dir, project_path
from dl_pipeline.common.registry import append_registry_row
from dl_pipeline.common.seed import seed_everything
from dl_pipeline.data.datamodule import FaceDataModule
from dl_pipeline.training.progress import AsciiTQDMProgressBar
from dl_pipeline.training.lightning_module import FaceClassifierModule
from dl_pipeline.training.monitoring import resolve_monitor_config


def main() -> None:
    parser = argparse.ArgumentParser(description="训练第一版分类模型。")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    seed_everything(config["seed"])
    monitor_config = resolve_monitor_config(config["train"])
    loss_config = config.get("loss", {})

    splits_dir = project_path(config["data"]["splits_dir"])
    output_root = ensure_dir(project_path("outputs", config["experiment_name"]))
    logger = CSVLogger(save_dir=str(output_root / "logs"), name="")
    train_df = pd.read_csv(splits_dir / "train.csv")

    datamodule = FaceDataModule(
        train_csv=splits_dir / "train.csv",
        val_csv=splits_dir / "val.csv",
        test_csv=splits_dir / "test.csv",
        image_size=config["data"]["face_size"],
        batch_size=config["train"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        use_horizontal_flip=config["augmentation"]["use_horizontal_flip"],
        use_affine=config["augmentation"]["use_affine"],
        use_degradation_pack=config["augmentation"].get("use_degradation_pack", False),
        normalization=config["data"]["normalization"],
    )
    datamodule.setup()

    model = FaceClassifierModule(
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
        monitor_metric=monitor_config.metric,
        pretrained_repo_id=config["model"].get("pretrained_repo_id"),
        freeze_backbone=config["model"].get("freeze_backbone", False),
        unfreeze_last_stage=config["model"].get("unfreeze_last_stage", False),
        unfreeze_stage_count=config["model"].get("unfreeze_stage_count", 0),
        loss_name=loss_config.get("name", "cross_entropy"),
        loss_target_labels=loss_config.get("target_labels"),
        arcface_scale=loss_config.get("arcface_scale", 30.0),
        arcface_margin=loss_config.get("arcface_margin", 0.5),
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_root / "checkpoints",
        filename="best",
        monitor=monitor_config.metric,
        mode=monitor_config.mode,
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor=monitor_config.metric,
        patience=config["train"]["early_stopping_patience"],
        mode=monitor_config.mode,
    )
    progress_bar = AsciiTQDMProgressBar(refresh_rate=1)

    trainer = L.Trainer(
        max_epochs=config["train"]["max_epochs"],
        accelerator=config["train"]["accelerator"],
        devices=config["train"]["devices"],
        precision=config["train"]["precision"],
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping, progress_bar],
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=False,
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=datamodule)

    metrics = {
        "best_model_path": checkpoint_callback.best_model_path,
        f"best_{monitor_config.metric}": float(checkpoint_callback.best_model_score.cpu().item())
        if checkpoint_callback.best_model_score is not None
        else None,
    }
    (output_root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    append_registry_row(
        project_path("reports", "experiments", "registry.csv"),
        {
            "experiment_name": config["experiment_name"],
            "stage": "train",
            "best_model_path": checkpoint_callback.best_model_path,
            "best_val_acc": metrics.get("best_val_acc"),
            "config_path": config["config_path"],
        },
    )
    print(f"训练完成，最佳模型: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
