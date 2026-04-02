from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_pipeline.common.config import load_experiment_config
from dl_pipeline.common.paths import project_path
from dl_pipeline.data.datamodule import FaceDataModule
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
        loss_name=loss_config.get("name", "cross_entropy"),
        loss_target_labels=loss_config.get("target_labels"),
        arcface_scale=loss_config.get("arcface_scale", 30.0),
        arcface_margin=loss_config.get("arcface_margin", 0.5),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="导出指定 checkpoint 的 train/val/test embedding。")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    datamodule = FaceDataModule(
        train_csv=project_path(config["data"]["splits_dir"]) / "train.csv",
        val_csv=project_path(config["data"]["splits_dir"]) / "val.csv",
        test_csv=project_path(config["data"]["splits_dir"]) / "test.csv",
        image_size=config["data"]["face_size"],
        batch_size=config["train"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        use_horizontal_flip=config["augmentation"]["use_horizontal_flip"],
        use_affine=config["augmentation"]["use_affine"],
        use_degradation_pack=config["augmentation"].get("use_degradation_pack", False),
        normalization=config["data"]["normalization"],
    )
    datamodule.setup()

    device = torch.device("cuda" if torch.cuda.is_available() and config["train"]["accelerator"] != "cpu" else "cpu")
    model = _load_model_from_checkpoint(config, args.checkpoint).to(device)

    train_embeddings, train_labels = _collect_embeddings(model, datamodule.train_dataloader(), device)
    val_embeddings, val_labels = _collect_embeddings(model, datamodule.val_dataloader(), device)
    test_embeddings, test_ids = _collect_embeddings(model, datamodule.predict_dataloader(), device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config_path": str(args.config),
            "checkpoint_path": str(args.checkpoint),
            "train_embeddings": train_embeddings,
            "train_labels": train_labels,
            "val_embeddings": val_embeddings,
            "val_labels": val_labels,
            "test_embeddings": test_embeddings,
            "test_ids": test_ids,
        },
        output_path,
    )
    print(f"embedding 已导出: {output_path}")


if __name__ == "__main__":
    main()
