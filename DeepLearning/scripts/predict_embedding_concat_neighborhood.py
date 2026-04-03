"""
ViT + IR101（或多 backbone）embedding 在 L2 归一化后沿特征维拼接，再 L2 归一化，
在同一联合空间上做与 exp_047 相同的 neighborhood_aware + 可选 hflip TTA。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_pipeline.common.config import load_experiment_config
from dl_pipeline.common.paths import project_path, ensure_dir
from dl_pipeline.common.registry import append_registry_row
from dl_pipeline.data.datamodule import FaceDataModule
from dl_pipeline.inference.prototype import (
    combine_embedding_sets,
    compute_class_prototypes,
    neighborhood_aware_predictions,
)
from dl_pipeline.inference.submission import build_submission_dataframe, save_submission_dataframe
from dl_pipeline.inference.tta import extract_tta_features
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


def _load_model_from_checkpoint(config, checkpoint_path: str) -> FaceClassifierModule:
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
        pretrained_checkpoint_path=config["model"].get("pretrained_checkpoint_path"),
        freeze_backbone=config["model"].get("freeze_backbone", False),
        unfreeze_last_stage=config["model"].get("unfreeze_last_stage", False),
        unfreeze_stage_count=config["model"].get("unfreeze_stage_count", 0),
        unfreeze_cvlface_norm=config["model"].get("unfreeze_cvlface_norm", True),
        unfreeze_cvlface_feature=config["model"].get("unfreeze_cvlface_feature", True),
        loss_name=loss_config.get("name", "cross_entropy"),
        loss_target_labels=loss_config.get("target_labels"),
        arcface_scale=loss_config.get("arcface_scale", 30.0),
        arcface_margin=loss_config.get("arcface_margin", 0.5),
        cosface_scale=loss_config.get("cosface_scale", loss_config.get("arcface_scale", 30.0)),
        cosface_margin=loss_config.get("cosface_margin", 0.35),
    )


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_path(path_value)


def _concat_branch_embeddings(*branches: torch.Tensor) -> torch.Tensor:
    """各分支先 L2 归一化，再拼接，再对拼接向量 L2 归一化（平衡各 backbone 尺度）。"""
    normed = [F.normalize(e, p=2, dim=1) for e in branches]
    z = torch.cat(normed, dim=1)
    return F.normalize(z, p=2, dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="多 backbone embedding 拼接 + neighborhood_aware。")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    experiment_name = config["experiment_name"]
    ec = config["embedding_concat_neighborhood"]
    members = ec["members"]
    if len(members) < 2:
        raise ValueError("embedding_concat_neighborhood.members 至少需要 2 个 backbone。")

    prototype_labels = ec.get("prototype_labels", [1, 2])
    other_label = ec.get("other_label", 0)
    threshold = float(ec.get("threshold", 0.55))
    use_tta = ec.get("tta_horizontal_flip", True)
    nb = ec.get("neighborhood_aware", {})
    top_k = int(nb.get("top_k", 15))
    base_weight = float(nb.get("base_weight", 0.5))

    splits_dir = project_path(config["data"]["splits_dir"])
    test_df = pd.read_csv(splits_dir / "test.csv")
    output_root = ensure_dir(project_path("outputs", experiment_name))

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

    # 训练集默认 shuffle=True，多 backbone 依次提特征时顺序不一致会导致标签错位；提 embedding 时用确定性顺序。
    train_loader_ordered = DataLoader(
        datamodule.train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and config["train"]["accelerator"] != "cpu" else "cpu"
    )

    train_parts: list[torch.Tensor] = []
    val_parts: list[torch.Tensor] = []
    test_parts: list[torch.Tensor] = []
    checkpoint_paths: list[str] = []
    ref_train_labels = ref_val_labels = ref_test_ids = None

    for member in members:
        mcfg = load_experiment_config(_resolve_path(str(member["config_path"])))
        ckpt = str(_resolve_path(str(member["checkpoint_path"])))
        checkpoint_paths.append(ckpt)
        model = _load_model_from_checkpoint(mcfg, ckpt).to(device)
        tr_e, tr_l = _collect_embeddings(
            model, train_loader_ordered, device, use_horizontal_flip_tta=use_tta
        )
        va_e, va_l = _collect_embeddings(
            model, datamodule.val_dataloader(), device, use_horizontal_flip_tta=use_tta
        )
        te_e, te_ids = _collect_embeddings(
            model, datamodule.predict_dataloader(), device, use_horizontal_flip_tta=use_tta
        )
        if ref_train_labels is None:
            ref_train_labels, ref_val_labels, ref_test_ids = tr_l, va_l, te_ids
        else:
            if not torch.equal(ref_train_labels, tr_l):
                raise ValueError("不同成员的 train_labels 不一致。")
            if not torch.equal(ref_val_labels, va_l):
                raise ValueError("不同成员的 val_labels 不一致。")
            if not torch.equal(ref_test_ids, te_ids):
                raise ValueError("不同成员的 test_ids 不一致。")
        train_parts.append(tr_e)
        val_parts.append(va_e)
        test_parts.append(te_e)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    train_embeddings = _concat_branch_embeddings(*train_parts)
    val_embeddings = _concat_branch_embeddings(*val_parts)
    test_embeddings = _concat_branch_embeddings(*test_parts)
    train_labels = ref_train_labels
    val_labels = ref_val_labels

    prototypes = compute_class_prototypes(
        train_embeddings, train_labels, prototype_labels=prototype_labels
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
        [(train_embeddings, train_labels), (val_embeddings, val_labels)]
    )
    final_prototypes = compute_class_prototypes(
        all_gallery_embeddings, all_gallery_labels, prototype_labels=prototype_labels
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
        "mode": "embedding_concat_neighborhood",
        "prototype_labels": prototype_labels,
        "other_label": other_label,
        "tta_horizontal_flip": use_tta,
        "selected_threshold": threshold,
        "val_accuracy": float(val_accuracy),
        "neighborhood_aware": {
            "top_k": top_k,
            "base_weight": base_weight,
            "neighbor_weight": float(1.0 - base_weight),
        },
        "member_configs": [str(m["config_path"]) for m in members],
        "checkpoint_paths": checkpoint_paths,
        "concat_dim": int(train_embeddings.shape[1]),
        "mean_val_final_score": float(val_final_scores.mean().item()),
        "mean_test_final_score": float(test_final_scores.mean().item()),
    }
    (output_root / "prototype_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    id_to_prediction = {
        int(sample_id): int(pred)
        for sample_id, pred in zip(ref_test_ids.tolist(), test_predictions.tolist())
    }
    ordered_predictions = [id_to_prediction[int(sample_id)] for sample_id in test_df["id"].tolist()]

    submission = build_submission_dataframe(test_df, ordered_predictions)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = project_path(
        config["data"]["submissions_dir"],
        f"{timestamp}_{experiment_name}_submission.csv",
    )
    save_submission_dataframe(submission, submission_path)

    append_registry_row(
        project_path("reports", "experiments", "registry.csv"),
        {
            "experiment_name": experiment_name,
            "stage": "predict",
            "submission_path": str(submission_path),
            "checkpoint_path": json.dumps(checkpoint_paths, ensure_ascii=False),
            "config_path": config["config_path"],
            "tta_horizontal_flip": use_tta,
        },
    )
    print(f"embedding_concat_neighborhood submission 已生成: {submission_path}")


if __name__ == "__main__":
    main()
