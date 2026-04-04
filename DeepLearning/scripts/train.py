from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import lightning as L
import pandas as pd
import torch
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
from dl_pipeline.training.shadow_probe import ShadowProbeEpochCallback, resolve_shadow_probe_config
from dl_pipeline.models.classifier import (
    imprint_cosine_classifier_weight_from_loader,
    imprint_linear_head_from_loader,
)


def _resolve_train_device(config: dict) -> torch.device:
    acc = str(config["train"].get("accelerator", "auto")).lower()
    if acc in ("gpu", "cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda", 0)
        return torch.device("cpu")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="训练第一版分类模型。")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # 训练前半段会加载权重/构建模型，可能耗时数分钟且此前脚本几乎无输出；显式 flush 避免被缓冲。
    print(f"[train] 启动，读取配置: {args.config}", flush=True)
    config = load_experiment_config(args.config)
    print(f"[train] 实验: {config['experiment_name']} | 输出: outputs/{config['experiment_name']}", flush=True)
    seed_everything(config["seed"])
    monitor_config = resolve_monitor_config(config["train"])
    loss_config = config.get("loss", {})

    splits_dir = project_path(config["data"]["splits_dir"])
    output_root = ensure_dir(project_path("outputs", config["experiment_name"]))
    logger = CSVLogger(save_dir=str(output_root / "logs"), name="")
    train_df = pd.read_csv(splits_dir / "train.csv")
    num_classes = int(train_df["class"].nunique())
    use_full_train = bool(config["data"].get("use_full_train", False))

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
        use_full_train=use_full_train,
    )
    datamodule.setup()
    print(
        f"[train] 数据已就绪 | train={len(datamodule.train_dataset)} val={len(datamodule.val_dataset)} "
        f"| num_workers={config['data']['num_workers']}",
        flush=True,
    )

    print("[train] 构建模型（若首次运行可能下载预训练权重，请稍候）…", flush=True)
    model = FaceClassifierModule(
        model_family=config["model"]["family"],
        backbone_name=config["model"]["backbone_name"],
        num_classes=num_classes,
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
        learning_rate=config["train"]["learning_rate"],
        backbone_learning_rate=config["train"].get("backbone_learning_rate"),
        weight_decay=config["train"]["weight_decay"],
        scheduler_name=config["train"]["scheduler"],
        max_epochs=config["train"]["max_epochs"],
        monitor_metric=monitor_config.metric,
        pretrained_repo_id=config["model"].get("pretrained_repo_id"),
        pretrained_checkpoint_path=config["model"].get("pretrained_checkpoint_path"),
        freeze_backbone=config["model"].get("freeze_backbone", False),
        unfreeze_last_stage=config["model"].get("unfreeze_last_stage", False),
        unfreeze_stage_count=config["model"].get("unfreeze_stage_count", 0),
        unfreeze_cvlface_norm=config["model"].get("unfreeze_cvlface_norm", True),
        unfreeze_cvlface_feature=config["model"].get("unfreeze_cvlface_feature", True),
        iresnet_finetune_mode=config["model"].get("iresnet_finetune_mode", "bn_only"),
        loss_name=loss_config.get("name", "cross_entropy"),
        loss_target_labels=loss_config.get("target_labels"),
        arcface_scale=loss_config.get("arcface_scale", 30.0),
        arcface_margin=loss_config.get("arcface_margin", 0.5),
        cosface_scale=loss_config.get("cosface_scale", loss_config.get("arcface_scale", 30.0)),
        cosface_margin=loss_config.get("cosface_margin", 0.35),
        label_smoothing=float(loss_config.get("label_smoothing", 0.0)),
        ovr_threshold=float(loss_config.get("ovr_threshold", 0.5)),
        supcon_weight=float(loss_config.get("supcon_weight", 0.0)),
        supcon_temperature=float(loss_config.get("supcon_temperature", 0.1)),
    )
    print("[train] 模型构建完成。", flush=True)

    if config["model"].get("weight_imprinting", False):
        if config["model"]["family"] != "arcface_iresnet":
            raise ValueError("model.weight_imprinting 当前仅支持 family: arcface_iresnet")
        dev = _resolve_train_device(config)
        model.to(dev)
        loss_name = loss_config.get("name", "cross_entropy")
        if loss_name == "cosface":
            imprint_cosine_classifier_weight_from_loader(
                model.model,
                model.cosface_head.weight,
                datamodule.train_dataloader(),
                dev,
                num_classes=num_classes,
            )
            print(
                f"[train] Weight imprinting 已在 {dev} 上完成（各类原型写入 cosface_head.weight）。",
                flush=True,
            )
        else:
            imprint_linear_head_from_loader(
                model.model,
                datamodule.train_dataloader(),
                dev,
                num_classes=num_classes,
            )
            print(
                f"[train] Weight imprinting 已在 {dev} 上完成（各类原型写入 Linear.weight）。",
                flush=True,
            )

    progress_bar = AsciiTQDMProgressBar(refresh_rate=1)
    shadow_probe_callback = None
    shadow_probe_config = resolve_shadow_probe_config(config, output_root)
    if shadow_probe_config is not None:
        shadow_probe_callback = ShadowProbeEpochCallback(
            probe_config=shadow_probe_config,
            image_size=int(config["data"]["face_size"]),
            normalization=str(config["data"]["normalization"]),
            batch_size=int(config["train"]["batch_size"]),
            num_workers=int(config["data"]["num_workers"]),
        )
    if use_full_train:
        checkpoint_callback = ModelCheckpoint(
            dirpath=output_root / "checkpoints",
            save_last=True,
            save_top_k=0,
        )
        callbacks = [checkpoint_callback, progress_bar]
    else:
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
        callbacks = [checkpoint_callback, early_stopping, progress_bar]
    if shadow_probe_callback is not None:
        callbacks.append(shadow_probe_callback)

    accumulate = int(config["train"].get("accumulate_grad_batches", 1))
    trainer_kwargs: dict = dict(
        max_epochs=config["train"]["max_epochs"],
        accelerator=config["train"]["accelerator"],
        devices=config["train"]["devices"],
        precision=config["train"]["precision"],
        logger=logger,
        callbacks=callbacks,
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=False,
        log_every_n_steps=1,
        accumulate_grad_batches=accumulate,
    )
    clip = config["train"].get("gradient_clip_val")
    if clip is not None:
        trainer_kwargs["gradient_clip_val"] = float(clip)
    trainer = L.Trainer(**trainer_kwargs)
    print(
        f"[train] 开始 fit | max_epochs={config['train']['max_epochs']} "
        f"| accelerator={config['train']['accelerator']} devices={config['train']['devices']} "
        f"| accumulate_grad_batches={accumulate} "
        f"| monitor={monitor_config.metric} ({monitor_config.mode})",
        flush=True,
    )
    trainer.fit(model, datamodule=datamodule)

    if use_full_train:
        last_path = checkpoint_callback.last_model_path
        if not last_path:
            last_path = str(output_root / "checkpoints" / "last.ckpt")
        best_model_path = last_path
        best_metric_value = None
    else:
        best_model_path = checkpoint_callback.best_model_path
        best_metric_value = (
            float(checkpoint_callback.best_model_score.cpu().item())
            if checkpoint_callback.best_model_score is not None
            else None
        )

    metrics = {
        "best_model_path": best_model_path,
        "use_full_train": use_full_train,
        f"best_{monitor_config.metric}": best_metric_value,
    }
    (output_root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    append_registry_row(
        project_path("reports", "experiments", "registry.csv"),
        {
            "experiment_name": config["experiment_name"],
            "stage": "train",
            "best_model_path": best_model_path,
            "best_val_acc": metrics.get("best_val_acc"),
            "config_path": config["config_path"],
        },
    )
    print(f"训练完成，最佳模型: {best_model_path}")


if __name__ == "__main__":
    main()
