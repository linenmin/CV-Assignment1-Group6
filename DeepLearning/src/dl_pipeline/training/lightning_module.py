from __future__ import annotations

import lightning as L
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from dl_pipeline.models.classifier import ArcFaceHead, build_classifier


class FaceClassifierModule(L.LightningModule):
    def __init__(
        self,
        model_family: str,
        backbone_name: str,
        num_classes: int,
        pretrained: bool,
        dropout: float,
        learning_rate: float,
        backbone_learning_rate: float | None,
        weight_decay: float,
        scheduler_name: str,
        max_epochs: int,
        monitor_metric: str = "val_acc",
        pretrained_repo_id: str | None = None,
        pretrained_checkpoint_path: str | None = None,
        freeze_backbone: bool = False,
        unfreeze_last_stage: bool = False,
        unfreeze_stage_count: int = 0,
        unfreeze_cvlface_norm: bool = True,
        unfreeze_cvlface_feature: bool = True,
        loss_name: str = "cross_entropy",
        loss_target_labels: list[int] | None = None,
        arcface_scale: float = 30.0,
        arcface_margin: float = 0.5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = build_classifier(
            model_family=model_family,
            backbone_name=backbone_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            pretrained_repo_id=pretrained_repo_id,
            pretrained_checkpoint_path=pretrained_checkpoint_path,
            freeze_backbone=freeze_backbone,
            unfreeze_last_stage=unfreeze_last_stage,
            unfreeze_stage_count=unfreeze_stage_count,
            unfreeze_cvlface_norm=unfreeze_cvlface_norm,
            unfreeze_cvlface_feature=unfreeze_cvlface_feature,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.loss_name = loss_name
        self.loss_target_labels = list(loss_target_labels or [])
        self.loss_target_label_to_index = {
            int(label): index for index, label in enumerate(self.loss_target_labels)
        }

        if self.loss_name == "arcface":
            if len(self.loss_target_labels) < 2:
                raise ValueError("ArcFace 模式至少需要两个 target labels。")
            self.arcface_head = ArcFaceHead(
                in_features=self._resolve_feature_dim(),
                num_classes=len(self.loss_target_labels),
                scale=arcface_scale,
                margin=arcface_margin,
            )
            self.val_target_acc = MulticlassAccuracy(num_classes=len(self.loss_target_labels))
            self.val_target_f1 = MulticlassF1Score(
                num_classes=len(self.loss_target_labels),
                average="macro",
            )
        else:
            self.val_acc = MulticlassAccuracy(num_classes=num_classes)
            self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

    def forward(self, images):
        return self.model(images)

    def extract_features(self, images):
        if hasattr(self.model, "extract_features"):
            return self.model.extract_features(images)
        if hasattr(self.model, "forward_features"):
            features = self.model.forward_features(images)
            if features.ndim > 2:
                features = features.mean(dim=(-2, -1))
            return features
        raise ValueError("当前模型不支持提取 embedding 特征。")

    def _resolve_feature_dim(self) -> int:
        classifier = getattr(self.model, "classifier", None)
        if isinstance(classifier, nn.Sequential) and len(classifier) > 0 and hasattr(classifier[-1], "in_features"):
            return int(classifier[-1].in_features)
        if hasattr(classifier, "in_features"):
            return int(classifier.in_features)
        raise ValueError("无法解析模型 feature dim，不能构建 ArcFace head。")

    def _select_arcface_targets(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.loss_target_labels:
            raise ValueError("ArcFace 模式缺少 target labels。")

        selected_feature_list = []
        selected_label_list = []
        for raw_label, mapped_label in self.loss_target_label_to_index.items():
            current_mask = labels == raw_label
            if current_mask.any():
                selected_feature_list.append(features[current_mask])
                selected_label_list.append(
                    torch.full(
                        (int(current_mask.sum().item()),),
                        mapped_label,
                        device=labels.device,
                        dtype=torch.long,
                    )
                )

        if not selected_feature_list:
            return (
                torch.empty((0, features.shape[1]), device=features.device, dtype=features.dtype),
                torch.empty((0,), device=labels.device, dtype=torch.long),
            )

        return torch.cat(selected_feature_list, dim=0), torch.cat(selected_label_list, dim=0)

    def _compute_loss_and_predictions(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if self.loss_name != "arcface":
            logits = self(images)
            loss = self.criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)
            return loss, preds, labels, int(labels.shape[0])

        features = self.extract_features(images)
        target_features, target_labels = self._select_arcface_targets(features, labels)
        if target_labels.numel() == 0:
            zero_loss = features.sum() * 0.0
            empty_preds = torch.empty((0,), device=labels.device, dtype=torch.long)
            return zero_loss, empty_preds, target_labels, 0

        logits = self.arcface_head(target_features, target_labels)
        loss = self.criterion(logits, target_labels)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, target_labels, int(target_labels.shape[0])

    def training_step(self, batch, batch_idx):
        images, labels = batch
        loss, _, _, batch_size = self._compute_loss_and_predictions(images, labels)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=max(batch_size, 1),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        loss, preds, metric_labels, batch_size = self._compute_loss_and_predictions(images, labels)
        if self.loss_name == "arcface":
            if batch_size == 0:
                return
            self.val_target_acc.update(preds, metric_labels)
            self.val_target_f1.update(preds, metric_labels)
            self.log("val_target_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
            return

        self.val_acc.update(preds, metric_labels)
        self.val_f1.update(preds, metric_labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

    def on_validation_epoch_end(self):
        if self.loss_name == "arcface":
            self.log("val_target_acc", self.val_target_acc.compute(), prog_bar=True)
            self.log("val_target_f1", self.val_target_f1.compute(), prog_bar=True)
            self.val_target_acc.reset()
            self.val_target_f1.reset()
            return

        self.log("val_acc", self.val_acc.compute(), prog_bar=True)
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        self.val_acc.reset()
        self.val_f1.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images, sample_ids = batch
        logits = self(images)
        preds = torch.argmax(logits, dim=1)
        return {"ids": sample_ids, "preds": preds}

    def _build_optimizer_param_groups(self):
        backbone_learning_rate = self.hparams.backbone_learning_rate
        if backbone_learning_rate is None or not hasattr(self.model, "backbone"):
            return [
                {
                    "params": [parameter for parameter in self.parameters() if parameter.requires_grad],
                    "lr": self.hparams.learning_rate,
                }
            ]

        backbone_params = [parameter for parameter in self.model.backbone.parameters() if parameter.requires_grad]
        if not backbone_params:
            return [
                {
                    "params": [parameter for parameter in self.parameters() if parameter.requires_grad],
                    "lr": self.hparams.learning_rate,
                }
            ]

        backbone_param_ids = {id(parameter) for parameter in backbone_params}
        head_params = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in backbone_param_ids
        ]
        return [
            {"params": backbone_params, "lr": backbone_learning_rate},
            {"params": head_params, "lr": self.hparams.learning_rate},
        ]

    def configure_optimizers(self):
        optimizer = AdamW(
            self._build_optimizer_param_groups(),
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.scheduler_name == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        if self.hparams.scheduler_name == "plateau":
            scheduler_mode = "min" if self.hparams.monitor_metric == "val_loss" else "max"
            scheduler = ReduceLROnPlateau(optimizer, mode=scheduler_mode, factor=0.5, patience=2)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hparams.monitor_metric,
                },
            }
        return optimizer
