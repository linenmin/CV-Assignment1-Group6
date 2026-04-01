from __future__ import annotations

import lightning as L
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from dl_pipeline.models.classifier import build_classifier


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
        pretrained_repo_id: str | None = None,
        freeze_backbone: bool = False,
        unfreeze_last_stage: bool = False,
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
            freeze_backbone=freeze_backbone,
            unfreeze_last_stage=unfreeze_last_stage,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, labels)
        self.val_f1.update(preds, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
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
            scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_acc"}}
        return optimizer
