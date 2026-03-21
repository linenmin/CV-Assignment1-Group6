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
        backbone_name: str,
        num_classes: int,
        pretrained: bool,
        dropout: float,
        learning_rate: float,
        weight_decay: float,
        scheduler_name: str,
        max_epochs: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = build_classifier(backbone_name, num_classes, pretrained, dropout)
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

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.scheduler_name == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        if self.hparams.scheduler_name == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_acc"}}
        return optimizer
