from __future__ import annotations

import timm
import torch.nn as nn

from dl_pipeline.models.cvlface import load_cvlface_backbone


class CVLFaceClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int,
        dropout: float,
        freeze_backbone: bool,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        if isinstance(features, tuple):
            features = features[0]
        return self.classifier(features)


def build_classifier(
    model_family: str,
    backbone_name: str,
    num_classes: int,
    pretrained: bool,
    dropout: float,
    pretrained_repo_id: str | None = None,
    freeze_backbone: bool = False,
) -> nn.Module:
    if model_family == "cvlface":
        if not pretrained_repo_id:
            raise ValueError("cvlface 模型需要提供 pretrained_repo_id。")
        backbone = load_cvlface_backbone(pretrained_repo_id)
        return CVLFaceClassifier(
            backbone=backbone,
            feature_dim=512,
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )

    return timm.create_model(
        backbone_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
    )
