from __future__ import annotations

import timm
import torch.nn as nn


def build_classifier(
    backbone_name: str,
    num_classes: int,
    pretrained: bool,
    dropout: float,
) -> nn.Module:
    return timm.create_model(
        backbone_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
    )
